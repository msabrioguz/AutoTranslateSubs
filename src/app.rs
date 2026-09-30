use crate::models::{AppConfig, Language, SubtitleFile};
use crate::subtitle_parser::{find_subtitle_files, parse_subtitle_file, write_subtitle_file};
use crate::ollama_client::OllamaClient;
use eframe::egui;
use egui::{RichText, ScrollArea, Ui, Vec2};
use rfd::FileDialog;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Accent color: selections, links, info toasts.
const ACCENT: egui::Color32 = egui::Color32::from_rgb(63, 125, 237);
/// Fill for affirmative actions (start button).
const SUCCESS_FILL: egui::Color32 = egui::Color32::from_rgb(40, 167, 69);
/// Fill for destructive actions (stop button).
const DANGER_FILL: egui::Color32 = egui::Color32::from_rgb(210, 52, 60);
/// Readable error text on dark backgrounds.
const ERR_TEXT: egui::Color32 = egui::Color32::from_rgb(255, 99, 99);
/// Readable success text on dark backgrounds.
const OK_TEXT: egui::Color32 = egui::Color32::from_rgb(80, 200, 120);

const TOAST_DURATION: Duration = Duration::from_secs(5);
const MAX_TOASTS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ToastKind {
    Info,
    Success,
    Error,
}

#[derive(Debug, Clone)]
struct Toast {
    message: String,
    kind: ToastKind,
    created: Instant,
}

#[derive(Debug, Clone, PartialEq)]
enum AppTab {
    Files,
    Translation,
    Settings,
    About,
    Dependencies,
}

#[derive(Debug, Clone)]
struct TranslationJob {
    file_path: PathBuf,
    subtitle_file: SubtitleFile,
    status: JobStatus,
    progress: f32,
    message: String,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct JobSummary {
    file_path: PathBuf,
    entry_count: usize,
    status: JobStatus,
    progress: f32,
    message: String,
    error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum JobStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Skipped,
}

type SharedJobs = Arc<Mutex<Vec<TranslationJob>>>;

fn is_language_code(segment: &str) -> bool {
    matches!(
        Language::from_code(segment),
        Some(lang) if lang != Language::Auto
    )
}

/// Naming standard: `<base>.<lang>.<ext>` (e.g. `003 Setting up the input.en.vtt`).
///
/// Splits a file stem into `(base, language_code)` when it ends with a
/// **non-standard** language suffix like `_en` or `-en` (`input_en` ->
/// `("input", "en")`). Returns `None` for anything else, so `green` or
/// `kitchen` are never mistaken for language markers.
fn split_nonstandard_lang_suffix(stem: &str) -> Option<(&str, &'static str)> {
    let (base, suffix) = stem.rsplit_once(['_', '-'])?;
    if base.is_empty() {
        return None;
    }
    let lang = Language::from_code(suffix)?;
    if lang == Language::Auto {
        return None;
    }
    Some((base, lang.code()))
}

/// Returns why `input_path` is considered already translated for `target_lang`,
/// or `None` when it still needs translation.
///
/// A file is considered translated when either:
/// 1. its name already ends with the target language code
///    (`video.tr.srt` with target Turkish), or
/// 2. the output file for it already exists on disk and is not empty
///    (`video.tr.srt` present for input `video.en.srt`).
fn already_translated_reason(input_path: &Path, target_lang: Language) -> Option<&'static str> {
    if target_lang == Language::Auto {
        return None;
    }

    // Rule 1: the file name already carries the target language code
    // (standard `video.tr.srt` or non-standard `video_tr.srt` / `video-tr.srt`)
    if let Some(stem) = input_path.file_stem().map(|s| s.to_string_lossy()) {
        if let Some((_, last)) = stem.rsplit_once('.') {
            if last.eq_ignore_ascii_case(target_lang.code()) {
                return Some("Dosya adında hedef dil kodu zaten var");
            }
        }
        if let Some((_, code)) = split_nonstandard_lang_suffix(&stem) {
            if code == target_lang.code() {
                return Some("Dosya adında hedef dil kodu zaten var");
            }
        }
    }

    // Rule 2: the translated output already exists on disk
    let output = AutoTranslateApp::generate_output_path(input_path, target_lang);
    if std::fs::metadata(&output).map(|m| m.len() > 0).unwrap_or(false) {
        return Some("Çıktı dosyası diskte mevcut");
    }

    None
}

/// Applies the "skip already translated" policy to prepared jobs.
///
/// - `Pending` jobs whose output already exists become `Skipped`.
/// - `Skipped` jobs become `Pending` again when the policy no longer applies
///   (checkbox turned off, target language changed, output deleted).
///
/// Returns `(newly_skipped, resumed)` counts.
fn apply_skip_policy_to(
    jobs: &mut [TranslationJob],
    target_lang: Language,
    skip_enabled: bool,
) -> (usize, usize) {
    let mut skipped = 0;
    let mut resumed = 0;

    for job in jobs.iter_mut() {
        let reason = if skip_enabled {
            already_translated_reason(&job.file_path, target_lang)
        } else {
            None
        };

        match (job.status.clone(), reason) {
            (JobStatus::Pending, Some(reason)) => {
                job.status = JobStatus::Skipped;
                job.message = reason.to_string();
                skipped += 1;
            }
            (JobStatus::Skipped, Some(reason)) => {
                job.message = reason.to_string();
            }
            (JobStatus::Skipped, None) => {
                job.status = JobStatus::Pending;
                job.message.clear();
                resumed += 1;
            }
            _ => {}
        }
    }

    (skipped, resumed)
}

/// Renames the original subtitle file to the standard `name.<source_code>.<ext>`
/// form (e.g. `video.srt` -> `video.en.srt`) after a successful translation.
/// A non-standard suffix with the same code is normalized instead of appended
/// (`input_en.vtt` -> `input.en.vtt`).
///
/// Returns `Ok(Some(new_path))` when the file was renamed, `Ok(None)` when no
/// rename is needed or possible (Auto language, the name is already standard,
/// or the target name is already taken).
fn rename_original_with_code(
    original: &Path,
    source_lang: Language,
) -> std::io::Result<Option<PathBuf>> {
    if source_lang == Language::Auto {
        return Ok(None);
    }
    let code = source_lang.code();

    let Some(stem) = original.file_stem().map(|s| s.to_string_lossy().to_string()) else {
        return Ok(None);
    };
    let Some(ext) = original.extension().map(|e| e.to_string_lossy().to_string()) else {
        return Ok(None);
    };

    // Already standard: "video.en.srt"
    if stem.rsplit('.').next() == Some(code) {
        return Ok(None);
    }

    // Non-standard suffix carrying the source code: normalize it to the
    // standard form ("input_en.vtt" -> "input.en.vtt"). A suffix with a
    // different code is kept as part of the base name and the code is
    // appended as usual ("input_en.srt" + fr -> "input_en.fr.srt").
    let base = match split_nonstandard_lang_suffix(&stem) {
        Some((base, suffix_code)) if suffix_code == code => base,
        _ => &stem,
    };

    let mut new_path = original.to_path_buf();
    new_path.set_file_name(format!("{}.{}.{}", base, code, ext));
    if new_path == original || new_path.exists() {
        return Ok(None);
    }

    std::fs::rename(original, &new_path)?;
    Ok(Some(new_path))
}

#[derive(Debug, Clone)]
struct ProgressUpdate {
    job_index: usize,
    progress: f32,
    message: String,
}

/// Applies the app-wide visual theme: dark visuals with a custom accent,
/// rounded widgets and slightly roomier spacing.
fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    {
        let visuals = &mut style.visuals;
        *visuals = egui::Visuals::dark();
        visuals.hyperlink_color = ACCENT;
        visuals.selection.bg_fill = ACCENT;
        visuals.selection.stroke = egui::Stroke::new(1.0_f32, ACCENT);
        visuals.error_fg_color = ERR_TEXT;
        visuals.window_rounding = egui::Rounding::same(8.0);
        visuals.menu_rounding = egui::Rounding::same(6.0);
        for widget in [
            &mut visuals.widgets.noninteractive,
            &mut visuals.widgets.inactive,
            &mut visuals.widgets.hovered,
            &mut visuals.widgets.active,
            &mut visuals.widgets.open,
        ] {
            widget.rounding = egui::Rounding::same(6.0);
        }
    }
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 4.0);
    ctx.set_style(style);
}

/// Opens the log file with the operating system's default application.
fn open_log_file(path: &Path) {
    let mut command = if cfg!(windows) {
        let mut command = std::process::Command::new("cmd");
        command.args(["/C", "start", ""]);
        command
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else {
        std::process::Command::new("xdg-open")
    };
    let _ = command.arg(path).spawn();
}

enum InitUpdate {
    Models(Result<Vec<String>, String>),
    Connection(ConnectionStatus),
}

pub struct AutoTranslateApp {
    config: AppConfig,
    config_path: PathBuf,
    log_path: PathBuf,
    
    selected_files: Vec<PathBuf>,
    translation_jobs: SharedJobs,
    current_job_index: Option<usize>,
    
    tab: AppTab,
    
    ollama_client: Option<OllamaClient>,
    models_loaded: bool,
    connection_status: ConnectionStatus,
    
    translation_thread: Option<thread::JoinHandle<()>>,
    stop_translation: Arc<Mutex<bool>>,
    
    show_file_dialog: bool,
    show_folder_dialog: bool,
    
    log_messages: Vec<String>,
    log_sender: std::sync::mpsc::Sender<String>,
    log_receiver: std::sync::mpsc::Receiver<String>,
    
    progress_sender: std::sync::mpsc::Sender<ProgressUpdate>,
    progress_receiver: std::sync::mpsc::Receiver<ProgressUpdate>,

    init_sender: std::sync::mpsc::Sender<InitUpdate>,
    init_receiver: std::sync::mpsc::Receiver<InitUpdate>,

    toasts: Vec<Toast>,
    models_loading: bool,
    connection_checking: bool,
    announce_model_load: bool,
    announce_connection: bool,
    confirm_clear: bool,
    preview_status: Option<(String, ToastKind)>,
}

#[derive(Debug, Clone, PartialEq)]
enum ConnectionStatus {
    Unknown,
    Connected,
    Disconnected,
    Error(String),
}

impl Default for AutoTranslateApp {
    fn default() -> Self {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("auto-translate-subs");
        std::fs::create_dir_all(&config_dir).ok();
        let config_path = config_dir.join("config.json");
        let log_path = crate::logger::init(&config_dir.join("logs"));
        
        let config = if config_path.exists() {
            let loaded = std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok());
            match &loaded {
                Some(_) => crate::logger::log(
                    crate::logger::Level::Info,
                    "app",
                    &format!("config loaded: {}", config_path.display()),
                ),
                None => crate::logger::log(
                    crate::logger::Level::Error,
                    "app",
                    &format!("config unreadable, using defaults: {}", config_path.display()),
                ),
            }
            loaded.unwrap_or_default()
        } else {
            crate::logger::log(
                crate::logger::Level::Info,
                "app",
                "config not found, using defaults",
            );
            AppConfig::default()
        };
        
        let (log_sender, log_receiver) = std::sync::mpsc::channel();
        
        let (progress_sender, progress_receiver) = std::sync::mpsc::channel();
        
        let (init_sender, init_receiver) = std::sync::mpsc::channel();
        
        Self {
            config,
            config_path,
            log_path,
            selected_files: Vec::new(),
            translation_jobs: Arc::new(Mutex::new(Vec::new())),
            current_job_index: None,
            tab: AppTab::Files,
            ollama_client: None,
            models_loaded: false,
            connection_status: ConnectionStatus::Unknown,
            translation_thread: None,
            stop_translation: Arc::new(Mutex::new(false)),
            show_file_dialog: false,
            show_folder_dialog: false,
            log_messages: Vec::new(),
            log_sender,
            log_receiver,
            progress_sender,
            progress_receiver,
            init_sender,
            init_receiver,
            toasts: Vec::new(),
            models_loading: false,
            connection_checking: false,
            announce_model_load: false,
            announce_connection: false,
            confirm_clear: false,
            preview_status: None,
        }
    }
}

impl AutoTranslateApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_theme(&cc.egui_ctx);
        let mut app = Self::default();
        app.init_ollama_client();
        app.load_models();
        app.check_connection();
        app
    }
    
    fn init_ollama_client(&mut self) {
        match OllamaClient::new(&self.config) {
            Ok(client) => self.ollama_client = Some(client),
            Err(e) => self.log(&format!("Ollama client error: {}", e)),
        }
    }
    
    fn sync_client_config(&mut self) {
        if let Some(client) = &mut self.ollama_client {
            client.set_config(&self.config);
        }
    }

    fn load_models(&mut self) {
        self.sync_client_config();
        let Some(client) = self.ollama_client.clone() else {
            self.models_loading = false;
            self.announce_model_load = false;
            self.notify(
                ToastKind::Error,
                "Ollama istemcisi hazır değil; Ayarlar bölümünü kontrol edin".to_string(),
            );
            return;
        };
        let tx = self.init_sender.clone();
        self.models_loading = true;
        self.log("Loading models...");
        thread::spawn(move || {
            let _ = tx.send(InitUpdate::Models(
                client.fetch_models().map_err(|e| e.to_string()),
            ));
        });
    }

    fn check_connection(&mut self) {
        self.sync_client_config();
        let Some(client) = self.ollama_client.clone() else {
            self.connection_checking = false;
            self.announce_connection = false;
            return;
        };
        self.connection_checking = true;
        let tx = self.init_sender.clone();
        thread::spawn(move || {
            let status = match client.test_connection() {
                Ok(true) => ConnectionStatus::Connected,
                Ok(false) => ConnectionStatus::Disconnected,
                Err(e) => ConnectionStatus::Error(e.to_string()),
            };
            let _ = tx.send(InitUpdate::Connection(status));
        });
    }
    
    fn save_config(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.config) {
            std::fs::write(&self.config_path, json).ok();
        }
    }
    
    fn log(&mut self, msg: &str) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        let formatted = format!("[{}] {}", timestamp, msg);
        self.log_messages.push(formatted);
        if self.log_messages.len() > 100 {
            self.log_messages.remove(0);
        }
        crate::logger::log(crate::logger::level_for(msg), "app", msg);
    }

    /// Logs `message` and shows it as a toast notification. A duplicate of
    /// the currently visible toast only refreshes its timeout.
    fn notify(&mut self, kind: ToastKind, message: String) {
        self.log(&message);
        if let Some(last) = self.toasts.last_mut() {
            if last.kind == kind && last.message == message {
                last.created = Instant::now();
                return;
            }
        }
        self.toasts.push(Toast {
            message,
            kind,
            created: Instant::now(),
        });
        while self.toasts.len() > MAX_TOASTS {
            self.toasts.remove(0);
        }
    }

    /// Returns `(finished, total, overall)`, where `finished` counts jobs
    /// that no longer need work and `overall` is the 0..1 fraction of work
    /// done, including partial progress of the job in flight.
    fn progress_stats(&self) -> (usize, usize, f32) {
        let jobs = self.translation_jobs.lock().unwrap();
        let total = jobs.len();
        if total == 0 {
            return (0, 0, 0.0);
        }
        let mut finished = 0usize;
        let mut work = 0.0f32;
        for job in jobs.iter() {
            match job.status {
                JobStatus::Completed
                | JobStatus::Skipped
                | JobStatus::Failed
                | JobStatus::Cancelled => {
                    finished += 1;
                    work += 1.0;
                }
                JobStatus::InProgress => work += job.progress.clamp(0.0, 1.0),
                JobStatus::Pending => {}
            }
        }
        (finished, total, work / total as f32)
    }

    /// Draws the toast notifications in the bottom-right corner of the UI.
    /// Toasts disappear on their own or when clicked.
    fn render_toasts(ctx: &egui::Context, toasts: &mut Vec<Toast>) {
        toasts.retain(|t| t.created.elapsed() < TOAST_DURATION);
        if toasts.is_empty() {
            return;
        }
        let mut dismissed: Vec<usize> = Vec::new();
        egui::Area::new(egui::Id::new("toast_area"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -56.0))
            .show(ctx, |ui| {
                for (idx, toast) in toasts.iter().enumerate() {
                    let color = match toast.kind {
                        ToastKind::Info => ACCENT,
                        ToastKind::Success => OK_TEXT,
                        ToastKind::Error => ERR_TEXT,
                    };
                    let icon = match toast.kind {
                        ToastKind::Info => "ℹ️",
                        ToastKind::Success => "✅",
                        ToastKind::Error => "❌",
                    };
                    egui::Frame::popup(ui.style())
                        .stroke(egui::Stroke::new(1.0_f32, color))
                        .rounding(egui::Rounding::same(6.0))
                        .show(ui, |ui| {
                            ui.set_max_width(360.0);
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(icon).color(color));
                                ui.label(&toast.message);
                            });
                            let response = ui
                                .interact(
                                    ui.min_rect(),
                                    ui.id().with(("toast", idx)),
                                    egui::Sense::click(),
                                )
                                .on_hover_text("Kapatmak için tıklayın");
                            if response.clicked() {
                                dismissed.push(idx);
                            }
                        });
                    ui.add_space(6.0);
                }
            });
        for idx in dismissed.into_iter().rev() {
            toasts.remove(idx);
        }
    }
    
    fn add_files(&mut self, files: Vec<PathBuf>) {
        for file in files {
            let is_subtitle = file.extension().is_some_and(|ext| {
                let ext = ext.to_string_lossy();
                ext.eq_ignore_ascii_case("srt") || ext.eq_ignore_ascii_case("vtt")
            });
            if is_subtitle && !self.selected_files.contains(&file) {
                crate::logger::log(
                    crate::logger::Level::Info,
                    "files",
                    &format!("selected: {}", file.display()),
                );
                self.selected_files.push(file);
            }
        }
    }

    fn add_folder(&mut self, folder: PathBuf) {
        match find_subtitle_files(&folder) {
            Ok(files) => {
                let count = files.len();
                self.add_files(files);
                self.log(&format!("Found {} subtitle files in folder", count));
            }
            Err(e) => {
                self.notify(ToastKind::Error, format!("Klasör taranamadı: {}", e));
            }
        }
    }
    
    /// Applies file renames performed by the translation thread to the file
    /// selection, so the list stays in sync with the files on disk.
    fn sync_renamed_selection(&mut self) {
        let renamed: Vec<(usize, PathBuf)> = {
            let jobs = self.translation_jobs.lock().unwrap();
            if jobs.len() != self.selected_files.len() {
                return;
            }
            self.selected_files
                .iter()
                .zip(jobs.iter())
                .enumerate()
                .filter(|(_, (selected, job))| *selected != &job.file_path)
                .map(|(idx, (_, job))| (idx, job.file_path.clone()))
                .collect()
        };

        for (idx, new_path) in renamed {
            self.log(&format!("Source file renamed to: {}", new_path.display()));
            self.selected_files[idx] = new_path;
        }
    }

    /// Switches tabs. The Translation tab redirects to Files when no file is
    /// selected, and refreshes the job list automatically when the selection
    /// changed since the last preparation.
    fn select_tab(&mut self, tab: AppTab) {
        crate::logger::log(
            crate::logger::Level::Info,
            "ui",
            &format!("tab -> {:?}", tab),
        );
        if tab == AppTab::Translation {
            if self.selected_files.is_empty() {
                self.tab = AppTab::Files;
                self.notify(
                    ToastKind::Info,
                    "Dosya seçilmedi; önce 'Dosyalar' sekmesinden dosya ekleyin".to_string(),
                );
                return;
            }
            self.tab = AppTab::Translation;
            if !self.jobs_match_selection() {
                self.prepare_translation_jobs();
            }
            return;
        }
        self.tab = tab;
    }

    /// True when the prepared jobs correspond exactly to `selected_files`.
    fn jobs_match_selection(&self) -> bool {
        let jobs = self.translation_jobs.lock().unwrap();
        jobs.len() == self.selected_files.len()
            && jobs
                .iter()
                .zip(self.selected_files.iter())
                .all(|(job, file)| &job.file_path == file)
    }

    fn prepare_translation_jobs(&mut self) {
        if self.is_running() {
            self.log("Translation is running; wait or stop it first");
            return;
        }

        let files = self.selected_files.clone();
        let mut new_jobs = Vec::new();
        let mut parse_errors = Vec::new();

        for file_path in files {
            match parse_subtitle_file(
                &file_path,
                self.config.source_language,
                self.config.target_language,
            ) {
                Ok(subtitle_file) => new_jobs.push(TranslationJob {
                    file_path,
                    subtitle_file,
                    status: JobStatus::Pending,
                    progress: 0.0,
                    message: String::new(),
                    error: None,
                }),
                Err(e) => {
                    let msg = format!("Failed to parse {}: {}", file_path.display(), e);
                    parse_errors.push(msg.clone());
                    let path = file_path.clone();
                    new_jobs.push(TranslationJob {
                        file_path,
                        subtitle_file: SubtitleFile {
                            path,
                            entries: Vec::new(),
                            source_language: self.config.source_language,
                            target_language: self.config.target_language,
                        },
                        status: JobStatus::Failed,
                        progress: 0.0,
                        message: String::new(),
                        error: Some(msg),
                    });
                }
            }
        }

        let job_count = new_jobs.len();
        {
            let mut jobs = self.translation_jobs.lock().unwrap();
            jobs.clear();
            jobs.extend(new_jobs);
        }

        let (skipped, resumed) = self.apply_skip_policy();
        if skipped > 0 {
            self.log(&format!("Skipped {} already translated file(s)", skipped));
        }
        if resumed > 0 {
            self.log(&format!("Resumed {} previously skipped file(s)", resumed));
        }

        let parse_error_count = parse_errors.len();
        for err in parse_errors {
            self.log(&err);
        }
        if parse_error_count > 0 {
            self.notify(
                ToastKind::Error,
                format!(
                    "{} dosya çözümlenemedi; ayrıntılar uygulama günlüğünde",
                    parse_error_count
                ),
            );
        }
        self.log(&format!("Prepared {} files for translation", job_count));
    }

    /// Applies the "skip already translated" policy to the prepared jobs.
    /// Returns `(newly_skipped, resumed)` counts.
    fn apply_skip_policy(&mut self) -> (usize, usize) {
        let target_lang = self.config.target_language;
        let skip_enabled = self.config.skip_translated;
        let mut jobs = self.translation_jobs.lock().unwrap();
        apply_skip_policy_to(&mut jobs, target_lang, skip_enabled)
    }
    
    fn start_translation(&mut self) {
        if self.is_running() {
            self.log("Translation is already running");
            return;
        }

        let needs_prepare = {
            let jobs = self.translation_jobs.lock().unwrap();
            jobs.is_empty() || jobs.len() != self.selected_files.len()
        };
        if needs_prepare {
            self.prepare_translation_jobs();
        }

        if self.translation_jobs.lock().unwrap().is_empty() {
            self.log("No files to translate");
            return;
        }

        // Re-check the skip policy at start time: outputs may have appeared
        // since the jobs were prepared, or the policy/target may have changed.
        let (skipped, resumed) = self.apply_skip_policy();
        if skipped > 0 {
            self.log(&format!("Skipped {} already translated file(s)", skipped));
        }
        if resumed > 0 {
            self.log(&format!("Resumed {} previously skipped file(s)", resumed));
        }
        let has_pending = self
            .translation_jobs
            .lock()
            .unwrap()
            .iter()
            .any(|j| j.status == JobStatus::Pending);
        if !has_pending {
            self.log("Nothing to translate: all files are already translated");
            return;
        }

        let pending_count = self
            .translation_jobs
            .lock()
            .unwrap()
            .iter()
            .filter(|j| j.status == JobStatus::Pending)
            .count();
        self.log(&format!(
            "Translation requested: {} file(s), {} -> {}, model '{}'",
            pending_count,
            self.config.source_language.name(),
            self.config.target_language.name(),
            self.config.selected_model
        ));

        self.sync_client_config();
        let Some(client) = self.ollama_client.clone() else {
            self.log("Ollama client is not initialized; check settings");
            return;
        };

        let jobs = self.translation_jobs.clone();
        let stop_flag = self.stop_translation.clone();
        let source_lang = self.config.source_language;
        let target_lang = self.config.target_language;
        let completion_sound = self.config.completion_sound;
        let log_sender = self.log_sender.clone();
        let progress_sender = self.progress_sender.clone();

        if let Ok(mut guard) = stop_flag.lock() {
            *guard = false;
        }

        let handle = thread::spawn(move || {
            let send_log = |msg: &str| {
                let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
                let _ = log_sender.send(format!("[{}] {}", timestamp, msg));
                crate::logger::log(crate::logger::level_for(msg), "worker", msg);
            };

            let send_progress = |job_index: usize, progress: f32, message: String| {
                let _ = progress_sender.send(ProgressUpdate {
                    job_index,
                    progress,
                    message,
                });
            };

            send_log("Translation started");

            loop {
                let should_stop = stop_flag.lock().map(|g| *g).unwrap_or(false);
                if should_stop {
                    let mut jobs_guard = jobs.lock().unwrap();
                    for job in jobs_guard.iter_mut() {
                        if job.status == JobStatus::InProgress {
                            job.status = JobStatus::Cancelled;
                            job.message = String::new();
                        }
                    }
                    send_log("Translation stopped");
                    break;
                }

                // Claim next pending job (brief lock)
                let job_data = {
                    let mut jobs_guard = jobs.lock().unwrap();
                    let idx = jobs_guard
                        .iter()
                        .position(|j| j.status == JobStatus::Pending);
                    match idx {
                        Some(idx) => {
                            let job = &mut jobs_guard[idx];
                            job.status = JobStatus::InProgress;
                            job.progress = 0.05;
                            let texts: Vec<String> = job
                                .subtitle_file
                                .entries
                                .iter()
                                .map(|e| e.text.clone())
                                .collect();
                            let file_path = job.file_path.clone();
                            let subtitle_file = job.subtitle_file.clone();
                            Some((idx, texts, file_path, subtitle_file))
                        }
                        None => None,
                    }
                };

                let Some((idx, texts, file_path, subtitle_file)) = job_data else {
                    // No more pending jobs
                    break;
                };

                let job_started = std::time::Instant::now();
                let total_entries = texts.len();
                send_progress(
                    idx,
                    0.05,
                    format!("Starting translation of {} entries...", total_entries),
                );

                // Perform translation WITHOUT holding the lock
                let translations = client.translate_batch(
                    &texts,
                    source_lang,
                    target_lang,
                    |progress, message| {
                        send_progress(idx, progress, message);
                        !stop_flag.lock().map(|g| *g).unwrap_or(false)
                    },
                );

                match translations {
                    Ok((translations, stats)) => {
                        let file_name = file_path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        for diag in &stats.batch_diagnostics {
                            send_log(&format!("[diag] {}: {}", file_name, diag));
                        }
                        let untranslated = texts
                            .iter()
                            .zip(translations.iter())
                            .filter(|(src, tr)| !src.trim().is_empty() && tr.trim().is_empty())
                            .count();

                        let mut updated_subtitle = subtitle_file;
                        for (entry, translated) in
                            updated_subtitle.entries.iter_mut().zip(translations)
                        {
                            entry.translated_text = Some(translated);
                        }

                        send_progress(idx, 0.95, "Writing output file...".to_string());

                        let output_path = Self::generate_output_path(&file_path, target_lang);
                        send_log(&format!("Writing output to: {}", output_path.display()));
                        let write_result = write_subtitle_file(&updated_subtitle, &output_path);

                        // Verify the target file really exists on disk and is not empty
                        let verified = write_result.is_ok()
                            && std::fs::metadata(&output_path)
                                .map(|m| m.len() > 0)
                                .unwrap_or(false);

                        // Only after a successful verification the original file
                        // is renamed with the source language code
                        let renamed_original = if verified {
                            send_log(&format!("Verified output file: {}", output_path.display()));
                            match rename_original_with_code(&file_path, source_lang) {
                                Ok(Some(new_path)) => {
                                    send_log(&format!(
                                        "Renamed original to: {}",
                                        new_path.display()
                                    ));
                                    Some(new_path)
                                }
                                Ok(None) => {
                                    send_log("Original file left unchanged");
                                    None
                                }
                                Err(e) => {
                                    send_log(&format!("Could not rename original file: {}", e));
                                    None
                                }
                            }
                        } else {
                            None
                        };

                        let mut jobs_guard = jobs.lock().unwrap();
                        if let Some(job) = jobs_guard.get_mut(idx) {
                            job.subtitle_file = updated_subtitle;
                            if let Some(new_path) = renamed_original {
                                job.file_path = new_path;
                            }
                            match write_result {
                                Err(e) => {
                                    job.status = JobStatus::Failed;
                                    job.progress = 0.0;
                                    job.message = String::new();
                                    job.error = Some(e.to_string());
                                    send_log(&format!("Write failed: {}", e));
                                    send_progress(idx, 0.0, format!("Write failed: {}", e));
                                }
                                Ok(()) if !verified => {
                                    job.status = JobStatus::Failed;
                                    job.progress = 0.0;
                                    job.message = String::new();
                                    job.error = Some("Output file missing or empty after write"
                                        .to_string());
                                    send_log(&format!(
                                        "Output verification failed: {}",
                                        output_path.display()
                                    ));
                                    send_progress(
                                        idx,
                                        0.0,
                                        "Output verification failed".to_string(),
                                    );
                                }
                                Ok(()) => {
                                    if untranslated > 0 {
                                        send_log(&format!(
                                            "{} entries could not be translated; original text kept",
                                            untranslated
                                        ));
                                    }
                                    send_log(&format!(
                                        "Saved translation to: {}",
                                        output_path.display()
                                    ));
                                    send_log(&format!(
                                        "[stats] {}: {:.1}s total | {}",
                                        file_path
                                            .file_name()
                                            .unwrap_or_default()
                                            .to_string_lossy(),
                                        job_started.elapsed().as_secs_f32(),
                                        stats.summary()
                                    ));
                                    send_progress(idx, 1.0, "Completed".to_string());
                                    job.status = JobStatus::Completed;
                                    job.progress = 1.0;
                                    job.message = String::new();
                                    job.error = None;
                                }
                            }
                            if completion_sound
                                && matches!(
                                    job.status,
                                    JobStatus::Completed | JobStatus::Failed
                                )
                            {
                                crate::sound::play_completion();
                            }
                        }
                    }
                    Err(e) => {
                        let cancelled =
                            stop_flag.lock().map(|g| *g).unwrap_or(false);
                        let mut jobs_guard = jobs.lock().unwrap();
                        if let Some(job) = jobs_guard.get_mut(idx) {
                            job.subtitle_file = subtitle_file;
                            job.message = String::new();
                            if cancelled {
                                job.status = JobStatus::Cancelled;
                                job.progress = 0.0;
                                job.error = None;
                            } else {
                                job.status = JobStatus::Failed;
                                job.progress = 0.0;
                                job.error = Some(e.to_string());
                                send_log(&format!("Translation error: {}", e));
                                send_progress(idx, 0.0, format!("Translation error: {}", e));
                            }
                            if completion_sound && job.status == JobStatus::Failed {
                                crate::sound::play_completion();
                            }
                        }
                    }
                }
            }
        });

        self.translation_thread = Some(handle);
    }
    
    /// Builds the output path as `dosya.hedefDil.srt` (e.g. `video.tr.srt`).
    /// The source file is never used as the output path.
    fn generate_output_path(input_path: &Path, target_lang: Language) -> PathBuf {
        let mut output = input_path.to_path_buf();
        let target_code = target_lang.code();
        // Keep the input format (`.srt` stays `.srt`, `.vtt` stays `.vtt`)
        let ext = input_path
            .extension()
            .map(|e| e.to_string_lossy().to_string())
            .filter(|e| !e.is_empty())
            .unwrap_or_else(|| "srt".to_string());

        if let Some(stem) = output.file_stem() {
            let stem_str = stem.to_string_lossy().to_string();

            // Strip a non-standard language suffix first: "input_en" -> "input",
            // so the output follows the `<base>.<lang>.<ext>` standard
            // ("003 Setting up the input.tr.vtt").
            let stem_str = match split_nonstandard_lang_suffix(&stem_str) {
                Some((base, _)) => base.to_string(),
                None => stem_str,
            };

            // Remove source language code if present (e.g., "video.en" -> "video")
            let base_name = match stem_str.rfind('.') {
                Some(pos) if is_language_code(&stem_str[pos + 1..]) => &stem_str[..pos],
                _ => &stem_str,
            };

            // Build the full name at once: set_extension() would replace the
            // language code ("video.tr" -> "video.srt") and overwrite the source.
            let new_name = format!("{}.{}.{}", base_name, target_code, ext);
            output.set_file_name(new_name);

            // Never overwrite the source file (e.g. video.tr.srt -> Turkish)
            if output == input_path {
                let fallback = format!("{}.{}.{}", stem_str, target_code, ext);
                output.set_file_name(fallback);
            }
        } else {
            output.set_extension(ext);
        }
        output
    }
    
    fn stop_translation(&mut self) {
        if let Ok(mut guard) = self.stop_translation.lock() {
            *guard = true;
        }
        self.log("Translation stop requested");
    }
    
    fn update_job_status(&mut self) {
        if let Some(handle) = self.translation_thread.take() {
            if handle.is_finished() {
                handle.join().ok();
                let (completed, failed, cancelled) = {
                    let jobs = self.translation_jobs.lock().unwrap();
                    let count = |status: &JobStatus| {
                        jobs.iter().filter(|j| &j.status == status).count()
                    };
                    (
                        count(&JobStatus::Completed),
                        count(&JobStatus::Failed),
                        count(&JobStatus::Cancelled),
                    )
                };
                if cancelled > 0 {
                    self.notify(
                        ToastKind::Info,
                        format!("Çeviri durduruldu ({} dosya tamamlandı)", completed),
                    );
                } else if failed > 0 {
                    self.notify(
                        ToastKind::Error,
                        format!(
                            "Çeviri bitti: {} başarılı, {} hatalı",
                            completed, failed
                        ),
                    );
                } else {
                    self.notify(
                        ToastKind::Success,
                        format!("Çeviri tamamlandı: {} dosya", completed),
                    );
                }
            } else {
                self.translation_thread = Some(handle);
            }
        }
    }

    fn is_running(&self) -> bool {
        self.translation_thread
            .as_ref()
            .is_some_and(|h| !h.is_finished())
    }

    fn job_summaries(&self) -> Vec<JobSummary> {
        self.translation_jobs
            .lock()
            .unwrap()
            .iter()
            .map(|j| JobSummary {
                file_path: j.file_path.clone(),
                entry_count: j.subtitle_file.entries.len(),
                status: j.status.clone(),
                progress: j.progress,
                message: j.message.clone(),
                error: j.error.clone(),
            })
            .collect()
    }

    fn get_job(&self, idx: usize) -> Option<TranslationJob> {
        self.translation_jobs.lock().unwrap().get(idx).cloned()
    }
}

impl eframe::App for AutoTranslateApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Receive log messages from translation thread
        while let Ok(msg) = self.log_receiver.try_recv() {
            self.log_messages.push(msg);
            if self.log_messages.len() > 100 {
                self.log_messages.remove(0);
            }
        }
        
        while let Ok(progress) = self.progress_receiver.try_recv() {
            let mut jobs_guard = self.translation_jobs.lock().unwrap();
            if let Some(job) = jobs_guard.get_mut(progress.job_index) {
                job.progress = progress.progress;
                job.message = progress.message;
            }
        }

        while let Ok(update) = self.init_receiver.try_recv() {
            match update {
                InitUpdate::Models(Ok(models)) => {
                    let count = models.len();
                    self.config.available_models = models;
                    self.models_loaded = true;
                    self.models_loading = false;
                    if !self.config.available_models.contains(&self.config.selected_model) {
                        if let Some(first) = self.config.available_models.first() {
                            self.config.selected_model = first.clone();
                        }
                    }
                    if self.announce_model_load {
                        self.announce_model_load = false;
                        self.notify(ToastKind::Success, format!("{} model yüklendi", count));
                    } else {
                        self.log(&format!("Loaded {} models", count));
                    }
                }
                InitUpdate::Models(Err(e)) => {
                    self.connection_status = ConnectionStatus::Error(e.clone());
                    self.models_loaded = true;
                    self.models_loading = false;
                    if self.announce_model_load {
                        self.announce_model_load = false;
                        self.notify(ToastKind::Error, format!("Modeller yüklenemedi: {}", e));
                    } else {
                        self.log(&format!("Failed to load models: {}", e));
                    }
                }
                InitUpdate::Connection(status) => {
                    self.connection_checking = false;
                    match &status {
                        ConnectionStatus::Connected => {
                            self.log("Ollama connected");
                            if self.announce_connection {
                                self.announce_connection = false;
                                self.notify(
                                    ToastKind::Success,
                                    "Ollama bağlantısı başarılı".to_string(),
                                );
                            }
                        }
                        ConnectionStatus::Disconnected => {
                            self.announce_connection = false;
                            self.log("Ollama is not reachable");
                            self.notify(
                                ToastKind::Error,
                                "Ollama sunucusuna ulaşılamıyor; sunucunun çalıştığından emin olun"
                                    .to_string(),
                            );
                        }
                        ConnectionStatus::Error(e) => {
                            self.announce_connection = false;
                            self.log(&format!("Connection error: {}", e));
                            self.notify(ToastKind::Error, format!("Bağlantı hatası: {}", e));
                        }
                        ConnectionStatus::Unknown => {}
                    }
                    self.connection_status = status;
                }
            }
        }
        
        self.update_job_status();
        self.sync_renamed_selection();

        // The preview window must be closed before the app can quit
        if self.current_job_index.is_some() && ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.notify(
                ToastKind::Error,
                "Çıkıştan önce önizleme penceresini kapatın".to_string(),
            );
        }
        
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("Dosya", |ui| {
                    if ui.button("Dosya Seç").clicked() {
                        self.show_file_dialog = true;
                        ui.close_menu();
                    }
                    if ui.button("Klasör Seç").clicked() {
                        self.show_folder_dialog = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Çıkış").clicked() {
                        if self.current_job_index.is_some() {
                            self.notify(
                                ToastKind::Error,
                                "Çıkıştan önce önizleme penceresini kapatın".to_string(),
                            );
                        } else {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                });
                ui.separator();
                if ui
                    .selectable_label(self.tab == AppTab::Files, "Dosyalar")
                    .clicked()
                {
                    self.select_tab(AppTab::Files);
                }
                if ui
                    .selectable_label(self.tab == AppTab::Translation, "Çeviri")
                    .clicked()
                {
                    self.select_tab(AppTab::Translation);
                }
                if ui
                    .selectable_label(self.tab == AppTab::Settings, "Ayarlar")
                    .clicked()
                {
                    self.select_tab(AppTab::Settings);
                }
                ui.separator();
                if ui
                    .selectable_label(self.tab == AppTab::About, "Hakkında")
                    .clicked()
                {
                    self.select_tab(AppTab::About);
                }
                if ui
                    .selectable_label(self.tab == AppTab::Dependencies, "Bağımlılıklar")
                    .clicked()
                {
                    self.select_tab(AppTab::Dependencies);
                }
            });
        });
        
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.connection_checking {
                    ui.spinner();
                    ui.label("Bağlantı kontrol ediliyor");
                } else {
                    let status_text = match &self.connection_status {
                        ConnectionStatus::Connected => "🟢 Ollama Bağlı",
                        ConnectionStatus::Disconnected => "🔴 Ollama Bağlı Değil",
                        ConnectionStatus::Error(e) => &format!("🔴 Hata: {}", e),
                        ConnectionStatus::Unknown => "⚪ Bağlantı Kontrol Ediliyor",
                    };
                    ui.label(status_text);
                }
                ui.separator();
                ui.label(format!("Model: {}", self.config.selected_model));
                ui.separator();
                ui.label(format!("Dosya: {}", self.selected_files.len()));
                let (finished, total, _) = self.progress_stats();
                if total > 0 {
                    ui.separator();
                    ui.label(format!("İlerleme: {}/{}", finished, total));
                }
            });
        });
        
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.tab {
                AppTab::Files => self.render_files_tab(ui),
                AppTab::Translation => self.render_translation_tab(ui),
                AppTab::Settings => self.render_settings_tab(ui),
                AppTab::About => self.render_about_tab(ui),
                AppTab::Dependencies => self.render_dependencies_tab(ui),
            }
        });

        if self.confirm_clear {
            egui::Window::new("Listeyi Temizle")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.label(
                        "Seçili dosyalar ve hazırlanmış çeviri işleri listeden kaldırılacak.",
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Evet, Temizle").clicked() {
                            self.selected_files.clear();
                            self.translation_jobs.lock().unwrap().clear();
                            self.confirm_clear = false;
                            self.notify(
                                ToastKind::Info,
                                "Dosya listesi temizlendi".to_string(),
                            );
                        }
                        if ui.button("Vazgeç").clicked() {
                            self.confirm_clear = false;
                        }
                    });
                });
        }

        if self.show_file_dialog {
            if let Some(files) = FileDialog::new()
                .add_filter("Subtitle Files (SRT, VTT)", &["srt", "vtt"])
                .pick_files()
            {
                self.add_files(files);
            }
            self.show_file_dialog = false;
        }
        
        if self.show_folder_dialog {
            if let Some(folder) = FileDialog::new().pick_folder() {
                self.add_folder(folder);
            }
            self.show_folder_dialog = false;
        }

        self.show_preview_window(ctx);
        Self::render_toasts(ctx, &mut self.toasts);

        ctx.request_repaint_after(Duration::from_millis(100));
    }
    
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop_translation();
        if let Some(handle) = self.translation_thread.take() {
            // Wait briefly for the thread to finish; detach if an HTTP
            // request is still in flight so the app can close promptly.
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while !handle.is_finished() && std::time::Instant::now() < deadline {
                thread::sleep(Duration::from_millis(50));
            }
            if handle.is_finished() {
                handle.join().ok();
            }
        }
        self.save_config();
    }
}

impl AutoTranslateApp {
    fn render_files_tab(&mut self, ui: &mut Ui) {
        ui.heading("Altyazı Dosyaları");
        ui.separator();
        
        ui.horizontal(|ui| {
            if ui.button("📁 Dosya Seç").clicked() {
                self.show_file_dialog = true;
            }
            if ui.button("📂 Klasör Seç").clicked() {
                self.show_folder_dialog = true;
            }
            let is_running = self.is_running();
            if ui
                .add_enabled(!is_running, egui::Button::new("🗑️ Temizle"))
                .clicked()
            {
                self.confirm_clear = true;
            }
        });
        
        ui.separator();
        
        ScrollArea::vertical().show(ui, |ui| {
            let summaries = self.job_summaries();
            if self.selected_files.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        RichText::new(
                            "Henüz dosya seçilmedi. Yukarıdaki butonları kullanarak SRT/VTT dosyaları ekleyin.",
                        )
                        .weak(),
                    );
                });
            } else {
                egui::Grid::new("file_list").striped(true).show(ui, |ui| {
                    ui.label(RichText::new("#").strong());
                    ui.label(RichText::new("Dosya Adı").strong());
                    ui.label(RichText::new("Konum").strong());
                    ui.label(RichText::new("Satır Sayısı").strong());
                    ui.end_row();
                    
                    for (idx, file) in self.selected_files.iter().enumerate() {
                        ui.label((idx + 1).to_string());
                        ui.label(file.file_name().unwrap_or_default().to_string_lossy().to_string());
                        ui.label(file.parent().unwrap_or(Path::new("")).to_string_lossy().to_string());
                        
                        if let Some(job) = summaries.get(idx) {
                            ui.label(job.entry_count.to_string());
                        } else {
                            ui.label("-");
                        }
                        ui.end_row();
                    }
                });
            }
        });
    }
    
    fn render_translation_tab(&mut self, ui: &mut Ui) {
        ui.heading("Çeviri İşlemleri");
        ui.separator();
        
        ui.horizontal(|ui| {
            egui::ComboBox::from_label("Kaynak Dil")
                .selected_text(self.config.source_language.name())
                .show_ui(ui, |ui| {
                    for lang in Language::all() {
                        ui.selectable_value(&mut self.config.source_language, lang, lang.name());
                    }
                });
            
            ui.add_space(10.0);
            
            egui::ComboBox::from_label("Hedef Dil")
                .selected_text(self.config.target_language.name())
                .show_ui(ui, |ui| {
                    for lang in Language::all() {
                        if lang != Language::Auto {
                            ui.selectable_value(&mut self.config.target_language, lang, lang.name());
                        }
                    }
                });
        });
        
        ui.separator();
        
        ui.horizontal(|ui| {
            let jobs = self.job_summaries();
            let can_start = !jobs.is_empty() 
                && jobs.iter().any(|j| j.status == JobStatus::Pending);
            let is_running = self.is_running();
            
            if ui
                .add_enabled(
                    can_start && !is_running,
                    egui::Button::new("▶ Çeviriyi Başlat").fill(SUCCESS_FILL),
                )
                .clicked()
            {
                self.start_translation();
            }
            
            if ui
                .add_enabled(
                    is_running,
                    egui::Button::new("⏹ Durdur").fill(DANGER_FILL),
                )
                .clicked()
            {
                self.stop_translation();
            }
            
            if ui
                .add_enabled(!is_running, egui::Button::new("🔄 Yeniden Hazırla"))
                .clicked()
            {
                self.prepare_translation_jobs();
            }
        });
        
        ui.separator();

        let (finished, total, overall) = self.progress_stats();
        if total > 0 {
            ui.horizontal(|ui| {
                ui.strong("Genel İlerleme");
                ui.label(RichText::new(format!("{finished}/{total} dosya")).weak());
            });
            ui.add(egui::ProgressBar::new(overall).show_percentage());
            ui.separator();
        }

        ScrollArea::vertical().show(ui, |ui| {
            let jobs = self.job_summaries();
            if jobs.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        RichText::new("Çevrilecek dosya yok. 'Dosyalar' sekmesinden dosya ekleyin.")
                            .weak(),
                    );
                });
            } else {
                for (idx, job) in jobs.iter().enumerate() {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            let file_name = job.file_path.file_name().unwrap_or_default().to_string_lossy().to_string();
                            ui.strong(file_name);
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                match job.status {
                                    JobStatus::Pending => ui.label(RichText::new("⏳ Bekliyor").color(egui::Color32::YELLOW)),
                                    JobStatus::InProgress => ui.label(RichText::new("🔄 Çevriliyor...").color(egui::Color32::BLUE)),
                                    JobStatus::Completed => ui.label(RichText::new("✅ Tamamlandı").color(egui::Color32::GREEN)),
                                    JobStatus::Failed => ui.label(RichText::new("❌ Hata").color(egui::Color32::RED)),
                                    JobStatus::Cancelled => ui.label(RichText::new("⛔ İptal").color(egui::Color32::GRAY)),
                                    JobStatus::Skipped => ui.label(RichText::new("⏭ Atlandı").color(egui::Color32::from_rgb(120, 170, 120))),
                                }
                            });
                        });
                        
                        if job.status == JobStatus::Skipped && !job.message.is_empty() {
                            ui.label(RichText::new(&job.message).weak().size(11.0));
                        }
                        
                        if job.status == JobStatus::InProgress {
                            ui.add(egui::ProgressBar::new(job.progress).show_percentage());
                            if !job.message.is_empty() {
                                ui.label(RichText::new(&job.message).weak().size(11.0));
                            }
                        }
                        
                        if let Some(error) = &job.error {
                            ui.colored_label(ERR_TEXT, format!("Hata: {}", error));
                        }
                        
                        if (job.status == JobStatus::Completed
                            || job.status == JobStatus::Failed)
                            && ui.button("👁 Önizle").clicked()
                        {
                            self.preview_status = None;
                            self.current_job_index = Some(idx);
                        }
                    });
                    ui.add_space(5.0);
                }
            }
        });
    }
    
    fn show_preview_window(&mut self, ctx: &egui::Context) {
        let Some(job_idx) = self.current_job_index else {
            return;
        };
        let Some(job) = self.get_job(job_idx) else {
            self.current_job_index = None;
            return;
        };

        let file_name = job
            .file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let window_title = format!("Önizleme: {}", file_name);
        let jobs_mutex = self.translation_jobs.clone();
        let log_messages = &mut self.log_messages;
        let preview_status = &mut self.preview_status;
        let mut close_requested = false;

        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("translation_preview"),
            egui::ViewportBuilder::default()
                .with_title(window_title.clone())
                .with_inner_size([1000.0, 600.0])
                .with_min_inner_size([640.0, 400.0]),
            |preview_ctx, class| {
                // The user closed the native preview window
                if class != egui::ViewportClass::Embedded
                    && preview_ctx.input(|i| i.viewport().close_requested())
                {
                    close_requested = true;
                    return;
                }

                let content = |ui: &mut Ui| {
                ui.horizontal(|ui| {
                    ui.strong("Orijinal");
                    ui.add_space(20.0);
                    ui.strong("Çeviri (Düzenlenebilir)");
                });
                
                ui.separator();
                
                ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new(format!("preview_grid_{}", job_idx))
                        .striped(true)
                        .min_col_width(300.0)
                        .show(ui, |ui| {
                            for (pos, entry) in job.subtitle_file.entries.iter().enumerate() {
                                ui.label(format!("{}", entry.index));
                                ui.label(
                                    RichText::new(format!(
                                        "{} --> {}",
                                        crate::subtitle_parser::format_duration(entry.start_time),
                                        crate::subtitle_parser::format_duration(entry.end_time)
                                    ))
                                    .monospace()
                                    .size(10.0),
                                );

                                ui.add(
                                    egui::TextEdit::multiline(&mut entry.text.as_str())
                                        .desired_width(300.0)
                                        .desired_rows(2)
                                        .interactive(false),
                                );

                                let mut translated = entry.translated_text.clone().unwrap_or_default();
                                let response = ui.add(
                                    egui::TextEdit::multiline(&mut translated)
                                        .desired_width(300.0)
                                        .desired_rows(2),
                                );

                                if response.changed() {
                                    if let Ok(mut jobs_guard) = jobs_mutex.lock() {
                                        if let Some(job_mut) = jobs_guard.get_mut(job_idx) {
                                            if let Some(entry_mut) =
                                                job_mut.subtitle_file.entries.get_mut(pos)
                                            {
                                                entry_mut.translated_text = Some(translated);
                                            }
                                        }
                                    }
                                }

                                ui.end_row();
                            }
                        });
                });
                
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("💾 Kaydet").clicked() {
                        if let Ok(mut jobs_guard) = jobs_mutex.lock() {
                            if let Some(job_mut) = jobs_guard.get_mut(job_idx) {
                                let target_lang = job_mut.subtitle_file.target_language;
                                let output_path = Self::generate_output_path(&job_mut.file_path, target_lang);
                                let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                                if let Err(e) = write_subtitle_file(&job_mut.subtitle_file, &output_path) {
                                    log_messages.push(format!("[{}] Kaydetme hatası: {}", ts, e));
                                    crate::logger::log(
                                        crate::logger::Level::Error,
                                        "preview",
                                        &format!("Kaydetme hatası: {}", e),
                                    );
                                    *preview_status =
                                        Some((format!("Kaydetme hatası: {}", e), ToastKind::Error));
                                } else {
                                    log_messages.push(format!("[{}] Dosya kaydedildi: {}", ts, output_path.display()));
                                    crate::logger::log(
                                        crate::logger::Level::Info,
                                        "preview",
                                        &format!("Dosya kaydedildi: {}", output_path.display()),
                                    );
                                    *preview_status = Some((
                                        format!("Dosya kaydedildi: {}", output_path.display()),
                                        ToastKind::Success,
                                    ));
                                }
                            }
                        }
                    }
                    if ui.button("Kapat").clicked() {
                        close_requested = true;
                    }
                });
                if let Some((message, kind)) = preview_status.as_ref() {
                    let color = match kind {
                        ToastKind::Error => ERR_TEXT,
                        ToastKind::Success => OK_TEXT,
                        ToastKind::Info => ACCENT,
                    };
                    ui.colored_label(color, message);
                }
                };

                match class {
                    egui::ViewportClass::Embedded => {
                        egui::Window::new(window_title.clone())
                            .default_size(Vec2::new(1000.0, 600.0))
                            .show(preview_ctx, content);
                    }
                    _ => {
                        egui::CentralPanel::default().show(preview_ctx, content);
                    }
                }
            },
        );

        if close_requested {
            self.current_job_index = None;
        }
    }
    
    fn render_about_tab(&mut self, ui: &mut Ui) {
        ui.heading("Hakkında");
        ui.separator();

        let version = env!("CARGO_PKG_VERSION");
        ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!("Auto Translate Subs v{}", version))
                    .strong()
                    .size(16.0),
            );
            ui.add_space(6.0);
            ui.label(
                "SRT ve VTT altyazı dosyalarını yerel Ollama modelleriyle çeviren bir masaüstü uygulamasıdır. \
                 Çeviri tamamen bilgisayarınızda yapılır; altyazı içerikleri hiçbir çevrimiçi servise gönderilmez.",
            );

            ui.add_space(12.0);
            ui.group(|ui| {
                ui.strong("Özellikler");
                ui.label("• Toplu dosya ve klasör seçimi");
                ui.label("• Chunk'lı hızlı toplu çeviri + eksik sonuçlar için tek tek yedek yol");
                ui.label("• İlerleme çubukları, duraklatma/iptal");
                ui.label("• Ayrı pencerede düzenlenebilir önizleme ve kaydetme");
                ui.label("• Çıktı dosyası doğrulanır, orijinal dosyaya dil kodu eklenir");
                ui.label("• Orijinal dosyalar asla ezilmez (çıktı: `dosya.tr.srt`)");
            });

            ui.add_space(8.0);
            ui.group(|ui| {
                ui.strong("Teknolojiler");
                ui.label("Rust ile geliştirilmiştir · egui/eframe arayüzü · Ollama yerel model sunucusu");
            });

            ui.add_space(8.0);
            ui.group(|ui| {
                ui.strong("Bağlantı");
                ui.horizontal(|ui| {
                    ui.label("Ollama:");
                    ui.hyperlink_to("https://ollama.com", "https://ollama.com");
                });
            });
        });
    }

    fn render_dependencies_tab(&mut self, ui: &mut Ui) {
        ui.heading("Bağımlılıklar Kurulumu");
        ui.separator();

        let selected_model = self.config.selected_model.clone();
        ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(4.0);
            ui.label("Uygulamanın çalışması için aşağıdaki bağımlılıklar gerekir:");
            ui.add_space(8.0);

            ui.group(|ui| {
                ui.strong("1. Ollama (zorunlu)");
                ui.label(
                    "Yerel yapay zekâ sunucusudur; çeviriler bu sunucuda çalışan model tarafından yapılır.",
                );
                ui.horizontal(|ui| {
                    ui.label("İndirme:");
                    ui.hyperlink_to("https://ollama.com/download", "https://ollama.com/download");
                });
                ui.label("Kurulum sonrası doğrulama:");
                ui.code("ollama --version");
            });

            ui.add_space(6.0);
            ui.group(|ui| {
                ui.strong("2. Model indirme (zorunlu)");
                ui.label(
                    "Seçili model henüz indirilmemişse çeviri başarısız olur. Örnek kurulum:",
                );
                ui.code("ollama pull llama3.2");
                ui.label(format!("Uygulamadaki seçili model: {}", selected_model));
                ui.label(
                    "İndirilen modelleri `ollama list` ile, uygulamadaki listeyi Ayarlar > Modelleri Yenile ile görebilirsiniz.",
                );
            });

            ui.add_space(6.0);
            ui.group(|ui| {
                ui.strong("3. Sunucu adresi");
                ui.label("Varsayılan: http://localhost:11434 (Ayarlar sekmesinden değiştirilebilir).");
                ui.label(
                    "Ollama başka bir bilgisayarda çalışıyorsa IP adresini girin ve 11434 portunu \
                     (TCP) güvenlik duvarında açın.",
                );
            });

            ui.add_space(6.0);
            ui.group(|ui| {
                ui.strong("4. Bağlantıyı test etme");
                ui.label(
                    "Ayarlar sekmesindeki \"🔌 Bağlantıyı Test Et\" ve \"🔄 Modelleri Yenile\" \
                     düğmeleriyle kurulumu doğrulayın.",
                );
            });

            ui.add_space(6.0);
            ui.group(|ui| {
                ui.strong("5. Windows notu");
                ui.label(
                    "Bazı temiz Windows kurulumlarında Microsoft Visual C++ Redistributable gerekebilir:",
                );
                ui.horizontal(|ui| {
                    ui.hyperlink_to(
                        "En son sürüm",
                        "https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist",
                    );
                });
            });
        });
    }

    fn render_settings_tab(&mut self, ui: &mut Ui) {
        ui.heading("Ayarlar");
        ui.separator();
        
        ui.group(|ui| {
            ui.strong("Ollama Bağlantısı");
            ui.horizontal(|ui| {
                ui.label("Sunucu URL:");
                ui.text_edit_singleline(&mut self.config.ollama_url);
            });
            
            ui.horizontal(|ui| {
                let refresh = ui.add_enabled(
                    !self.models_loading,
                    egui::Button::new("🔄 Modelleri Yenile"),
                );
                if refresh.clicked() {
                    self.announce_model_load = true;
                    self.load_models();
                }
                let test = ui.add_enabled(
                    !self.connection_checking,
                    egui::Button::new("🔌 Bağlantıyı Test Et"),
                );
                if test.clicked() {
                    self.announce_connection = true;
                    self.check_connection();
                }
                if self.models_loading {
                    ui.spinner();
                }
            });
            
            if !self.models_loaded {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Modeller yükleniyor…");
                });
            } else if self.config.available_models.is_empty() {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "Model bulunamadı — Ollama'nın çalıştığından emin olun",
                );
            } else {
                egui::ComboBox::from_label("Model Seç")
                    .selected_text(&self.config.selected_model)
                    .show_ui(ui, |ui| {
                        for model in &self.config.available_models {
                            ui.selectable_value(
                                &mut self.config.selected_model,
                                model.clone(),
                                model,
                            );
                        }
                    });
            }
        });
        
        ui.separator();
        
        ui.group(|ui| {
            ui.strong("Dil Ayarları");
            ui.horizontal(|ui| {
                egui::ComboBox::from_label("Varsayılan Kaynak Dil")
                    .selected_text(self.config.source_language.name())
                    .show_ui(ui, |ui| {
                        for lang in Language::all() {
                            ui.selectable_value(&mut self.config.source_language, lang, lang.name());
                        }
                    });
            });
            
            ui.horizontal(|ui| {
                egui::ComboBox::from_label("Varsayılan Hedef Dil")
                    .selected_text(self.config.target_language.name())
                    .show_ui(ui, |ui| {
                        for lang in Language::all() {
                            if lang != Language::Auto {
                                ui.selectable_value(&mut self.config.target_language, lang, lang.name());
                            }
                        }
                    });
            });
        });
        
        ui.separator();
        
        ui.group(|ui| {
            ui.strong("Genel");
            let resp = ui.checkbox(
                &mut self.config.skip_translated,
                "Çevrilmiş dosyaları atla (yeniden çevirme yapma)",
            );
            if resp.changed() {
                let (skipped, resumed) = self.apply_skip_policy();
                if skipped > 0 {
                    self.log(&format!("Skipped {} already translated file(s)", skipped));
                }
                if resumed > 0 {
                    self.log(&format!("Resumed {} previously skipped file(s)", resumed));
                }
            }
            let sound_resp = ui.checkbox(
                &mut self.config.completion_sound,
                "Her dosya bitince uyar sesi çal",
            );
            if sound_resp.changed() && self.config.completion_sound {
                crate::sound::play_completion();
                self.notify(ToastKind::Success, "Uyarı sesi açıldı".to_string());
            }
            if ui.button("Ayarları Kaydet").clicked() {
                self.save_config();
                self.notify(ToastKind::Success, "Ayarlar kaydedildi".to_string());
            }
        });
        
        ui.separator();
        
        ui.group(|ui| {
            ui.strong("Uygulama Günlüğü");
            ui.label(
                RichText::new(format!("Dosya: {}", self.log_path.display()))
                    .monospace()
                    .size(10.0)
                    .weak(),
            );
            ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    for msg in &self.log_messages {
                        ui.label(RichText::new(msg).monospace().size(11.0));
                    }
                });
            ui.horizontal(|ui| {
                if ui.button("📂 Log Dosyasını Aç").clicked() {
                    open_log_file(&self.log_path);
                }
                if ui.button("Görüntüyü Temizle (dosya korunur)").clicked() {
                    self.log_messages.clear();
                }
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_path_adds_target_code() {
        assert_eq!(
            AutoTranslateApp::generate_output_path(Path::new("video.srt"), Language::Turkish),
            PathBuf::from("video.tr.srt")
        );
        assert_eq!(
            AutoTranslateApp::generate_output_path(
                Path::new("1 - Introduction.srt"),
                Language::Turkish
            ),
            PathBuf::from("1 - Introduction.tr.srt")
        );
    }

    #[test]
    fn output_path_replaces_source_language_code() {
        assert_eq!(
            AutoTranslateApp::generate_output_path(Path::new("video.en.srt"), Language::Turkish),
            PathBuf::from("video.tr.srt")
        );
        assert_eq!(
            AutoTranslateApp::generate_output_path(Path::new("video.tr.srt"), Language::English),
            PathBuf::from("video.en.srt")
        );
    }

    #[test]
    fn output_path_never_overwrites_source() {
        // Source already ends with the target language code
        let out = AutoTranslateApp::generate_output_path(
            Path::new("video.tr.srt"),
            Language::Turkish,
        );
        assert_ne!(out, PathBuf::from("video.tr.srt"));
        assert_eq!(out, PathBuf::from("video.tr.tr.srt"));
    }

    #[test]
    fn output_path_keeps_directory() {
        assert_eq!(
            AutoTranslateApp::generate_output_path(
                Path::new("C:/subs/video.srt"),
                Language::Turkish
            ),
            PathBuf::from("C:/subs/video.tr.srt")
        );
    }

    #[test]
    fn output_path_preserves_file_format() {
        assert_eq!(
            AutoTranslateApp::generate_output_path(Path::new("video.vtt"), Language::Turkish),
            PathBuf::from("video.tr.vtt")
        );
        assert_eq!(
            AutoTranslateApp::generate_output_path(Path::new("video.en.vtt"), Language::Turkish),
            PathBuf::from("video.tr.vtt")
        );
        assert_eq!(
            AutoTranslateApp::generate_output_path(Path::new("video.VTT"), Language::Turkish),
            PathBuf::from("video.tr.VTT")
        );
    }

    fn temp_dir_for(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ats_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn rename_adds_source_language_code() {
        let dir = temp_dir_for("rename_adds");
        let src = dir.join("video.srt");
        std::fs::write(&src, "content").unwrap();

        let result = rename_original_with_code(&src, Language::English).unwrap();

        assert_eq!(result, Some(dir.join("video.en.srt")));
        assert!(!src.exists());
        assert!(dir.join("video.en.srt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rename_skips_when_code_already_present() {
        let dir = temp_dir_for("rename_existing_code");
        let src = dir.join("video.en.srt");
        std::fs::write(&src, "content").unwrap();

        let result = rename_original_with_code(&src, Language::English).unwrap();

        assert_eq!(result, None);
        assert!(src.exists());
        assert!(!dir.join("video.en.en.srt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rename_skips_when_target_name_taken() {
        let dir = temp_dir_for("rename_target_taken");
        let src = dir.join("video.srt");
        let occupied = dir.join("video.en.srt");
        std::fs::write(&src, "content").unwrap();
        std::fs::write(&occupied, "other").unwrap();

        let result = rename_original_with_code(&src, Language::English).unwrap();

        assert_eq!(result, None);
        assert!(src.exists());
        assert_eq!(
            std::fs::read_to_string(&occupied).unwrap(),
            "other",
            "existing file must not be overwritten"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rename_skips_auto_language() {
        let dir = temp_dir_for("rename_auto");
        let src = dir.join("video.srt");
        std::fs::write(&src, "content").unwrap();

        let result = rename_original_with_code(&src, Language::Auto).unwrap();

        assert_eq!(result, None);
        assert!(src.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn make_job(path: PathBuf) -> TranslationJob {
        TranslationJob {
            file_path: path,
            subtitle_file: SubtitleFile::default(),
            status: JobStatus::Pending,
            progress: 0.0,
            message: String::new(),
            error: None,
        }
    }

    #[test]
    fn skip_reason_target_code_in_name() {
        let dir = temp_dir_for("skip_name");
        let file = dir.join("video.tr.srt");

        assert_eq!(
            already_translated_reason(&file, Language::Turkish),
            Some("Dosya adında hedef dil kodu zaten var")
        );
        // Different target code: not translated by rule 1 and no output on disk
        assert_eq!(already_translated_reason(&file, Language::English), None);
        // Auto target can never be detected
        assert_eq!(already_translated_reason(&file, Language::Auto), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skip_reason_output_exists() {
        let dir = temp_dir_for("skip_output");
        let src = dir.join("video.en.srt");
        std::fs::write(&src, "original").unwrap();

        assert_eq!(already_translated_reason(&src, Language::Turkish), None);

        std::fs::write(dir.join("video.tr.srt"), "translated").unwrap();
        assert_eq!(
            already_translated_reason(&src, Language::Turkish),
            Some("Çıktı dosyası diskte mevcut")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skip_reason_ignores_empty_output() {
        let dir = temp_dir_for("skip_empty_output");
        let src = dir.join("video.en.srt");
        std::fs::write(&src, "original").unwrap();
        std::fs::write(dir.join("video.tr.srt"), "").unwrap();

        assert_eq!(already_translated_reason(&src, Language::Turkish), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skip_policy_skips_and_resumes() {
        let dir = temp_dir_for("skip_policy");
        let mut jobs = vec![
            make_job(dir.join("video.tr.srt")), // rule 1: target code in name
            make_job(dir.join("other.srt")),    // nothing on disk
        ];

        let (skipped, resumed) = apply_skip_policy_to(&mut jobs, Language::Turkish, true);
        assert_eq!((skipped, resumed), (1, 0));
        assert_eq!(jobs[0].status, JobStatus::Skipped);
        assert!(!jobs[0].message.is_empty());
        assert_eq!(jobs[1].status, JobStatus::Pending);

        // Policy off: previously skipped jobs become pending again
        let (skipped, resumed) = apply_skip_policy_to(&mut jobs, Language::Turkish, false);
        assert_eq!((skipped, resumed), (0, 1));
        assert_eq!(jobs[0].status, JobStatus::Pending);
        assert!(jobs[0].message.is_empty());

        // Policy on again: skipped once more
        let (skipped, resumed) = apply_skip_policy_to(&mut jobs, Language::Turkish, true);
        assert_eq!((skipped, resumed), (1, 0));
        assert_eq!(jobs[0].status, JobStatus::Skipped);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skip_policy_resumes_when_target_language_changes() {
        let dir = temp_dir_for("skip_policy_lang");
        let mut jobs = vec![make_job(dir.join("video.tr.srt"))];

        let (skipped, _) = apply_skip_policy_to(&mut jobs, Language::Turkish, true);
        assert_eq!(skipped, 1);
        assert_eq!(jobs[0].status, JobStatus::Skipped);

        // Same file, different target: "video.tr.srt" -> output "video.en.srt"
        // does not exist, so it must be translated again
        let (_, resumed) = apply_skip_policy_to(&mut jobs, Language::English, true);
        assert_eq!(resumed, 1);
        assert_eq!(jobs[0].status, JobStatus::Pending);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_nonstandard_suffix_detects_lang_codes() {
        assert_eq!(
            split_nonstandard_lang_suffix("003 Setting up the input_en"),
            Some(("003 Setting up the input", "en"))
        );
        assert_eq!(
            split_nonstandard_lang_suffix("003 Setting up the input-en"),
            Some(("003 Setting up the input", "en"))
        );
        assert_eq!(
            split_nonstandard_lang_suffix("INPUT_TR"),
            Some(("INPUT", "tr"))
        );
        assert_eq!(split_nonstandard_lang_suffix("my-file_en"), Some(("my-file", "en")));
        // No separator: never guessed
        assert_eq!(split_nonstandard_lang_suffix("green"), None);
        assert_eq!(split_nonstandard_lang_suffix("kitchen"), None);
        // Unknown or excluded codes
        assert_eq!(split_nonstandard_lang_suffix("input_xx"), None);
        assert_eq!(split_nonstandard_lang_suffix("input_auto"), None);
        // Empty base
        assert_eq!(split_nonstandard_lang_suffix("_en"), None);
    }

    #[test]
    fn output_path_strips_nonstandard_suffix() {
        assert_eq!(
            AutoTranslateApp::generate_output_path(
                Path::new("003 Setting up the input_en.vtt"),
                Language::Turkish
            ),
            PathBuf::from("003 Setting up the input.tr.vtt")
        );
        assert_eq!(
            AutoTranslateApp::generate_output_path(Path::new("input-en.srt"), Language::Turkish),
            PathBuf::from("input.tr.srt")
        );
        assert_eq!(
            AutoTranslateApp::generate_output_path(Path::new("input_en.vtt"), Language::English),
            PathBuf::from("input.en.vtt")
        );
    }

    #[test]
    fn skip_reason_nonstandard_target_code_in_name() {
        let dir = temp_dir_for("skip_nonstd");
        let file = dir.join("003 Setting up the input_tr.vtt");

        assert_eq!(
            already_translated_reason(&file, Language::Turkish),
            Some("Dosya adında hedef dil kodu zaten var")
        );
        // Different target and no output on disk
        assert_eq!(already_translated_reason(&file, Language::English), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rename_normalizes_nonstandard_suffix() {
        let dir = temp_dir_for("rename_nonstd");
        let src = dir.join("003 Setting up the input_en.vtt");
        std::fs::write(&src, "content").unwrap();

        let result = rename_original_with_code(&src, Language::English).unwrap();

        assert_eq!(
            result,
            Some(dir.join("003 Setting up the input.en.vtt"))
        );
        assert!(!src.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rename_keeps_mismatched_suffix_and_appends_code() {
        let dir = temp_dir_for("rename_mismatch");
        let src = dir.join("input_en.srt");
        std::fs::write(&src, "content").unwrap();

        let result = rename_original_with_code(&src, Language::German).unwrap();

        assert_eq!(result, Some(dir.join("input_en.de.srt")));
        assert!(!src.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn toasts_render_in_headless_context() {
        let ctx = egui::Context::default();
        let mut toasts = vec![
            Toast {
                message: "bilgi".to_string(),
                kind: ToastKind::Info,
                created: Instant::now(),
            },
            Toast {
                message: "başarı".to_string(),
                kind: ToastKind::Success,
                created: Instant::now(),
            },
            Toast {
                message: "hata".to_string(),
                kind: ToastKind::Error,
                created: Instant::now(),
            },
        ];

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            AutoTranslateApp::render_toasts(ctx, &mut toasts);
        });

        assert_eq!(toasts.len(), 3, "fresh toasts must stay visible");
    }

    #[test]
    fn expired_toasts_are_removed_before_rendering() {
        let ctx = egui::Context::default();
        let mut toasts = vec![Toast {
            message: "eski".to_string(),
            kind: ToastKind::Info,
            created: Instant::now()
                .checked_sub(TOAST_DURATION + Duration::from_secs(1))
                .unwrap(),
        }];

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            AutoTranslateApp::render_toasts(ctx, &mut toasts);
        });

        assert!(toasts.is_empty(), "expired toast must be dropped");
    }
}