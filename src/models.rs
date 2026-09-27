use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleEntry {
    pub index: usize,
    pub start_time: Duration,
    pub end_time: Duration,
    pub text: String,
    pub translated_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubtitleFile {
    pub path: PathBuf,
    pub entries: Vec<SubtitleEntry>,
    pub source_language: Language,
    pub target_language: Language,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Language {
    #[default]
    English,
    German,
    French,
    Spanish,
    Italian,
    Portuguese,
    Russian,
    Chinese,
    Japanese,
    Korean,
    Turkish,
    Auto,
}

impl Language {
    pub fn all() -> Vec<Language> {
        vec![
            Language::Auto,
            Language::English,
            Language::German,
            Language::French,
            Language::Spanish,
            Language::Italian,
            Language::Portuguese,
            Language::Russian,
            Language::Chinese,
            Language::Japanese,
            Language::Korean,
            Language::Turkish,
        ]
    }

    pub fn from_code(code: &str) -> Option<Language> {
        Language::all()
            .into_iter()
            .find(|l| l.code().eq_ignore_ascii_case(code))
    }

    pub fn code(&self) -> &'static str {
        match self {
            Language::Auto => "auto",
            Language::English => "en",
            Language::German => "de",
            Language::French => "fr",
            Language::Spanish => "es",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Russian => "ru",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Turkish => "tr",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Language::Auto => "Otomatik Algıla",
            Language::English => "İngilizce",
            Language::German => "Almanca",
            Language::French => "Fransızca",
            Language::Spanish => "İspanyolca",
            Language::Italian => "İtalyanca",
            Language::Portuguese => "Portekizce",
            Language::Russian => "Rusça",
            Language::Chinese => "Çince",
            Language::Japanese => "Japonca",
            Language::Korean => "Korece",
            Language::Turkish => "Türkçe",
        }
    }
}

fn default_skip_translated() -> bool {
    true
}

fn default_completion_sound() -> bool {
    true
}

fn default_max_input_tokens() -> usize {
    3000
}

fn default_context_before() -> usize {
    2
}

fn default_context_after() -> usize {
    2
}

fn default_max_concurrent_requests() -> usize {
    // Conservative default: the measured baseline is sequential (1 request
    // at a time). Raising this requires benchmark evidence on the user's
    // system (Task 07 benchmarks 1/2/3; the value is never guessed up).
    1
}

fn default_max_retries() -> u32 {
    // Task 08's suggested policy: attempt 1 immediate, attempt 2 after a
    // short backoff, attempt 3 after a longer one, then fail — i.e. two
    // retries on top of the first attempt. 0 disables retries.
    2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub ollama_url: String,
    pub selected_model: String,
    pub available_models: Vec<String>,
    pub source_language: Language,
    pub target_language: Language,
    pub last_directory: Option<PathBuf>,
    /// Skip files whose translation already exists (default on). Old config
    /// files without this field keep working thanks to the serde default.
    #[serde(default = "default_skip_translated")]
    pub skip_translated: bool,
    /// Play a system chime when a file finishes (default on).
    #[serde(default = "default_completion_sound")]
    pub completion_sound: bool,
    /// Approximate input-token budget per batch request. Batches accumulate
    /// estimated tokens until this budget would be exceeded (default 3000).
    #[serde(default = "default_max_input_tokens")]
    pub max_input_tokens: usize,
    /// Source lines shown before the batch as reference context (0 disables).
    /// Context lines are only reference: the model never returns them.
    #[serde(default = "default_context_before")]
    pub context_before: usize,
    /// Source lines shown after the batch as reference context (0 disables).
    #[serde(default = "default_context_after")]
    pub context_after: usize,
    /// Upper bound for parallel Ollama batch requests (1 = sequential).
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
    /// Extra attempts after the first failed request (0 disables retries).
    /// Transient failures back off exponentially before each retry;
    /// permanent failures are never retried.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            selected_model: "llama3.2".to_string(),
            available_models: vec![],
            source_language: Language::English,
            target_language: Language::Turkish,
            last_directory: None,
            skip_translated: default_skip_translated(),
            completion_sound: default_completion_sound(),
            max_input_tokens: default_max_input_tokens(),
            context_before: default_context_before(),
            context_after: default_context_after(),
            max_concurrent_requests: default_max_concurrent_requests(),
            max_retries: default_max_retries(),
        }
    }
}

impl AppConfig {
    /// Location of the persistent config file
    /// (`<config dir>/auto-translate-subs/config.json`).
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("auto-translate-subs")
            .join("config.json")
    }

    /// Loads the config stored at `path`. Returns `None` when the file is
    /// missing, unreadable or cannot be parsed.
    pub fn load(path: &Path) -> Option<AppConfig> {
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    pub options: TranslationOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub num_predict: i32,
}

impl Default for TranslationOptions {
    fn default() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.9,
            num_predict: 4096,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResponse {
    pub response: String,
    pub done: bool,
    /// Ollama timing metadata (nanoseconds; optional)
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTranslationRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    pub options: TranslationOptions,
    /// Ollama structured-output JSON schema (the `format` field). `None`
    /// falls back to prompt-only shaping; omitted from the wire when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTranslationEntry {
    /// Subtitle id owned by the application. Accepts both `id` (structured
    /// output target shape) and the legacy `index` key on input.
    #[serde(alias = "id")]
    pub index: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTranslationResponse {
    pub translations: Vec<BatchTranslationEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelsResponse {
    pub models: Vec<OllamaModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_missing_config_returns_none() {
        assert!(AppConfig::load(Path::new("definitely-missing-config.json")).is_none());
    }

    #[test]
    fn language_from_code() {
        assert_eq!(Language::from_code("en"), Some(Language::English));
        assert_eq!(Language::from_code("TR"), Some(Language::Turkish));
        assert_eq!(Language::from_code("auto"), Some(Language::Auto));
        assert_eq!(Language::from_code("xx"), None);
        assert_eq!(Language::from_code("hd"), None);
        assert_eq!(Language::from_code("01"), None);
    }

    #[test]
    fn translation_response_tolerates_missing_timing_fields() {
        // Older Ollama responses without timing metadata must still parse
        let r: TranslationResponse =
            serde_json::from_str(r#"{"response":"hi","done":true}"#).unwrap();
        assert_eq!(r.response, "hi");
        assert_eq!(r.prompt_eval_count, None);
        assert_eq!(r.eval_count, None);

        let r: TranslationResponse = serde_json::from_str(
            r#"{"response":"hi","done":true,"total_duration":1000,"prompt_eval_count":10,"eval_count":20}"#,
        )
        .unwrap();
        assert_eq!(r.prompt_eval_count, Some(10));
        assert_eq!(r.eval_count, Some(20));
    }

    #[test]
    fn batch_entry_deserializes_id_alias() {
        let e: BatchTranslationEntry = serde_json::from_str(r#"{"id":7,"text":"hi"}"#).unwrap();
        assert_eq!(e.index, 7);
        let e: BatchTranslationEntry = serde_json::from_str(r#"{"index":7,"text":"hi"}"#).unwrap();
        assert_eq!(e.index, 7);
    }

    #[test]
    fn batch_request_serializes_format_only_when_set() {
        let with = BatchTranslationRequest {
            model: "m".into(),
            prompt: "p".into(),
            stream: false,
            options: TranslationOptions::default(),
            format: Some(serde_json::json!({"type": "object"})),
        };
        let v = serde_json::to_value(&with).unwrap();
        assert!(v.get("format").is_some(), "schema must reach the wire");

        let without = BatchTranslationRequest {
            format: None,
            ..with
        };
        let v = serde_json::to_value(&without).unwrap();
        assert!(
            v.get("format").is_none(),
            "no format key when structured output is off"
        );
    }

    #[test]
    fn context_defaults_apply_when_keys_missing() {
        let mut v = serde_json::to_value(AppConfig::default()).unwrap();
        v.as_object_mut().unwrap().remove("context_before");
        v.as_object_mut().unwrap().remove("context_after");
        let cfg: AppConfig = serde_json::from_value(v).unwrap();
        assert_eq!(cfg.context_before, 2);
        assert_eq!(cfg.context_after, 2);
    }

    #[test]
    fn concurrency_defaults_to_sequential() {
        assert_eq!(AppConfig::default().max_concurrent_requests, 1);
        let mut v = serde_json::to_value(AppConfig::default()).unwrap();
        v.as_object_mut().unwrap().remove("max_concurrent_requests");
        let cfg: AppConfig = serde_json::from_value(v).unwrap();
        assert_eq!(
            cfg.max_concurrent_requests, 1,
            "old config files must stay sequential"
        );
    }

    #[test]
    fn retries_default_to_the_three_attempt_policy() {
        assert_eq!(AppConfig::default().max_retries, 2);
        let mut v = serde_json::to_value(AppConfig::default()).unwrap();
        v.as_object_mut().unwrap().remove("max_retries");
        let cfg: AppConfig = serde_json::from_value(v).unwrap();
        assert_eq!(
            cfg.max_retries, 2,
            "old config files inherit the suggested 3-attempt policy"
        );
    }
}
