//! Persistent file logging.
//!
//! Every log line ends up in `<config>/logs/app.log` (rotated once to
//! `app.log.1` at startup when oversized) in addition to the in-app view:
//!
//! ```text
//! 2026-09-27 13:51:47 [INFO ] [worker] [13:51:47] Translation started
//! 2026-09-27 13:51:47 [ERROR] [app] Write failed: access denied
//! ```

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Info,
    Error,
}

impl Level {
    fn as_str(self) -> &'static str {
        match self {
            Level::Info => "INFO",
            Level::Error => "ERROR",
        }
    }
}

static LOG_FILE: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();

const MAX_BYTES: u64 = 5 * 1024 * 1024;

/// Opens `dir/app.log` (rotating a previous oversized file to `app.log.1`)
/// and writes a session header. Call once at startup; later calls keep the
/// already open handle. Returns the log file path.
pub fn init(dir: &Path) -> PathBuf {
    let _ = std::fs::create_dir_all(dir);
    let path = dir.join("app.log");
    rotate_if_needed(&path, MAX_BYTES);
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok();
    let _ = LOG_FILE.set(Mutex::new(file));
    log(Level::Info, "app", "==== session started ====");
    path
}

/// Appends one formatted line to the log file. Silently does nothing when
/// the logger is not initialized or the file cannot be written.
pub fn log(level: Level, target: &str, message: &str) {
    let Some(slot) = LOG_FILE.get() else { return };
    let Ok(mut guard) = slot.lock() else { return };
    let Some(file) = guard.as_mut() else { return };
    let timestamp = chrono::Local::now()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let _ = writeln!(file, "{}", format_line(&timestamp, level, target, message));
    let _ = file.flush();
}

/// Picks a level from the message text so existing call sites do not have to
/// be rewritten. Everything is INFO unless the message clearly reports a
/// failure (English or Turkish).
pub fn level_for(message: &str) -> Level {
    let m = message.to_ascii_lowercase();
    const ERROR_WORDS: [&str; 6] = [
        "error",
        "failed",
        "failure",
        "cannot",
        "could not",
        "hatas", // "hata", "hatası", "hatası:"
    ];
    if ERROR_WORDS.iter().any(|w| m.contains(w)) {
        Level::Error
    } else {
        Level::Info
    }
}

pub fn format_line(timestamp: &str, level: Level, target: &str, message: &str) -> String {
    format!("{} [{:<5}] [{}] {}", timestamp, level.as_str(), target, message)
}

/// Renames `path` to `<path>.1` (replacing an old backup) when it exceeds
/// `max_bytes`.
fn rotate_if_needed(path: &Path, max_bytes: u64) {
    let Ok(meta) = std::fs::metadata(path) else {
        return;
    };
    if meta.len() <= max_bytes {
        return;
    }
    let backup = path.with_extension("log.1");
    let _ = std::fs::remove_file(&backup);
    let _ = std::fs::rename(path, backup);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_line_contains_all_parts() {
        assert_eq!(
            format_line("2026-09-27 10:00:00", Level::Error, "worker", "boom"),
            "2026-09-27 10:00:00 [ERROR] [worker] boom"
        );
        assert_eq!(
            format_line("2026-09-27 10:00:00", Level::Info, "app", "hello"),
            "2026-09-27 10:00:00 [INFO ] [app] hello"
        );
    }

    #[test]
    fn level_for_detects_failures_only() {
        assert_eq!(level_for("Failed to parse response"), Level::Error);
        assert_eq!(level_for("Write failed: access denied"), Level::Error);
        assert_eq!(level_for("Could not rename original file"), Level::Error);
        assert_eq!(level_for("Kaydetme hatası: disk full"), Level::Error);
        assert_eq!(level_for("Translation started"), Level::Info);
        assert_eq!(
            level_for("Skipped 3 already translated file(s)"),
            Level::Info
        );
        assert_eq!(level_for("Saved translation to: x.srt"), Level::Info);
    }

    #[test]
    fn rotate_moves_oversized_file_to_backup() {
        let dir = std::env::temp_dir().join(format!("ats_log_rotate_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("app.log");

        std::fs::write(&path, vec![b'x'; 64]).unwrap();
        rotate_if_needed(&path, 32);
        assert!(!path.exists(), "oversized file must be rotated away");
        assert!(path.with_extension("log.1").exists());

        // Small file stays untouched
        std::fs::write(&path, b"small").unwrap();
        rotate_if_needed(&path, 32);
        assert!(path.exists());
        assert_eq!(std::fs::read(&path).unwrap(), b"small");

        // Old backup is replaced, not appended to
        std::fs::write(&path, vec![b'y'; 64]).unwrap();
        rotate_if_needed(&path, 32);
        assert_eq!(std::fs::read(path.with_extension("log.1")).unwrap(), vec![b'y'; 64]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
