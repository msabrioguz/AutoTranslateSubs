//! Persistent SQLite cache for translated subtitle lines.
//!
//! Key = (hash of normalized source text, source language, target language,
//! model). The hash narrows the lookup; the full stored text must match
//! exactly, so hash collisions degrade to a cache miss and can never return a
//! translation of a different line.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::models::Language;

/// SQLite-backed translation cache. Cheap to clone: all clones share one
/// connection behind a mutex, which serializes concurrent reads/writes.
#[derive(Clone)]
pub struct TranslationCache {
    conn: Arc<Mutex<Connection>>,
}

impl TranslationCache {
    /// Opens (creating it if needed) the cache database at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
        }
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open cache database {}", path.display()))?;
        // WAL: a crashed/interrupted run leaves a consistent database and only
        // loses at most the last transaction; readers never block the writer.
        conn.pragma_update(None, "journal_mode", "WAL")
            .context("failed to enable WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .context("failed to set synchronous mode")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS translation_cache (
                id INTEGER PRIMARY KEY,
                source_hash TEXT NOT NULL,
                source_text TEXT NOT NULL,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                model TEXT NOT NULL,
                translation TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE (source_hash, source_language, target_language, model)
            );",
        )
        .context("failed to create translation_cache table")?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Returns the cached translation for `source_text`, or `None` on a miss.
    ///
    /// A hit requires the normalized text, language pair **and** model to all
    /// match; translations from another model or language pair are never
    /// returned. Empty text is always a miss.
    pub fn get(
        &self,
        source_text: &str,
        source: Language,
        target: Language,
        model: &str,
    ) -> Result<Option<String>> {
        let text = normalize(source_text);
        if text.is_empty() {
            return Ok(None);
        }
        let hash = hash_text(&text);
        let conn = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("cache mutex poisoned"))?;
        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT source_text, translation FROM translation_cache
                 WHERE source_hash = ?1 AND source_language = ?2
                   AND target_language = ?3 AND model = ?4",
                params![hash, source.code(), target.code(), model],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .context("cache lookup failed")?;
        Ok(match row {
            // Hash collision with different text: treat as a miss. The stored
            // row stays valid for its own text.
            Some((stored, translation)) if stored == text => Some(translation),
            _ => None,
        })
    }

    /// Stores a translation. Best effort for the caller: empty inputs are
    /// skipped, and an existing row for the same key is replaced.
    pub fn put(
        &self,
        source_text: &str,
        translation: &str,
        source: Language,
        target: Language,
        model: &str,
    ) -> Result<()> {
        let text = normalize(source_text);
        if text.is_empty() || translation.trim().is_empty() {
            return Ok(());
        }
        let hash = hash_text(&text);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0) as i64;
        let conn = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("cache mutex poisoned"))?;
        conn.execute(
            "INSERT INTO translation_cache
                (source_hash, source_text, source_language, target_language,
                 model, translation, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(source_hash, source_language, target_language, model)
             DO UPDATE SET source_text = excluded.source_text,
                           translation = excluded.translation,
                           created_at = excluded.created_at",
            params![
                hash,
                text,
                source.code(),
                target.code(),
                model,
                translation,
                now
            ],
        )
        .context("cache insert failed")?;
        Ok(())
    }
}

/// Normalizes source text for caching: unified line endings + trimmed ends.
/// Interior whitespace (including newlines inside an entry) is preserved.
pub fn normalize(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim()
        .to_string()
}

/// FNV-1a 64-bit over UTF-8 bytes, formatted as 16 lowercase hex chars.
/// Spec-defined and stable across runs, platforms and Rust versions (unlike
/// `DefaultHasher`). Not cryptographic — collisions are handled by the exact
/// text comparison in [`TranslationCache::get`].
pub fn hash_text(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ats_cache_{}_{}.sqlite", name, std::process::id()))
    }

    fn cleanup(path: &Path) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{}", path.display(), suffix));
        }
    }

    fn open_temp(name: &str) -> (TranslationCache, PathBuf) {
        let path = temp_path(name);
        cleanup(&path);
        (TranslationCache::open(&path).expect("open cache"), path)
    }

    #[test]
    fn hit_after_put() {
        let (cache, path) = open_temp("hit");
        cache
            .put(
                "Hello world",
                "Merhaba dünya",
                Language::English,
                Language::Turkish,
                "m1",
            )
            .unwrap();
        let hit = cache
            .get("Hello world", Language::English, Language::Turkish, "m1")
            .unwrap();
        assert_eq!(hit.as_deref(), Some("Merhaba dünya"));
        cleanup(&path);
    }

    #[test]
    fn miss_when_absent() {
        let (cache, path) = open_temp("miss");
        let got = cache
            .get("Never stored", Language::English, Language::Turkish, "m1")
            .unwrap();
        assert_eq!(got, None);
        cleanup(&path);
    }

    #[test]
    fn duplicate_text_keeps_one_row() {
        let (cache, path) = open_temp("dup");
        for _ in 0..3 {
            cache
                .put(
                    "Same line",
                    "Aynı satır",
                    Language::English,
                    Language::Turkish,
                    "m1",
                )
                .unwrap();
        }
        let count: i64 = {
            let conn = cache.conn.lock().unwrap();
            conn.query_row("SELECT COUNT(*) FROM translation_cache", [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(count, 1);
        let hit = cache
            .get("Same line", Language::English, Language::Turkish, "m1")
            .unwrap();
        assert_eq!(hit.as_deref(), Some("Aynı satır"));
        cleanup(&path);
    }

    #[test]
    fn different_model_is_a_miss() {
        let (cache, path) = open_temp("model");
        cache
            .put(
                "Line",
                "Satır",
                Language::English,
                Language::Turkish,
                "model-a",
            )
            .unwrap();
        assert_eq!(
            cache
                .get("Line", Language::English, Language::Turkish, "model-b")
                .unwrap(),
            None
        );
        assert_eq!(
            cache
                .get("Line", Language::English, Language::Turkish, "model-a")
                .unwrap()
                .as_deref(),
            Some("Satır")
        );
        cleanup(&path);
    }

    #[test]
    fn different_language_pair_is_a_miss() {
        let (cache, path) = open_temp("langs");
        cache
            .put("Line", "Satır", Language::English, Language::Turkish, "m1")
            .unwrap();
        // Same source, different target
        assert_eq!(
            cache
                .get("Line", Language::English, Language::German, "m1")
                .unwrap(),
            None
        );
        // Reversed pair
        assert_eq!(
            cache
                .get("Line", Language::Turkish, Language::English, "m1")
                .unwrap(),
            None
        );
        cleanup(&path);
    }

    #[test]
    fn empty_text_and_empty_translation_are_skipped() {
        let (cache, path) = open_temp("empty");
        cache
            .put("", "boş", Language::English, Language::Turkish, "m1")
            .unwrap();
        cache
            .put(
                "Some line",
                "   ",
                Language::English,
                Language::Turkish,
                "m1",
            )
            .unwrap();
        assert_eq!(
            cache
                .get("", Language::English, Language::Turkish, "m1")
                .unwrap(),
            None
        );
        assert_eq!(
            cache
                .get("Some line", Language::English, Language::Turkish, "m1")
                .unwrap(),
            None
        );
        let count: i64 = {
            let conn = cache.conn.lock().unwrap();
            conn.query_row("SELECT COUNT(*) FROM translation_cache", [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(count, 0);
        cleanup(&path);
    }

    #[test]
    fn unicode_text_roundtrips_with_normalization() {
        let (cache, path) = open_temp("unicode");
        cache
            .put(
                "  Merhaba\r\nDünya \u{1F30D}  ",
                "Hello\nWorld",
                Language::Turkish,
                Language::English,
                "m1",
            )
            .unwrap();
        let hit = cache
            .get(
                "Merhaba\nDünya \u{1F30D}",
                Language::Turkish,
                Language::English,
                "m1",
            )
            .unwrap();
        assert_eq!(hit.as_deref(), Some("Hello\nWorld"));
        cleanup(&path);
    }

    #[test]
    fn hash_collision_with_different_text_is_a_miss() {
        let (cache, path) = open_temp("collision");
        // Simulate a collision: a row whose hash was computed for "alpha"
        // but whose stored text is "beta".
        {
            let conn = cache.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO translation_cache
                    (source_hash, source_text, source_language, target_language,
                     model, translation, created_at)
                 VALUES (?1, 'beta', 'en', 'tr', 'm1', 'yanlis', 0)",
                params![hash_text("alpha")],
            )
            .unwrap();
        }
        // Looking up "alpha" finds the row by hash but must miss on text.
        assert_eq!(
            cache
                .get("alpha", Language::English, Language::Turkish, "m1")
                .unwrap(),
            None
        );
        // "beta" hashes differently, so the shadowed row is unreachable —
        // a collision can only ever cause misses, never a wrong translation.
        assert_eq!(
            cache
                .get("beta", Language::English, Language::Turkish, "m1")
                .unwrap(),
            None
        );
        // Storing the real "alpha" replaces the shadowed row and heals it.
        cache
            .put("alpha", "dogru", Language::English, Language::Turkish, "m1")
            .unwrap();
        assert_eq!(
            cache
                .get("alpha", Language::English, Language::Turkish, "m1")
                .unwrap()
                .as_deref(),
            Some("dogru")
        );
        cleanup(&path);
    }

    #[test]
    fn entries_survive_reopen() {
        let (cache, path) = open_temp("persist");
        cache
            .put(
                "Persisted",
                "Kalıcı",
                Language::English,
                Language::Turkish,
                "m1",
            )
            .unwrap();
        drop(cache);
        let reopened = TranslationCache::open(&path).unwrap();
        assert_eq!(
            reopened
                .get("Persisted", Language::English, Language::Turkish, "m1")
                .unwrap()
                .as_deref(),
            Some("Kalıcı")
        );
        cleanup(&path);
    }

    #[test]
    fn concurrent_reads_and_writes_are_safe() {
        let (cache, path) = open_temp("concurrent");
        let mut handles = Vec::new();
        for t in 0..4 {
            let cache = cache.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..25 {
                    let text = format!("line {} thread {}", i, t);
                    cache
                        .put(&text, "çeviri", Language::English, Language::Turkish, "m1")
                        .unwrap();
                    let _ = cache
                        .get(&text, Language::English, Language::Turkish, "m1")
                        .unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().expect("thread panicked");
        }
        assert_eq!(
            cache
                .get(
                    "line 0 thread 0",
                    Language::English,
                    Language::Turkish,
                    "m1"
                )
                .unwrap()
                .as_deref(),
            Some("çeviri")
        );
        cleanup(&path);
    }
}
