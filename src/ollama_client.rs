use crate::models::{
    AppConfig, BatchTranslationEntry, BatchTranslationRequest, BatchTranslationResponse, Language,
    OllamaModelsResponse, TranslationRequest, TranslationResponse,
};
use anyhow::{Context, Result};
use reqwest::blocking::Client;
use std::sync::Arc;
use std::time::Duration;

const TRANSLATE_CHUNK_SIZE: usize = 25;

/// Timing/token counters collected while translating one file.
#[derive(Debug, Default, Clone)]
pub struct RequestStats {
    pub batch_requests: u32,
    pub batch_seconds: f64,
    pub batch_entries: u32,
    /// Entries actually written to results by batch responses
    pub batch_covered: u32,
    /// Chunks mapped by position because the model returned wrong indexes
    pub batch_positional: u32,
    pub batch_prompt_tokens: u64,
    pub batch_gen_tokens: u64,
    /// Batch request errors or unparseable responses (trigger fallback)
    pub batch_failures: u32,
    pub single_requests: u32,
    pub single_seconds: f64,
    pub single_prompt_tokens: u64,
    pub single_gen_tokens: u64,
    /// One line per problematic batch chunk: parsed/covered counts, a sample
    /// of returned indexes and a preview of the raw model response.
    pub batch_diagnostics: Vec<String>,
}

impl RequestStats {
    pub fn summary(&self) -> String {
        format!(
            "batch: {} req, {:.1}s, {}/{} covered, {} prompt tok, {} gen tok, {} fail, {} positional | \
             single: {} req, {:.1}s, {} prompt tok, {} gen tok",
            self.batch_requests,
            self.batch_seconds,
            self.batch_covered,
            self.batch_entries,
            self.batch_prompt_tokens,
            self.batch_gen_tokens,
            self.batch_failures,
            self.batch_positional,
            self.single_requests,
            self.single_seconds,
            self.single_prompt_tokens,
            self.single_gen_tokens,
        )
    }
}

#[derive(Clone)]
pub struct OllamaClient {
    client: Arc<Client>,
    config: AppConfig,
}

impl OllamaClient {
    pub fn new(config: &AppConfig) -> Result<Self> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(300))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client: Arc::new(client),
            config: config.clone(),
        })
    }

    pub fn set_config(&mut self, config: &AppConfig) {
        self.config = config.clone();
    }

    pub fn fetch_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.config.ollama_url);
        let response = self
            .client
            .get(&url)
            .send()
            .with_context(|| format!("Failed to connect to Ollama at {}", self.config.ollama_url))?;

        if !response.status().is_success() {
            anyhow::bail!("Ollama returned error: {}", response.status());
        }

        let models_response: OllamaModelsResponse =
            response.json().context("Failed to parse models response")?;

        Ok(models_response.models.into_iter().map(|m| m.name).collect())
    }

    /// Translates a single text with a plain-text prompt (no JSON involved).
    fn translate_single(
        &self,
        text: &str,
        source_lang: Language,
        target_lang: Language,
    ) -> Result<TranslationResponse> {
        let prompt = self.build_translation_prompt(text, source_lang, target_lang);

        let request = TranslationRequest {
            model: self.config.selected_model.clone(),
            prompt,
            stream: false,
            options: Default::default(),
        };

        let url = format!("{}/api/generate", self.config.ollama_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .with_context(|| "Failed to send translation request to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().unwrap_or_default();
            anyhow::bail!("Ollama translation failed: {} - {}", status, error_body);
        }

        let translation_response: TranslationResponse = response
            .json()
            .context("Failed to parse translation response")?;

        Ok(translation_response)
    }

    /// Sends one batch request for `chunk`; returns the raw model response.
    fn request_batch_chunk(
        &self,
        chunk: &[(usize, &String)],
        source_lang: Language,
        target_lang: Language,
    ) -> Result<TranslationResponse> {
        let prompt = self.build_batch_translation_prompt(chunk, source_lang, target_lang);

        let request = BatchTranslationRequest {
            model: self.config.selected_model.clone(),
            prompt,
            stream: false,
            format: "json".to_string(),
            options: Default::default(),
        };

        let url = format!("{}/api/generate", self.config.ollama_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .with_context(|| "Failed to send batch translation request to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().unwrap_or_default();
            anyhow::bail!("Ollama batch translation failed: {} - {}", status, error_body);
        }

        let translation_response: TranslationResponse = response
            .json()
            .context("Failed to parse batch translation response")?;

        Ok(translation_response)
    }

    /// Translates `texts` in chunks of `TRANSLATE_CHUNK_SIZE` entries.
    ///
    /// Fast path: one batch request per chunk. Any entry the batch response
    /// does not cover (parse failure, partial/short response, empty text)
    /// is then translated one-by-one with plain-text requests, so every
    /// non-empty text is translated in order.
    ///
    /// Every request is timed and token counts from Ollama are collected in
    /// the returned [`RequestStats`].
    ///
    /// `on_progress(fraction, message)` is called before and after each
    /// request; return `false` to abort (returns an error in that case).
    pub fn translate_batch<F>(
        &self,
        texts: &[String],
        source_lang: Language,
        target_lang: Language,
        mut on_progress: F,
    ) -> Result<(Vec<String>, RequestStats)>
    where
        F: FnMut(f32, String) -> bool,
    {
        let mut stats = RequestStats::default();

        if texts.is_empty() {
            return Ok((Vec::new(), stats));
        }

        // Filter out empty texts but keep track of indices
        let non_empty: Vec<(usize, &String)> = texts
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.trim().is_empty())
            .collect();

        if non_empty.is_empty() {
            return Ok((vec![String::new(); texts.len()], stats));
        }

        let total = non_empty.len();
        let mut results = vec![String::new(); texts.len()];

        let count_done = |results: &[String]| -> usize {
            results
                .iter()
                .zip(texts.iter())
                .filter(|(r, t)| !t.trim().is_empty() && !r.trim().is_empty())
                .count()
        };
        let fraction = |done: usize| 0.1 + 0.8 * done as f32 / total as f32;

        for chunk in non_empty.chunks(TRANSLATE_CHUNK_SIZE) {
            let done = count_done(&results);
            let message = format!(
                "Translating entries {}-{} of {}...",
                done + 1,
                done + chunk.len(),
                total
            );
            if !on_progress(fraction(done), message) {
                anyhow::bail!("translation interrupted");
            }

            // Fast path: batch request. Failures fall back to single requests.
            let started = std::time::Instant::now();
            let batch_resp = self.request_batch_chunk(chunk, source_lang, target_lang);
            let elapsed = started.elapsed().as_secs_f64();
            stats.batch_requests += 1;
            stats.batch_seconds += elapsed;
            stats.batch_entries += chunk.len() as u32;

            let mut parsed_ok = false;
            let mut chunk_covered = 0u32;
            let chunk_first = chunk.first().map(|(i, _)| *i).unwrap_or(0);
            let chunk_last = chunk.last().map(|(i, _)| *i).unwrap_or(0);
            let preview = |raw: &str| -> String {
                raw.chars()
                    .take(200)
                    .collect::<String>()
                    .replace(['\n', '\r'], " ")
            };
            match batch_resp {
                Ok(resp) => {
                    stats.batch_prompt_tokens += resp.prompt_eval_count.unwrap_or(0);
                    stats.batch_gen_tokens += resp.eval_count.unwrap_or(0);
                    match parse_batch_response(&resp.response) {
                        Ok(entries) => {
                            parsed_ok = true;
                            let parsed_count = entries.len();
                            let idx_sample: Vec<usize> =
                                entries.iter().map(|e| e.index).take(10).collect();
                            let (covered, positional) =
                                map_batch_entries(entries, chunk, &mut results);
                            chunk_covered = covered;
                            stats.batch_covered += covered;
                            if positional {
                                stats.batch_positional += 1;
                            }
                            if covered < chunk.len() as u32 {
                                stats.batch_diagnostics.push(format!(
                                    "chunk {}-{}: parsed {}, covered {}, idx {:?} | {}",
                                    chunk_first,
                                    chunk_last,
                                    parsed_count,
                                    covered,
                                    idx_sample,
                                    preview(&resp.response)
                                ));
                            }
                        }
                        Err(e) => {
                            stats.batch_failures += 1;
                            stats.batch_diagnostics.push(format!(
                                "chunk {}-{}: parse error: {} | {}",
                                chunk_first,
                                chunk_last,
                                e,
                                preview(&resp.response)
                            ));
                        }
                    }
                }
                Err(e) => {
                    stats.batch_failures += 1;
                    stats
                        .batch_diagnostics
                        .push(format!("chunk {}-{}: request error: {}", chunk_first, chunk_last, e));
                }
            }

            let done = count_done(&results);
            let msg = if parsed_ok {
                format!(
                    "Batch: {} entries in {:.1}s ({} covered, {}/{} done)",
                    chunk.len(),
                    elapsed,
                    chunk_covered,
                    done,
                    total
                )
            } else {
                format!(
                    "Batch failed in {:.1}s, falling back to single requests ({}/{} done)",
                    elapsed, done, total
                )
            };
            if !on_progress(fraction(done), msg) {
                anyhow::bail!("translation interrupted");
            }

            // Fallback: translate every still-missing entry of this chunk one by one
            for (idx, text) in chunk {
                if !results[*idx].trim().is_empty() {
                    continue;
                }
                let done = count_done(&results);
                let message = format!("Translating {}/{}...", done + 1, total);
                if !on_progress(fraction(done), message) {
                    anyhow::bail!("translation interrupted");
                }

                let started = std::time::Instant::now();
                let resp = self.translate_single(text, source_lang, target_lang)?;
                let elapsed = started.elapsed().as_secs_f64();
                stats.single_requests += 1;
                stats.single_seconds += elapsed;
                stats.single_prompt_tokens += resp.prompt_eval_count.unwrap_or(0);
                stats.single_gen_tokens += resp.eval_count.unwrap_or(0);
                results[*idx] = clean_single_response(&resp.response);

                let done = count_done(&results);
                let message = format!(
                    "Single {}/{} in {:.1}s (fallback)",
                    done,
                    total,
                    elapsed
                );
                if !on_progress(fraction(done), message) {
                    anyhow::bail!("translation interrupted");
                }
            }
        }

        Ok((results, stats))
    }

    fn build_translation_prompt(
        &self,
        text: &str,
        source_lang: Language,
        target_lang: Language,
    ) -> String {
        let source_name = source_lang.name();
        let target_name = target_lang.name();

        format!(
            "You are a professional subtitle translator. Translate the following text from {} to {}.\n\
             Important rules:\n\
             1. Keep the translation natural and idiomatic for subtitles\n\
             2. Preserve formatting, line breaks, and timing cues\n\
             3. Do not add explanations or notes\n\
             4. Return ONLY the translated text\n\
             5. Keep proper names and technical terms as-is when appropriate\n\
             \n\
             Text to translate:\n{}",
            source_name, target_name, text
        )
    }

    fn build_batch_translation_prompt(
        &self,
        entries: &[(usize, &String)],
        source_lang: Language,
        target_lang: Language,
    ) -> String {
        let source_name = source_lang.name();
        let target_name = target_lang.name();

        let entries_json = serde_json::to_string(
            &entries
                .iter()
                .map(|(i, t)| serde_json::json!({"index": *i, "text": *t}))
                .collect::<Vec<_>>(),
        )
        .unwrap_or_default();

        format!(
            "You are a professional subtitle translator. Translate the following texts from {} to {}.\n\
             \n\
             Texts to translate (JSON array):\n\
             {}\n\
             \n\
             Rules:\n\
             1. Keep translations natural and idiomatic for subtitles\n\
             2. Preserve line breaks\n\
             3. Keep proper names and technical terms as-is when appropriate\n\
             4. Your ENTIRE response must be a single valid JSON array and nothing else.\n\
                The first character of your response must be '[' and the last character ']'.\n\
                Do not use markdown, comments, or explanatory text.\n\
             5. Include exactly one object per input text, keeping the same index numbers.\n\
             \n\
             Example output:\n\
             [{{\"index\": 0, \"text\": \"translated text 1\"}}, {{\"index\": 1, \"text\": \"translated text 2\"}}]",
            source_name, target_name, entries_json
        )
    }

    pub fn test_connection(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.config.ollama_url);
        let response = self.client.get(&url).send().with_context(|| {
            format!("Cannot reach Ollama at {}", self.config.ollama_url)
        })?;

        Ok(response.status().is_success())
    }
}

/// Cleans a single-text translation response: trims whitespace and strips
/// wrapping double quotes that some models add around the whole answer.
fn clean_single_response(raw: &str) -> String {
    let t = raw.trim();
    if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
        // Parse as a JSON string so escapes like \" are handled correctly
        if let Ok(s) = serde_json::from_str::<String>(t) {
            return s;
        }
        return t[1..t.len() - 1].trim().to_string();
    }
    t.to_string()
}

/// Maps a parsed batch response onto `results` for one chunk.
///
/// Strategy, in order of confidence:
/// 1. **Index mapping** when every returned index belongs to this chunk and
///    is distinct — no ambiguity (covers perfect and truncated responses).
/// 2. **Positional mapping** when the count matches, indexes are distinct,
///    but *none* of them belong to this chunk: the model renumbered them
///    (e.g. every chunk restarts at 0). Entry `k` is assumed to be the
///    translation of `chunk[k]`.
/// 3. **Partial index mapping** for mixed/ambiguous responses: only entries
///    whose index belongs to *this* chunk are accepted (each slot once), so
///    wrong indexes can never overwrite another chunk's translations.
///    Everything else is left for the single-request fallback.
///
/// Returns `(covered, used_positional)`. Entries with empty text are skipped.
fn map_batch_entries(
    entries: Vec<BatchTranslationEntry>,
    chunk: &[(usize, &String)],
    results: &mut [String],
) -> (u32, bool) {
    if chunk.is_empty() {
        return (0, false);
    }
    let chunk_indices: Vec<usize> = chunk.iter().map(|(i, _)| *i).collect();
    let mut covered = 0u32;

    let mut seen = std::collections::HashSet::with_capacity(entries.len());
    let distinct = entries.iter().all(|e| seen.insert(e.index));
    let all_in_chunk = entries.iter().all(|e| chunk_indices.contains(&e.index));
    let none_in_chunk = entries.iter().all(|e| !chunk_indices.contains(&e.index));

    if all_in_chunk && distinct {
        for e in entries {
            if !e.text.trim().is_empty() {
                results[e.index] = e.text;
                covered += 1;
            }
        }
        return (covered, false);
    }

    if entries.len() == chunk.len() && distinct && none_in_chunk {
        // Model renumbered its indexes: fall back to position
        for (e, (idx, _)) in entries.into_iter().zip(chunk) {
            if !e.text.trim().is_empty() {
                results[*idx] = e.text;
                covered += 1;
            }
        }
        return (covered, true);
    }

    // Mixed or ambiguous response: accept only this chunk's indexes, each once
    for e in entries {
        if chunk_indices.contains(&e.index)
            && !e.text.trim().is_empty()
            && results[e.index].is_empty()
        {
            results[e.index] = e.text;
            covered += 1;
        }
    }
    (covered, false)
}

fn parse_batch_response(raw: &str) -> Result<Vec<BatchTranslationEntry>> {
    let trimmed = raw.trim();

    // 1. Expected format: a JSON array of entries
    if let Ok(entries) = serde_json::from_str::<Vec<BatchTranslationEntry>>(trimmed) {
        return Ok(entries);
    }

    // 2. Wrapped format: {"translations": [...]}
    if let Ok(wrapper) = serde_json::from_str::<BatchTranslationResponse>(trimmed) {
        return Ok(wrapper.translations);
    }

    // 3. Single object: {"index": 0, "text": "..."} (model returned only one entry)
    if let Ok(entry) = serde_json::from_str::<BatchTranslationEntry>(trimmed) {
        return Ok(vec![entry]);
    }

    // 4. Model returned multiple JSON objects (e.g., separated by newlines)
    let mut entries = Vec::new();
    for obj in extract_json_objects(trimmed) {
        if let Ok(entry) = serde_json::from_str::<BatchTranslationEntry>(&obj) {
            entries.push(entry);
        }
    }
    if !entries.is_empty() {
        return Ok(entries);
    }

    let preview: String = trimmed.chars().take(300).collect();
    anyhow::bail!("Failed to parse batch translation response: {}", preview);
}

/// Extracts balanced `{...}` substrings from text, respecting strings and escapes.
fn extract_json_objects(s: &str) -> Vec<String> {
    let mut objects = Vec::new();
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escape = false;

    for (i, c) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match c {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' if !in_string && depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    if let Some(st) = start.take() {
                        objects.push(s[st..=i].to_string());
                    }
                }
            }
            _ => {}
        }
    }
    objects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_array_format() {
        let raw = r#"[{"index": 0, "text": "Merhaba"}, {"index": 1, "text": "Dünya"}]"#;
        let entries = parse_batch_response(raw).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].index, 0);
        assert_eq!(entries[0].text, "Merhaba");
        assert_eq!(entries[1].index, 1);
        assert_eq!(entries[1].text, "Dünya");
    }

    #[test]
    fn parse_wrapped_format() {
        let raw = r#"{"translations": [{"index": 0, "text": "Merhaba"}]}"#;
        let entries = parse_batch_response(raw).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "Merhaba");
    }

    #[test]
    fn parse_single_object_format() {
        // Model returned a single object instead of an array (observed in the wild)
        let raw = "{\n\"index\": 0,\n\"text\": \"Kendi korku hayatta kalma oyununuzu yaratmayı\\nhayal ettiniz mi?\"\n}";
        let entries = parse_batch_response(raw).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].index, 0);
        assert!(entries[0].text.contains("hayal ettiniz mi?"));
    }

    #[test]
    fn parse_newline_separated_objects() {
        let raw = "{\"index\": 0, \"text\": \"Birinci\"}\n{\"index\": 1, \"text\": \"İkinci\"}\n{\"index\": 2, \"text\": \"Üçüncü\"}";
        let entries = parse_batch_response(raw).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].text, "Birinci");
        assert_eq!(entries[1].text, "İkinci");
        assert_eq!(entries[2].text, "Üçüncü");
    }

    #[test]
    fn parse_object_with_nested_quotes() {
        let raw = "{\"index\": 0, \"text\": \"Ali \\\"Veli\\\" dedi\"}\n{\"index\": 1, \"text\": \"Second\"}";
        let entries = parse_batch_response(raw).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Ali \"Veli\" dedi");
    }

    #[test]
    fn parse_invalid_formats() {
        assert!(parse_batch_response("not json").is_err());
        assert!(parse_batch_response("\"just a string\"").is_err());
        assert!(parse_batch_response("").is_err());
        assert!(parse_batch_response("42").is_err());
    }

    #[test]
    fn parse_skips_missing_indices() {
        // Model returned fewer entries than expected; caller keeps empty strings
        let raw = r#"[{"index": 0, "text": "only first"}]"#;
        let entries = parse_batch_response(raw).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].index, 0);
    }

    #[test]
    fn clean_response_trims_and_strips_quotes() {
        assert_eq!(clean_single_response("  Merhaba Dünya \n"), "Merhaba Dünya");
        assert_eq!(clean_single_response("\"Merhaba\""), "Merhaba");
        // Wrapped JSON string with escapes is unescaped
        assert_eq!(
            clean_single_response("\"Ali \\\"Veli\\\" dedi\""),
            "Ali \"Veli\" dedi"
        );
        // Legitimate quotes inside only, untouched
        assert_eq!(clean_single_response("\"Merhaba\" dedi"), "\"Merhaba\" dedi");
        // No wrapping quotes at both ends -> untouched
        assert_eq!(clean_single_response("Merhaba \"Veli\""), "Merhaba \"Veli\"");
    }

    fn entry(index: usize, text: &str) -> BatchTranslationEntry {
        BatchTranslationEntry {
            index,
            text: text.to_string(),
        }
    }

    fn chunk_of(indices: &[usize], texts: &[&str]) -> Vec<(usize, String)> {
        indices
            .iter()
            .zip(texts.iter())
            .map(|(i, t)| (*i, t.to_string()))
            .collect()
    }

    #[test]
    fn map_uses_indexes_when_model_keeps_them() {
        let owned = chunk_of(&[25, 26], &["a", "b"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        let mut results = vec![String::new(); 30];

        let (covered, positional) =
            map_batch_entries(vec![entry(25, "x"), entry(26, "y")], &chunk, &mut results);

        assert_eq!((covered, positional), (2, false));
        assert_eq!(results[25], "x");
        assert_eq!(results[26], "y");
    }

    #[test]
    fn map_falls_back_to_position_when_model_renumbers() {
        // Chunk covers global indexes 25..27 but the model restarts at 0
        let owned = chunk_of(&[25, 26, 27], &["a", "b", "c"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        let mut results = vec![String::new(); 30];

        let (covered, positional) = map_batch_entries(
            vec![entry(0, "x"), entry(1, "y"), entry(2, "z")],
            &chunk,
            &mut results,
        );

        assert_eq!((covered, positional), (3, true));
        assert_eq!(results[25], "x");
        assert_eq!(results[26], "y");
        assert_eq!(results[27], "z");
        assert!(results[0].is_empty(), "must not touch other chunks' slots");
    }

    #[test]
    fn map_partial_response_only_accepts_own_chunk_indexes() {
        let owned = chunk_of(&[25, 26, 27], &["a", "b", "c"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        let mut results = vec![String::new(); 30];
        results[0] = "keep-me".to_string();

        // Truncated/wrong response: one foreign index, two correct ones
        let (covered, positional) = map_batch_entries(
            vec![entry(0, "WRONG"), entry(25, "x"), entry(27, "z")],
            &chunk,
            &mut results,
        );

        assert_eq!((covered, positional), (2, false));
        assert_eq!(results[0], "keep-me", "foreign index must be ignored");
        assert_eq!(results[25], "x");
        assert!(results[26].is_empty(), "missing entry left for fallback");
        assert_eq!(results[27], "z");
    }

    #[test]
    fn map_skips_empty_texts() {
        let owned = chunk_of(&[0, 1], &["a", "b"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        let mut results = vec![String::new(); 5];

        let (covered, _) =
            map_batch_entries(vec![entry(0, "x"), entry(1, "  ")], &chunk, &mut results);

        assert_eq!(covered, 1);
        assert_eq!(results[0], "x");
        assert!(results[1].is_empty());
    }
}