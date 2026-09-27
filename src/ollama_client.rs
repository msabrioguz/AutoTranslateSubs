use crate::batch_builder::estimate_tokens;
use crate::models::{
    AppConfig, BatchTranslationEntry, BatchTranslationRequest, BatchTranslationResponse, Language,
    OllamaModelsResponse, TranslationRequest, TranslationResponse,
};
use crate::translation_cache::TranslationCache;
use anyhow::{Context, Result};
use reqwest::blocking::Client;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Entries per batch request. Public so the benchmark can report it.
pub const TRANSLATE_CHUNK_SIZE: usize = 25;

/// Timing/token counters collected while translating one file.
#[derive(Debug, Default, Clone)]
pub struct RequestStats {
    pub batch_requests: u32,
    pub batch_seconds: f64,
    pub batch_entries: u32,
    /// Entries actually written to results by batch responses
    pub batch_covered: u32,
    pub batch_prompt_tokens: u64,
    pub batch_gen_tokens: u64,
    /// Model generation time reported by Ollama (`eval_duration`, seconds)
    pub batch_eval_seconds: f64,
    /// Batch request errors or unparseable responses after retries
    /// (trigger fallback)
    pub batch_failures: u32,
    pub single_requests: u32,
    pub single_seconds: f64,
    pub single_prompt_tokens: u64,
    pub single_gen_tokens: u64,
    /// Model generation time reported by Ollama (`eval_duration`, seconds)
    pub single_eval_seconds: f64,
    /// Entries served from the persistent translation cache
    pub cache_hits: u32,
    /// Entries whose cache lookup found nothing (or cache was disabled)
    pub cache_misses: u32,
    /// Extra attempts beyond the first for transient failures (Task 08).
    /// Each retry is a separate HTTP call; its cost is included in
    /// `batch_seconds`/`single_seconds` wall time and shown separately via
    /// `retry_backoff_seconds`.
    pub retries: u32,
    /// Time spent sleeping in backoff between retries (seconds, measured).
    pub retry_backoff_seconds: f64,
    /// Response attempts rejected by validation (Task 08 checks: missing/
    /// foreign/duplicate ids, empty translation) before anything was written.
    /// Each rejected attempt counts once — retried attempts included, so the
    /// number shows how often validation saved the output.
    pub validation_failures: u32,
    /// One line per problematic batch chunk: failure reason (parse/validation
    /// error, HTTP status) and, when available, a preview of the raw model
    /// response.
    pub batch_diagnostics: Vec<String>,
}

impl RequestStats {
    pub fn summary(&self) -> String {
        format!(
            "batch: {} req, {:.1}s, {}/{} covered, {} prompt tok, {} gen tok, {} fail | \
             single: {} req, {:.1}s, {} prompt tok, {} gen tok | cache: {} hit, {} miss | \
             retries: {} ({:.1}s backoff)",
            self.batch_requests,
            self.batch_seconds,
            self.batch_covered,
            self.batch_entries,
            self.batch_prompt_tokens,
            self.batch_gen_tokens,
            self.batch_failures,
            self.single_requests,
            self.single_seconds,
            self.single_prompt_tokens,
            self.single_gen_tokens,
            self.cache_hits,
            self.cache_misses,
            self.retries,
            self.retry_backoff_seconds,
        )
    }
}

/// Idle HTTP connections kept alive per host.
///
/// The app talks to exactly one host (Ollama) and has at most three
/// concurrent callers (model-load thread, connection check, one job worker);
/// the request path inside a job is sequential. reqwest's default of 2 idle
/// connections would force a fresh TCP handshake as soon as a third caller
/// overlaps, so 8 keeps connections warm with bounded headroom.
const POOL_MAX_IDLE_PER_HOST: usize = 8;

/// Hard cap on the estimated tokens spent on reference context per request.
///
/// Rationale: the runtime `num_ctx` is not set by the app (server default
/// applies; worst-case floor 2048), the model declares 131 072 — 512 tokens
/// is ≤25% of the worst case and ≤0.4% of the declared window, so context can
/// never crowd out the translation payload. Context is additionally folded
/// into the `max_input_tokens` batch budget (see `build_context`).
const CONTEXT_TOKEN_CAP: usize = 512;

#[derive(Clone)]
pub struct OllamaClient {
    client: Arc<Client>,
    config: AppConfig,
    /// Persistent translation cache. `None` when disabled (`--no-cache`,
    /// e2e test) or when the database cannot be opened — caching is
    /// best-effort and never blocks a translation run.
    cache: Option<TranslationCache>,
}

impl OllamaClient {
    pub fn new(config: &AppConfig) -> Result<Self> {
        Self::build(config, Duration::from_secs(300))
    }

    fn build(config: &AppConfig, request_timeout: Duration) -> Result<Self> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(request_timeout)
            .pool_max_idle_per_host(POOL_MAX_IDLE_PER_HOST)
            .build()
            .context("Failed to create HTTP client")?;

        let cache_path = AppConfig::config_path()
            .parent()
            .map(|dir| dir.join("cache.sqlite"));
        let cache = cache_path.and_then(|path| TranslationCache::open(&path).ok());

        Ok(Self {
            client: Arc::new(client),
            config: config.clone(),
            cache,
        })
    }

    /// Test-only constructor with a short request timeout, used to exercise
    /// timeout classification of the retry logic against a mock server
    /// (production uses 300s — long batches must never be cut off).
    #[cfg(test)]
    fn with_request_timeout(config: &AppConfig, timeout: Duration) -> Result<Self> {
        Self::build(config, timeout)
    }

    /// Turns the translation cache off for this client (used by the
    /// benchmark's `--no-cache` mode and the e2e test).
    pub fn disable_cache(&mut self) {
        self.cache = None;
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
    ) -> Result<TranslationResponse, Failure> {
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
            .map_err(|e| classify_reqwest("Failed to send translation request to Ollama", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().unwrap_or_default();
            return Err(classify_status(
                status,
                "Ollama translation failed",
                &error_body,
            ));
        }

        let translation_response: TranslationResponse = response
            .json()
            .map_err(|e| classify_reqwest("Failed to parse translation response", e))?;

        Ok(translation_response)
    }

    /// Sends one batch request for `chunk`; returns the raw model response.
    ///
    /// Uses Ollama structured output (`format` = JSON schema) with
    /// `minItems`/`maxItems` pinned to the chunk size. Plain
    /// `format: "json"` (and a schema without item-count bounds) makes this
    /// model end after the first object — measured live on 2026-09-27 —
    /// while the bounded schema forces the full array (25/25 covered).
    /// `validate_chunk_entries` still checks the result before it is used.
    fn request_batch_chunk(
        &self,
        chunk: &[(usize, &String)],
        context_before: &[String],
        context_after: &[String],
        source_lang: Language,
        target_lang: Language,
    ) -> Result<TranslationResponse, Failure> {
        let prompt = self.build_batch_translation_prompt(
            chunk,
            context_before,
            context_after,
            source_lang,
            target_lang,
        );

        let request = BatchTranslationRequest {
            model: self.config.selected_model.clone(),
            prompt,
            stream: false,
            options: Default::default(),
            format: Some(batch_output_schema(chunk.len())),
        };

        let url = format!("{}/api/generate", self.config.ollama_url);
        let response = self.client.post(&url).json(&request).send().map_err(|e| {
            classify_reqwest("Failed to send batch translation request to Ollama", e)
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().unwrap_or_default();
            return Err(classify_status(
                status,
                "Ollama batch translation failed",
                &error_body,
            ));
        }

        let translation_response: TranslationResponse = response
            .json()
            .map_err(|e| classify_reqwest("Failed to parse batch translation response", e))?;

        Ok(translation_response)
    }

    /// Translates `texts` in token-budgeted batches (see
    /// [`crate::batch_builder::build_batches`]): items accumulate until the
    /// configured `max_input_tokens` budget would be exceeded, with
    /// `TRANSLATE_CHUNK_SIZE` as a hard ceiling per request.
    ///
    /// Fast path: one batch request per batch. Every response is validated
    /// before it is written (requested ids present, no foreign/duplicate
    /// ids, no empty translations — Task 08) and transient failures
    /// (connection reset, timeout, 5xx, malformed or invalid responses) are
    /// retried with bounded exponential backoff up to `config.max_retries`
    /// times; permanent failures (e.g. 404) are never retried. Any entry
    /// still missing after retries is translated one-by-one with plain-text
    /// requests, so every non-empty text ends up translated in input order.
    ///
    /// Returns `Err` (instead of partial/corrupt output) if a single request
    /// exhausts its attempts or if the final vector cannot be reconstructed
    /// one-for-one (`validate_complete_output`).
    ///
    /// With `config.max_concurrent_requests > 1` a fixed pool of up to that
    /// many worker threads processes the batches in parallel (spawned once
    /// per call; the pool size IS the concurrency limit, so no unbounded
    /// request spawning is possible — a semaphore would need an async
    /// runtime this blocking client does not have). Output ordering is
    /// independent of completion order: every entry is written to its own
    /// index under a lock, so the returned vector always matches the input
    /// order. A failed or interrupted run stops the pool; entries of other
    /// in-flight batches are left untouched (they are already valid).
    ///
    /// Every request is timed and token counts from Ollama are collected in
    /// the returned [`RequestStats`].
    ///
    /// `on_progress(fraction, message)` is called before and after each
    /// request (serialized across workers); return `false` to abort (returns
    /// an error in that case).
    pub fn translate_batch<F>(
        &self,
        texts: &[String],
        source_lang: Language,
        target_lang: Language,
        on_progress: F,
    ) -> Result<(Vec<String>, RequestStats)>
    where
        F: FnMut(f32, String) -> bool + Send,
    {
        if texts.is_empty() {
            return Ok((Vec::new(), RequestStats::default()));
        }

        // Filter out empty texts but keep track of indices
        let non_empty: Vec<(usize, &String)> = texts
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.trim().is_empty())
            .collect();

        if non_empty.is_empty() {
            return Ok((vec![String::new(); texts.len()], RequestStats::default()));
        }

        let total = non_empty.len();
        let results = Mutex::new(vec![String::new(); texts.len()]);
        let stats = Mutex::new(RequestStats::default());

        // Cache lookup happens before any Ollama request: hits are prefilled
        // and their entries never reach a request. Any cache problem degrades
        // to a miss so a broken cache can never fail a translation run.
        // (Only this pre-spawn phase may hold two locks at once; the workers
        // always take `results` before `stats` and never the reverse.)
        {
            let mut results_guard = results.lock().unwrap();
            let mut stats_guard = stats.lock().unwrap();
            if let Some(cache) = &self.cache {
                for (idx, text) in &non_empty {
                    match cache.get(text, source_lang, target_lang, &self.config.selected_model) {
                        Ok(Some(hit)) => {
                            results_guard[*idx] = hit;
                            stats_guard.cache_hits += 1;
                        }
                        Ok(None) | Err(_) => stats_guard.cache_misses += 1,
                    }
                }
            }
        }

        // Only misses are sent to Ollama, grouped by the input-token budget
        // (`max_input_tokens`) with TRANSLATE_CHUNK_SIZE as a hard ceiling.
        let pending: Vec<(usize, &String)> = {
            let results_guard = results.lock().unwrap();
            non_empty
                .iter()
                .copied()
                .filter(|(idx, _)| results_guard[*idx].trim().is_empty())
                .collect()
        };
        let pending_refs: Vec<(usize, &str)> = pending
            .iter()
            .map(|(idx, text)| (*idx, text.as_str()))
            .collect();
        let batches = crate::batch_builder::build_batches(
            &pending_refs,
            self.config.max_input_tokens,
            TRANSLATE_CHUNK_SIZE,
        );

        let count_done = |results: &[String]| -> usize {
            results
                .iter()
                .zip(texts.iter())
                .filter(|(r, t)| !t.trim().is_empty() && !r.trim().is_empty())
                .count()
        };
        let fraction = |done: usize| 0.1 + 0.8 * done as f32 / total as f32;
        let workers = effective_concurrency(self.config.max_concurrent_requests, batches.len());
        let retry_policy = RetryPolicy::from_config(&self.config);

        let progress = Mutex::new(on_progress);
        let queue = Mutex::new(batches.into_iter());
        let stop = AtomicBool::new(false);
        let fatal: Mutex<Option<anyhow::Error>> = Mutex::new(None);

        // Records the first fatal error and stops the pool; later failures
        // are ignored (first error wins). Workers check `stop` between units
        // of work, so in-flight requests finish normally before joining.
        let record_fatal = |err: anyhow::Error| {
            {
                let mut slot = fatal.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(err);
                }
            }
            stop.store(true, Ordering::SeqCst);
        };

        // Fixed-size worker pool: the pool size IS the concurrency limit, so
        // no semaphore/async runtime is needed and no unbounded tasks can be
        // spawned (Tokio would require rewriting this blocking client).
        // Each worker pulls the next batch from the shared queue until it is
        // empty or `stop` is set.
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    while !stop.load(Ordering::SeqCst) {
                        let next = queue.lock().unwrap().next();
                        let Some(positions) = next else {
                            break;
                        };
                        if stop.load(Ordering::SeqCst) {
                            break;
                        }
                        let chunk: Vec<(usize, &String)> =
                            positions.into_iter().map(|p| pending[p]).collect();

                        let done = {
                            let r = results.lock().unwrap();
                            count_done(&r)
                        };
                        let message = format!(
                            "Translating entries {}-{} of {}...",
                            done + 1,
                            done + chunk.len(),
                            total
                        );
                        if !(progress.lock().unwrap())(fraction(done), message) {
                            record_fatal(anyhow::anyhow!("translation interrupted"));
                            break;
                        }

                        // Fast path: batch request (no locks are held during
                        // I/O). The response is parsed AND validated before
                        // anything is written; transient failures (network,
                        // 5xx, malformed/invalid responses) are retried with
                        // bounded backoff, permanent ones are not. When the
                        // batch still fails, singles recover the chunk.
                        let item_est: usize = chunk.iter().map(|(_, t)| estimate_tokens(t)).sum();
                        let (ctx_before, ctx_after) = build_context(
                            texts,
                            &chunk,
                            self.config.context_before,
                            self.config.context_after,
                            item_est,
                            self.config.max_input_tokens,
                        );
                        let chunk_first = chunk.first().map(|(i, _)| *i).unwrap_or(0);
                        let chunk_last = chunk.last().map(|(i, _)| *i).unwrap_or(0);
                        let preview = |raw: &str| -> String {
                            raw.chars()
                                .take(200)
                                .collect::<String>()
                                .replace(['\n', '\r'], " ")
                        };
                        let started = std::time::Instant::now();
                        let mut attempt_report = RetryReport::default();
                        let batch_result = with_retry(&retry_policy, &mut attempt_report, || {
                            let resp = self.request_batch_chunk(
                                &chunk,
                                &ctx_before,
                                &ctx_after,
                                source_lang,
                                target_lang,
                            )?;
                            // Token counters include every attempt, so retry
                            // cost stays visible in resource metrics.
                            {
                                let mut s = stats.lock().unwrap();
                                s.batch_prompt_tokens += resp.prompt_eval_count.unwrap_or(0);
                                s.batch_gen_tokens += resp.eval_count.unwrap_or(0);
                                s.batch_eval_seconds +=
                                    resp.eval_duration.unwrap_or(0) as f64 / 1_000_000_000.0;
                            }
                            let entries = parse_batch_response(&resp.response).map_err(|e| {
                                Failure::transient(format!(
                                    "parse error: {} | {}",
                                    e,
                                    preview(&resp.response)
                                ))
                            })?;
                            validate_chunk_entries(&entries, &chunk).map_err(|failure| {
                                stats.lock().unwrap().validation_failures += 1;
                                Failure::Transient(format!(
                                    "{} | {}",
                                    failure.message(),
                                    preview(&resp.response)
                                ))
                            })?;
                            Ok(entries)
                        });
                        let elapsed = started.elapsed().as_secs_f64();
                        {
                            let mut s = stats.lock().unwrap();
                            s.batch_requests += 1;
                            s.batch_seconds += elapsed;
                            s.batch_entries += chunk.len() as u32;
                            s.retries += attempt_report.retries;
                            s.retry_backoff_seconds += attempt_report.backoff_seconds;
                        }

                        let mut parsed_ok = false;
                        let mut chunk_covered = 0u32;
                        match batch_result {
                            Ok(entries) => {
                                parsed_ok = true;
                                {
                                    // Global lock order of the pool:
                                    // `results` before `stats`.
                                    let mut r = results.lock().unwrap();
                                    chunk_covered = map_batch_entries(entries, &mut r);
                                    {
                                        let mut s = stats.lock().unwrap();
                                        s.batch_covered += chunk_covered;
                                    }

                                    // Persist newly translated entries (best effort).
                                    if let Some(cache) = &self.cache {
                                        for (idx, text) in &chunk {
                                            if !r[*idx].trim().is_empty() {
                                                let _ = cache.put(
                                                    text,
                                                    &r[*idx],
                                                    source_lang,
                                                    target_lang,
                                                    &self.config.selected_model,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            Err(failure) => {
                                let mut s = stats.lock().unwrap();
                                s.batch_failures += 1;
                                s.batch_diagnostics.push(format!(
                                    "chunk {}-{}: {}",
                                    chunk_first,
                                    chunk_last,
                                    failure.message()
                                ));
                            }
                        }

                        let done = {
                            let r = results.lock().unwrap();
                            count_done(&r)
                        };
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
                        if !(progress.lock().unwrap())(fraction(done), msg) {
                            record_fatal(anyhow::anyhow!("translation interrupted"));
                            break;
                        }

                        // Fallback: translate every still-missing entry of this
                        // chunk one by one. Singles share the retry policy;
                        // a failure after all attempts (or an interrupt) is
                        // fatal for the run, so invalid output can never be
                        // returned.
                        for (idx, text) in &chunk {
                            if stop.load(Ordering::SeqCst) {
                                break;
                            }
                            {
                                let r = results.lock().unwrap();
                                if !r[*idx].trim().is_empty() {
                                    continue;
                                }
                            }
                            let done = {
                                let r = results.lock().unwrap();
                                count_done(&r)
                            };
                            let message = format!("Translating {}/{}...", done + 1, total);
                            if !(progress.lock().unwrap())(fraction(done), message) {
                                record_fatal(anyhow::anyhow!("translation interrupted"));
                                break;
                            }

                            let started = std::time::Instant::now();
                            let mut attempt_report = RetryReport::default();
                            let single_result =
                                with_retry(&retry_policy, &mut attempt_report, || {
                                    let resp =
                                        self.translate_single(text, source_lang, target_lang)?;
                                    {
                                        let mut s = stats.lock().unwrap();
                                        s.single_prompt_tokens +=
                                            resp.prompt_eval_count.unwrap_or(0);
                                        s.single_gen_tokens += resp.eval_count.unwrap_or(0);
                                        s.single_eval_seconds +=
                                            resp.eval_duration.unwrap_or(0) as f64
                                                / 1_000_000_000.0;
                                    }
                                    let translated = clean_single_response(&resp.response);
                                    if translated.trim().is_empty() {
                                        stats.lock().unwrap().validation_failures += 1;
                                        return Err(Failure::transient(format!(
                                            "empty translation for entry {}",
                                            idx
                                        )));
                                    }
                                    Ok(translated)
                                });
                            let elapsed = started.elapsed().as_secs_f64();
                            {
                                let mut s = stats.lock().unwrap();
                                s.single_requests += 1;
                                s.single_seconds += elapsed;
                                s.retries += attempt_report.retries;
                                s.retry_backoff_seconds += attempt_report.backoff_seconds;
                            }
                            match single_result {
                                Ok(translated) => {
                                    let mut r = results.lock().unwrap();
                                    r[*idx] = translated;
                                    if let Some(cache) = &self.cache {
                                        let _ = cache.put(
                                            text,
                                            &r[*idx],
                                            source_lang,
                                            target_lang,
                                            &self.config.selected_model,
                                        );
                                    }
                                }
                                Err(failure) => {
                                    record_fatal(failure.into_error());
                                    break;
                                }
                            }

                            let done = {
                                let r = results.lock().unwrap();
                                count_done(&r)
                            };
                            let message = format!(
                                "Single {}/{} in {:.1}s (fallback)",
                                done, total, elapsed
                            );
                            if !(progress.lock().unwrap())(fraction(done), message) {
                                record_fatal(anyhow::anyhow!("translation interrupted"));
                                break;
                            }
                        }
                    }
                });
            }
        });

        if let Some(err) = fatal.into_inner().unwrap() {
            return Err(err);
        }

        let results = results.into_inner().unwrap();
        let stats = stats.into_inner().unwrap();
        // Only a completely reconstructable file may be handed back: wrong
        // counts or untranslated slots are an error, never written output.
        validate_complete_output(texts, &results)?;
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
        context_before: &[String],
        context_after: &[String],
        source_lang: Language,
        target_lang: Language,
    ) -> String {
        let source_name = source_lang.name();
        let target_name = target_lang.name();

        // The response shape is enforced by the JSON schema on the request
        // (`format`), so the prompt only carries the translation task, the
        // id-echo rule and — when present — the context disclaimer.
        let has_context = !context_before.is_empty() || !context_after.is_empty();
        let input = if has_context {
            serde_json::json!({
                "context_before": context_before,
                "translate": entries
                    .iter()
                    .map(|(i, t)| serde_json::json!({"id": *i, "text": *t}))
                    .collect::<Vec<_>>(),
                "context_after": context_after,
            })
            .to_string()
        } else {
            serde_json::to_string(
                &entries
                    .iter()
                    .map(|(i, t)| serde_json::json!({"id": *i, "text": *t}))
                    .collect::<Vec<_>>(),
            )
            .unwrap_or_default()
        };
        let label = if has_context {
            "Input (JSON):"
        } else {
            "Texts to translate:"
        };
        let rules = "Rules:\n\
             1. Keep translations natural and idiomatic for subtitles\n\
             2. Preserve line breaks\n\
             3. Keep proper names and technical terms as-is when appropriate\n\
             4. Echo the same id values you were given.";
        let rules = if has_context {
            format!(
                "{}\n\
                 5. context_before and context_after are reference only: do not translate them \
                 and do not include them in your response.",
                rules
            )
        } else {
            rules.to_string()
        };

        format!(
            "You are a professional subtitle translator. Translate the following texts from {} to {}.\n\
             \n\
             {}\n\
             {}\n\
             \n\
             {}",
            source_name, target_name, label, input, rules
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

/// Writes a *validated* batch response onto `results` for one chunk.
///
/// The caller runs [`validate_chunk_entries`] first: ids are exactly this
/// chunk's, distinct and non-empty, so every entry lands on its own input
/// index and the output reconstructs in the original order (foreign ids can
/// never touch another chunk's slots). Returns the number of entries
/// written.
fn map_batch_entries(entries: Vec<BatchTranslationEntry>, results: &mut [String]) -> u32 {
    let mut covered = 0u32;
    for e in entries {
        debug_assert!(e.index < results.len(), "validated id out of bounds");
        results[e.index] = e.text;
        covered += 1;
    }
    covered
}

/// Effective worker count for one translation run: at least 1, never more
/// than the configured limit and never more than there are batches (no idle
/// workers are spawned).
fn effective_concurrency(configured: usize, batch_count: usize) -> usize {
    configured.max(1).min(batch_count.max(1))
}

/// Classified request failure — decides whether another attempt is worth it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Failure {
    /// Retrying may help: connection reset, timeout, 5xx/408/429, or a
    /// malformed/unusable model response (including validation failures).
    Transient(String),
    /// Retrying cannot help: other 4xx statuses and local request-builder
    /// errors. Never retried.
    Permanent(String),
}

impl Failure {
    fn transient(msg: impl Into<String>) -> Self {
        Failure::Transient(msg.into())
    }

    fn permanent(msg: impl Into<String>) -> Self {
        Failure::Permanent(msg.into())
    }

    fn message(&self) -> &str {
        match self {
            Failure::Transient(m) | Failure::Permanent(m) => m,
        }
    }

    fn into_error(self) -> anyhow::Error {
        match self {
            Failure::Transient(m) | Failure::Permanent(m) => anyhow::anyhow!(m),
        }
    }
}

/// Maps a `reqwest` transport/decode error onto a retry class.
fn classify_reqwest(context: &str, e: reqwest::Error) -> Failure {
    let mut msg = format!("{}: {}", context, e);
    let mut src = std::error::Error::source(&e);
    while let Some(s) = src {
        msg.push_str(&format!(": {}", s));
        src = s.source();
    }
    if e.is_builder() || e.is_redirect() {
        Failure::permanent(msg)
    } else {
        // connect / timeout / request / body / decode: worth another attempt.
        Failure::transient(msg)
    }
}

/// Maps a non-success HTTP status onto a retry class: server errors and
/// 408/429 are transient, everything else (e.g. 404 model not found) is
/// permanent.
fn classify_status(status: reqwest::StatusCode, context: &str, body: &str) -> Failure {
    let msg = format!("{}: {} - {}", context, status, body);
    let code = status.as_u16();
    if status.is_server_error() || code == 408 || code == 429 {
        Failure::transient(msg)
    } else {
        Failure::permanent(msg)
    }
}

/// Bounded exponential backoff: `base * 2^retry_index` capped at `max`.
/// `retry_index` is 0-based, so the first retry waits `base` (attempt 1 is
/// immediate, matching the Task 08 policy sketch).
fn backoff_delay(base: Duration, max: Duration, retry_index: u32) -> Duration {
    let factor = 1u32 << retry_index.min(20); // 2^20 ≈ 1M, far beyond sane configs
    base.checked_mul(factor).unwrap_or(max).min(max)
}

/// Retry configuration for one translation run.
struct RetryPolicy {
    max_retries: u32,
    base_delay: Duration,
    max_delay: Duration,
}

impl RetryPolicy {
    fn from_config(config: &AppConfig) -> Self {
        // Test builds use tiny delays so retry tests finish in milliseconds;
        // production backs off 500ms, 1s, 2s, … capped at 5s.
        #[cfg(test)]
        let (base_delay, max_delay) = (Duration::from_millis(5), Duration::from_millis(40));
        #[cfg(not(test))]
        let (base_delay, max_delay) = (Duration::from_millis(500), Duration::from_secs(5));
        Self {
            max_retries: config.max_retries,
            base_delay,
            max_delay,
        }
    }
}

/// Accumulates the retry cost of one `with_retry` call (feeds `RequestStats`).
#[derive(Debug, Default, Clone, Copy)]
struct RetryReport {
    retries: u32,
    backoff_seconds: f64,
}

/// Runs `op` with bounded exponential backoff (Task 08).
///
/// Transient failures sleep `backoff_delay(base, max, retry_index)` before
/// the next attempt, up to `policy.max_retries` retries; the final error
/// states how many attempts were made. Permanent failures return
/// immediately. `report` accumulates retries executed and time slept across
/// calls (callers zero it per logical request).
fn with_retry<T>(
    policy: &RetryPolicy,
    report: &mut RetryReport,
    mut op: impl FnMut() -> Result<T, Failure>,
) -> Result<T, Failure> {
    let mut retries = 0u32;
    loop {
        let failure = match op() {
            Ok(value) => {
                report.retries += retries;
                return Ok(value);
            }
            Err(failure) => failure,
        };
        let retryable = matches!(failure, Failure::Transient(_));
        if !retryable || retries >= policy.max_retries {
            let failure = match failure {
                Failure::Transient(msg) => {
                    Failure::Transient(format!("{} (after {} attempt(s))", msg, retries + 1))
                }
                permanent => permanent,
            };
            report.retries += retries;
            return Err(failure);
        }
        let wait = backoff_delay(policy.base_delay, policy.max_delay, retries);
        std::thread::sleep(wait);
        report.backoff_seconds += wait.as_secs_f64();
        retries += 1;
    }
}

/// Collects the reference context for one batch: up to `before` non-empty
/// source lines before the chunk's lowest index and `after` non-empty lines
/// after its highest index (chunk entries themselves are never context).
/// Results are ordered oldest-first on both sides.
fn collect_context(
    texts: &[String],
    chunk: &[(usize, &String)],
    before: usize,
    after: usize,
) -> (Vec<String>, Vec<String>) {
    if chunk.is_empty() || (before == 0 && after == 0) {
        return (Vec::new(), Vec::new());
    }
    let min = chunk.iter().map(|(i, _)| *i).min().unwrap_or(0);
    let max = chunk.iter().map(|(i, _)| *i).max().unwrap_or(0);

    let mut ctx_before = Vec::new();
    let mut i = min;
    while i > 0 && ctx_before.len() < before {
        i -= 1;
        if !texts[i].trim().is_empty() {
            ctx_before.push(texts[i].clone());
        }
    }
    ctx_before.reverse();

    let mut ctx_after = Vec::new();
    let mut i = max + 1;
    while i < texts.len() && ctx_after.len() < after {
        if !texts[i].trim().is_empty() {
            ctx_after.push(texts[i].clone());
        }
        i += 1;
    }

    (ctx_before, ctx_after)
}

/// Builds the budgeted reference context for one batch: `collect_context`
/// output trimmed so that
/// * context estimated tokens never exceed [`CONTEXT_TOKEN_CAP`], and
/// * context + chunk estimated tokens stay within `budget`
///   (token-aware batching compatibility — context can only *reduce* the
///   room left for translate items, never inflate the request).
///
/// Trimming drops the farthest lines first (context_after's tail, then
/// context_before's head).
fn build_context(
    texts: &[String],
    chunk: &[(usize, &String)],
    before: usize,
    after: usize,
    item_est: usize,
    budget: usize,
) -> (Vec<String>, Vec<String>) {
    let (mut ctx_before, mut ctx_after) = collect_context(texts, chunk, before, after);
    let allowed = CONTEXT_TOKEN_CAP.min(budget.saturating_sub(item_est));

    let est = |v: &Vec<String>| -> usize { v.iter().map(|t| estimate_tokens(t)).sum() };
    loop {
        if est(&ctx_before) + est(&ctx_after) <= allowed {
            break;
        }
        if ctx_after.pop().is_some() {
            continue; // farthest after-side line
        }
        if !ctx_before.is_empty() {
            ctx_before.remove(0); // farthest before-side line
            continue;
        }
        break; // both sides empty (cannot exceed a non-negative budget)
    }
    (ctx_before, ctx_after)
}

/// Structured-output schema for one batch response: a wrapper object holding
/// exactly `count` translation items with application-owned ids.
///
/// The `minItems`/`maxItems` bounds are load-bearing: without them (and with
/// plain `format: "json"`) this model closes the response after the first
/// object — measured live on 2026-09-27 (3-entry probe: 1/3 without bounds,
/// 3/3 with; 25-entry chunk: 25/25 with).
fn batch_output_schema(count: usize) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "integer" },
                        "text": { "type": "string" }
                    },
                    "required": ["id", "text"]
                }
            }
        },
        "required": ["translations"]
    })
}

/// Accepts only unambiguous entry lists: every id must be unique. Duplicate
/// ids are rejected so the caller falls back to single requests instead of
/// guessing which translation wins.
fn validate_entries(entries: Vec<BatchTranslationEntry>) -> Result<Vec<BatchTranslationEntry>> {
    let mut seen = std::collections::HashSet::with_capacity(entries.len());
    for e in &entries {
        if !seen.insert(e.index) {
            anyhow::bail!("duplicate id {} in batch translation response", e.index);
        }
    }
    Ok(entries)
}

/// Checks a parsed batch response against the requested chunk (Task 08
/// validation requirements 1–4): every requested id must be present, no
/// foreign ids may appear, no id may repeat, and no translation may be
/// empty. Violations become a *transient* failure so the batch is retried
/// (or recovered by single requests afterwards) — an unvalidated response is
/// never written to the output.
fn validate_chunk_entries(
    entries: &[BatchTranslationEntry],
    chunk: &[(usize, &String)],
) -> Result<(), Failure> {
    let requested: std::collections::HashSet<usize> = chunk.iter().map(|(i, _)| *i).collect();
    let mut returned = std::collections::HashSet::with_capacity(entries.len());
    let mut unexpected = Vec::new();
    let mut empty = Vec::new();
    for e in entries {
        if !requested.contains(&e.index) {
            unexpected.push(e.index);
        }
        if !returned.insert(e.index) {
            // Normally rejected earlier by `validate_entries`; repeated here
            // so this function is self-contained.
            return Err(Failure::transient(format!(
                "duplicate id {} in batch translation response",
                e.index
            )));
        }
        if e.text.trim().is_empty() {
            empty.push(e.index);
        }
    }
    let mut missing: Vec<usize> = requested.difference(&returned).copied().collect();
    missing.sort_unstable();
    if !missing.is_empty() || !unexpected.is_empty() || !empty.is_empty() {
        return Err(Failure::transient(format!(
            "batch validation failed: missing ids {:?}, unexpected ids {:?}, empty translations {:?}",
            missing, unexpected, empty
        )));
    }
    Ok(())
}

/// Final gate before results leave `translate_batch` (Task 08 requirements
/// 5 and 7): the output must match the input one-for-one and every non-empty
/// source line must have a translation. Only a fully reconstructable file
/// may be handed back for writing.
fn validate_complete_output(texts: &[String], results: &[String]) -> Result<(), anyhow::Error> {
    if results.len() != texts.len() {
        anyhow::bail!(
            "output count mismatch: {} entries in, {} out",
            texts.len(),
            results.len()
        );
    }
    let missing: Vec<usize> = texts
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.trim().is_empty())
        .filter(|(i, _)| results[*i].trim().is_empty())
        .map(|(i, _)| i)
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "translation incomplete: {} of {} entries untranslated (first indexes: {:?})",
            missing.len(),
            texts.len(),
            &missing[..missing.len().min(10)]
        );
    }
    Ok(())
}

fn parse_batch_response(raw: &str) -> Result<Vec<BatchTranslationEntry>> {
    let trimmed = raw.trim();

    // 1. Expected format: a JSON array of entries
    if let Ok(entries) = serde_json::from_str::<Vec<BatchTranslationEntry>>(trimmed) {
        return validate_entries(entries);
    }

    // 2. Wrapper format: {"translations": [{"id": ..., "text": ...}]}
    if let Ok(wrapper) = serde_json::from_str::<BatchTranslationResponse>(trimmed) {
        return validate_entries(wrapper.translations);
    }

    // 3. Single object: {"id": 0, "text": "..."} (model returned only one entry)
    if let Ok(entry) = serde_json::from_str::<BatchTranslationEntry>(trimmed) {
        return validate_entries(vec![entry]);
    }

    // 4. Model returned multiple JSON objects (e.g., separated by newlines)
    let mut entries = Vec::new();
    for obj in extract_json_objects(trimmed) {
        if let Ok(entry) = serde_json::from_str::<BatchTranslationEntry>(&obj) {
            entries.push(entry);
        }
    }
    if !entries.is_empty() {
        return validate_entries(entries);
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
    fn clone_shares_the_same_http_client() {
        // The app clones OllamaClient per background thread; every clone must
        // reuse the one connection pool, not build a new client.
        let a = OllamaClient::new(&AppConfig::default()).expect("client");
        let b = a.clone();
        assert!(Arc::ptr_eq(&a.client, &b.client));
    }

    #[test]
    fn set_config_reuses_the_http_client() {
        // Settings edits (model/URL/language changes) must not churn the
        // connection pool: only the config is swapped, never the client.
        let mut client = OllamaClient::new(&AppConfig::default()).expect("client");
        let before = Arc::clone(&client.client);
        let changed = AppConfig {
            ollama_url: "http://localhost:19999".to_string(),
            ..AppConfig::default()
        };
        client.set_config(&changed);
        assert!(Arc::ptr_eq(&before, &client.client));
    }

    #[test]
    fn effective_concurrency_bounds_the_worker_pool() {
        // Default (1) must stay sequential, 0 must clamp up to 1, higher
        // values are capped by the batch count (no idle workers spawn).
        assert_eq!(effective_concurrency(1, 5), 1);
        assert_eq!(effective_concurrency(0, 5), 1);
        assert_eq!(effective_concurrency(4, 5), 4);
        assert_eq!(effective_concurrency(4, 2), 2);
        assert_eq!(effective_concurrency(8, 0), 1);
        assert_eq!(effective_concurrency(usize::MAX, 3), 3);
    }

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
    fn parse_accepts_target_shape_wrapper() {
        let raw = r#"{"translations": [{"id": 1, "text": "Merhaba."}, {"id": 2, "text": "Nereye gidiyorsun?"}]}"#;
        let entries = parse_batch_response(raw).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].index, 1);
        assert_eq!(entries[1].index, 2);
    }

    #[test]
    fn parse_rejects_duplicate_ids() {
        // Ambiguous output: which translation of the same line wins?
        // Rejected entirely so the chunk falls back to single requests.
        let array = r#"[{"index": 0, "text": "a"}, {"index": 0, "text": "b"}]"#;
        assert!(parse_batch_response(array).is_err());
        let wrapped = r#"{"translations": [{"id": 0, "text": "a"}, {"id": 0, "text": "b"}]}"#;
        assert!(parse_batch_response(wrapped).is_err());
        let objects = "{\"id\": 0, \"text\": \"a\"}\n{\"id\": 0, \"text\": \"b\"}";
        assert!(parse_batch_response(objects).is_err());
    }

    #[test]
    fn parse_rejects_entry_without_id() {
        assert!(parse_batch_response(r#"[{"text": "orphan"}]"#).is_err());
        assert!(parse_batch_response(r#"{"translations": [{"text": "no id"}]}"#).is_err());
    }

    #[test]
    fn batch_output_schema_pins_item_count() {
        let schema = batch_output_schema(7);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["required"][0], "translations");
        let arr = &schema["properties"]["translations"];
        assert_eq!(arr["type"], "array");
        assert_eq!(arr["minItems"], 7, "grammar must force all items");
        assert_eq!(arr["maxItems"], 7);
        assert_eq!(arr["items"]["properties"]["id"]["type"], "integer");
        assert_eq!(arr["items"]["required"][0], "id");
        assert_eq!(arr["items"]["required"][1], "text");
    }

    #[test]
    fn batch_prompt_uses_ids_and_leaves_json_to_the_schema() {
        let text = "Hello".to_string();
        let entries = [(0usize, &text)];
        let client = OllamaClient::new(&AppConfig::default()).expect("client");
        let prompt = client.build_batch_translation_prompt(
            &entries,
            &[],
            &[],
            Language::English,
            Language::Turkish,
        );

        assert!(prompt.contains(r#""id":0"#), "input must use app-owned ids");
        assert!(!prompt.contains("Example output"), "no example needed");
        assert!(
            !prompt.contains("ENTIRE response"),
            "shape rules belong to the schema"
        );
        assert!(prompt.contains("Echo the same id values"));
        assert!(!prompt.contains("context_before"), "no context keys at 0");
        assert!(!prompt.contains("reference only"), "no context rule at 0");
    }

    #[test]
    fn batch_prompt_with_context_is_object_shaped_and_disclaims_context() {
        let text = "Target line".to_string();
        let entries = [(42usize, &text)];
        let ctx_before = vec![
            "Where are you?".to_string(),
            "I thought you were leaving.".to_string(),
        ];
        let ctx_after = vec!["Let's go.".to_string()];
        let client = OllamaClient::new(&AppConfig::default()).expect("client");
        let prompt = client.build_batch_translation_prompt(
            &entries,
            &ctx_before,
            &ctx_after,
            Language::English,
            Language::Turkish,
        );

        assert!(prompt.contains(r#""context_before""#));
        assert!(prompt.contains(r#""translate""#));
        assert!(prompt.contains(r#""context_after""#));
        assert!(prompt.contains(r#""id":42"#));
        assert!(prompt.contains("Where are you?"));
        assert!(prompt.contains("Let's go."));
        assert!(
            prompt.contains("reference only"),
            "context must be explicitly reference-only"
        );
        assert!(prompt.contains("do not include them in your response"));
    }

    #[test]
    fn collect_context_grabs_nearest_non_empty_lines() {
        let texts: Vec<String> = ["a", "", "b", "TARGET", "c", "d", "e"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let chunk = [(3usize, &texts[3])];
        let (before, after) = collect_context(&texts, &chunk, 2, 2);
        assert_eq!(before, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(after, vec!["c".to_string(), "d".to_string()]);
    }

    #[test]
    fn collect_context_clamps_at_file_edges_and_disabled() {
        let texts: Vec<String> = ["first", "second"].iter().map(|s| s.to_string()).collect();
        let head = [(0usize, &texts[0])];
        let (before, after) = collect_context(&texts, &head, 5, 5);
        assert!(before.is_empty(), "no lines before file start");
        assert_eq!(after, vec!["second".to_string()]);

        let tail = [(1usize, &texts[1])];
        let (before, after) = collect_context(&texts, &tail, 5, 5);
        assert_eq!(before, vec!["first".to_string()]);
        assert!(after.is_empty(), "no lines after file end");

        let (before, after) = collect_context(&texts, &head, 0, 0);
        assert!(before.is_empty() && after.is_empty(), "0 disables context");
    }

    #[test]
    fn build_context_enforces_token_cap_and_budget() {
        let line = "x".repeat(600); // 150 estimated tokens each
        let texts: Vec<String> = (0..7).map(|_| line.clone()).collect();
        let chunk = [(3usize, &texts[3])];

        // 4 context lines = 600 est tok > CONTEXT_TOKEN_CAP (512): the
        // farthest after-side line is dropped first -> 3 lines (450) kept.
        let (before, after) = build_context(&texts, &chunk, 2, 2, 100, 3000);
        assert_eq!(before.len(), 2);
        assert_eq!(after.len(), 1);
        assert_eq!(after[0], texts[4], "nearest after-line survives");

        // Tight budget: items already use 2950 of 3000 -> context allowed
        // 50 tokens -> every line (150) is dropped, request never grows.
        let (before, after) = build_context(&texts, &chunk, 2, 2, 2950, 3000);
        assert!(before.is_empty() && after.is_empty());
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
    fn map_writes_validated_entries_by_index() {
        let owned = chunk_of(&[25, 26], &["a", "b"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        validate_chunk_entries(&[entry(25, "x"), entry(26, "y")], &chunk).expect("valid");
        let mut results = vec![String::new(); 30];

        let covered = map_batch_entries(vec![entry(25, "x"), entry(26, "y")], &mut results);

        assert_eq!(covered, 2);
        assert_eq!(results[25], "x");
        assert_eq!(results[26], "y");
        assert!(results[0].is_empty(), "other chunks' slots stay untouched");
    }

    #[test]
    fn validate_accepts_the_exact_requested_chunk() {
        let owned = chunk_of(&[25, 26], &["a", "b"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        assert!(validate_chunk_entries(&[entry(25, "x"), entry(26, "y")], &chunk).is_ok());
        // Response order must not matter.
        assert!(validate_chunk_entries(&[entry(26, "y"), entry(25, "x")], &chunk).is_ok());
    }

    #[test]
    fn validate_rejects_missing_id() {
        let owned = chunk_of(&[0, 1], &["a", "b"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        let err = validate_chunk_entries(&[entry(0, "x")], &chunk).unwrap_err();
        assert!(
            err.message().contains("missing ids [1]"),
            "unexpected message: {}",
            err.message()
        );
    }

    #[test]
    fn validate_rejects_unexpected_id() {
        let owned = chunk_of(&[0, 1], &["a", "b"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        let err =
            validate_chunk_entries(&[entry(0, "x"), entry(1, "y"), entry(2, "WRONG")], &chunk)
                .unwrap_err();
        assert!(
            err.message().contains("unexpected ids [2]"),
            "unexpected message: {}",
            err.message()
        );
    }

    #[test]
    fn validate_rejects_duplicate_id() {
        let owned = chunk_of(&[0, 1], &["a", "b"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        let err = validate_chunk_entries(&[entry(0, "x"), entry(0, "y")], &chunk).unwrap_err();
        assert!(
            err.message().contains("duplicate id 0"),
            "unexpected message: {}",
            err.message()
        );
    }

    #[test]
    fn validate_rejects_empty_translation() {
        let owned = chunk_of(&[0, 1], &["a", "b"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        let err = validate_chunk_entries(&[entry(0, "x"), entry(1, "   ")], &chunk).unwrap_err();
        assert!(
            err.message().contains("empty translations [1]"),
            "unexpected message: {}",
            err.message()
        );
    }

    #[test]
    fn validate_rejects_renumbered_response() {
        // Model restarts ids at 0 for a chunk at 25..27: nothing matches.
        // Since Task 08 this fails validation (retry, then single fallback)
        // instead of being written positionally.
        let owned = chunk_of(&[25, 26, 27], &["a", "b", "c"]);
        let chunk: Vec<(usize, &String)> = owned.iter().map(|(i, t)| (*i, t)).collect();
        let err = validate_chunk_entries(&[entry(0, "x"), entry(1, "y"), entry(2, "z")], &chunk)
            .unwrap_err();
        let msg = err.message().to_string();
        assert!(msg.contains("missing ids [25, 26, 27]"), "{}", msg);
        assert!(msg.contains("unexpected ids [0, 1, 2]"), "{}", msg);
    }

    #[test]
    fn complete_output_requires_count_and_coverage() {
        let texts = vec!["a".to_string(), "   ".to_string(), "c".to_string()];
        let short = vec![
            "x".to_string(),
            String::new(),
            "z".to_string(),
            "extra".to_string(),
        ];
        assert!(
            validate_complete_output(&texts, &short).is_err(),
            "count mismatch must fail"
        );
        let gaps = vec!["x".to_string(), String::new(), String::new()];
        let err = validate_complete_output(&texts, &gaps)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 of 3"), "{}", err);
        // The empty source line may stay empty; everything else must exist.
        let ok = vec!["x".to_string(), String::new(), "z".to_string()];
        assert!(validate_complete_output(&texts, &ok).is_ok());
    }

    #[test]
    fn backoff_delay_is_exponential_and_bounded() {
        let base = Duration::from_millis(500);
        let max = Duration::from_secs(5);
        assert_eq!(backoff_delay(base, max, 0), Duration::from_millis(500));
        assert_eq!(backoff_delay(base, max, 1), Duration::from_secs(1));
        assert_eq!(backoff_delay(base, max, 2), Duration::from_secs(2));
        assert_eq!(backoff_delay(base, max, 3), Duration::from_secs(4));
        assert_eq!(backoff_delay(base, max, 4), max, "capped at max");
        assert_eq!(
            backoff_delay(base, max, 100),
            max,
            "cap holds for any index"
        );
    }

    #[test]
    #[ignore = "needs a running local Ollama: cargo test -- --ignored"]
    fn e2e_batch_covers_all_entries() {
        let config_path = dirs::config_dir()
            .unwrap_or_default()
            .join("auto-translate-subs")
            .join("config.json");
        let config: AppConfig = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        if config.selected_model.is_empty() {
            eprintln!("skipped: no model in {}", config_path.display());
            return;
        }
        let mut client = OllamaClient::new(&config).expect("client");
        // Keep the e2e assertions deterministic: measure the pipeline, not
        // whatever happens to be in the persistent cache from earlier runs.
        client.disable_cache();
        let texts: Vec<String> = (0..5)
            .map(|i| format!("Subtitle line number {} with some sample English text.", i))
            .collect();

        let (results, stats) = client
            .translate_batch(&texts, Language::English, Language::Turkish, |_, _| true)
            .expect("translate_batch");
        eprintln!("{}", stats.summary());

        assert_eq!(stats.batch_requests, 1);
        assert_eq!(
            stats.batch_covered, 5,
            "batch must cover every entry: {:?}",
            stats.batch_diagnostics
        );
        assert_eq!(stats.batch_failures, 0);
        assert_eq!(
            stats.single_requests, 0,
            "single fallback should not trigger: {}",
            stats.summary()
        );
        for (i, r) in results.iter().enumerate() {
            assert!(!r.trim().is_empty(), "entry {} not translated", i);
        }
    }

    #[test]
    #[ignore = "needs a running local Ollama: cargo test -- --ignored"]
    fn e2e_concurrent_batches_keep_order() {
        // Concurrency 2 with three batches: two workers share the queue and
        // finish out of order, but every entry must still be translated into
        // its own input position (index-keyed writes, never append order).
        let config_path = dirs::config_dir()
            .unwrap_or_default()
            .join("auto-translate-subs")
            .join("config.json");
        let mut config: AppConfig = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        if config.selected_model.is_empty() {
            eprintln!("skipped: no model in {}", config_path.display());
            return;
        }
        config.max_concurrent_requests = 2;
        let mut client = OllamaClient::new(&config).expect("client");
        client.disable_cache();
        let texts: Vec<String> = (0..60)
            .map(|i| format!("Subtitle line number {} with some sample English text.", i))
            .collect();

        let (results, stats) = client
            .translate_batch(&texts, Language::English, Language::Turkish, |_, _| true)
            .expect("translate_batch");
        eprintln!("{}", stats.summary());

        // 60 entries split by the 25-per-request ceiling: 25 + 25 + 10.
        assert_eq!(stats.batch_requests, 3);
        assert_eq!(
            stats.batch_covered, 60,
            "every entry must be covered under concurrency: {:?}",
            stats.batch_diagnostics
        );
        assert_eq!(stats.batch_failures, 0);
        for (i, r) in results.iter().enumerate() {
            assert!(!r.trim().is_empty(), "entry {} not translated", i);
        }
    }

    // ---- Mock Ollama server for retry/recovery tests (Task 08) ----

    use std::collections::VecDeque;
    use std::io::{Read, Write as IoWrite};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::AtomicUsize;

    /// Scripted behavior for one accepted connection.
    enum Step {
        /// Non-2xx status with a JSON error body.
        Status(u16),
        /// 200 with the given body.
        Body(String),
        /// Sleep before answering — the client times out first.
        SleepThenBody(Duration, String),
        /// Close the connection without any response (connection reset).
        Reset,
    }

    /// Minimal HTTP server: every connection pops the next scripted step, so
    /// tests fully control the failure sequence. Each connection is handled
    /// in its own thread so a slow step cannot delay the next (retried)
    /// connection.
    struct MockOllama {
        url: String,
        hits: Arc<AtomicUsize>,
    }

    impl MockOllama {
        fn start(steps: Vec<Step>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
            let url = format!("http://{}", listener.local_addr().expect("mock addr"));
            let hits = Arc::new(AtomicUsize::new(0));
            let steps = Arc::new(Mutex::new(VecDeque::from(steps)));
            let thread_hits = Arc::clone(&hits);
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { break };
                    let hits = Arc::clone(&thread_hits);
                    let steps = Arc::clone(&steps);
                    std::thread::spawn(move || {
                        hits.fetch_add(1, Ordering::SeqCst);
                        read_http_request(&mut stream);
                        let step = steps.lock().unwrap().pop_front();
                        match step {
                            Some(Step::Status(code)) => {
                                let _ =
                                    write_http_response(&mut stream, code, r#"{"error":"boom"}"#);
                            }
                            Some(Step::Body(body)) => {
                                let _ = write_http_response(&mut stream, 200, &body);
                            }
                            Some(Step::SleepThenBody(delay, body)) => {
                                std::thread::sleep(delay);
                                let _ = write_http_response(&mut stream, 200, &body);
                            }
                            // Drop = connection closed without a response.
                            Some(Step::Reset) | None => {}
                        }
                    });
                }
            });
            Self { url, hits }
        }

        fn hit_count(&self) -> usize {
            self.hits.load(Ordering::SeqCst)
        }
    }

    fn read_http_request(stream: &mut TcpStream) {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            match stream.read(&mut tmp) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&buf[..pos]);
                        let mut content_length = 0usize;
                        for line in headers.lines() {
                            if let Some((k, v)) = line.split_once(':') {
                                if k.eq_ignore_ascii_case("content-length") {
                                    content_length = v.trim().parse().unwrap_or(0);
                                }
                            }
                        }
                        if buf.len() - (pos + 4) >= content_length {
                            break;
                        }
                    }
                }
            }
        }
    }

    fn write_http_response(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
        let reason = match status {
            200 => "OK",
            404 => "Not Found",
            500 => "Internal Server Error",
            _ => "Status",
        };
        let response = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{}",
            status,
            reason,
            body.len(),
            body
        );
        stream
            .write_all(response.as_bytes())
            .and_then(|()| stream.flush())
    }

    /// Wraps a model payload the way Ollama's `/api/generate` does
    /// (`response` holds the model output, `done` marks completion).
    fn ollama_body(response: &str) -> String {
        serde_json::json!({ "response": response, "done": true }).to_string()
    }

    fn valid_batch_body() -> String {
        ollama_body(
            &serde_json::json!({
                "translations": [
                    {"id": 0, "text": "bir"},
                    {"id": 1, "text": "iki"}
                ]
            })
            .to_string(),
        )
    }

    fn two_texts() -> Vec<String> {
        vec!["alpha".to_string(), "beta".to_string()]
    }

    fn mock_config(url: &str, max_retries: u32) -> AppConfig {
        AppConfig {
            ollama_url: url.to_string(),
            selected_model: "test-model".to_string(),
            max_retries,
            ..AppConfig::default()
        }
    }

    fn mock_client(url: &str, max_retries: u32) -> OllamaClient {
        let mut client = OllamaClient::new(&mock_config(url, max_retries)).expect("mock client");
        client.disable_cache();
        client
    }

    fn translate_with(
        client: &OllamaClient,
        texts: &[String],
    ) -> Result<(Vec<String>, RequestStats)> {
        client.translate_batch(texts, Language::English, Language::Turkish, |_, _| true)
    }

    #[test]
    fn transient_http_failure_is_retried_until_success() {
        let server = MockOllama::start(vec![Step::Status(500), Step::Body(valid_batch_body())]);
        let client = mock_client(&server.url, 2);

        let (results, stats) = translate_with(&client, &two_texts()).expect("retry succeeds");

        assert_eq!(server.hit_count(), 2, "one failed attempt + one retry");
        assert_eq!(stats.retries, 1);
        assert!(stats.retry_backoff_seconds > 0.0, "backoff is measured");
        assert_eq!(
            stats.batch_requests, 1,
            "retries are reported separately from logical requests"
        );
        assert_eq!(stats.batch_failures, 0);
        assert_eq!(results[0], "bir");
        assert_eq!(results[1], "iki");
    }

    #[test]
    fn connection_reset_is_retried() {
        let server = MockOllama::start(vec![Step::Reset, Step::Body(valid_batch_body())]);
        let client = mock_client(&server.url, 2);

        let (results, stats) = translate_with(&client, &two_texts()).expect("retry succeeds");

        assert_eq!(server.hit_count(), 2);
        assert_eq!(stats.retries, 1);
        assert_eq!(results[1], "iki");
    }

    #[test]
    fn timeout_is_transient_and_retried() {
        let server = MockOllama::start(vec![
            Step::SleepThenBody(Duration::from_secs(3), valid_batch_body()),
            Step::Body(valid_batch_body()),
        ]);
        let mut client = OllamaClient::with_request_timeout(
            &mock_config(&server.url, 2),
            Duration::from_millis(300),
        )
        .expect("client");
        client.disable_cache();

        let (results, stats) = translate_with(&client, &two_texts()).expect("timeout retried");

        assert_eq!(stats.retries, 1, "timeout counts as a transient failure");
        assert_eq!(stats.batch_failures, 0);
        assert_eq!(server.hit_count(), 2);
        assert_eq!(results[0], "bir");
    }

    #[test]
    fn malformed_response_is_retried() {
        let server = MockOllama::start(vec![
            Step::Body(ollama_body("definitely not json")),
            Step::Body(valid_batch_body()),
        ]);
        let client = mock_client(&server.url, 2);

        let (results, stats) = translate_with(&client, &two_texts()).expect("retry succeeds");

        assert_eq!(stats.retries, 1);
        assert_eq!(stats.batch_failures, 0);
        assert_eq!(server.hit_count(), 2);
        assert_eq!(results[0], "bir");
    }

    #[test]
    fn missing_id_is_rejected_retried_and_recovered() {
        let partial = serde_json::json!({"translations": [{"id": 0, "text": "bir"}]}).to_string();
        let server = MockOllama::start(vec![
            Step::Body(ollama_body(&partial)),
            Step::Body(valid_batch_body()),
        ]);
        let client = mock_client(&server.url, 2);

        let (results, stats) = translate_with(&client, &two_texts()).expect("retry succeeds");

        assert_eq!(
            stats.retries, 1,
            "a response missing an id is invalid and retried: {:?}",
            stats.batch_diagnostics
        );
        assert_eq!(
            stats.validation_failures, 1,
            "the rejected response is counted"
        );
        assert_eq!(stats.batch_failures, 0);
        assert_eq!(results[0], "bir");
        assert_eq!(results[1], "iki");
    }

    #[test]
    fn duplicate_id_is_rejected_and_retried() {
        let dup = serde_json::json!({
            "translations": [
                {"id": 0, "text": "a"},
                {"id": 0, "text": "b"}
            ]
        })
        .to_string();
        let server = MockOllama::start(vec![
            Step::Body(ollama_body(&dup)),
            Step::Body(valid_batch_body()),
        ]);
        let client = mock_client(&server.url, 2);

        let (results, stats) = translate_with(&client, &two_texts()).expect("retry succeeds");

        assert_eq!(stats.retries, 1);
        assert_eq!(stats.batch_failures, 0);
        assert_eq!(server.hit_count(), 2);
        assert_eq!(results[0], "bir");
    }

    #[test]
    fn unexpected_id_is_rejected_and_retried() {
        let foreign = serde_json::json!({
            "translations": [
                {"id": 0, "text": "bir"},
                {"id": 1, "text": "iki"},
                {"id": 99, "text": "WRONG"}
            ]
        })
        .to_string();
        let server = MockOllama::start(vec![
            Step::Body(ollama_body(&foreign)),
            Step::Body(valid_batch_body()),
        ]);
        let client = mock_client(&server.url, 2);

        let (results, stats) = translate_with(&client, &two_texts()).expect("retry succeeds");

        assert_eq!(stats.retries, 1);
        assert_eq!(stats.batch_failures, 0);
        assert_eq!(server.hit_count(), 2);
        assert_eq!(results[0], "bir");
        assert_eq!(results[1], "iki");
    }

    #[test]
    fn max_retries_are_reached_then_the_run_fails() {
        // max_retries = 1 → exactly 2 attempts per request: the batch gets
        // 2, then the first single fallback gets 2 more and fails the run.
        let server = MockOllama::start(vec![
            Step::Status(500),
            Step::Status(500),
            Step::Status(500),
            Step::Status(500),
        ]);
        let client = mock_client(&server.url, 1);

        let err = translate_with(&client, &two_texts()).unwrap_err();

        assert_eq!(
            server.hit_count(),
            4,
            "no attempt may happen beyond max_retries"
        );
        let msg = err.to_string();
        assert!(msg.contains("after 2 attempt"), "unexpected error: {}", msg);
    }

    #[test]
    fn permanent_failure_is_not_retried() {
        let server = MockOllama::start(vec![Step::Status(404), Step::Status(404)]);
        let client = mock_client(&server.url, 5); // generous budget: still 0 retries

        let err = translate_with(&client, &two_texts()).unwrap_err();

        assert_eq!(
            server.hit_count(),
            2,
            "404 is permanent: 1 batch attempt + 1 single attempt, no retries"
        );
        let msg = err.to_string();
        assert!(msg.contains("404"), "unexpected error: {}", msg);
    }

    #[test]
    fn empty_single_translation_is_retried_then_fails() {
        let server = MockOllama::start(vec![
            Step::Status(500),
            Step::Status(500), // batch: 2 attempts (max_retries = 1)
            Step::Body(ollama_body("   ")),
            Step::Body(ollama_body("   ")), // single: 2 attempts
        ]);
        let client = mock_client(&server.url, 1);

        let err = translate_with(&client, &two_texts()).unwrap_err();

        assert_eq!(server.hit_count(), 4);
        let msg = err.to_string();
        assert!(
            msg.contains("empty translation"),
            "unexpected error: {}",
            msg
        );
    }
}
