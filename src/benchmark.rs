//! Baseline benchmark mode (`--bench`).
//!
//! Measures the translation pipeline against one subtitle file without writing
//! any output file. The report is a stable markdown block (fixed metric order,
//! one row per metric) so later runs can be compared row by row.

use crate::models::{AppConfig, Language};
use crate::ollama_client::{OllamaClient, RequestStats, TRANSLATE_CHUNK_SIZE};
use crate::subtitle_parser::parse_subtitle_file;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Instant;

/// Parsed `--bench` command line.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchArgs {
    pub file: PathBuf,
    pub runs: usize,
    pub out: Option<PathBuf>,
    pub warmup: bool,
    /// Disable the persistent translation cache (pipeline-parity runs).
    pub no_cache: bool,
    /// Override the reference-context size for both sides (benchmarking
    /// context 0 vs N without touching the stored config).
    pub context: Option<usize>,
    /// Override the parallel request limit (benchmarking concurrency 1/2/3
    /// without touching the stored config). `None` keeps the stored config.
    pub concurrency: Option<usize>,
    /// Override the per-request token budget `max_input_tokens` (benchmarking
    /// 1000/2000/3000/4000 without touching the stored config).
    pub tokens: Option<usize>,
    /// Run the Task 09 comparison matrix (one-factor-at-a-time sweeps over
    /// token budget, concurrency, cache and context) instead of one run.
    pub matrix: bool,
    /// Write a machine-readable JSON result next to the markdown report.
    pub json: Option<PathBuf>,
}

/// Parses benchmark arguments from the full process argument list.
///
/// Expected shape:
/// `program --bench <file> [--runs <n>] [--out <file>] [--no-warmup]
/// [--no-cache] [--context <n>] [--concurrency <n>] [--tokens <n>]
/// [--matrix] [--json <file>]`
pub fn parse_args(args: &[String]) -> Result<BenchArgs> {
    let bench_pos = args
        .iter()
        .position(|a| a == "--bench")
        .context("missing --bench flag")?;
    let file = args
        .get(bench_pos + 1)
        .filter(|v| !v.starts_with("--"))
        .context("--bench requires a subtitle file path")?;

    let mut runs = 1usize;
    let mut out = None;
    let mut warmup = true;
    let mut no_cache = false;
    let mut context = None;
    let mut concurrency = None;
    let mut tokens = None;
    let mut matrix = false;
    let mut json = None;
    let mut i = bench_pos + 2;
    while i < args.len() {
        match args[i].as_str() {
            "--runs" => {
                let value = args.get(i + 1).context("--runs requires a value")?;
                runs = value
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n >= 1)
                    .context("--runs must be a positive integer")?;
                i += 2;
            }
            "--out" => {
                let value = args.get(i + 1).context("--out requires a path")?;
                out = Some(PathBuf::from(value));
                i += 2;
            }
            "--context" => {
                let value = args.get(i + 1).context("--context requires a value")?;
                context = Some(
                    value
                        .parse::<usize>()
                        .ok()
                        .context("--context must be a non-negative integer")?,
                );
                i += 2;
            }
            "--concurrency" => {
                let value = args.get(i + 1).context("--concurrency requires a value")?;
                concurrency = Some(
                    value
                        .parse::<usize>()
                        .ok()
                        .filter(|n| *n >= 1)
                        .context("--concurrency must be a positive integer")?,
                );
                i += 2;
            }
            "--no-warmup" => {
                warmup = false;
                i += 1;
            }
            "--no-cache" => {
                no_cache = true;
                i += 1;
            }
            "--tokens" => {
                let value = args.get(i + 1).context("--tokens requires a value")?;
                tokens = Some(
                    value
                        .parse::<usize>()
                        .ok()
                        .filter(|n| *n >= 1)
                        .context("--tokens must be a positive integer")?,
                );
                i += 2;
            }
            "--matrix" => {
                matrix = true;
                i += 1;
            }
            "--json" => {
                let value = args.get(i + 1).context("--json requires a path")?;
                json = Some(PathBuf::from(value));
                i += 2;
            }
            other => anyhow::bail!("unknown benchmark flag: {}", other),
        }
    }

    Ok(BenchArgs {
        file: PathBuf::from(file),
        runs,
        out,
        warmup,
        no_cache,
        context,
        concurrency,
        tokens,
        matrix,
        json,
    })
}

/// Raw result of one benchmark translation run.
struct RunResult {
    /// Wall time of `translate_batch` (excludes parsing and file output).
    elapsed: f64,
    stats: RequestStats,
    uncovered: usize,
}

/// Input facts measured once, before any request.
struct InputFacts {
    entries: usize,
    source_chars: usize,
    est_input_tokens: usize,
    parse_seconds: f64,
}

/// Aggregates over all runs of one benchmark invocation.
#[derive(Debug, Clone, PartialEq)]
struct Aggregate {
    mean_elapsed: f64,
    min_elapsed: f64,
    max_elapsed: f64,
    total_requests: u32,
    successful_requests: u32,
    batch_requests: u32,
    single_requests: u32,
    batch_failures: u32,
    batch_covered: u32,
    batch_entries: u32,
    avg_batch_size: f64,
    avg_request_seconds: f64,
    request_seconds: f64,
    eval_seconds: f64,
    prompt_tokens: u64,
    gen_tokens: u64,
    subs_per_sec: f64,
    out_tok_per_req_sec: f64,
    out_tok_per_eval_sec: f64,
    max_uncovered: usize,
    retries: u32,
    retry_backoff_seconds: f64,
    validation_failures: u32,
    cache_hits: u32,
    cache_misses: u32,
}

fn aggregate(runs: &[RunResult], entries: usize) -> Aggregate {
    let n = runs.len() as f64;
    let mean_elapsed = runs.iter().map(|r| r.elapsed).sum::<f64>() / n;
    let min_elapsed = runs.iter().map(|r| r.elapsed).fold(f64::MAX, f64::min);
    let max_elapsed = runs.iter().map(|r| r.elapsed).fold(f64::MIN, f64::max);

    let batch_requests: u32 = runs.iter().map(|r| r.stats.batch_requests).sum();
    let single_requests: u32 = runs.iter().map(|r| r.stats.single_requests).sum();
    let batch_failures: u32 = runs.iter().map(|r| r.stats.batch_failures).sum();
    let batch_entries: u32 = runs.iter().map(|r| r.stats.batch_entries).sum();
    let batch_covered: u32 = runs.iter().map(|r| r.stats.batch_covered).sum();
    let request_seconds: f64 = runs
        .iter()
        .map(|r| r.stats.batch_seconds + r.stats.single_seconds)
        .sum();
    let eval_seconds: f64 = runs
        .iter()
        .map(|r| r.stats.batch_eval_seconds + r.stats.single_eval_seconds)
        .sum();
    let prompt_tokens: u64 = runs
        .iter()
        .map(|r| r.stats.batch_prompt_tokens + r.stats.single_prompt_tokens)
        .sum();
    let gen_tokens: u64 = runs
        .iter()
        .map(|r| r.stats.batch_gen_tokens + r.stats.single_gen_tokens)
        .sum();

    let total_requests = batch_requests + single_requests;
    Aggregate {
        mean_elapsed,
        min_elapsed,
        max_elapsed,
        total_requests,
        successful_requests: total_requests.saturating_sub(batch_failures),
        batch_requests,
        single_requests,
        batch_failures,
        batch_covered,
        batch_entries,
        avg_batch_size: if batch_requests > 0 {
            batch_entries as f64 / batch_requests as f64
        } else {
            0.0
        },
        avg_request_seconds: if total_requests > 0 {
            request_seconds / total_requests as f64
        } else {
            0.0
        },
        request_seconds,
        eval_seconds,
        prompt_tokens,
        gen_tokens,
        subs_per_sec: entries as f64 / mean_elapsed,
        out_tok_per_req_sec: if request_seconds > 0.0 {
            gen_tokens as f64 / request_seconds
        } else {
            f64::NAN
        },
        out_tok_per_eval_sec: if eval_seconds > 0.0 {
            gen_tokens as f64 / eval_seconds
        } else {
            f64::NAN
        },
        max_uncovered: runs.iter().map(|r| r.uncovered).max().unwrap_or(0),
        retries: runs.iter().map(|r| r.stats.retries).sum(),
        retry_backoff_seconds: runs.iter().map(|r| r.stats.retry_backoff_seconds).sum(),
        validation_failures: runs.iter().map(|r| r.stats.validation_failures).sum(),
        cache_hits: runs.iter().map(|r| r.stats.cache_hits).sum(),
        cache_misses: runs.iter().map(|r| r.stats.cache_misses).sum(),
    }
}

/// Cache hit ratio over all lookups; `None` when nothing looked up (cache
/// disabled) so the report can say `n/a` instead of inventing a rate.
fn cache_hit_ratio(hits: u32, misses: u32) -> Option<f64> {
    let total = hits + misses;
    (total > 0).then(|| hits as f64 / total as f64)
}

/// Prompt tokens per second of request wall time (counterpart of the
/// output-token rate; cache hits contribute tokens without model time).
fn input_tok_per_req_sec(prompt_tokens: u64, request_seconds: f64) -> f64 {
    if request_seconds > 0.0 {
        prompt_tokens as f64 / request_seconds
    } else {
        f64::NAN
    }
}

/// Logical requests completed per second of mean elapsed time.
fn requests_per_sec(total_requests: u32, mean_elapsed: f64) -> f64 {
    if mean_elapsed > 0.0 {
        total_requests as f64 / mean_elapsed
    } else {
        f64::NAN
    }
}

/// Formats `value` with one decimal; non-finite values become `n/a`.
fn fmt1(value: f64) -> String {
    if value.is_finite() {
        format!("{:.1}", value)
    } else {
        "n/a".to_string()
    }
}

/// Short git description (`<sha>` or `<sha> (tracked changes)`), best effort.
fn git_info() -> String {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let Some(mut commit) = commit else {
        return "unknown".to_string();
    };
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    if dirty {
        commit.push_str(" (tracked changes)");
    }
    commit
}

fn run_meta_line(label: &str, value: &str) -> String {
    format!("- {}: {}", label, value)
}

/// One cell of the Task 09 benchmark matrix (a point in the
/// token-budget / concurrency / cache / context space).
#[derive(Debug, Clone, PartialEq)]
struct CellDims {
    label: String,
    tokens: usize,
    concurrency: usize,
    cache_on: bool,
    context: usize,
}

/// One-factor-at-a-time sweep over the Task 09 dimensions around the
/// baseline: baseline + three token budgets + two concurrency levels + the
/// primed cache cell + two context levels (nine cells for default inputs).
///
/// The budget/concurrency/context cells always run uncached — a warm cache
/// would skip the very requests those dimensions are meant to measure. The
/// cache dimension itself is measured as one primed ON cell against the
/// uncached baseline.
fn matrix_cells(tokens: usize, concurrency: usize, context: usize) -> Vec<CellDims> {
    let mut cells = vec![CellDims {
        label: "baseline".to_string(),
        tokens,
        concurrency,
        cache_on: false,
        context,
    }];
    for t in [1000usize, 2000, 3000, 4000] {
        if t != tokens {
            cells.push(CellDims {
                label: format!("tokens-{}", t),
                tokens: t,
                concurrency,
                cache_on: false,
                context,
            });
        }
    }
    for c in [1usize, 2, 3] {
        if c != concurrency {
            cells.push(CellDims {
                label: format!("concurrency-{}", c),
                tokens,
                concurrency: c,
                cache_on: false,
                context,
            });
        }
    }
    cells.push(CellDims {
        label: "cache-on".to_string(),
        tokens,
        concurrency,
        cache_on: true,
        context,
    });
    for cx in [0usize, 2, 4] {
        if cx != context {
            cells.push(CellDims {
                label: format!("context-{}", cx),
                tokens,
                concurrency,
                cache_on: false,
                context: cx,
            });
        }
    }
    cells
}

#[allow(clippy::too_many_arguments)]
fn render_report(
    config: &AppConfig,
    args: &BenchArgs,
    facts: &InputFacts,
    runs: &[RunResult],
    agg: &Aggregate,
    dims: Option<&CellDims>,
) -> String {
    let mut out = String::new();
    match dims {
        Some(d) => out.push_str(&format!("## Matrix cell: {}\n\n", d.label)),
        None => out.push_str("## Baseline benchmark run\n\n"),
    }
    let cache_on = dims.map_or(!args.no_cache, |d| d.cache_on);
    out.push_str(&run_meta_line(
        "date",
        &chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
    ));
    out.push('\n');
    out.push_str(&run_meta_line("git", &git_info()));
    out.push('\n');
    out.push_str(&run_meta_line(
        "environment",
        &format!(
            "os {}, {} logical cpus",
            std::env::consts::OS,
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0)
        ),
    ));
    out.push('\n');
    out.push_str(&run_meta_line(
        "model",
        &format!("{} @ {}", config.selected_model, config.ollama_url),
    ));
    out.push('\n');
    out.push_str(&run_meta_line(
        "languages",
        &format!(
            "{} ({}) -> {} ({})",
            config.source_language.name(),
            config.source_language.code(),
            config.target_language.name(),
            config.target_language.code()
        ),
    ));
    out.push('\n');
    out.push_str(&run_meta_line("file", &args.file.display().to_string()));
    out.push('\n');
    out.push_str(&run_meta_line(
        "runs",
        &format!(
            "{} (warmup: {})",
            args.runs,
            if args.warmup { "on" } else { "off" }
        ),
    ));
    out.push('\n');
    out.push_str(&run_meta_line("cache", if cache_on { "on" } else { "off" }));
    out.push('\n');
    if cache_on {
        out.push_str(&run_meta_line(
            "cache prime",
            "one full translation before timed runs (hit path measured)",
        ));
        out.push('\n');
    }
    out.push_str(&run_meta_line(
        "batch limits",
        &format!(
            "{} input tok, max {} entries/request",
            config.max_input_tokens, TRANSLATE_CHUNK_SIZE
        ),
    ));
    out.push('\n');
    out.push_str(&run_meta_line(
        "context",
        &format!(
            "before {}, after {}{}",
            config.context_before,
            config.context_after,
            if args.context.is_some() {
                " (--context override)"
            } else {
                ""
            }
        ),
    ));
    out.push('\n');
    out.push_str(&run_meta_line(
        "concurrency",
        &format!(
            "{}{}",
            config.max_concurrent_requests,
            if args.concurrency.is_some() {
                " (--concurrency override)"
            } else {
                ""
            }
        ),
    ));
    out.push('\n');

    let rows: [(&str, String, &str); 28] = [
        ("subtitle count", facts.entries.to_string(), "measured"),
        (
            "total source characters",
            facts.source_chars.to_string(),
            "measured",
        ),
        (
            "estimated input tokens (chars/4)",
            facts.est_input_tokens.to_string(),
            "estimate",
        ),
        (
            "actual prompt tokens (Ollama)",
            agg.prompt_tokens.to_string(),
            "measured",
        ),
        (
            "actual output tokens (Ollama)",
            agg.gen_tokens.to_string(),
            "measured",
        ),
        (
            "translation requests",
            agg.total_requests.to_string(),
            "measured",
        ),
        (
            "successful requests (usable response)",
            agg.successful_requests.to_string(),
            "measured",
        ),
        (
            "failed/unparseable batch responses",
            agg.batch_failures.to_string(),
            "measured",
        ),
        (
            "validation failures (rejected responses)",
            agg.validation_failures.to_string(),
            "measured",
        ),
        (
            "batch entries covered / sent",
            format!("{}/{}", agg.batch_covered, agg.batch_entries),
            "measured",
        ),
        (
            "single fallback requests",
            agg.single_requests.to_string(),
            "measured",
        ),
        ("retries", agg.retries.to_string(), "measured"),
        (
            "retry backoff time",
            format!("{} s", fmt1(agg.retry_backoff_seconds)),
            "measured",
        ),
        ("cache hits", agg.cache_hits.to_string(), "measured"),
        ("cache misses", agg.cache_misses.to_string(), "measured"),
        (
            "cache hit ratio",
            match cache_hit_ratio(agg.cache_hits, agg.cache_misses) {
                Some(ratio) => format!("{}%", fmt1(ratio * 100.0)),
                None => "n/a".to_string(),
            },
            "derived",
        ),
        (
            "total elapsed (mean)",
            format!("{} s", fmt1(agg.mean_elapsed)),
            "measured",
        ),
        (
            "elapsed min / max",
            format!("{} / {} s", fmt1(agg.min_elapsed), fmt1(agg.max_elapsed)),
            "measured",
        ),
        (
            "parse time (excluded from translation metrics)",
            format!("{:.3} s", facts.parse_seconds),
            "measured",
        ),
        (
            "average batch size",
            format!("{} entries/req", fmt1(agg.avg_batch_size)),
            "derived",
        ),
        (
            "average request duration",
            format!("{} s", fmt1(agg.avg_request_seconds)),
            "derived",
        ),
        ("queue wait", "n/a".to_string(), "single job, no queue"),
        (
            "subtitles/sec (entries / mean elapsed)",
            fmt1(agg.subs_per_sec),
            "derived",
        ),
        (
            "requests/sec (requests / mean elapsed)",
            fmt1(requests_per_sec(agg.total_requests, agg.mean_elapsed)),
            "derived",
        ),
        (
            "input tokens/sec (request wall time)",
            fmt1(input_tok_per_req_sec(
                agg.prompt_tokens,
                agg.request_seconds,
            )),
            "derived",
        ),
        (
            "output tokens/sec (request wall time)",
            fmt1(agg.out_tok_per_req_sec),
            "derived",
        ),
        (
            "output tokens/sec (model eval only)",
            fmt1(agg.out_tok_per_eval_sec),
            "derived",
        ),
        (
            "uncovered entries (max over runs)",
            agg.max_uncovered.to_string(),
            "measured",
        ),
    ];

    out.push_str("\n| Metric | Value | Kind |\n|---|---|---|\n");
    for (metric, value, kind) in &rows {
        out.push_str(&format!("| {} | {} | {} |\n", metric, value, kind));
    }

    out.push_str("\n### Per-run detail\n\n");
    out.push_str(
        "\n| run | elapsed s | requests | batch s | single s | prompt tok | gen tok | eval s | subs/s | uncovered | cache h | cache m |\n",
    );
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (i, run) in runs.iter().enumerate() {
        let s = &run.stats;
        let requests = s.batch_requests + s.single_requests;
        let gen = s.batch_gen_tokens + s.single_gen_tokens;
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            i + 1,
            fmt1(run.elapsed),
            requests,
            fmt1(s.batch_seconds),
            fmt1(s.single_seconds),
            s.batch_prompt_tokens + s.single_prompt_tokens,
            gen,
            fmt1(s.batch_eval_seconds + s.single_eval_seconds),
            fmt1(facts.entries as f64 / run.elapsed),
            run.uncovered,
            s.cache_hits,
            s.cache_misses,
        ));
    }

    let diagnostics: Vec<(usize, &String)> = runs
        .iter()
        .enumerate()
        .flat_map(|(i, r)| r.stats.batch_diagnostics.iter().map(move |d| (i + 1, d)))
        .collect();
    if !diagnostics.is_empty() {
        out.push_str("\n### Diagnostics\n\n");
        for (run_no, diag) in diagnostics {
            out.push_str(&format!("- run {}: {}\n", run_no, diag));
        }
    }

    out
}

fn texts_of(subtitle: &crate::models::SubtitleFile) -> Vec<String> {
    subtitle.entries.iter().map(|e| e.text.clone()).collect()
}

fn uncovered_count(source: &[String], translated: &[String]) -> usize {
    source
        .iter()
        .zip(translated.iter())
        .filter(|(src, tr)| !src.trim().is_empty() && tr.trim().is_empty())
        .count()
}

/// Runs the benchmark described by `args` and prints/writes the report.
pub fn run(args: &BenchArgs) -> Result<()> {
    let mut config = AppConfig::load(&AppConfig::config_path()).unwrap_or_default();
    if let Some(n) = args.context {
        config.context_before = n;
        config.context_after = n;
    }
    if let Some(n) = args.concurrency {
        config.max_concurrent_requests = n;
    }
    if let Some(n) = args.tokens {
        config.max_input_tokens = n;
    }
    let source: Language = config.source_language;
    let target: Language = config.target_language;

    let parse_started = Instant::now();
    let subtitle = parse_subtitle_file(&args.file, source, target)
        .with_context(|| format!("failed to parse {}", args.file.display()))?;
    let parse_seconds = parse_started.elapsed().as_secs_f64();

    let texts = texts_of(&subtitle);
    let entries = texts.len();
    anyhow::ensure!(
        entries > 0,
        "no subtitle entries found in {}",
        args.file.display()
    );
    let source_chars: usize = texts.iter().map(|t| t.chars().count()).sum();
    let facts = InputFacts {
        entries,
        source_chars,
        est_input_tokens: source_chars / 4,
        parse_seconds,
    };

    // Connectivity check once, with the base configuration.
    let mut client = OllamaClient::new(&config)?;
    check_model(&client, &config)?;

    if args.matrix {
        return run_matrix(args, &config, &facts, &texts, source, target);
    }

    if args.no_cache {
        client.disable_cache();
    }
    let cache_on = !args.no_cache;
    let results = execute_runs(&client, args, &texts, source, target, cache_on)?;
    let agg = aggregate(&results, entries);
    let report = render_report(&config, args, &facts, &results, &agg, None);
    println!("{}", report);

    if let Some(out) = &args.out {
        write_text(out, &report, "report")?;
    }
    if let Some(path) = &args.json {
        let value = single_json(args, &config, &facts, &results, &agg, cache_on);
        write_text(path, &to_pretty(&value)?, "json result")?;
    }
    Ok(())
}

/// Checks Ollama reachability and warns when the selected model is missing.
fn check_model(client: &OllamaClient, config: &AppConfig) -> Result<()> {
    let models = client.fetch_models().with_context(|| {
        format!(
            "Ollama unreachable at {} (is `ollama serve` running?)",
            config.ollama_url
        )
    })?;
    if !models.contains(&config.selected_model) {
        eprintln!(
            "warning: selected model '{}' not listed by /api/tags",
            config.selected_model
        );
    }
    Ok(())
}

/// Warmup, optional cache priming and the timed runs against `client`.
///
/// When the cache is on, one full translation of the file is done *before*
/// the timed runs so they measure the cache-hit path (a cold first run would
/// mix misses into the timing). Prime stats are discarded.
fn execute_runs(
    client: &OllamaClient,
    args: &BenchArgs,
    texts: &[String],
    source: Language,
    target: Language,
    cache_on: bool,
) -> Result<Vec<RunResult>> {
    if args.warmup {
        client
            .translate_batch(&["Warmup.".to_string()], source, target, |_, _| true)
            .context("warmup request failed")?;
    }
    if cache_on {
        client
            .translate_batch(texts, source, target, |_, _| true)
            .context("cache prime failed")?;
        eprintln!("cache primed: timed runs measure the hit path");
    }

    let mut results = Vec::with_capacity(args.runs);
    for i in 0..args.runs {
        let started = Instant::now();
        let (translated, stats) = client
            .translate_batch(texts, source, target, |_, _| true)
            .with_context(|| format!("benchmark run {} failed", i + 1))?;
        let elapsed = started.elapsed().as_secs_f64();
        let uncovered = uncovered_count(texts, &translated);
        eprintln!(
            "run {}/{}: {:.1}s, {} requests, {} uncovered",
            i + 1,
            args.runs,
            elapsed,
            stats.batch_requests + stats.single_requests,
            uncovered,
        );
        results.push(RunResult {
            elapsed,
            stats,
            uncovered,
        });
    }
    Ok(results)
}

/// Task 09 matrix mode: runs every sweep cell, prints one combined report
/// (meta + comparison table + full per-cell reports) and optional JSON.
fn run_matrix(
    args: &BenchArgs,
    base_config: &AppConfig,
    facts: &InputFacts,
    texts: &[String],
    source: Language,
    target: Language,
) -> Result<()> {
    let cells = matrix_cells(
        base_config.max_input_tokens,
        base_config.max_concurrent_requests,
        base_config.context_before,
    );
    eprintln!(
        "matrix: {} cells x {} run(s) (the cache-on cell adds one priming run)",
        cells.len(),
        args.runs
    );

    let mut cell_markdown = String::new();
    let mut cell_jsons = Vec::with_capacity(cells.len());
    let mut comparison: Vec<(CellDims, Aggregate)> = Vec::with_capacity(cells.len());
    for (i, dims) in cells.iter().enumerate() {
        eprintln!(
            "cell {}/{}: {} — tokens={}, concurrency={}, cache={}, context={}",
            i + 1,
            cells.len(),
            dims.label,
            dims.tokens,
            dims.concurrency,
            if dims.cache_on { "on" } else { "off" },
            dims.context,
        );
        let mut config = base_config.clone();
        config.max_input_tokens = dims.tokens;
        config.max_concurrent_requests = dims.concurrency;
        config.context_before = dims.context;
        config.context_after = dims.context;
        let mut client = OllamaClient::new(&config)?;
        if !dims.cache_on {
            client.disable_cache();
        }
        let results = execute_runs(&client, args, texts, source, target, dims.cache_on)?;
        let agg = aggregate(&results, facts.entries);
        cell_markdown.push_str(&render_report(
            &config,
            args,
            facts,
            &results,
            &agg,
            Some(dims),
        ));
        cell_markdown.push('\n');
        cell_jsons.push(cell_json(
            Some(dims),
            &config,
            dims.cache_on,
            &results,
            &agg,
        ));
        comparison.push((dims.clone(), agg));
    }

    let mut report = String::new();
    report.push_str("## Benchmark matrix run\n\n");
    report.push_str(&run_meta_line(
        "date",
        &chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
    ));
    report.push('\n');
    report.push_str(&run_meta_line("git", &git_info()));
    report.push('\n');
    report.push_str(&run_meta_line(
        "environment",
        &format!(
            "os {}, {} logical cpus",
            std::env::consts::OS,
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0)
        ),
    ));
    report.push('\n');
    report.push_str(&run_meta_line(
        "model",
        &format!(
            "{} @ {}",
            base_config.selected_model, base_config.ollama_url
        ),
    ));
    report.push('\n');
    report.push_str(&run_meta_line("file", &args.file.display().to_string()));
    report.push('\n');
    report.push_str(&run_meta_line(
        "cells",
        &format!("{} (one-factor-at-a-time sweep)", comparison.len()),
    ));
    report.push('\n');
    report.push_str(&run_meta_line(
        "runs",
        &format!(
            "{} per cell (warmup: {})",
            args.runs,
            if args.warmup { "on" } else { "off" }
        ),
    ));
    report.push('\n');
    report.push_str(&run_meta_line(
        "sweep",
        "budget/concurrency/context cells uncached; cache cell primed",
    ));
    report.push('\n');
    report.push_str(&render_comparison(&comparison));
    report.push('\n');
    report.push_str(&cell_markdown);

    println!("{}", report);
    if let Some(out) = &args.out {
        write_text(out, &report, "matrix report")?;
    }
    if let Some(path) = &args.json {
        let value = matrix_json(args, base_config, facts, cell_jsons);
        write_text(path, &to_pretty(&value)?, "matrix json")?;
    }
    Ok(())
}

/// Side-by-side metric table over all matrix cells (the human summary).
fn render_comparison(cells: &[(CellDims, Aggregate)]) -> String {
    let mut out = String::from("### Comparison\n\n");
    out.push_str(
        "| cell | tokens | concurrency | cache | context | mean s | subs/s | requests/s | input tok/s | output tok/s (eval) | avg latency s | cache hits/misses | retries | validation failures | uncovered |\n",
    );
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (dims, agg) in cells {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {} |\n",
            dims.label,
            dims.tokens,
            dims.concurrency,
            if dims.cache_on { "on" } else { "off" },
            dims.context,
            fmt1(agg.mean_elapsed),
            fmt1(agg.subs_per_sec),
            fmt1(requests_per_sec(agg.total_requests, agg.mean_elapsed)),
            fmt1(input_tok_per_req_sec(
                agg.prompt_tokens,
                agg.request_seconds
            )),
            fmt1(agg.out_tok_per_eval_sec),
            fmt1(agg.avg_request_seconds),
            agg.cache_hits,
            agg.cache_misses,
            agg.retries,
            agg.validation_failures,
            agg.max_uncovered,
        ));
    }
    out
}

/// Writes `content` to `path`, creating parent directories.
fn write_text(path: &std::path::Path, content: &str, what: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    std::fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))?;
    eprintln!("{} written to {}", what, path.display());
    Ok(())
}

/// JSON number or `null` for NaN/infinity (never invalid JSON).
fn jnum(value: f64) -> serde_json::Value {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

fn to_pretty(value: &serde_json::Value) -> Result<String> {
    Ok(serde_json::to_string_pretty(value)?)
}

/// Run-level telemetry (one timed run).
fn runs_json(runs: &[RunResult]) -> Vec<serde_json::Value> {
    runs.iter()
        .map(|r| {
            let s = &r.stats;
            serde_json::json!({
                "elapsed_s": jnum(r.elapsed),
                "requests": s.batch_requests + s.single_requests,
                "batch_s": jnum(s.batch_seconds),
                "single_s": jnum(s.single_seconds),
                "prompt_tokens": s.batch_prompt_tokens + s.single_prompt_tokens,
                "gen_tokens": s.batch_gen_tokens + s.single_gen_tokens,
                "eval_s": jnum(s.batch_eval_seconds + s.single_eval_seconds),
                "uncovered": r.uncovered,
                "cache_hits": s.cache_hits,
                "cache_misses": s.cache_misses,
                "retries": s.retries,
                "retry_backoff_s": jnum(s.retry_backoff_seconds),
                "validation_failures": s.validation_failures,
            })
        })
        .collect()
}

/// The Task 09 metric set in machine-readable form.
fn metrics_json(agg: &Aggregate) -> serde_json::Value {
    serde_json::json!({
        "total_time_mean_s": jnum(agg.mean_elapsed),
        "total_time_min_s": jnum(agg.min_elapsed),
        "total_time_max_s": jnum(agg.max_elapsed),
        "subtitles_per_sec": jnum(agg.subs_per_sec),
        "input_tokens_per_sec": jnum(input_tok_per_req_sec(
            agg.prompt_tokens,
            agg.request_seconds,
        )),
        "output_tokens_per_req_sec": jnum(agg.out_tok_per_req_sec),
        "output_tokens_per_eval_sec": jnum(agg.out_tok_per_eval_sec),
        "requests_per_sec": jnum(requests_per_sec(
            agg.total_requests,
            agg.mean_elapsed,
        )),
        "request_count": agg.total_requests,
        "avg_request_latency_s": jnum(agg.avg_request_seconds),
        "queue_wait_s": serde_json::Value::Null,
        "cache_hits": agg.cache_hits,
        "cache_misses": agg.cache_misses,
        "cache_hit_ratio": match cache_hit_ratio(agg.cache_hits, agg.cache_misses) {
            Some(ratio) => jnum(ratio),
            None => serde_json::Value::Null,
        },
        "retry_count": agg.retries,
        "retry_backoff_s": jnum(agg.retry_backoff_seconds),
        "validation_failures": agg.validation_failures,
        "batch_failures": agg.batch_failures,
        "uncovered_max": agg.max_uncovered,
    })
}

fn meta_json(args: &BenchArgs, config: &AppConfig, facts: &InputFacts) -> serde_json::Value {
    serde_json::json!({
        "date": chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        "git": git_info(),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "os": std::env::consts::OS,
        "cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "model": config.selected_model,
        "ollama_url": config.ollama_url,
        "file": args.file.display().to_string(),
        "runs": args.runs,
        "warmup": args.warmup,
        "entries": facts.entries,
        "source_chars": facts.source_chars,
        "est_input_tokens": facts.est_input_tokens,
        "parse_seconds": jnum(facts.parse_seconds),
    })
}

fn dims_json(config: &AppConfig, cache_on: bool) -> serde_json::Value {
    serde_json::json!({
        "tokens": config.max_input_tokens,
        "concurrency": config.max_concurrent_requests,
        "cache": if cache_on { "on" } else { "off" },
        "cache_prime": cache_on,
        "context": config.context_before,
    })
}

/// One benchmark cell: dimensions + raw runs + aggregate metrics.
fn cell_json(
    dims: Option<&CellDims>,
    config: &AppConfig,
    cache_on: bool,
    runs: &[RunResult],
    agg: &Aggregate,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "dims": dims_json(config, cache_on),
        "runs": runs_json(runs),
        "metrics": metrics_json(agg),
    });
    if let Some(d) = dims {
        value["label"] = serde_json::json!(d.label);
    }
    value
}

/// Machine-readable single-run result (`--json`).
fn single_json(
    args: &BenchArgs,
    config: &AppConfig,
    facts: &InputFacts,
    runs: &[RunResult],
    agg: &Aggregate,
    cache_on: bool,
) -> serde_json::Value {
    serde_json::json!({
        "schema": "auto-translate-subs/bench/1",
        "meta": meta_json(args, config, facts),
        "cell": cell_json(None, config, cache_on, runs, agg),
    })
}

/// Machine-readable matrix result (`--matrix --json`).
fn matrix_json(
    args: &BenchArgs,
    config: &AppConfig,
    facts: &InputFacts,
    cells: Vec<serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "schema": "auto-translate-subs/bench-matrix/1",
        "meta": meta_json(args, config, facts),
        "cells": cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn sample_stats() -> RequestStats {
        RequestStats {
            batch_requests: 2,
            batch_seconds: 30.0,
            batch_entries: 50,
            batch_covered: 30,
            batch_prompt_tokens: 1_000,
            batch_gen_tokens: 2_000,
            batch_eval_seconds: 18.0,
            batch_failures: 0,
            single_requests: 0,
            single_seconds: 0.0,
            single_prompt_tokens: 0,
            single_gen_tokens: 0,
            single_eval_seconds: 0.0,
            cache_hits: 7,
            cache_misses: 43,
            retries: 3,
            retry_backoff_seconds: 0.5,
            validation_failures: 5,
            batch_diagnostics: Vec::new(),
        }
    }

    #[test]
    fn parse_full_args() {
        let parsed = parse_args(&args(&[
            "prog",
            "--bench",
            "file.srt",
            "--runs",
            "3",
            "--out",
            "report.md",
            "--no-warmup",
            "--no-cache",
            "--context",
            "0",
            "--concurrency",
            "2",
            "--tokens",
            "2000",
        ]))
        .unwrap();
        assert_eq!(
            parsed,
            BenchArgs {
                file: PathBuf::from("file.srt"),
                runs: 3,
                out: Some(PathBuf::from("report.md")),
                warmup: false,
                no_cache: true,
                context: Some(0),
                concurrency: Some(2),
                tokens: Some(2000),
                matrix: false,
                json: None,
            }
        );
    }

    #[test]
    fn parse_concurrency_override() {
        let parsed =
            parse_args(&args(&["prog", "--bench", "f.srt", "--concurrency", "3"])).unwrap();
        assert_eq!(parsed.concurrency, Some(3));
        let parsed = parse_args(&args(&["prog", "--bench", "f.srt"])).unwrap();
        assert_eq!(parsed.concurrency, None, "no flag keeps the stored config");
        assert!(
            parse_args(&args(&["prog", "--bench", "f.srt", "--concurrency", "0"])).is_err(),
            "zero workers is meaningless"
        );
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--concurrency", "x"])).is_err());
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--concurrency"])).is_err());
    }

    #[test]
    fn parse_context_override() {
        let parsed = parse_args(&args(&["prog", "--bench", "f.srt", "--context", "2"])).unwrap();
        assert_eq!(parsed.context, Some(2));
        let parsed = parse_args(&args(&["prog", "--bench", "f.srt"])).unwrap();
        assert_eq!(parsed.context, None, "no flag keeps the stored config");
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--context", "x"])).is_err());
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--context"])).is_err());
    }

    #[test]
    fn parse_defaults() {
        let parsed = parse_args(&args(&["prog", "--bench", "file.srt"])).unwrap();
        assert_eq!(parsed.runs, 1);
        assert_eq!(parsed.out, None);
        assert!(parsed.warmup);
        assert!(!parsed.no_cache);
        assert_eq!(parsed.concurrency, None);
        assert_eq!(parsed.tokens, None);
        assert!(!parsed.matrix);
        assert_eq!(parsed.json, None);
    }

    #[test]
    fn parse_tokens_matrix_json_flags() {
        let parsed = parse_args(&args(&["prog", "--bench", "f.srt", "--tokens", "4000"])).unwrap();
        assert_eq!(parsed.tokens, Some(4000), "token budget override");
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--tokens", "0"])).is_err());
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--tokens", "x"])).is_err());
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--tokens"])).is_err());

        let parsed = parse_args(&args(&[
            "prog", "--bench", "f.srt", "--matrix", "--json", "res.json",
        ]))
        .unwrap();
        assert!(parsed.matrix, "--matrix switch");
        assert_eq!(parsed.json, Some(PathBuf::from("res.json")), "--json path");
        let plain = parse_args(&args(&["prog", "--bench", "f.srt"])).unwrap();
        assert!(!plain.matrix);
        assert_eq!(plain.json, None);
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--json"])).is_err());
    }

    #[test]
    fn parse_requires_file_path() {
        assert!(parse_args(&args(&["prog", "--bench"])).is_err());
        assert!(parse_args(&args(&["prog", "--bench", "--runs", "2"])).is_err());
        assert!(parse_args(&args(&["prog"])).is_err());
    }

    #[test]
    fn parse_rejects_bad_runs_and_unknown_flags() {
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--runs", "0"])).is_err());
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--runs", "abc"])).is_err());
        assert!(parse_args(&args(&["prog", "--bench", "f.srt", "--frobnicate"])).is_err());
    }

    #[test]
    fn aggregate_computes_derived_metrics() {
        let runs = vec![RunResult {
            elapsed: 32.0,
            stats: sample_stats(),
            uncovered: 0,
        }];
        let agg = aggregate(&runs, 30);
        assert_eq!(agg.total_requests, 2);
        assert_eq!(agg.successful_requests, 2);
        assert!((agg.avg_batch_size - 25.0).abs() < 1e-9);
        assert!((agg.avg_request_seconds - 15.0).abs() < 1e-9);
        assert!((agg.subs_per_sec - 30.0 / 32.0).abs() < 1e-9);
        assert!((agg.out_tok_per_req_sec - 2_000.0 / 30.0).abs() < 1e-9);
        assert!((agg.out_tok_per_eval_sec - 2_000.0 / 18.0).abs() < 1e-9);
        assert_eq!(agg.max_uncovered, 0);
        assert_eq!(agg.retries, 3);
        assert!((agg.retry_backoff_seconds - 0.5).abs() < 1e-9);
        assert_eq!(agg.validation_failures, 5);
        assert_eq!(agg.cache_hits, 7);
        assert_eq!(agg.cache_misses, 43);
    }

    #[test]
    fn aggregate_counts_failures_out_of_successful() {
        let mut stats = sample_stats();
        stats.batch_failures = 1;
        let runs = vec![RunResult {
            elapsed: 32.0,
            stats,
            uncovered: 1,
        }];
        let agg = aggregate(&runs, 30);
        assert_eq!(agg.total_requests, 2);
        assert_eq!(agg.successful_requests, 1);
        assert_eq!(agg.max_uncovered, 1);
    }

    #[test]
    fn report_contains_stable_metric_rows() {
        let runs = vec![RunResult {
            elapsed: 32.0,
            stats: sample_stats(),
            uncovered: 0,
        }];
        let agg = aggregate(&runs, 30);
        let facts = InputFacts {
            entries: 30,
            source_chars: 800,
            est_input_tokens: 200,
            parse_seconds: 0.004,
        };
        let args = BenchArgs {
            file: PathBuf::from("sample.srt"),
            runs: 1,
            out: None,
            warmup: true,
            no_cache: false,
            context: None,
            concurrency: None,
            tokens: None,
            matrix: false,
            json: None,
        };
        let report = render_report(&AppConfig::default(), &args, &facts, &runs, &agg, None);
        for needle in [
            "## Baseline benchmark run",
            "- cache: on",
            "- context: before 2, after 2",
            "- concurrency: 1",
            "| subtitle count | 30 | measured |",
            "| estimated input tokens (chars/4) | 200 | estimate |",
            "| batch entries covered / sent | 30/50 | measured |",
            "| cache hits | 7 | measured |",
            "| cache misses | 43 | measured |",
            "| cache hit ratio | 14.0% | derived |",
            "| validation failures (rejected responses) | 5 | measured |",
            "| output tokens/sec (model eval only) | 111.1 | derived |",
            "| input tokens/sec (request wall time) | 33.3 | derived |",
            "| requests/sec (requests / mean elapsed) | 0.1 | derived |",
            "| retries | 3 | measured |",
            "| retry backoff time | 0.5 s | measured |",
            "### Per-run detail",
            "| 1 | 32.0 | 2 | 30.0 | 0.0 | 1000 | 2000 | 18.0 | 0.9 | 0 | 7 | 43 |",
        ] {
            assert!(
                report.contains(needle),
                "report missing row: {}\n{}",
                needle,
                report
            );
        }
    }

    fn bench_args() -> BenchArgs {
        BenchArgs {
            file: PathBuf::from("sample.srt"),
            runs: 1,
            out: None,
            warmup: true,
            no_cache: false,
            context: None,
            concurrency: None,
            tokens: None,
            matrix: false,
            json: None,
        }
    }

    #[test]
    fn matrix_cells_sweep_layout() {
        let cells = matrix_cells(3000, 1, 2);
        assert_eq!(cells.len(), 9, "one baseline + 8 sweep cells");
        let labels: Vec<&str> = cells.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "baseline",
                "tokens-1000",
                "tokens-2000",
                "tokens-4000",
                "concurrency-2",
                "concurrency-3",
                "cache-on",
                "context-0",
                "context-4",
            ]
        );
        assert_eq!(
            cells[0],
            CellDims {
                label: "baseline".to_string(),
                tokens: 3000,
                concurrency: 1,
                cache_on: false,
                context: 2,
            }
        );
        let cache = cells.iter().find(|c| c.label == "cache-on").unwrap();
        assert!(cache.cache_on, "the cache cell is the only ON cell");
        assert_eq!(
            (cache.tokens, cache.concurrency, cache.context),
            (3000, 1, 2),
            "cache cell keeps the baseline dimensions"
        );
        for cell in &cells {
            if cell.label != "cache-on" {
                assert!(
                    !cell.cache_on,
                    "sweep cells must issue real requests: {}",
                    cell.label
                );
            }
        }

        // A custom baseline skips its own value inside each sweep.
        let custom = matrix_cells(4000, 3, 0);
        assert_eq!(custom.len(), 9);
        assert!(custom.iter().any(|c| c.label == "tokens-3000"));
        assert!(!custom.iter().any(|c| c.label == "tokens-4000"));
        assert!(!custom.iter().any(|c| c.label == "concurrency-3"));
        assert!(!custom.iter().any(|c| c.label == "context-0"));
    }

    #[test]
    fn metric_helper_functions() {
        assert_eq!(cache_hit_ratio(7, 43), Some(0.14));
        assert_eq!(cache_hit_ratio(10, 0), Some(1.0));
        assert_eq!(cache_hit_ratio(0, 0), None, "no lookup: n/a, not 0%");
        assert!((requests_per_sec(4, 8.0) - 0.5).abs() < 1e-9);
        assert!(requests_per_sec(0, 0.0).is_nan());
        assert!((input_tok_per_req_sec(1_000, 10.0) - 100.0).abs() < 1e-9);
        assert!(input_tok_per_req_sec(1_000, 0.0).is_nan());
    }

    #[test]
    fn single_json_exposes_task09_metrics() {
        let runs = vec![RunResult {
            elapsed: 32.0,
            stats: sample_stats(),
            uncovered: 0,
        }];
        let agg = aggregate(&runs, 30);
        let facts = InputFacts {
            entries: 30,
            source_chars: 800,
            est_input_tokens: 200,
            parse_seconds: 0.004,
        };
        let value = single_json(
            &bench_args(),
            &AppConfig::default(),
            &facts,
            &runs,
            &agg,
            false,
        );

        assert_eq!(value["schema"], "auto-translate-subs/bench/1");
        assert_eq!(value["cell"]["dims"]["tokens"], 3000);
        assert_eq!(value["cell"]["dims"]["cache"], "off");
        assert_eq!(value["cell"]["dims"]["concurrency"], 1);
        assert_eq!(value["cell"]["dims"]["context"], 2);
        assert_eq!(value["cell"]["runs"].as_array().unwrap().len(), 1);
        assert_eq!(value["cell"]["runs"][0]["retries"], 3);

        let m = &value["cell"]["metrics"];
        assert_eq!(m["request_count"], 2);
        assert_eq!(m["retry_count"], 3);
        assert_eq!(m["validation_failures"], 5);
        assert_eq!(m["cache_hits"], 7);
        assert!((m["cache_hit_ratio"].as_f64().unwrap() - 0.14).abs() < 1e-9);
        assert!(
            m["queue_wait_s"].is_null(),
            "single job has no queue: recorded as null"
        );
    }

    #[test]
    fn json_rates_serialize_null_instead_of_nan() {
        let runs = vec![RunResult {
            elapsed: 0.0,
            stats: RequestStats::default(),
            uncovered: 0,
        }];
        let agg = aggregate(&runs, 0);
        let m = metrics_json(&agg);
        assert!(m["subtitles_per_sec"].is_null(), "0/0 must not be NaN");
        assert!(m["cache_hit_ratio"].is_null(), "no lookups: null");
        assert!(m["input_tokens_per_sec"].is_null());
        let text = serde_json::to_string(&m).expect("always valid JSON");
        assert!(!text.contains("NaN"));
    }

    #[test]
    fn matrix_json_lists_cells_with_labels() {
        let runs = vec![RunResult {
            elapsed: 32.0,
            stats: sample_stats(),
            uncovered: 0,
        }];
        let agg = aggregate(&runs, 30);
        let facts = InputFacts {
            entries: 30,
            source_chars: 800,
            est_input_tokens: 200,
            parse_seconds: 0.004,
        };
        let cells: Vec<serde_json::Value> = matrix_cells(3000, 1, 2)
            .iter()
            .map(|dims| {
                let config = AppConfig {
                    max_input_tokens: dims.tokens,
                    max_concurrent_requests: dims.concurrency,
                    context_before: dims.context,
                    context_after: dims.context,
                    ..AppConfig::default()
                };
                cell_json(Some(dims), &config, dims.cache_on, &runs, &agg)
            })
            .collect();
        let value = matrix_json(&bench_args(), &AppConfig::default(), &facts, cells);

        assert_eq!(value["schema"], "auto-translate-subs/bench-matrix/1");
        let listed = value["cells"].as_array().unwrap();
        assert_eq!(listed.len(), 9);
        assert_eq!(listed[0]["label"], "baseline");
        assert_eq!(listed[0]["dims"]["tokens"], 3000);
        let cache_cell = listed.iter().find(|c| c["label"] == "cache-on").unwrap();
        assert_eq!(cache_cell["dims"]["cache"], "on");
        assert_eq!(cache_cell["dims"]["cache_prime"], true);
        assert_eq!(listed[6]["metrics"]["validation_failures"], 5);
    }
}
