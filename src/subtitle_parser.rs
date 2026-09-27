use crate::models::{Language, SubtitleEntry, SubtitleFile};
use anyhow::{Context, Result};
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    Srt,
    Vtt,
}

/// Detects the subtitle format from the file extension (defaults to SRT).
pub fn format_from_path(path: &Path) -> SubtitleFormat {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("vtt") => SubtitleFormat::Vtt,
        _ => SubtitleFormat::Srt,
    }
}

pub fn parse_subtitle_file(
    path: &Path,
    source_lang: Language,
    target_lang: Language,
) -> Result<SubtitleFile> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    let entries = match format_from_path(path) {
        SubtitleFormat::Vtt => parse_vtt_content(&content)?,
        SubtitleFormat::Srt => parse_srt_content(&content)?,
    };

    Ok(SubtitleFile {
        path: path.to_path_buf(),
        entries,
        source_language: source_lang,
        target_language: target_lang,
    })
}

fn parse_srt_content(content: &str) -> Result<Vec<SubtitleEntry>> {
    // Normalize line endings first
    let content = content.replace("\r\n", "\n").replace('\r', "\n");
    
    // Try primary parser (double newline separator)
    let entries = parse_srt_blocks(&content)?;
    
    // If we got very few entries, try alternative parsing (single blank line separator)
    if entries.len() < 2 {
        return parse_srt_content_single_newline(&content);
    }
    
    Ok(entries)
}

fn parse_srt_blocks(content: &str) -> Result<Vec<SubtitleEntry>> {
    let mut entries = Vec::new();
    
    // Split by double newline, but filter out empty blocks
    let blocks: Vec<&str> = content
        .split("\n\n")
        .map(|b| b.trim())
        .filter(|b| !b.is_empty())
        .collect();
    
    for block in blocks {
        let lines: Vec<&str> = block.lines().collect();
        if lines.len() < 3 {
            continue;
        }
        
        let index = lines[0].trim().parse::<usize>().unwrap_or(entries.len() + 1);
        let time_line = lines[1].trim();
        let text = lines[2..].join("\n").trim().to_string();
        
        if text.is_empty() {
            continue;
        }
        
        let Ok((start_time, end_time)) = parse_time_line(time_line) else {
            // Skip malformed entry instead of failing the whole file
            continue;
        };
        
        entries.push(SubtitleEntry {
            index,
            start_time,
            end_time,
            text,
            translated_text: None,
        });
    }
    
    Ok(entries)
}

fn parse_srt_content_single_newline(content: &str) -> Result<Vec<SubtitleEntry>> {
    let mut entries = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    
    while i < lines.len() {
        // Skip empty lines
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() {
            break;
        }
        
        // Parse index
        let index_line = lines[i].trim();
        let index: usize = index_line.parse().unwrap_or(entries.len() + 1);
        i += 1;
        
        // Skip empty lines
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() {
            break;
        }
        
        // Parse time line
        let time_line = lines[i].trim();
        if !time_line.contains(" --> ") {
            // Not a valid time line, skip
            i += 1;
            continue;
        }
        i += 1;
        
        // Parse text lines (until empty line or end or next index)
        let mut text_lines = Vec::new();
        while i < lines.len() {
            let line = lines[i].trim();
            if line.is_empty() {
                i += 1;
                break;
            }
            // Check if next line looks like an index (just a number followed by time line)
            if line.parse::<usize>().is_ok() && i + 1 < lines.len() && lines[i + 1].contains(" --> ") {
                break;
            }
            text_lines.push(lines[i]);
            i += 1;
        }
        
        if text_lines.is_empty() {
            continue;
        }
        
        let text = text_lines.join("\n").trim().to_string();
        let Ok((start_time, end_time)) = parse_time_line(time_line) else {
            // Skip malformed entry instead of failing the whole file
            continue;
        };
        
        entries.push(SubtitleEntry {
            index,
            start_time,
            end_time,
            text,
            translated_text: None,
        });
    }
    
    Ok(entries)
}

/// Parses WebVTT content: `WEBVTT` header, optional NOTE/STYLE/REGION blocks,
/// cue identifiers, cue settings after the timestamps, and `MM:SS.mmm`
/// timestamps without hours.
fn parse_vtt_content(content: &str) -> Result<Vec<SubtitleEntry>> {
    let content = content.replace("\r\n", "\n").replace('\r', "\n");
    let content = content.strip_prefix('\u{feff}').unwrap_or(&content);
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    // Signature line and its metadata (until the first blank line)
    if i < lines.len() && lines[i].trim_start().starts_with("WEBVTT") {
        i += 1;
        while i < lines.len() && !lines[i].trim().is_empty() {
            i += 1;
        }
    }

    let mut entries = Vec::new();
    while i < lines.len() {
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() {
            break;
        }

        // Skip NOTE / STYLE / REGION blocks
        let first = lines[i].trim();
        if first.starts_with("NOTE") || first.starts_with("STYLE") || first.starts_with("REGION") {
            while i < lines.len() && !lines[i].trim().is_empty() {
                i += 1;
            }
            continue;
        }

        // An optional cue identifier may precede the timing line
        let mut time_line_idx = i;
        if !lines[i].contains("-->") {
            time_line_idx = i + 1;
            if time_line_idx >= lines.len() || !lines[time_line_idx].contains("-->") {
                while i < lines.len() && !lines[i].trim().is_empty() {
                    i += 1;
                }
                continue;
            }
        }

        let time_line = lines[time_line_idx].trim();
        i = time_line_idx + 1;

        let mut text_lines = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() {
            text_lines.push(lines[i]);
            i += 1;
        }
        let text = text_lines.join("\n").trim().to_string();
        if text.is_empty() {
            continue;
        }

        let Ok((start_time, end_time)) = parse_time_line(time_line) else {
            continue;
        };

        entries.push(SubtitleEntry {
            index: entries.len() + 1,
            start_time,
            end_time,
            text,
            translated_text: None,
        });
    }

    Ok(entries)
}

fn parse_time_line(line: &str) -> Result<(Duration, Duration)> {
    let (start_part, rest) = line
        .split_once("-->")
        .ok_or_else(|| anyhow::anyhow!("Invalid time format: {}", line))?;

    // The end timestamp may be followed by WebVTT cue settings
    // (e.g. `align:start position:0%`)
    let end_part = rest.split_whitespace().next().unwrap_or("");

    let start = parse_timestamp(start_part.trim())?;
    let end = parse_timestamp(end_part)?;

    Ok((start, end))
}

fn parse_timestamp(ts: &str) -> Result<Duration> {
    let ts = ts.replace(',', ".");
    let parts: Vec<&str> = ts.split(':').collect();

    let (hours, minutes, seconds) = match parts.as_slice() {
        [h, m, s] => (h.parse::<u64>()?, m.parse::<u64>()?, s.parse::<f64>()?),
        // WebVTT allows timestamps without hours (MM:SS.mmm)
        [m, s] => (0, m.parse::<u64>()?, s.parse::<f64>()?),
        _ => anyhow::bail!("Invalid timestamp format: {}", ts),
    };

    let total_millis = (hours * 3600 + minutes * 60) as f64 * 1000.0 + seconds * 1000.0;
    Ok(Duration::from_millis(total_millis as u64))
}

pub fn format_duration(d: Duration) -> String {
    let total_millis = d.as_millis();
    let hours = total_millis / 3_600_000;
    let minutes = (total_millis % 3_600_000) / 60_000;
    let seconds = (total_millis % 60_000) / 1000;
    let millis = total_millis % 1000;
    
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}

/// Falls back to the original text when translation is missing or empty
fn effective_text(entry: &SubtitleEntry) -> &str {
    match entry.translated_text.as_deref() {
        Some(t) if !t.trim().is_empty() => t,
        _ => entry.text.as_str(),
    }
}

pub fn write_srt_file(subtitle_file: &SubtitleFile, output_path: &Path) -> Result<()> {
    let mut content = String::new();
    
    for entry in &subtitle_file.entries {
        content.push_str(&entry.index.to_string());
        content.push('\n');
        content.push_str(&format_duration(entry.start_time));
        content.push_str(" --> ");
        content.push_str(&format_duration(entry.end_time));
        content.push('\n');
        content.push_str(effective_text(entry));
        content.push_str("\n\n");
    }
    
    std::fs::write(output_path, content)
        .with_context(|| format!("Failed to write file: {}", output_path.display()))?;
    
    Ok(())
}

/// WebVTT timestamp: `HH:MM:SS.mmm` (dot instead of comma, no cue index)
fn format_duration_vtt(d: Duration) -> String {
    let total_millis = d.as_millis();
    let hours = total_millis / 3_600_000;
    let minutes = (total_millis % 3_600_000) / 60_000;
    let seconds = (total_millis % 60_000) / 1000;
    let millis = total_millis % 1000;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

pub fn write_vtt_file(subtitle_file: &SubtitleFile, output_path: &Path) -> Result<()> {
    let mut content = String::from("WEBVTT\n\n");

    for entry in &subtitle_file.entries {
        content.push_str(&format_duration_vtt(entry.start_time));
        content.push_str(" --> ");
        content.push_str(&format_duration_vtt(entry.end_time));
        content.push('\n');
        content.push_str(effective_text(entry));
        content.push_str("\n\n");
    }

    std::fs::write(output_path, content)
        .with_context(|| format!("Failed to write file: {}", output_path.display()))?;

    Ok(())
}

/// Writes the file in the format matching its extension (`.srt` or `.vtt`).
pub fn write_subtitle_file(subtitle_file: &SubtitleFile, output_path: &Path) -> Result<()> {
    match format_from_path(output_path) {
        SubtitleFormat::Vtt => write_vtt_file(subtitle_file, output_path),
        SubtitleFormat::Srt => write_srt_file(subtitle_file, output_path),
    }
}

pub fn find_subtitle_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    
    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().is_some_and(|ext| {
                let ext = ext.to_string_lossy();
                ext.eq_ignore_ascii_case("srt") || ext.eq_ignore_ascii_case("vtt")
            })
        })
    {
        files.push(entry.path().to_path_buf());
    }
    
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Language;
    use std::path::Path;
    use std::path::PathBuf;
    
    fn get_test_path(filename: &str) -> std::path::PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or(".".to_string());
        // Try tests/fixtures (committed), then test_subs, then target/release/examples
        let fixtures = Path::new(&manifest_dir)
            .join("tests/fixtures")
            .join(filename);
        if fixtures.exists() {
            return fixtures;
        }
        let test_subs = Path::new(&manifest_dir).join("test_subs").join(filename);
        if test_subs.exists() {
            return test_subs;
        }
        Path::new(&manifest_dir).join("target/release/examples").join(filename)
    }
    
    #[test]
    fn test_parse_introduction_srt() {
        let path = get_test_path("1 - Introduction.srt");
        let result = parse_subtitle_file(&path, Language::English, Language::Turkish);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        
        let file = result.unwrap();
        assert_eq!(file.entries.len(), 25, "Expected 25 entries, got {}", file.entries.len());
        
        // Check first entry
        let e1 = &file.entries[0];
        assert_eq!(e1.index, 1);
        assert_eq!(e1.text, "Have you ever dreamed of creating your own\nhorror survival game?");
        
        // Check multi-line entry
        let e2 = &file.entries[1];
        assert_eq!(e2.index, 2);
        assert_eq!(e2.text, "If the answer is yes, you're in a great\nplace because in this course you will");
        
        // Check last entry
        let e25 = &file.entries[24];
        assert_eq!(e25.index, 25);
        assert_eq!(e25.text, "If that sounds interesting, I will see you\nin the first lesson.");
    }
    
    #[test]
    fn test_parse_sample_srt() {
        let path = get_test_path("sample.srt");
        let result = parse_subtitle_file(&path, Language::English, Language::Turkish);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        
        let file = result.unwrap();
        assert!(!file.entries.is_empty());
    }
    
    #[test]
    fn test_parse_windows_line_endings() {
        let content = "1\r\n00:00:01,000 --> 00:00:02,000\r\nHello\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,000\r\nWorld\r\n";
        let entries = parse_srt_content(content).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Hello");
        assert_eq!(entries[1].text, "World");
    }
    
    #[test]
    fn test_parse_multiple_blank_lines() {
        let content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n\n\n2\n00:00:03,000 --> 00:00:04,000\nWorld\n";
        let entries = parse_srt_content(content).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Hello");
        assert_eq!(entries[1].text, "World");
    }
    
    #[test]
    fn test_parse_single_blank_line() {
        let content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,000\nWorld\n";
        let entries = parse_srt_content(content).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Hello");
        assert_eq!(entries[1].text, "World");
    }

    #[test]
    fn test_parse_vtt_content() {
        let content = concat!(
            "WEBVTT - Example\n",
            "\n",
            "NOTE this comment\n",
            "spans two lines\n",
            "\n",
            "1\n",
            "00:00:01.000 --> 00:00:04.000 align:start position:0%\n",
            "Hello VTT\n",
            "\n",
            "00:00:05.500 --> 00:00:07.250\n",
            "Second cue\n",
            "multi line\n"
        );
        let entries = parse_vtt_content(content).unwrap();
        assert_eq!(entries.len(), 2);

        assert_eq!(entries[0].index, 1);
        assert_eq!(entries[0].text, "Hello VTT");
        assert_eq!(entries[0].start_time, Duration::from_millis(1000));
        assert_eq!(entries[0].end_time, Duration::from_millis(4000));

        assert_eq!(entries[1].text, "Second cue\nmulti line");
        assert_eq!(entries[1].start_time, Duration::from_millis(5500));
        assert_eq!(entries[1].end_time, Duration::from_millis(7250));
    }

    #[test]
    fn test_parse_vtt_without_hours_and_bom() {
        let content = "\u{feff}WEBVTT\n\n01:05.500 --> 01:07.000\nNo hours here\n";
        let entries = parse_vtt_content(content).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].start_time, Duration::from_millis(65_500));
        assert_eq!(entries[0].end_time, Duration::from_millis(67_000));
    }

    #[test]
    fn test_format_from_path() {
        assert_eq!(format_from_path(Path::new("a.srt")), SubtitleFormat::Srt);
        assert_eq!(format_from_path(Path::new("a.vtt")), SubtitleFormat::Vtt);
        assert_eq!(format_from_path(Path::new("a.VTT")), SubtitleFormat::Vtt);
        assert_eq!(format_from_path(Path::new("a.txt")), SubtitleFormat::Srt);
        assert_eq!(format_from_path(Path::new("a")), SubtitleFormat::Srt);
    }

    #[test]
    fn test_write_vtt_file() {
        let file = SubtitleFile {
            path: PathBuf::from("out.vtt"),
            entries: vec![SubtitleEntry {
                index: 1,
                start_time: Duration::from_millis(1500),
                end_time: Duration::from_millis(3250),
                text: "Original text".to_string(),
                translated_text: Some("Çevrilmiş metin".to_string()),
            }],
            source_language: Language::English,
            target_language: Language::Turkish,
        };

        let out = std::env::temp_dir().join(format!("ats_vtt_write_{}.vtt", std::process::id()));
        write_subtitle_file(&file, &out).unwrap();
        let written = std::fs::read_to_string(&out).unwrap();
        let _ = std::fs::remove_file(&out);

        assert!(written.starts_with("WEBVTT\n\n"));
        assert!(written.contains("00:00:01.500 --> 00:00:03.250"));
        assert!(written.contains("Çevrilmiş metin"));
        assert!(!written.contains("Original text"));
    }

    #[test]
    fn test_write_vtt_falls_back_to_original_when_translation_empty() {
        let file = SubtitleFile {
            path: PathBuf::from("out.vtt"),
            entries: vec![SubtitleEntry {
                index: 1,
                start_time: Duration::from_millis(0),
                end_time: Duration::from_millis(1000),
                text: "Original text".to_string(),
                translated_text: Some("   ".to_string()),
            }],
            source_language: Language::English,
            target_language: Language::Turkish,
        };

        let out = std::env::temp_dir().join(format!("ats_vtt_fallback_{}.vtt", std::process::id()));
        write_subtitle_file(&file, &out).unwrap();
        let written = std::fs::read_to_string(&out).unwrap();
        let _ = std::fs::remove_file(&out);

        assert!(written.contains("Original text"));
    }
    
    #[test]
    fn test_parse_multiline_text() {
        let content = "1\n00:00:01,000 --> 00:00:02,000\nLine 1\nLine 2\nLine 3\n\n2\n00:00:03,000 --> 00:00:04,000\nSingle line\n";
        let entries = parse_srt_content(content).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Line 1\nLine 2\nLine 3");
        assert_eq!(entries[1].text, "Single line");
    }
    
    #[test]
    fn test_parse_trailing_whitespace() {
        let content = "1\n00:00:01,000 --> 00:00:02,000\n  Hello  \n\n2\n00:00:03,000 --> 00:00:04,000\nWorld\n";
        let entries = parse_srt_content(content).unwrap();
        assert_eq!(entries[0].text, "Hello");
    }
    
    #[test]
    fn test_skip_invalid_timestamp() {
        let content = "1\n00:00:01,000 --> 00:00:02,000\nValid\n\n2\nbad time line\nBroken\n\n3\n00:00:03,000 --> 00:00:04,000\nAlso valid\n";
        let entries = parse_srt_content(content).unwrap();
        assert_eq!(entries.len(), 2, "Malformed entry should be skipped");
        assert_eq!(entries[0].text, "Valid");
        assert_eq!(entries[1].text, "Also valid");
    }
    
    #[test]
    fn test_write_falls_back_to_original_when_translation_empty() {
        let file = SubtitleFile {
            path: "test.srt".into(),
            entries: vec![
                SubtitleEntry {
                    index: 1,
                    start_time: Duration::from_secs(1),
                    end_time: Duration::from_secs(2),
                    text: "Original text".into(),
                    translated_text: Some(String::new()),
                },
                SubtitleEntry {
                    index: 2,
                    start_time: Duration::from_secs(3),
                    end_time: Duration::from_secs(4),
                    text: "Second line".into(),
                    translated_text: None,
                },
                SubtitleEntry {
                    index: 3,
                    start_time: Duration::from_secs(5),
                    end_time: Duration::from_secs(6),
                    text: "Third line".into(),
                    translated_text: Some("ÃœÃ§Ã¼ncÃ¼ satÄ±r".into()),
                },
            ],
            source_language: Language::English,
            target_language: Language::Turkish,
        };
        
        let out = std::env::temp_dir().join("ats_test_fallback.srt");
        write_srt_file(&file, &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        let _ = std::fs::remove_file(&out);
        
        assert!(content.contains("Original text"), "Empty translation should fall back to original");
        assert!(content.contains("Second line"), "Missing translation should fall back to original");
        assert!(content.contains("ÃœÃ§Ã¼ncÃ¼ satÄ±r"), "Valid translation should be used");
    }

    #[test]
    fn write_preserves_timestamps_and_count() {
        // Task 08 requirements 6/7: attaching translations must never touch
        // timestamps or the entry count — parse → write → parse keeps every
        // start/end time identical.
        let content =
            "1\n00:00:01,000 --> 00:00:02,500\nHello\n\n2\n00:00:03,250 --> 00:00:04,750\nWorld\n";
        let parsed = parse_srt_content(content).unwrap();
        assert_eq!(parsed.len(), 2);

        let file = SubtitleFile {
            path: "roundtrip.srt".into(),
            entries: parsed
                .into_iter()
                .map(|mut e| {
                    e.translated_text = Some(format!("T{}", e.index));
                    e
                })
                .collect(),
            source_language: Language::English,
            target_language: Language::Turkish,
        };

        let out = std::env::temp_dir().join(format!("ats_ts_roundtrip_{}.srt", std::process::id()));
        write_srt_file(&file, &out).unwrap();
        let written = parse_srt_content(&std::fs::read_to_string(&out).unwrap()).unwrap();
        let _ = std::fs::remove_file(&out);

        assert_eq!(
            written.len(),
            file.entries.len(),
            "subtitle count must not change"
        );
        for (before, after) in file.entries.iter().zip(written.iter()) {
            assert_eq!(before.start_time, after.start_time, "start time changed");
            assert_eq!(before.end_time, after.end_time, "end time changed");
        }
        assert_eq!(written[0].text, "T1", "translation must be written");
        assert_eq!(written[1].text, "T2", "translation must be written");
    }
}
