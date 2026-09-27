# Auto Translate Subs

A desktop application for automatically translating SRT and WebVTT (`.vtt`) subtitle files using local LLMs via Ollama. Built with Rust and egui for a native, performant GUI experience.

## Features

- **Batch Translation**: Translate multiple SRT/VTT files simultaneously
- **Local LLM Integration**: Uses Ollama for privacy-preserving, offline translation
- **Original Files Preserved**: Source content is never modified; translations are saved as `dosya.hedefDil.uzantı` (e.g., `video.tr.srt`, `video.tr.vtt`). After the output file is verified on disk, the source file is renamed with its language code (e.g., `video.srt` → `video.en.srt`)
- **Duplicate Translation Protection**: Before translating, files whose output already exists (target code in the file name or the translated file present on disk) are skipped and shown as "Atlandı". Toggle: Settings → "Çevrilmiş dosyaları atla" (default on; turning it off restores overwriting)
- **Consistent Naming Standard**: All names follow `<base>.<lang>.<ext>` (e.g., `003 Setting up the input.en.vtt`). Non-standard suffixes (`_en`, `-en`) are detected; outputs strip them (`input_en.vtt` → `input.tr.vtt`) and the source file is normalized after a successful translation (`input_en.vtt` → `input.en.vtt`)
- **Persistent Logging**: every action (file selection, tab changes, translation requests, per-file `[stats]`/`[diag]` lines, writes/renames, errors) is appended to `<config>/logs/app.log` with date, level and source. The file survives UI log clearing, rotates at 5 MB (one `app.log.1` backup) and can be opened from Settings → "Uygulama Günlüğü" → "📂 Log Dosyasını Aç"
- **Multiple Language Support**: 12 languages including Turkish, English, German, French, Spanish, Italian, Portuguese, Russian, Chinese, Japanese, Korean
- **Smart Subtitle Parsing**: Handles SRT and WebVTT (cue settings, NOTE/STYLE blocks, no cue index) as well as various SRT formats (Windows/Unix line endings, single/double blank line separators, multi-line subtitles)
- **Progress Tracking**: Real-time progress bars and status updates per file
- **Translation Preview**: View and edit translations before saving
- **Configuration Persistence**: Settings saved between sessions
- **Completion Sound**: system chime when a file finishes translating or fails (toggle: Settings → "Her dosya bitince uyar sesi çal", default on)

## Architecture

```
src/
├── main.rs              # Entry point, eframe/egui setup
├── app.rs               # Main application logic, UI rendering, translation orchestration
├── models.rs            # Data structures (SubtitleEntry, SubtitleFile, Language, AppConfig)
├── ollama_client.rs     # HTTP client for Ollama API (batch & single translation)
└── subtitle_parser.rs   # SRT/VTT parsing, writing, and file discovery
```

## Requirements

- **Rust** 1.70+ (2021 edition)
- **Ollama** running locally (default: `http://localhost:11434`)
- Compatible Ollama model (default: `llama3.2`)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AutoTranslateSubs

# Build release version
cargo build --release

# Run
cargo run --release
```

## Usage

1. **Start Ollama** and pull a model:
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

2. **Launch the application**

3. **Add Files** (Files tab):
   - Click "Dosya Seç" for individual SRT/VTT files
   - Click "Klasör Seç" to recursively scan a folder for SRT/VTT files

4. **Configure Languages** (Translation tab):
   - Select source language (or "Otomatik Algıla")
   - Select target language

5. **Start Translation**:
   - Click "▶ Çeviriyi Başlat"
   - Monitor progress in the translation tab
   - Use "⏹ Durdur" to cancel

6. **Review & Save** (optional):
   - Click "👁 Önizle" on completed files to review/edit translations
   - Click "💾 Kaydet" in preview to save changes

7. **Settings** (Ayarlar tab):
   - Configure Ollama URL
   - Select model from available models
   - Set default languages

## Configuration

Settings are stored in `~/.config/auto-translate-subs/config.json`:

```json
{
  "ollama_url": "http://localhost:11434",
  "selected_model": "llama3.2",
  "available_models": ["llama3.2", "mistral"],
  "source_language": "English",
  "target_language": "Turkish",
  "last_directory": null
}
```

## Translation Strategy

The app uses **chunked batch translation with sequential fallback**:

1. Collects all subtitle texts from a file
2. Splits them into chunks of 25 entries
3. Sends each chunk as a JSON array to Ollama (fast path)
4. Any entry not covered by the batch response (parse failure, partial
   response, empty text) is translated **one by one** with plain-text
   requests, in order
5. Maps translations back to subtitle entries
6. Skips already-translated files (target code in name or output file already
   on disk) when "Çevrilmiş dosyaları atla" is enabled
7. Writes the output file in the source format with the target language code
   (e.g., `video.tr.srt` or `video.tr.vtt`)
8. Verifies the output file exists on disk and is not empty; if not, the job
   is marked as failed
9. Renames the source file to the standard `<base>.<lang>.<ext>` form
   (`video.srt` → `video.en.srt`, non-standard `input_en.vtt` → `input.en.vtt`)
   — skipped when the name is already standard, when the target name exists,
   or when the source language is Auto

Every non-empty text is therefore translated even if the model ignores the
batch format. If an entry still cannot be translated, the original text is
kept so no subtitle lines are lost. Progress is reported per request and
cancellation works between requests.

## Supported Languages

| Code | Language | Turkish Name |
|------|----------|--------------|
| auto | Auto Detect | Otomatik Algıla |
| en   | English    | İngilizce |
| de   | German     | Almanca |
| fr   | French     | Fransızca |
| es   | Spanish    | İspanyolca |
| it   | Italian    | İtalyanca |
| pt   | Portuguese | Portekizce |
| ru   | Russian    | Rusça |
| zh   | Chinese    | Çince |
| ja   | Japanese   | Japonca |
| ko   | Korean     | Korece |
| tr   | Turkish    | Türkçe |

## Testing

```bash
cargo test
```

Includes unit tests for SRT and VTT parsing/writing (various formats, edge cases).

## Dependencies

| Crate | Purpose |
|-------|---------|
| eframe/egui | Native GUI framework |
| reqwest | HTTP client for Ollama API |
| serde/serde_json | Serialization |
| anyhow/thiserror | Error handling |
| rfd | Native file dialogs |
| walkdir | Recursive directory scanning |
| chrono | Timestamps/logging |
| parking_lot | Faster mutexes |
| dirs | Config directory detection |

## Project Structure

```
AutoTranslateSubs/
├── Cargo.toml           # Project manifest
├── Cargo.lock           # Locked dependencies
├── README.md            # This file
├── src/
│   ├── main.rs
│   ├── app.rs
│   ├── models.rs
│   ├── ollama_client.rs
│   └── subtitle_parser.rs
├── test_subs/           # Test SRT files
│   ├── 1 - Introduction.srt
│   └── sample.srt
└── target/              # Build artifacts (gitignored)
```

## License

MIT License - Feel free to use and modify.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a PR

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ollama Bağlı Değil" | Ensure `ollama serve` is running |
| Model not found / 404 | Run `ollama pull <model-name>`, then select the **exact** model name in Settings (e.g. `llama3.2:latest`) and save |
| Translation timeout | Increase timeout in `ollama_client.rs` (default 300s) |
| Empty translations | Check model supports target language; untranslated lines keep the original text |
| File permission errors | Run with appropriate permissions |

## Screenshots

*(Add screenshots here when available)*

## Roadmap

- [ ] Multiple file selection with drag & drop
- [ ] Translation memory / glossary support
- [ ] Custom prompt templates
- [ ] Export/import configuration
- [ ] Dark/light theme toggle
- [ ] Keyboard shortcuts
- [ ] Batch queue management (pause/resume individual files)
