use serde::{Deserialize, Serialize};
use std::path::PathBuf;
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
        }
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTranslationEntry {
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
}