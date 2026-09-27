// Hide the console window in release builds; keep it in debug builds for logs/panics
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod logger;
mod models;
mod ollama_client;
mod sound;
mod subtitle_parser;

use app::AutoTranslateApp;
use eframe::egui;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Auto Translate Subs - Altyazı Çevirici"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Auto Translate Subs",
        options,
        Box::new(|cc| Ok(Box::new(AutoTranslateApp::new(cc)) as Box<dyn eframe::App>)),
    )
}