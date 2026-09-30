//! UI notification sounds (cross-platform).

use std::io::Cursor;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::OnceLock;

use rodio::DeviceSinkBuilder;

/// Completion chime embedded into the binary (`assets/notification.wav`).
static CHIME: &[u8] = include_bytes!("../assets/notification.wav");

/// Sender side of the background audio thread, created on first use.
static AUDIO_THREAD: OnceLock<Sender<()>> = OnceLock::new();

/// Plays the completion chime once (non-blocking).
///
/// The audio output device is opened once, on a dedicated thread. When no
/// output device is available the call is silently ignored.
pub fn play_completion() {
    let tx = AUDIO_THREAD.get_or_init(spawn_audio_thread);
    let _ = tx.send(());
}

fn spawn_audio_thread() -> Sender<()> {
    let (tx, rx) = mpsc::channel();
    let _ = std::thread::Builder::new()
        .name("notification-sound".to_owned())
        .spawn(move || run_audio(rx));
    tx
}

fn run_audio(rx: Receiver<()>) {
    let Ok(sink) = DeviceSinkBuilder::open_default_sink() else {
        return; // no usable output device: stay silent
    };
    while rx.recv().is_ok() {
        // Coalesce requests that piled up while the previous chime played.
        while rx.try_recv().is_ok() {}
        let Ok(player) = rodio::play(sink.mixer(), Cursor::new(CHIME)) else {
            continue;
        };
        player.sleep_until_end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::Source;

    #[test]
    fn embedded_chime_is_valid_audio() {
        let decoder = rodio::Decoder::new(Cursor::new(CHIME)).expect("chime must decode as WAV");
        assert_eq!(decoder.channels().get(), 1);
        assert_eq!(decoder.sample_rate().get(), 44100);

        let mut peak = 0.0f32;
        let mut samples = 0usize;
        for sample in decoder {
            peak = peak.max(sample.abs());
            samples += 1;
        }

        assert!(peak > 0.1, "chime must not be silent (peak {peak})");
        let seconds = samples as f32 / 44_100.0;
        assert!(
            (0.75..=0.85).contains(&seconds),
            "unexpected chime length {seconds}s"
        );
    }
}
