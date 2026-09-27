//! UI notification sounds.

/// Plays the system "Information" chime once (non-blocking). No-op on
/// non-Windows platforms.
pub fn play_completion() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Diagnostics::Debug::MessageBeep;
        use windows_sys::Win32::UI::WindowsAndMessaging::MB_ICONASTERISK;
        MessageBeep(MB_ICONASTERISK);
    }
}
