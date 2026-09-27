# Notlar

## Dağıtım öncesi yapılacaklar

- [ ] **Lisans denetimi**: Herkese açık dağıtıma çıkarken kullanılan cargo
  paketlerinin lisanslarını denetle (`cargo-deny` veya `cargo-license`).
  Çoğu crate MIT/Apache-2.0; copyleft (LGPL/GPL) çıkan olursa atıf/bağlantı
  şartı gerekebilir. Kullanıcının karar verdiği tarih: dağıtım zamanı.
- [ ] **VC++ Runtime**: Exe `vcruntime140.dll` bağlıyor; temiz sistemlerde
  VC++ Redistributable gerekebilir. Gerekirse `.cargo/config.toml`'a
  `-C target-feature=+crt-static` eklenerek statik bağlanabilir.
