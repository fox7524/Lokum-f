# Lokum-F: Gelecek Geliştirmeler (Next Steps)

## 1. RAG & Context Optimizasyonu (Worker Thread)
**Durum:** Beklemede (Yarın Yapılacak)
**Açıklama:** Arayüzde "Gönder" butonuna basıldığında tetiklenen `soru_sor` fonksiyonu içerisindeki ağır işlemler (dosya okuma ve FAISS vektör taraması), UI thread'inden çıkarılıp `AIWorker` (arka plan işçisi) içerisine taşınacak.
**Beklenen Çıktı:** Arayüzdeki (UI) donma/kilitlenme sorunları tamamen çözülecek ve uygulamanın "snappy" (akıcı) hissiyatı fuar standartlarına ulaşacak.

## 2. PyQt5'ten PyQt6'ya (veya PySide6) Geçiş
**Durum:** Beklemede (Yarın Yapılacak)
**Açıklama:** Uygulamanın grafik arayüz altyapısı modern PyQt6 mimarisine geçirilecek.
**Beklenen Çıktı:**
- Apple Silicon (M çipler) için tam native destek ve daha düşük CPU/RAM tüketimi.
- High-DPI (Retina) ekranlarda daha keskin ve kusursuz görüntü.
- Modern endüstri standartlarına tam uyum (Breaking changes olan Enum'lar, `exec()` metodları ve WebEngine modülleri güncellenecek).
