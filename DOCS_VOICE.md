# Lokum-F Ses Teknolojileri (STT & TTS) Dokümantasyonu

Bu belge, Lokum-F projesinde kullanılan sesli etkileşim teknolojilerini, modelleri ve performans optimizasyonlarını açıklar.

## 1. Speech-to-Text (STT) - Sesi Metne Çevirme
Lokum-F, Apple Silicon donanımı üzerinde en yüksek hızı ve doğruluğu elde etmek için **MLX Whisper** altyapısını kullanır.

### Kullanılan Model: `whisper-large-v3-turbo`
- **Neden Bu Model?**: Standart `large-v3` modelinin zekasına sahip olmasına rağmen, "turbo" mimarisi sayesinde Apple Silicon GPU'larında yaklaşık **8-10 kat daha hızlı** çalışır.
- **Dil Desteği**: Çok dilli (Multilingual) olsa da, Lokum-F'te Türkçe (`language="tr"`) olarak sabitlenmiştir. Bu, "Merhaba Raskolnikov" gibi kısa cümlelerin yanlışlıkla Kiril alfabesiyle (Kazakça/Tatarca) algılanmasını engeller.
- **Donanım Hızlandırması**: MLX kütüphanesi sayesinde işlem doğrudan MacBook GPU'su (M5 Pro) üzerinde gerçekleşir, CPU'yu yormaz.

### Alternatifler (Developer Mode'da Düşünülebilir):
1. **Whisper-Base (En Hızlı)**: Çok daha küçük bir modeldir (145MB). Işık hızında çalışır ancak Türkçe doğruluk payı düşüktür.
2. **Faster-Whisper (CTranslate2)**: M serisi işlemcilerde MLX kadar optimize değildir ancak geniş bir kullanım alanı vardır.
3. **Deepgram / Groq API (Bulut)**: Eğer yerel (local) hız yetmezse, bulut tabanlı API'ler 100-200ms gecikme ile (latency) sonuç verebilir.

---

## 2. Text-to-Speech (TTS) - Metni Seslendirme
Raskolnikov'un sesli yanıtları için hibrit bir yapı tercih edilmiştir.

### Birincil Motor: `edge-tts` (Microsoft Azure Neural)
- **Teknoloji**: Microsoft'un bulut tabanlı "Neural" seslerini ücretsiz bir şekilde yerel uygulamalara köprüleyen bir sistemdir.
- **Ses Seçenekleri (Developer Mode'dan değiştirilebilir)**:
    - **Ahmet (Erkek/Tok)**: Raskolnikov gibi ağırbaşlı karakterler için ideal.
    - **Emel (Kadın/Yumuşak)**: Daha nazik ve yardımcı bir persona için uygun.
    - **Guy/Aria/Sonia (İngilizce)**: Farklı aksanlar ve karakterler için eklenmiş global sesler.
- **Performans**: Metni milisaniyeler içinde MP3 olarak indirir ve anında oynatır.

### İkincil Motor (Fallback): macOS `say`
- **Teknoloji**: macOS'in yerleşik konuşma motoru.
- **Avantajı**: Tamamen çevrimdışı (offline) çalışır.
- **Karakter**: `Cem` sesi. Daha robotik olsa da internet kesildiğinde sistemin susmamasını sağlar.

---

## 3. Performans İpuçları
- **VRAM Yönetimi**: MLX modelleri belleği dinamik kullanır. Eğer sistem yavaşlarsa, Developer Mode üzerinden `Clear Cache Threshold` ayarını düşürerek belleği boşaltabilirsiniz.
- **Gecikme (Latency)**: İlk seslendirme, sunucuyla kurulan ilk bağlantı nedeniyle 1 saniye kadar sürebilir. Sonraki cümlelerde bu süre 200-300ms'ye düşer.

---
*Lokum-F: Raskolnikov Personası ve Modern STT/TTS Motoru*
