# 🍭 Lokum-F: Fuar ve Kiosk İçin İnteraktif Yapay Zeka Ekosistemi

Lokum-F, Apple Silicon (M1/M2/M3/M4/M5) mimarisi üzerinde çalışan, yerel (local) dil modellerini yüksek performanslı bir kullanıcı deneyimiyle birleştiren uçtan uca bir çözümdür. Orijinal LokumAI projesinin üzerine, fuar ortamlarında kullanılmak üzere tasarlanmış elit bir UX, sesli etkileşim ve persona yönetimi katmanı eklenerek oluşturulmuştur.

---

## 🏛 1. Mimari ve İzolasyon Felsefesi

### 1.1. `.lokumf` Klasörü (Persistence Layer)
Uygulamanın en temel kuralı, orijinal LokumAI projesini veya kullanıcının diğer verilerini bozmamaktır. Bu nedenle:
- **İzolasyon**: Tüm veritabanları (`app.db`), konfigürasyonlar (`config.json`), LoRA adaptörleri ve RAG veritabanları kullanıcının ana dizinindeki `.lokumf` klasöründe saklanır.
- **Dinamik Yol Yönetimi**: `lokum_paths.py` modülü sayesinde uygulama, çalıştırıldığı dizinden bağımsız olarak her zaman doğru persistans klasörüne erişir.

### 1.2. Multithreading (QThread) Yapısı
Arayüzün asla donmaması (snappy feeling) için uygulama ağır yükleri arka plan thread'lerine dağıtır:
- **AIWorker**: Metin üretimini (Inference) yönetir.
- **MicWorker**: Whisper tabanlı STT (Sesi metne çevirme) işlemini GPU üzerinde koşturur.
- **TTSWorker**: Seslendirme (Text-to-Speech) işlemini asenkron olarak yürütür.
- **RagIndexWorker**: Binlerce sayfalık dokümanı arayüzü kilitlemeden indeksler.

---

## 🎨 2. UX/UI Tasarımı: LM Studio Standartları

Lokum-F, bir "Python betiği" gibi görünmek yerine, profesyonel bir masaüstü uygulaması estetiğine sahiptir.

### 2.1. WebEngine (Chromium) Arayüzü
- **Hibrit Yapı**: PyQt5'in standart widget'ları yerine, sohbet alanı `QWebEngineView` (Chromium) kullanılarak HTML5/CSS3 ile çizilir.
- **LM Studio Stili**: Sohbet balonları, sağa dayalı kullanıcı mesajları ve sola dayalı, altında "Dinle" butonu olan asistan mesajları LM Studio estetiğine sadık kalınarak tasarlandı.
- **Snappy Response**: Mesajlar anında ekranda belirir, yapay zeka düşünürken (Thinking phase) üç noktalı animasyon (`dot pulse`) devreye girer.

### 2.2. Dinamik Giriş Barı (Input Capsule)
- **Bukalemun Buton**: Giriş çubuğunun sağındaki buton, durumuna göre ikon ve işlev değiştirir:
    - Boşken: **🎤 Mikrofon** (Bas-Konuş aktif).
    - Yazarken: **↑ Gönder** (Mesajı ilet).
    - İşlerken: **■ Durdur** (Üretimi kes).

---

## 🎙 3. Ses Teknolojileri (STT & TTS)

### 3.1. STT: MLX Whisper `large-v3-turbo`
- **Turbo Hız**: Apple Silicon GPU'larında optimize edilmiş model, sesi metne çevirirken milisaniyeler içinde sonuç verir.
- **Dil Zekası**: Türkçe dili (`language="tr"`) zorunlu kılınarak, kısa cümlelerdeki yanlış algılama (hallucination) sorunları ortadan kaldırılmıştır.
- **Seçenekler**: Developer Mode üzerinden "Base" (Işık hızı) veya "Small" (Hafif) modelleri de seçilebilir.

### 3.2. TTS: Edge-TTS & macOS Fallback
- **Ahmet & Emel**: Microsoft'un Neural ses motoru sayesinde, yapay zeka çok doğal bir Türkçe tonlamasıyla konuşur.
- **Offline Güvenlik**: İnternet bağlantısı kesilirse, uygulama susmaz; macOS'in yerel `say` (Cem) sesine otomatik geçer.
- **İsteğe Bağlı Seslendirme**: Her asistan cevabının altında bulunan "🔊 Dinle" butonu, kullanıcının sadece istediği yanıtları duymasını sağlar.

---

## 🧠 4. Zeka Katmanı: MLX, RAG ve LoRA

### 4.1. MLX Inference
- **Apple Native**: Apple Silicon GPU'sunun tüm gücünü kullanarak 7B, 14B ve hatta 32B modelleri verimli bir şekilde çalıştırır.
- **Quantization**: 4-bit ve 8-bit modellerle bellek kullanımını optimize eder.

### 4.2. Gelişmiş RAG (Retrieval-Augmented Generation)
- **Embedding**: 768 boyutlu `paraphrase-multilingual-mpnet-base-v2` modeli ile Türkçe dokümanlarda semantik arama kalitesi en üst düzeye çıkarıldı.
- **Dosya Desteği**: PDF, DOCX, TXT ve ZIM (Wiki) arşivlerini doğrudan analiz edebilir.
- **Chunking**: 500 karakterlik parçalar ve 150 karakterlik kesişimlerle (overlap) bağlam kaybı minimuma indirildi.

### 4.3. Ultra Fine-Tuning (LoRA)
- **Tek Tıkla Eğitim**: Developer Mode içindeki "Ultra" ayarı ile Rank 32 ve Alpha 128 değerlerinde en kaliteli adaptör eğitimini başlatır.
- **Otomatik Fuse**: Eğitim bittiğinde adaptörü ana modele birleştirip (Fuse) LM Studio'da kullanıma hazır hale getirme özelliği.

---

## 🎭 5. Persona Yönetimi: Raskolnikov ve Ötesi

- **Varsayılan Persona**: Rodion Romanovich Raskolnikov (Suç ve Ceza).
- **Karakter Derinliği**: Model, Dostoevski'nin 19. yüzyıl Rus edebiyatı tonunda, felsefi, analiz odaklı ve hafif melankolik bir dille cevap verir.
- **Esneklik**: `prompts.json` üzerinden saniyeler içinde yeni bir persona (örneğin bir fuar asistanı veya teknik uzman) tanımlanabilir.

---

## 🛠 6. Developer Mode ve Güvenlik

- **Dev Panel**: RAG veritabanını sıfırlama, model yükleme/boşaltma, donanım stres testleri ve benchmark analizleri için merkezi kontrol alanı.
- **Güvenli Şifreleme**: Dev Mode erişimi, `lokum_paths.py` tarafından yönetilen yerel şifreleme ile korunur.
- **Chromium Sandbox Fix**: macOS üzerindeki WebEngine çökmelerini önlemek için `--no-sandbox` ve GPU optimizasyon bayrakları sisteme başlangıçta enjekte edilir.

---
*Lokum-F: Apple Silicon Donanımının Sınırlarını Zorlayan, Fuar Standlarının Yeni Yıldızı.*
