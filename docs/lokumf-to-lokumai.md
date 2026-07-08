# Lokum-F'den LokumAI'a Geçiş ve Entegrasyon Rehberi (Ultimate Prompt)

Bu belge, "Lokum-F" (Raskolnikov Personası / Fuar Sürümü) projesinde geliştirilen tüm kurumsal "Enterprise" özelliklerin, ana proje olan **LokumAI**'a nasıl taşınacağını tek bir devasa prompt olarak özetlemektedir. Bu belgeyi kopyalayıp yeni sohbette yapay zekaya verdiğinizde her şeyi tek seferde anlayıp uygulayacaktır.

---

## 🎯 THE ULTIMATE MIGRATION PROMPT (Bunu Kopyalayın)

**Bağlam (Context):**
Şu an "Lokum-F" adlı fuar/demo projemizde geliştirdiğimiz devasa özellikleri, ana projemiz olan "LokumAI" repo'suna entegre ediyoruz. Aşağıdaki 7 ana görevi sırasıyla ve eksiksiz olarak yerine getirmeni istiyorum. Projeyi bir "Vibecoded" (amatörce yazılmış) prototipten, "Enterprise/LM Studio" kalitesine taşıyacağız.

**Görev 1: Bağımlılıklar ve Mimari (PyQt6 Geçişi)**
- Uygulamanın grafik arayüz altyapısını hantal PyQt5'ten, Apple Silicon (M çipler) ile tam uyumlu **PyQt6**'ya geçir. 
- `requirements.txt` dosyasındaki PyQt5'i sil, yerine `PyQt6` ve `PyQt6-WebEngine` ekle.
- `main.py` içerisindeki eski PyQt5 enum yapılarını (`Qt.Dialog` -> `Qt.WindowType.Dialog`, `Qt.AlignCenter` -> `Qt.AlignmentFlag.AlignCenter` vb.) PyQt6'ya uygun şekilde güncelle. WebEngine view'deki `verticalScrollBar` gibi geçersiz metod çağrılarını temizle.

**Görev 2: UI Thread Optimizasyonu (Sıfır Donma)**
- Kullanıcı "Gönder" butonuna bastığında arayüzün (UI) donmasını tamamen engellemeliyiz.
- RAG aramalarını (FAISS vektör taraması) ve proje dosyası okumalarını UI thread'inden çıkarıp `AIWorker` (QThread tabanlı arka plan işçisi) içerisine taşı. Butona basıldığı an animasyonlar takılmadan akmalı.

**Görev 3: Premium CSS ve Arayüz (UI/UX) Refactoring**
- Uygulamanın o ucuz "vibecoded" hissini tamamen sil. Aşağıdaki "Apple / Vercel" tasarım prensiplerini CSS (QSS ve HTML) kodlarına entegre et:
  - InputBarFrame (giriş alanı) için tam hap formu (`border-radius: 24px`).
  - Dinamik Kapsül Buton (Sağdaki buton): İçi boşken Mikrofon (🎤), yazarken Gönder (⬆), AI düşünürken Durdur (■) şeklinde dinamik değişsin ve tam kavisli (`border-radius: 18px`, `padding-bottom: 2px`, tam merkezli) olsun. Hoparlör butonunu input'un yanından tamamen kaldır.
  - Chat Baloncukları (Bubbles): Sağ alt köşesi sivri eski tasarımı iptal et. `border-radius: 20px`, `box-shadow` ve hafif gölgeli, pürüzsüz (`-webkit-font-smoothing: antialiased`) typography kullan.
  - AI Rol Etiketi: Düz yazı yerine Lokum vurgu renginde bir "Robot/AI" SVG ikonu maskesi kullan.
  - Kod Blokları (Pre/Code): `border-radius: 12px` ve şık bir `box-shadow` ile GitHub/Vercel stili karanlık kod blokları yap.
  - "🔊 Dinle" Butonu: Mesajların altında her zaman görünmesin. Sadece mesaja fareyle gelindiğinde (hover) `opacity` ve `visibility` animasyonuyla şık bir şekilde ortaya çıksın (`speak://` JS köprüsünü kullan).
  - QScrollBar'ları 6px genişliğinde, transparan ve modern macOS stiline getir.
  - Tüm panellere (Sidebar, Header) derinlik katan hafif `box-shadow`'lar ekle.

**Görev 4: Hata ve Placeholder Temizliği**
- Kodun içinde unutulmuş sahte/geçici (placeholder) tüm yapıları temizle.
- Örneğin; "The user is..." ile başlayan bir promptta hardcoded "Fibonacci" yanıtı dönen sahte fonksiyonları sil.
- Arayüzdeki "Dummy buttons to satisfy existing references" yazılı arka planda duran sahte butonları kaldır.
- `finetune_engine.py` içindeki `except Exception: pass` gibi hataları sessizce yutan blokları düzelt ve log'a yazdıracak hale getir.

**Görev 5: Fine-Tune Skoru (Regex Düzeltmesi)**
- Apple MLX'in eğitim sonrasında bastığı log formatı değiştiği için sistem hep "%87" sahte skorunu veriyordu. 
- `main.py` içerisindeki Regex okuyucusunu `loss_matches = re.findall(r"Val(?:id)?\s*loss[:\s]*([\d\.]+)", log_text, re.IGNORECASE)` şeklinde güncelle. Son `[-1]` değeri alıp %80 altındaysa silecek, üstündeyse "⚡️ Fuse & Save" butonunu çıkaracak mantığı kur.

**Görev 6: Veritabanı ve Modeller**
- Raskolnikov karakteri için `mlx-community/whisper-large-v3-turbo` modelinin STT (Speech-to-Text) için `language="tr"` parametresiyle kullanılmasını sağla.
- Hazırladığımız 14.09 MB'lık yüksek kaliteli "Raskolnikov_HQ_Dataset" klasörünü LokumAI veri setlerine bağla.

**Görev 7: İsimlendirme**
- Tüm dosyalardaki ve arayüzdeki "Lokum-F" isimlerini "LokumAI" olarak güncelle (`.lokumf` klasörü -> `.lokumai` olacak).

Lütfen bu 7 adımı sırasıyla, eksiksiz ve tek bir hata bile bırakmadan "LokumAI" projesine entegre et. Hazır olduğunda başlayalım!
---
