# Lokum-F'den LokumAI'a Geçiş ve Entegrasyon Rehberi

Bu belge, "Lokum-F" (Raskolnikov Personası / Fuar Sürümü) projesinde geliştirilen tüm kurumsal "Enterprise" özelliklerin, ana proje olan **LokumAI**'a nasıl taşınacağını ve entegre edileceğini adım adım açıklamaktadır. 

Lokum-F üzerinde yapılan optimizasyonlar, arayüz modernizasyonları ve Apple Silicon (MLX) uyumlu motor geliştirmeleri sayesinde uygulama kusursuz bir seviyeye ulaşmıştır. Bu özelliklerin ana projeye aktarılması, LokumAI'ı sıradan bir sohbet botundan profesyonel bir yapay zeka stüdyosuna dönüştürecektir.

---

## 🚀 Ana Projeye (LokumAI) Aktarılacak Temel Yenilikler

Lokum-F'de başardığımız ve doğrudan ana projeye taşınması gereken 5 devrimsel özellik şunlardır:

### 1. PyQt6 Native Modernizasyonu & Arayüz Mimarisi
- **Yapılan Değişiklik:** Eski, hantal ve Mac'te sorun çıkaran `PyQt5` altyapısı tamamen `PyQt6`'ya geçirildi.
- **Entegrasyon Adımı:** LokumAI projesinin `requirements.txt` dosyasındaki `PyQt5` silinip `PyQt6` ve `PyQt6-WebEngine` eklenecek. Lokum-F'nin güncel `main.py` dosyasındaki tüm `Qt.WindowType.*`, `Qt.AlignmentFlag.*` gibi PyQt6 Enum yapılı kodlar doğrudan ana projeye kopyalanacak.

### 2. Worker Thread Optimizasyonu (Sıfır Donma)
- **Yapılan Değişiklik:** RAG aramaları (FAISS) ve ağır proje dosyası okumaları UI thread'inden alınıp `AIWorker` (Arka plan işçisi) içine taşındı.
- **Entegrasyon Adımı:** LokumAI'da "Gönder" butonuna basıldığında arayüzün donmaması için Lokum-F'deki güncel `AIWorker` sınıfı ve `soru_sor` fonksiyonunun hafifletilmiş hali birebir kullanılacak.

### 3. Dinamik "Kapsül" Input Bar & "🔊 Dinle" Özelliği
- **Yapılan Değişiklik:** LM Studio standartlarında; içi boşken Mikrofon (🔴), yazarken Gönder (↑), yapay zeka düşünürken Durdur (■) şeklini alan dinamik bir buton tasarlandı. Ayrıca her mesajın altına "🔊 Dinle" butonu eklenerek JS köprüsüyle (`speak://`) arka plandaki STT/TTS motoruna bağlandı.
- **Entegrasyon Adımı:** Lokum-F'nin `main.py` dosyasındaki `DynamicActionBtn` CSS stilleri ve JS köprüsü (`updateChat` vs.) LokumAI'ın arayüzüne entegre edilecek.

### 4. Gelişmiş RAG ve "Persona" Selector Sistemi
- **Yapılan Değişiklik:** `mpnet-base-v2` kullanılarak RAG motoru akıllandırıldı. Ayrıca Dev Mod içerisine eklenen "System Prompt (Persona) Selector" ile anında Raskolnikov, Yazılımcı vb. kişilikler arasında geçiş yapabilme imkanı sunuldu.
- **Entegrasyon Adımı:** `core/rag_engine.py` ve `.lokumf/personas/` klasör mantığı LokumAI'a taşınacak.

### 5. Otomatik Validasyon & Manuel Fuse (Fine-Tune Motoru)
- **Yapılan Değişiklik:** Fine-tune sonrasında eğitim logları okunarak başarı skoru hesaplanıyor. %80 altı puan alan "kalitesiz" adaptörler otomatik siliniyor, üstü alanlar için ise "⚡️ Fuse & Save" butonu çıkıyor.
- **Entegrasyon Adımı:** Lokum-F'nin `core/finetune_engine.py` dosyası ve UI'daki Training sekmesi kodları ana projeye entegre edilecek.

---

## 🛠 Geçiş (Migration) İçin Yol Haritası

1. **İsim Değişikliği:** Projedeki her yerdeki `Lokum-F` ibareleri, değişkenleri ve klasör isimleri `LokumAI` olarak güncellenecek. (Örn: `.lokumf` klasörü `.lokumai` olacak).
2. **Core Motorların Taşınması:** Lokum-F'nin `core/` klasörü (`finetune_engine.py`, `rag_engine.py` vb.) LokumAI'a doğrudan kopyalanacak.
3. **Veritabanı ve Dataset:** Hazırladığımız 14MB'lık devasa `Raskolnikov_HQ_Dataset`, ana projenin veri setleri klasörüne eklenecek.
4. **Bağımlılıkların (Requirements) Güncellenmesi:** `mlx_whisper`, `PyQt6` vb. modern kütüphaneler ana projenin kurulum listesine eklenecek.

Bu adımlar tamamlandığında LokumAI, fuar standartlarındaki en modern, en stabil ve en akıllı versiyonuna ulaşmış olacaktır.
