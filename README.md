# Lokum-F (Fuar Edition)

Lokum-F, orijinal LokumAI projesinin "Fuar ve Etkinlikler" (Kiosk Mode) için özel olarak modifiye edilmiş yepyeni ve bağımsız bir koludur. 

Lokum-F, sadece metin tabanlı bir asistan olmaktan çıkıp, etkinliklerde insanlarla doğrudan sesli ve görsel olarak etkileşime girebilen, önceden belirlenmiş bir karaktere (Persona) bürünebilen **interaktif bir yapay zeka avatar sistemine** dönüşmüştür.

## 🚀 Yeni Özellikler (Lokum-F Farkı)

*   **Sesli Girdi (Speech-to-Text):** Uygulama arayüzündeki mikrofon (🎤) butonu ve `mlx-whisper` entegrasyonu sayesinde, kullanıcıların sesli soruları milisaniyeler içinde metne dökülür.
*   **Animasyonlu Canlı Avatar (Yakında):** D-ID Streaming API entegrasyonu ile modelden gelen cevaplar anında sesli ve dudak senkronizasyonlu (lip-sync) bir 2D avatar videosuna dönüştürülür. Fuar katılımcıları yapay zeka ile yüz yüze konuşuyormuş hissi yaşar.
*   **Persona (Rol Yapma) Odaklı Sistem:** Model, basit bir asistan olmak yerine (Örn: Dostoyevski'nin Raskolnikov'u gibi) belirli bir kişiliğe bürünecek şekilde tasarlanmış ve `prompts.json` üzerinden bu role kilitlenmiştir.
*   **Tek Tıkla Model Birleştirme (Fuse):** Fine-Tune (İnce ayar) sekmesindeki yeni arayüz sayesinde, eğitim (training) bittiğinde terminale kod yazmaya gerek kalmadan tek tıkla model birleştirilir ve LM Studio klasörüne otomatik kaydedilir.
*   **Geliştirilmiş UX/DX:** Karmaşık eğitim parametreleri gizlenmiş, LM Studio modelleri şık ve okunabilir bir arayüzle listelenmiştir. Tamamen fuar operatörlerinin hızına uygun hale getirilmiştir.

## 🧠 Core Teknolojiler (LokumAI'den Miras)

Lokum-F, gücünü orijinal LokumAI mimarisinden alır:
*   **MLX (Apple Silicon):** Tüm eğitim ve RAG işlemleri Mac (M serisi) işlemcilerinde yerel ve ışık hızında çalışır.
*   **RAG (Retrieval-Augmented Generation):** PDF, DOCX ve ZIM dosyalarındaki devasa metinleri okuyup karakterin hafızasına ekler.
*   **Yerel LoRA Fine-Tuning:** SQLite veya JSONL dosyalarındaki verileri alıp, orijinal modele hiçbir zarar vermeden yepyeni yetenekler ve hafıza (Adapter) kazandırır.

## 🛠 Kurulum ve Kullanım

1. Python sanal ortamını oluşturun: `python3 -m venv .venv`
2. Aktif edin: `source .venv/bin/activate`
3. Gereksinimleri kurun: `pip install -r requirements.txt mlx-whisper sounddevice scipy numpy`
4. Uygulamayı başlatın: `python3 main.py`
