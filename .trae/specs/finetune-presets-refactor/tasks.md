# Tasks

- [x] Task 1: Mouse Wheel Etkileşimini Devre Dışı Bırakma
  - [x] SubTask 1.1: `main.py` içerisinde Fine-tune spinbox'ları (`lora_rank`, `lora_alpha`, `lora_iters`, `lora_batch`, `lora_layers`, `ft_max_seq`, `ft_steps_per_eval`, `ft_val_batches`, `ft_clear_cache_thr`) oluşturulurken, mouse tekerleği etkileşimini iptal eden yardımcı bir fonksiyon (örneğin `widget.wheelEvent = lambda e: e.ignore()`) yaz ve hepsine uygula.
- [x] Task 2: Preset Listesini Güncelleme
  - [x] SubTask 2.1: `main.py` içerisindeki `self.ft_preset.addItems([...])` listesini `["Ultra", "Good", "Mid", "Low", "Custom"]` olacak şekilde güncelle.
- [x] Task 3: Preset Değerlerini (Hardcoded) Yeniden Tanımlama
  - [x] SubTask 3.1: `_apply_ft_preset` metodunu güncelle. `Ultra`, `Good`, `Mid`, `Low` isimlerine göre yeni `setValue` atamalarını yap (Ultra için en yüksek kalite değerlerini kullan).
- [x] Task 4: QSettings ile Kayıt ve Geri Yükleme Mantığı
  - [x] SubTask 4.1: Uygulama başlatıldığında son seçilen preseti `QSettings`'ten okuyup `self.ft_preset`'e set etme mantığını ekle.
  - [x] SubTask 4.2: Eğer preset `Custom` ise, `lora_rank`, `lora_alpha` vb. tüm değerleri `QSettings` üzerinden (örneğin `ft_custom_rank` vb. anahtarlarla) okuyup arayüze set etme mantığını `_apply_ft_preset` içerisine (veya ayrı bir metoda) ekle.
- [x] Task 5: Manuel Değişimde Otomatik Custom'a Geçiş
  - [x] SubTask 5.1: `_is_updating_preset` bayrağı (flag) ekle. Programatik atamalarda tetiklenmeleri önlemek için.
  - [x] SubTask 5.2: Spinbox'ların `valueChanged` sinyallerini, preseti otomatik olarak `Custom`'a çeken ve güncel değerleri `QSettings`'e kaydeden bir metoda bağla.
