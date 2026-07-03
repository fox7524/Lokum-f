# Fine-Tuning (İnce Ayar) Rehberi: Train, Valid ve Fuse Mantığı

Bu belge, Lokum-F (ve genel MLX tabanlı LLM) projelerinde Fine-Tuning (İnce Ayar) sürecinin tam olarak nasıl işlediğini, arka planda nelerin yaşandığını ve terimlerin ne anlama geldiğini netleştirmek için hazırlanmıştır.

---

## 1. Fine-Tune (İnce Ayar) Nedir?

Mevcut, eğitilmiş devasa bir yapay zeka modelini (örneğin Mistral veya Qwen) alırız ve ona belirli bir formatta (JSONL) **"Özel Ders"** veririz. 
Örneğin; bir modele "Sen Raskolnikov'sun, sana şöyle sorulursa böyle cevap ver" şeklindeki binlerce diyalog örneğini gösteririz. Model, genel dünya bilgisini kaybetmeden senin istediğin spesifik görevi veya üslubu öğrenir.

MLX kütüphanesi bu işlemi **LoRA (Low-Rank Adaptation)** adı verilen bir yöntemle yapar. 
**LoRA'nın Mantığı:** Modelin milyarlarca parametresini (beyin hücrelerini) baştan yazmak yerine, orijinal beynin üzerine takılabilen küçük bir **"Yama" (Adapter)** oluşturur. Bu sayede işlem çok hızlı biter ve RAM/GPU'yu zorlamaz.

---

## 2. Eğitim Verisi (Dataset) ve Ayrışma (Train vs. Valid)

Fine-Tune işlemine başlamadan önce modelin çalışacağı veriyi (JSONL) sisteme veririz. Sistem bu veriyi her zaman iki parçaya böler (Genellikle %90 Train, %10 Valid).

### A) Train (Eğitim) Verisi (`train.jsonl`)
*   **Nedir?** Modelin bizzat üzerinden ders çalıştığı, okuyup ezberlediği ve mantığını kavramaya çalıştığı ana veri setidir.
*   **Nasıl Çalışır?** Model bu veriyi alır, okur, kendi içinde kurallar üretir ve hata yapa yapa doğruyu öğrenir.

### B) Validation / Valid (Doğrulama) Verisi (`valid.jsonl`)
*   **Nedir?** Modelin **deneme sınavıdır.** Train verisinin içinden rastgele seçilip ayrılmış, modelin eğitim sırasında *asla görmediği* sorulardır.
*   **Neden Gereklidir?** Eğer model sadece "Train" verisiyle çalışıp hiç test edilmezse, soruları mantığıyla anlamak yerine sadece ezberleyebilir (Buna **Overfitting / Aşırı Ezberleme** denir). Validation aşaması, modele daha önce hiç görmediği verileri göstererek "Sen gerçekten bu işin mantığını (Raskolnikov gibi düşünmeyi) öğrendin mi, yoksa sadece ezberledin mi?" diye sorar.
*   **Skor (Loss):** Doğrulama sonucunda sana bir "Test Loss" (Kayıp) puanı verir. Bu puan ne kadar düşükse (0'a ne kadar yakınsa), model senin konunu o kadar iyi öğrenmiş demektir.

---

## 3. Arayüzdeki Checkbox'ların (Kutucukların) Anlamı

Lokum-F arayüzünde "Train" ve "Validation" adında iki kutucuk vardır. Bunların kombinasyonları şu anlama gelir:

*   **Sadece "Train" İşaretliyse:** Model sadece `train.jsonl` dosyasını okur. İterasyonları tamamlar ve sana bir yama (Adapter) dosyası verir. Sınav (Validation) yapmaz. Hızlıdır ama modelin ne kadar iyi öğrendiğini bilemezsin.
*   **Hem "Train" Hem "Validation" İşaretliyse (Önerilen):** Sistem önce eğitimi (Train) tamamlar, yamayı kaydeder ve **hemen ardından** otomatik olarak bu yamayı kullanarak deneme sınavını (Validation) başlatır. Sonuç olarak sana "Test Loss" değerini söyler.
*   **Sadece "Validation" İşaretliyse:** Önceden eğittiğin ama test etmeyi unuttuğun bir yamayı (Adapter) sonradan test etmek için kullanılır. Yeni bir öğrenme (eğitim) yapmaz, sadece var olanı sınava sokar.

---

## 4. Fuse (Birleştirme) Nedir?

Eğitim başarıyla bittiğinde (`run_...` klasörü oluştuğunda) elinde aslında **eksik bir parça** vardır. O klasörün içindeki `adapters.safetensors` dosyası, tek başına bir hiçtir. O sadece küçük bir "Yamadır".

LM Studio, Ollama veya Lokum-F sohbet arayüzü bu yamayı tek başına çalıştıramaz. Onların tam ve bütün bir modele ihtiyacı vardır.

İşte **Fuse (Birleştirme)** işlemi burada devreye girer:
1.  Orijinal "Ana Modeli" (Mistral Nemo) alır.
2.  Senin eğittiğin "Yamayı" (Adapter) alır.
3.  İkisini bir fırına koyup eritir (Fuse) ve bu yamayı orijinal modelin hücrelerine kalıcı olarak entegre eder.
4.  Ortaya **yepyeni, bağımsız ve çalışmaya hazır** bir model çıkar (Örn: Raskolnikov-12B). 

Lokum-F'teki "⚡️ Fuse & Save to LM Studio" butonu tam olarak bu işi saniyeler içinde arka planda yapar ve modeli senin LM Studio kütüphanene kaydeder.
