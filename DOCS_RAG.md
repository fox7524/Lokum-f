# RAG (Retrieval-Augmented Generation) Rehberi

Bu belge, Lokum-F (ve modern LLM'lerin) en büyük silahı olan RAG sisteminin nasıl çalıştığını anlatmaktadır.

---

## 1. RAG Nedir?

RAG, yapay zeka modellerinin "Açık Kitap Sınavına" girmesidir.

Normalde bir yapay zeka (LLM) modeli, sadece eğitildiği tarihe kadar olan dünya bilgisini hatırlar. Eğer ona çok özel bir PDF'ten, senin yazdığın bir makaleden veya gizli bir şirket belgesinden soru sorarsan, "Bilmiyorum" der ya da yalan uydurur (Halüsinasyon).

**RAG (Bilgi Geri Çağırma ile Zenginleştirilmiş Üretim)** ise modele şunu der:
*"Bana sadece ezberlediklerinle cevap verme. Ben sana koca bir klasör dolusu PDF, DOCX ve ZIM dosyası verdim. Önce benim sorumu al, o klasörün içinde Google gibi arama yap, bulduğun paragrafları oku, sonra o paragraflara dayanarak bana cevap ver."*

---

## 2. Lokum-F'te RAG Nasıl Çalışır?

Lokum-F'te RAG menüsü iki aşamadan oluşur: Veri Yükleme (Ingest) ve Arama/Sohbet.

### Aşama 1: Ingest (Veriyi Öğütme ve İndeksleme)
1.  Sen uygulamaya "RATA" klasörünü (İçinde PDF'ler olan klasör) gösterirsin.
2.  Sistem bu PDF'leri okur.
3.  Ancak koca kitabı modele tek seferde veremezsin (hafızası yetmez). Bu yüzden sistem kitabı **"Chunk"** (Parça) adı verilen küçük paragraflara böler (Örn: 500 kelimelik parçalar).
4.  Bu parçaların her birini "Vektör" adı verilen matematiksel koordinatlara çevirir ve bir veri tabanına (Faiss/Chroma) kaydeder. Artık elinde aranabilir bir kütüphane vardır.

### Aşama 2: Geri Çağırma (Retrieval) ve Sohbet
1.  Sen mikrofondan veya klavyeden "Raskolnikov'un ahlaki teorisi nedir?" diye sorarsın.
2.  Sistem bu soruyu da matematiğe (vektöre) çevirir ve senin kütüphanende "Bu soruya matematiksel olarak en çok benzeyen (anlam olarak en yakın) 3-4 paragrafı bul" der.
3.  O 3-4 paragraf (Chunk) bulunur.
4.  Sistem, LLM'e (Yapay Zeka Modeline) arka planda şu gizli komutu gönderir:
    *"Kullanıcının sorusu: Raskolnikov'un teorisi nedir?*
    *Lütfen sadece şu metinlere bakarak cevap ver: [Sistemin bulduğu o 3-4 paragraf]"*
5.  Model, tam olarak senin PDF'indeki kelimeleri ve felsefeyi kullanarak mükemmel ve hatasız bir cevap üretir.

---

## 3. RAG vs. Fine-Tune (Hangisi Ne Zaman Kullanılır?)

Bu ikisi genellikle birbirine karıştırılır. Aslında amaçları tamamen farklıdır ve Lokum-F bu ikisini aynı anda kullanarak mükemmelliğe ulaşır.

| Özellik | RAG (Açık Kitap Sınavı) | Fine-Tune (Özel Ders / Beyin Yıkama) |
| :--- | :--- | :--- |
| **Amaç** | Modele "Bilgi" vermek. Spesifik belgelere bakarak doğru cevaplar üretmesini sağlamak. | Modele "Üslup ve Mantık" (Persona) vermek. Bir karakter gibi konuşmasını sağlamak. |
| **Hafıza Tipi** | Geçici hafıza. Soru soruldukça kitaba bakar, cevaplar, sonra unutur. | Kalıcı hafıza. Modelin nöronları (parametreleri) kalıcı olarak değişir. |
| **Hız** | Veriyi sisteme yüklemek (Ingest) çok hızlıdır (1-2 dakika). | Modeli eğitmek (Train) saatler sürebilir. |
| **Örnek Kullanım** | "Şu 1000 sayfalık PDF'te 3. bölümün özetini çıkar." | "Bana her zaman 19. yüzyıl depresif bir Rus genci gibi cevap ver." |

**Lokum-F'teki Muazzam Kombinasyon:**
Lokum-F'te biz **Fine-Tune** ile modelin beynini yıkayıp onu Raskolnikov yaptık (Kalıcı Persona). 
Sonra ona **RAG** ile Suç ve Ceza kitabını ve makaleleri verdik (Dış Hafıza).
Sonuç olarak: Model hem Raskolnikov gibi *konuşuyor*, hem de tam olarak Raskolnikov'un *bilmesi gereken* detayları kitaptan anında bulup sana söylüyor!
