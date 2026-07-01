## İndirme özeti

Bu klasöre **166 doküman** indirildi:

- **Project Gutenberg (Public Domain)**: `pg-*` (8 adet)
- **Wikibooks (CC BY-SA 4.0)**: `wb-*` (44 adet)  
  (TR: `tr.wikibooks.org` alt sayfaları + EN: `en.wikibooks.org` alt sayfaları)
- **Internet Archive (licenseurl publicdomain filtreli)**: `ia-*` (58 adet)  
  (Not: Erişim kısıtlı / “private” dosyaları otomatik eledim.)
- **Rust Edition Guide (MIT OR Apache-2.0)**: `rust-edition-guide-*` (56 adet markdown sayfası)

### Boyut
- `data/raw/` toplam: ~5.42 MB
- `data/txt/` toplam: ~5.38 MB
  (güncel: raw ~8.43 MB, txt ~8.40 MB)

### Lisans / yasal notlar
- `sources.json` her dosya için: kaynak URL + lisans + dosya hash’leri içerir.
- Gutenberg metinleri genelde Public Domain olsa da, Gutenberg’in kendi **license/terms** sayfasını referans aldım.
- Wikibooks içerikleri **CC BY-SA 4.0**: atıf ve share-alike şartları var. Fine-tune dağıtımı yapacaksan buna göre “attribution”/metaveri akışını planlamak gerekebilir.
- Internet Archive tarafında sadece metadata’da **licenseurl içinde “publicdomain”** geçen ve **indirilebilir (private olmayan)** text dosyaları aldım.

### Bir sonraki mantıklı adım
Eğer bunu gerçekten **pair-programmer** hedefiyle fine-tune edeceksen:
- edebiyat metinleri (Gutenberg) “yazınsal üslup” aşılar,
- Wikibooks ise daha “öğretici/teknik” tona yakındır.

İstersen bir sonraki adımda, sadece teknik olanları ayrı bir subset’e ayıracak şekilde `data/txt/` içinde bir `train/` ve `rag/` ayrımı da yapabilirim.
