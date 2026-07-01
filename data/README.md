## data klasörü (açık lisans/PD)

Bu klasör, **fine-tune + RAG** denemeleri için **telif riski düşük** (CC0 / Public Domain / açık lisans) metinleri içerir.

### Yapı
- `raw/`: Kaynaktan indirilen ham dosyalar (txt/html/pdf/zip vb.)
- `txt/`: Eğitim/RAG için normalize edilmiş düz metinler
- `sources.json`: Her dosyanın kaynağı + lisans + indirme tarihi + hash bilgisi

### Notlar (önemli)
- “Public Domain/CC0” içerikler bile **site kullanım şartları** açısından kısıt barındırabilir. Bu yüzden her öğe için `sources.json` içine hem **lisans** hem **kaynak linki** hem de varsa **lisans metni linkini** yazıyorum.
- Internet Archive/Open Library gibi yerlerde karışık lisans/erişim modeli olduğu için sadece **açıkça PD/CC** olduğu belirtilen öğeleri seçiyorum.

