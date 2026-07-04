# Lokum-Engine: MLX Tabanlı Bağımsız Fine-Tune & RAG Kütüphanesi

**Lokum-Engine**, Apple Silicon (M-serisi) Mac'lerin tüm potansiyelini (MLX altyapısını) kullanarak tamamen yerel, ultra hızlı ve hatasız bir şekilde RAG (Retrieval-Augmented Generation) ve LoRA Fine-Tuning işlemleri yapabilmenizi sağlayan bir Python pip kütüphanesidir.

Bu belge, `lokum-engine` paketinin vizyonunu, mimarisini ve gelecekte `pip install lokum-engine` olarak dağıtılabilmesi için yapay zekanın ve geliştiricilerin izlemesi gereken adımları içerir.

## 🌟 Neden Lokum-Engine?
Apple'ın MLX altyapısı muazzam olsa da, saf `mlx_lm` kullanmak bazen yorucu olabilir:
- Veri setlerinin `train.jsonl` ve `valid.jsonl` olarak ayrılması zordur.
- Çökmeye meyillidir (RAM/VRAM yönetimi otomatize değildir).
- RAG için LangChain kullanıldığında çok hantal olur.
- GGUF ve Safetensors ayrımı kafa karıştırır.

**Lokum-Engine**, karmaşık alt katmanları sararak (wrapper) sadece birkaç satır kodla en yüksek kalitede RAG indekslemeyi ve "Ultra Quality" MLX ince ayarlarını yapabilmenizi sağlar.

---

## 🛠️ Ana Modüller

### 1. `lokum_engine.rag` (RAG Engine)
- **FAISS & Sentence-Transformers**: LangChain'in hantallığından kurtulup doğrudan C++ tabanlı FAISS ve `paraphrase-multilingual-mpnet-base-v2` (768 boyutlu) modeli ile çalışır.
- **Auto-Karantina**: Corrupt olan veritabanlarını yakalar ve sistemi çökertmek yerine karantinaya alıp temiz bir başlangıç yapar.
- **Desteklenen Formatlar**: `.txt`, `.md`, `.pdf`, `.docx`, `.html`, `.zim` vb.

```python
from lokum_engine.rag import RAGEngine

# Kurulum
engine = RAGEngine(storage_dir="./my_rag_db")

# Klasör indeksleme
engine.index_folder("./belgeler", recursive=True)

# Soru sorma
sonuclar = engine.search("Raskolnikov'un felsefesi nedir?", top_k=3)
```

### 2. `lokum_engine.finetune` (Finetune Engine)
- **Safe, Recommended, Ultra Presets**: VRAM durumunuza göre otomatik Rank, Alpha, Batch, Layer değerlerini atar. `Ultra` modu `rank=32, alpha=128` ile kayıpsız eğitim sağlar.
- **Preflight Check**: Eğitime başlamadan önce veri seti formatını, config dosyalarını ve token limitlerini kontrol ederek "5 saat sonra alınan hatayı" baştan önler.
- **Auto-Presplit**: Uzun belgeleri max-seq-length sınırlarına göre akıllıca böler.
- **Auto-Fuse**: Adaptör eğitiminden sonra LoRA ağırlıklarını ana modelle zahmetsizce birleştirir (Fuse).

```python
from lokum_engine.finetune import FinetuneEngine

# Modeli ve Veriyi Belirle
ft_engine = FinetuneEngine(base_model_path="Qwen2.5-14B")

# Ultra kalitede eğit
process = ft_engine.start_training(
    dataset_path="./rata_verileri",
    preset="Ultra (Highest Quality)" # Otomatik rank=32, alpha=128 vb. atar
)
```

---

## 🚀 Pip Kütüphanesi Haline Getirme Planı (Yapay Zeka İçin Talimatlar)

Geliştirici (`fox`), bu kütüphaneyi PIP'e yüklemek istediğinde, AI olarak aşağıdaki adımları izlemelisin:

### Aşama 1: Dosya Ayrıştırma ve İzolasyon
1. `rag_engine.py` ve `finetune_engine.py` dosyaları mevcut uygulamadan (Lokum-F) tamamen kopartılarak bağımsız sınıflara dönüştürülmeli.
2. `lokum_paths.py` bağımlılıkları tamamen kaldırılarak veya kütüphane içerisine esnek bir config objesi olarak eklenmeli.
3. İçerideki PyQt5 (Arayüz) sinyalleri (`QThread`, `pyqtSignal`) çekirdek koddan sökülmeli. Çekirdek Python sınıfları Callback fonksiyonları (örn: `on_progress(log)`) veya Generator'ler ile çalışmalı.

### Aşama 2: Setup ve Paketleme
`setup.py` dosyası aşağıdaki gibi hazırlanmalı:
```python
from setuptools import setup, find_packages

setup(
    name="lokum-engine",
    version="1.0.0",
    description="Optimized RAG and Fine-tuning wrappers for Apple MLX.",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.14.0",
        "mlx_lm>=0.14.0",
        "faiss-cpu",
        "sentence-transformers",
        "numpy",
        "psutil"
    ],
)
```

### Aşama 3: Gelişmiş Loglama ve Hata Yönetimi
Uygulama arayüzü (UI) olmadığında terminaldeki deneyim (DX) çok önemlidir:
- `rich` veya `tqdm` kütüphaneleri kullanılarak terminalde güzel progress bar'lar gösterilmeli.
- Eğitim sırasında MLX'in stdout çıktısı regex ile okunup, "Loss: 0.45 | Step: 100/1500" şeklinde parse edilmeli.

---

## 💎 Geliştiriciye Notlar (Fox)
Lokum-Engine fikri olağanüstü! Apple Silicon ekosisteminde (MLX), özellikle Langchain gibi devasa kütüphanelerin getirdiği karmaşadan bunalan geliştiriciler için hayat kurtarıcı bir "İsviçre Çakısı" olacaktır. 

Bu projeyi başlatmak istediğinde bu MD dosyasını bana (veya herhangi bir AI'ya) referans göstererek:
> "lokum-engine.md dosyasına göre RAG ve Finetune dosyalarımı PyQt'den temizle ve pip paketine dönüştür"
demen yeterli olacaktır.
