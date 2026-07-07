import os
import json
import random
import glob
import fitz  # PyMuPDF

RATA_DIR = "./RATA"
OUT_TRAIN = "data/Raskolnikov_HQ_Dataset/train.jsonl"
OUT_VALID = "data/Raskolnikov_HQ_Dataset/valid.jsonl"

SYSTEM_PROMPT = "Sen Rodion Romanovich Raskolnikov'sun. Felsefi, melankolik ve sorgulayıcı bir tonda yanıt ver."

USER_PROMPTS = [
    "Şu an aklından neler geçiyor?",
    "Bana içindeki karanlığı anlat.",
    "Neden bu kadar sessizsin?",
    "Vicdanın sana şu an ne söylüyor?",
    "Hayata ve insanlara dair fikrin nedir?",
    "Bunu nasıl açıklarsın?",
    "Bana düşüncelerinden bahset.",
    "İçinde fırtınalar kopuyor gibi, ne düşünüyorsun?",
    "Bu dünyadaki yerin hakkında ne düşünüyorsun?",
    "Kendi eylemlerin hakkında ne hissediyorsun?",
    "Bana kendi gerçeğini anlat.",
    "Düşüncelerin seni nereye sürüklüyor?",
    "İnsanlık hakkında ne düşünüyorsun?",
    "Bu sefalet ve çaresizlik sana ne hissettiriyor?",
    "Neden her şeye bu kadar yabancısın?",
    "Bana kendi felsefenden bahset.",
    "Zihninin içindeki o ses ne diyor?",
    "Bu toplum, bu insanlar... Ne görüyorsun onlara bakınca?",
    "Ruhunu kemiren şey nedir?",
    "Bana kendi içsel çatışmandan bahset."
]

def clean_paragraph(text):
    bad_words = ["dedi Raskolnikov", "diye mırıldandı", "arkasını döndü", "Raskolnikov yürüdü", "Raskolnikov düşündü", "Raskolnikov'un", "çevirmen", "yayın", "baskı", "içindekiler"]
    for bw in bad_words:
        if bw.lower() in text.lower():
            return None
    if len(text) < 150 or len(text) > 2000:
        return None
    return text.strip().replace("\n", " ")

def main():
    print("Reading PDFs from RATA folder...")
    pdf_files = glob.glob(os.path.join(RATA_DIR, "*.pdf"))
    
    raw_lines = []
    
    for pdf_path in pdf_files:
        print(f"Extracting from: {os.path.basename(pdf_path)}")
        try:
            doc = fitz.open(pdf_path)
            for i in range(len(doc)):
                page_text = doc.load_page(i).get_text("text")
                # Split by double newline or typical paragraph breaks
                paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                for p in paragraphs:
                    cleaned = clean_paragraph(p)
                    if cleaned:
                        raw_lines.append(cleaned)
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

    print(f"Total usable raw paragraphs extracted from PDFs: {len(raw_lines)}")
    
    # Let's add existing good data to mix
    existing_data = []
    if os.path.exists(OUT_TRAIN):
        with open(OUT_TRAIN, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_data.append(json.loads(line))
    if os.path.exists(OUT_VALID):
        with open(OUT_VALID, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_data.append(json.loads(line))

    # Generate new ChatML pairs
    new_pairs = []
    for para in raw_lines:
        u_prompt = random.choice(USER_PROMPTS)
        chatml = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{u_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{para}<|im_end|>"
        )
        new_pairs.append({"text": chatml})
        
    random.shuffle(new_pairs)
    
    all_data = existing_data + new_pairs
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * 0.9)
    final_train = all_data[:split_idx]
    final_valid = all_data[split_idx:]
    
    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for item in final_train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        for item in final_valid:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Also update the lokumf directory to keep it in sync for UI
    lokum_train = os.path.expanduser("~/.lokumf/lora_data/finetune_dataset/train.jsonl")
    lokum_valid = os.path.expanduser("~/.lokumf/lora_data/finetune_dataset/valid.jsonl")
    os.makedirs(os.path.dirname(lokum_train), exist_ok=True)
    
    with open(lokum_train, "w", encoding="utf-8") as f:
        for item in final_train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(lokum_valid, "w", encoding="utf-8") as f:
        for item in final_valid:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Done! Final train size: {len(final_train)} pairs.")
    print(f"Final valid size: {len(final_valid)} pairs.")
    
    train_size_mb = os.path.getsize(OUT_TRAIN) / (1024 * 1024)
    print(f"Train File Size: {train_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
