import os
import json
import random

RAW_DATA_PATH = "data/RATA_dataset/train.jsonl"
OUT_TRAIN = "data/Raskolnikov_HQ_Dataset/train.jsonl"
OUT_VALID = "data/Raskolnikov_HQ_Dataset/valid.jsonl"

EXISTING_TRAIN = os.path.expanduser("~/.lokumf/lora_data/finetune_dataset/train.jsonl")
EXISTING_VALID = os.path.expanduser("~/.lokumf/lora_data/finetune_dataset/valid.jsonl")

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
    """
    Attempts to clean 3rd person narration slightly, 
    making it feel more like a first-person monologue or philosophical rant.
    """
    # We won't do advanced NLP here, just some basic heuristics to skip heavily narrated lines
    bad_words = ["dedi Raskolnikov", "diye mırıldandı", "arkasını döndü", "Raskolnikov yürüdü", "Raskolnikov düşündü", "Raskolnikov'un"]
    for bw in bad_words:
        if bw.lower() in text.lower():
            return None
            
    # If it's too short, ignore
    if len(text) < 150:
        return None
        
    return text.strip()

def main():
    print("Augmenting dataset to 3MB+...")
    
    # 1. Read existing good synthetic data
    existing_train = []
    existing_valid = []
    
    if os.path.exists(EXISTING_TRAIN):
        with open(EXISTING_TRAIN, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_train.append(json.loads(line))
                    
    if os.path.exists(EXISTING_VALID):
        with open(EXISTING_VALID, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_valid.append(json.loads(line))
                    
    print(f"Existing train size: {len(existing_train)}")
    print(f"Existing valid size: {len(existing_valid)}")
    
    # 2. Read raw book text
    raw_lines = []
    if os.path.exists(RAW_DATA_PATH):
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        cleaned = clean_paragraph(text)
                        if cleaned:
                            raw_lines.append(cleaned)
                    except:
                        pass
                        
    print(f"Usable raw paragraphs extracted: {len(raw_lines)}")
    
    # 3. Generate new ChatML pairs
    new_pairs = []
    for para in raw_lines:
        u_prompt = random.choice(USER_PROMPTS)
        chatml = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{u_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{para}<|im_end|>"
        )
        new_pairs.append({"text": chatml})
        
    # Let's shuffle and add to existing
    random.shuffle(new_pairs)
    
    # Calculate how many we need to add. 
    # Average pair is ~600 bytes. To get 3MB we need ~5000 pairs.
    # We will add everything we generated.
    
    split_idx = int(len(new_pairs) * 0.9)
    new_train = new_pairs[:split_idx]
    new_valid = new_pairs[split_idx:]
    
    final_train = existing_train + new_train
    final_valid = existing_valid + new_valid
    
    # Shuffle final
    random.shuffle(final_train)
    random.shuffle(final_valid)
    
    # Ensure dirs
    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for item in final_train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        for item in final_valid:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Done! Final train size: {len(final_train)} pairs.")
    print(f"Final valid size: {len(final_valid)} pairs.")
    
    train_size_mb = os.path.getsize(OUT_TRAIN) / (1024 * 1024)
    valid_size_mb = os.path.getsize(OUT_VALID) / (1024 * 1024)
    print(f"Train File Size: {train_size_mb:.2f} MB")
    print(f"Valid File Size: {valid_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
