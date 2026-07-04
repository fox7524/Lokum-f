import os
import glob
import fitz
import json
import random

def extract_text_from_pdf(pdf_path):
    """
    Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def clean_text(text):
    """
    Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
    """
    # Basic cleaning
    text = " ".join(text.split())
    return text

def chunk_text(text, chunk_size=800, overlap=100):
    """
    Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
    """
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # keep overlap
            overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_words
            current_length = sum(len(w) + 1 for w in current_chunk)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def main():
    """
    Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
    """
    rata_dir = "RATA"
    output_dir = "RATA_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = glob.glob(os.path.join(rata_dir, "*.pdf"))
    all_chunks = []
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        text = extract_text_from_pdf(pdf_file)
        text = clean_text(text)
        if text:
            chunks = chunk_text(text, chunk_size=1000, overlap=20) # words
            all_chunks.extend(chunks)
            
    # Remove too short chunks
    all_chunks = [c for c in all_chunks if len(c) > 50]
    
    # Shuffle for randomness
    random.seed(42)
    random.shuffle(all_chunks)
    
    # Split 85% train, 15% valid
    split_idx = int(len(all_chunks) * 0.85)
    train_chunks = all_chunks[:split_idx]
    valid_chunks = all_chunks[split_idx:]
    
    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for chunk in train_chunks:
            f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
            
    with open(valid_path, "w", encoding="utf-8") as f:
        for chunk in valid_chunks:
            f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
            
    print(f"Done! Generated {len(train_chunks)} training examples and {len(valid_chunks)} validation examples.")
    print(f"Saved to {train_path} and {valid_path}")

if __name__ == "__main__":
    main()
