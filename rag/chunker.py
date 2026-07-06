from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .normalize import normalize_text


@dataclass(frozen=True)
class TextChunk:
    text: str
    signature: str


def chunk_signature(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 120) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    # Anlamsal (Semantic) Bölme: Paragraflara, satırlara ve cümlelere göre ayır.
    # 1. Paragraflara ayır (\n\n)
    paragraphs = re.split(r'(?<=\n\n)', normalized)
    
    units = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            units.append(p)
        else:
            # 2. Satırlara ayır (\n)
            lines = re.split(r'(?<=\n)', p)
            for line in lines:
                if len(line) <= chunk_size:
                    units.append(line)
                else:
                    # 3. Cümlelere ayır (. ! ?)
                    sentences = re.split(r'(?<=[.!?])\s+', line)
                    for idx, s in enumerate(sentences):
                        if idx < len(sentences) - 1:
                            s += " "
                        if len(s) <= chunk_size:
                            units.append(s)
                        else:
                            # 4. Kelimelere ayır (boşluk)
                            words = re.split(r'(?<=\s)', s)
                            for w in words:
                                if len(w) <= chunk_size:
                                    units.append(w)
                                else:
                                    # 5. Karakterlere ayır
                                    for i in range(0, len(w), chunk_size):
                                        units.append(w[i:i+chunk_size])

    out: list[TextChunk] = []
    current_chunk_units = []
    current_len = 0
    
    i = 0
    while i < len(units):
        unit = units[i]
        
        if current_len + len(unit) > chunk_size and current_chunk_units:
            # Mevcut bloğu kaydet
            chunk_str = "".join(current_chunk_units).strip()
            if chunk_str:
                out.append(TextChunk(text=chunk_str, signature=chunk_signature(chunk_str)))
            
            # Overlap (Örtüşme) için geriye dön
            overlap_len = 0
            overlap_units = []
            for u in reversed(current_chunk_units):
                if overlap_len + len(u) <= overlap:
                    overlap_units.insert(0, u)
                    overlap_len += len(u)
                else:
                    break
            
            # Eğer overlap için uygun parça bulunamadıysa ama overlap isteniyorsa
            # (Örn. çok uzun bir cümle), en azından boş kalmasın diye son üniteyi alabiliriz
            # ama boyut kontrolünü bozmamak için sadece tam uyanları alıyoruz.
            current_chunk_units = overlap_units
            current_len = overlap_len
            
        current_chunk_units.append(unit)
        current_len += len(unit)
        i += 1
        
    if current_chunk_units:
        chunk_str = "".join(current_chunk_units).strip()
        if chunk_str:
            out.append(TextChunk(text=chunk_str, signature=chunk_signature(chunk_str)))
            
    return out
