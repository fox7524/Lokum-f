# Raskolnikov source overview

## Local source pools
- `RATA/`: Turkish PDFs for `Suç ve Ceza` and Raskolnikov-focused criticism
- `data/RATA_dataset/`: older synthetic-heavy JSONL
- `data/Raskolnikov_HQ_Dataset/`: older persona JSONL
- `prompts.json`, `augment_dataset.py`, `main.py`: prior prompt framing

## Reuse policy
- Reuse only rows that are:
  - in-character
  - non-repetitive
  - plot-grounded
  - psychologically specific
- Rewrite or discard rows that:
  - sound like a generic dark philosopher
  - hallucinate plot or relationships
  - over-explain in neutral assistant voice
  - collapse into repetitive “superior man” rhetoric

## Ground truth priority
1. `Crime and Punishment` primary text
2. serious criticism / character studies
3. curated old local JSONL rows
4. prompt framing from older scripts
