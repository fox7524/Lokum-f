# Victor Hugo LoRA dataset

This folder contains bilingual Victor Hugo supervised fine-tuning data generated from the existing RAG corpus.

Files:
- `train.jsonl`
- `valid.jsonl`
- `example_inventory.jsonl`
- `dataset_manifest.json`
- `source_map.json`

Policy:
- mostly Turkish user prompts
- meaningful English coverage
- cross-lingual examples included
- answer defaults to the user's language unless another language is requested
- outputs keep Victor Hugo in character rather than training a neutral biographer voice
- unsupported details are refused instead of invented
