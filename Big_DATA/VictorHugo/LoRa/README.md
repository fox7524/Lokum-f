# Victor Hugo LoRA dataset

This folder contains bilingual Victor Hugo supervised fine-tuning data generated from the existing RAG corpus.

Files:
- `chat_train.jsonl`
- `chat_valid.jsonl`
- `completion_train.jsonl`
- `completion_valid.jsonl`
- `example_inventory.jsonl`
- `dataset_manifest.json`
- `source_map.json`

Policy:
- mostly Turkish user prompts
- meaningful English coverage
- cross-lingual examples included
- answer defaults to the user's language unless another language is requested
- outputs aim for Hugo-like moral gravity without fake certainty
