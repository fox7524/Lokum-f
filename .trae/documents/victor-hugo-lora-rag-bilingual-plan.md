# Victor Hugo LoRA + Bilingual RAG Plan

> **For agentic workers:** Use `writing-plans` discipline for exact file-level decisions and `executing-plans` discipline for blocker-aware execution. Do not start implementation until this plan is approved.

## Summary

Build a reproducible Victor Hugo data pipeline under `Big_DATA/VictorHugo` that produces:

- a new `LoRa` folder with high-quality `train.jsonl` / `valid.jsonl` pairs for both:
  - chat-style supervised fine-tuning
  - completion-style supervised fine-tuning
- a corrective pass on the existing `RAG` corpus so it supports Turkish-heavy and English usage better
- engine-level support for more than raw `{"text": ...}` rows, while preserving compatibility with the current fine-tune flow

The design assumes the end goal is not merely “some Hugo text in JSONL,” but a dataset that teaches a base model to answer like a Victor Hugo assistant: historically grounded, stylistically aligned, language-aware, and less likely to hallucinate.

## Current State Analysis

### Existing Victor Hugo assets

Current corpus already exists under `Big_DATA/VictorHugo/RAG` and includes:

- `22` downloaded raw source files
- `9835` chunked corpus rows in `corpus/victor_hugo_rag_chunks.jsonl`
- structured metadata such as:
  - `metadata/victor_hugo_persona_guide.md`
  - `metadata/victor_hugo_timeline.json`
  - `works/complete_bibliography.md`

This is a strong base for retrieval and source-grounded prompt generation, but it is not yet optimized for:

- Turkish-heavy user queries
- bilingual title/alias retrieval
- LoRA-ready supervised examples
- native multi-format training rows

### Existing fine-tune format constraints

Current fine-tune utilities are text-only:

- `finetune/dataset_validator.py` only accepts rows with a non-empty `text` string
- `finetune/dataset_writer.py` only writes `{"text": ...}` rows
- tests in `tests/test_finetune_dataset_validator.py` only validate this single schema

This means the codebase can currently write plain-text JSONL, but it does not natively support:

- `messages`-based chat datasets
- `prompt` / `completion` datasets
- format detection or mixed-format validation

### Existing fine-tune engine strengths

Current engine behavior is still useful:

- `core/finetune_engine.py` already assumes `train.jsonl` and `valid.jsonl`
- there is existing safety work around ChatML pre-splitting in `tests/test_finetune_presplit_chatml.py`
- this suggests the codebase is already close to supporting richer training data, but needs a cleaner abstraction

### Existing RAG strengths

`core/rag_engine.py` already uses a multilingual embedding model, which is a strong starting point for Turkish and English retrieval.

The current main issue is therefore not multilingual embeddings. The bigger issues are:

- bilingual bridge metadata is missing
- title aliases across French / English / Turkish are missing
- prebuilt chunk metadata is not preserved richly enough during retrieval
- `.jsonl` corpus handling should be made explicit and reliable

## Research Grounding

This plan is based on three external constraints:

1. TRL / SFT training supports multiple dataset shapes, including:
   - `{"text": ...}`
   - conversational `{"messages": [...]}`
   - `{"prompt": ..., "completion": ...}`
   - conversational prompt-completion variants  
   Official guidance also recommends aligning training format with the model’s expected chat template and computing loss on assistant/completion tokens only when appropriate. [$TRAE_REF](https://huggingface.co/docs/trl/en/sft_trainer)

2. PEFT / LoRA quality depends more on dataset quality and consistency than raw volume, and multilingual or generalized targets benefit from diversity plus careful deduplication instead of noisy scale. [$TRAE_REF](https://ai.meta.com/blog/how-to-fine-tune-llms-peft-dataset-curation/)

3. The current sentence-transformers embedding model used by the project is already multilingual, so a first corrective pass should improve corpus content and metadata before changing embedding infrastructure. [$TRAE_REF](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)

## Skill Workflow Alignment

The implementation should follow these loaded-skill constraints:

- `brainstorming`
  - clarify major design choices before implementation
  - prefer a design that is coherent across data generation, validation, and retrieval
- `writing-plans`
  - make all file-level changes explicit
  - define verification before implementation
  - avoid placeholders and vague “handle edge cases later” planning
- `executing-plans`
  - execution must stop on format ambiguity, failing validation, or broken retrieval relevance
  - do not bulldoze through blockers by guessing

## Proposed Changes

### 1. Create a dedicated Victor Hugo LoRA workspace

#### Files

- Create `Big_DATA/VictorHugo/LoRa/README.md`
- Create `Big_DATA/VictorHugo/LoRa/chat_train.jsonl`
- Create `Big_DATA/VictorHugo/LoRa/chat_valid.jsonl`
- Create `Big_DATA/VictorHugo/LoRa/completion_train.jsonl`
- Create `Big_DATA/VictorHugo/LoRa/completion_valid.jsonl`
- Create `Big_DATA/VictorHugo/LoRa/dataset_manifest.json`
- Create `Big_DATA/VictorHugo/LoRa/source_map.json`
- Create `Big_DATA/VictorHugo/LoRa/example_inventory.jsonl`

#### What

Add a clean, user-visible dataset output folder separate from the RAG corpus.

#### Why

This keeps:

- source corpus
- retrieval bridge content
- supervised fine-tuning outputs

as separate concerns instead of mixing all artifacts in one folder.

#### How

Use a canonical internal inventory first, then render both chat and completion datasets from that inventory. Do not hand-author two unrelated corpora.

Recommended canonical example structure:

```json
{
  "id": "vh_000001",
  "source_docs": [
    "RAG/metadata/victor_hugo_timeline.json",
    "RAG/sources/raw_texts/les_miserables_en.txt"
  ],
  "topic": "exile",
  "language": "tr",
  "answer_language": "tr",
  "format_family": "qa",
  "system": "You are a careful Victor Hugo assistant grounded in source material.",
  "user": "Victor Hugo neden sürgüne gitti?",
  "assistant": "Victor Hugo, Louis-Napoléon'un darbesine karşı çıktığı için sürgüne gitti...",
  "completion_prompt": "Soru: Victor Hugo neden sürgüne gitti?\nYanıtı Türkçe ver.",
  "completion_target": "Victor Hugo, Louis-Napoléon'un darbesine karşı çıktığı için sürgüne gitti..."
}
```

### 2. Upgrade fine-tune dataset validation from text-only to multi-format

#### Files

- Modify `finetune/dataset_validator.py`
- Modify `tests/test_finetune_dataset_validator.py`

#### What

Extend validation to support these row families:

- text row
- chat row
- completion row

#### Why

Without this, the codebase cannot safely accept the datasets the user explicitly asked for.

#### How

Validator should:

- accept exactly one format per row
- reject empty strings
- reject malformed chat message arrays
- reject invalid roles
- reject empty prompt or completion
- reject mixed-schema rows such as:
  - `{"text": "...", "messages": [...]}`
  - `{"messages": [...], "completion": "..."}`

Recommended accepted formats:

```json
{"text":"Victor Hugo wrote ..."}
```

```json
{"messages":[
  {"role":"system","content":"You are Victor Hugo."},
  {"role":"user","content":"Sefiller ne anlatır?"},
  {"role":"assistant","content":"Sefiller, adalet, merhamet ve toplumsal sefalet üzerine büyük bir romandır..."}
]}
```

```json
{"prompt":"Explain Hugo's views on exile.","completion":"Hugo treated exile as both punishment and prophetic distance..."}
```

### 3. Upgrade dataset writing to format-aware JSONL writers

#### Files

- Modify `finetune/dataset_writer.py`
- Create `tests/test_finetune_dataset_writer.py`

#### What

Replace the single text-row writer abstraction with explicit writers for:

- text datasets
- chat datasets
- completion datasets

#### Why

The writer must not silently coerce all future datasets into `{"text": ...}`. Native schemas should stay native until a specific training path needs conversion.

#### How

Recommended public API shape:

```python
def write_text_jsonl_stream(path: Path, texts: Iterable[str]) -> int: ...
def write_chat_jsonl_stream(path: Path, rows: Iterable[dict]) -> int: ...
def write_completion_jsonl_stream(path: Path, rows: Iterable[dict]) -> int: ...
def detect_jsonl_format(path: Path) -> str: ...
```

Tests should verify:

- row count
- JSON validity
- exact stored schema
- UTF-8 Turkish content preservation
- non-ASCII punctuation preservation

### 4. Add a Victor Hugo dataset builder instead of hand-editing JSONL

#### Files

- Create `tools/build_victor_hugo_lora_dataset.py`
- Create `tests/test_victor_hugo_lora_builder.py`

#### What

Build a deterministic data generation script that reads from `Big_DATA/VictorHugo/RAG` and emits the LoRA outputs.

#### Why

This work should be reproducible. If later the prompt balance, filtering policy, or source coverage changes, the dataset must be rebuildable rather than manually patched.

#### How

The builder should:

- read the existing RAG metadata and raw texts
- read bridge files added in this plan
- construct a canonical example inventory
- deduplicate semantically similar prompt-response pairs
- split train/valid reproducibly with a fixed seed
- render both chat and completion JSONL outputs
- emit dataset statistics and source coverage

Recommended responsibilities inside the script:

- `load_hugo_sources()`
- `load_bridge_metadata()`
- `generate_example_inventory()`
- `dedupe_examples()`
- `split_train_valid(seed=...)`
- `render_chat_rows()`
- `render_completion_rows()`
- `write_manifest()`

### 5. Fix RAG for Turkish and English via bridge metadata

#### Files

- Create `Big_DATA/VictorHugo/RAG/metadata/victor_hugo_tr_en_glossary.md`
- Create `Big_DATA/VictorHugo/RAG/metadata/victor_hugo_tr_en_title_aliases.md`
- Create `Big_DATA/VictorHugo/RAG/metadata/victor_hugo_tr_en_retrieval_bridges.md`

#### What

Add bilingual bridge documents that explicitly connect:

- Turkish user vocabulary
- English retrieval vocabulary
- French canonical titles and names

#### Why

This is the lowest-risk way to improve Turkish-heavy retrieval quality against a corpus that is largely French and English without changing the embedding stack immediately.

#### How

These files should include:

- title aliases:
  - `Les Misérables` / `Les Miserables` / `Sefiller`
  - `Notre-Dame de Paris` / `The Hunchback of Notre-Dame` / Turkish common references
- concept bridges:
  - `sürgün` ↔ `exile`
  - `idam cezası` ↔ `death penalty`
  - `sefalet` ↔ `misery`
  - `cumhuriyetçilik` ↔ `republicanism`
  - `merhamet` ↔ `mercy`
- query anchor phrases for high-value Hugo topics:
  - exile
  - justice
  - punishment
  - poverty
  - Notre-Dame / cathedral
  - Les Misérables themes
  - rhetoric and style

### 6. Teach the RAG engine to ingest prebuilt `.jsonl` chunk corpora cleanly

#### Files

- Modify `core/rag_engine.py`
- Create `tests/test_rag_victor_hugo_bilingual.py`

#### What

Allow the engine to read and preserve prebuilt chunked JSONL rows like the existing `victor_hugo_rag_chunks.jsonl`.

#### Why

The current Victor Hugo corpus already exists in prechunked form. The system should be able to use that structure directly instead of relying only on raw-text re-chunking.

#### How

Add explicit support for rows like:

```json
{
  "chunk_id": "les_miserables_en__0001",
  "doc_path": "sources/raw_texts/les_miserables_en.txt",
  "doc_title": "les miserables en",
  "category": "sources",
  "language_guess": "en",
  "text": "..."
}
```

Persist these metadata keys into stored chunk metadata so retrieval tests can assert relevance by source and category, not just by non-empty output.

### 7. Add Turkish-heavy, English-capable LoRA example policies

#### Files

- Implement in `tools/build_victor_hugo_lora_dataset.py`
- Describe in `Big_DATA/VictorHugo/LoRa/README.md`
- Record in `Big_DATA/VictorHugo/LoRa/dataset_manifest.json`

#### What

Use a deliberate language distribution rather than random language mixing.

#### Why

The user explicitly wants English plus mostly Turkish usage. A random multilingual mix would be weaker and harder to audit.

#### How

Use this distribution in the generated inventory:

- `70%` Turkish prompts
- `20%` English prompts
- `10%` cross-lingual prompts

Default answer policy:

- answer in the user’s language unless the prompt explicitly requests another language

Cross-lingual slice should include:

- Turkish question → English answer requested
- English question → Turkish answer requested
- title alias disambiguation across French/English/Turkish
- “translate and explain like Hugo” examples

### 8. Define the content mix for a strong Victor Hugo persona dataset

#### Files

- Implement in `tools/build_victor_hugo_lora_dataset.py`
- Document in `Big_DATA/VictorHugo/LoRa/README.md`

#### What

Generate examples from distinct content buckets instead of relying on a single generic Q&A style.

#### Why

To make the LoRA actually feel like a Victor Hugo assistant, the dataset must include not just facts, but tone, rhetoric, restraint, style explanation, and work-level distinctions.

#### How

Required content buckets:

- biography and timeline
- works and title aliases
- political positions
- exile and republicanism
- death penalty and justice
- poverty and social conscience
- literary style and rhetoric
- themes by work
- correspondence and private voice
- “unknown / not enough evidence” restraint

Required output families:

- direct factual answers
- short essay answers
- quote-grounded explanations
- compare/contrast answers
- style explanation answers
- careful refusal / uncertainty answers

### 9. Add dataset audit outputs so quality is measurable

#### Files

- Create or generate `Big_DATA/VictorHugo/LoRa/dataset_manifest.json`
- Optionally create `Big_DATA/VictorHugo/LoRa/dataset_report.md`

#### What

Store quantitative audit metadata for every build.

#### Why

High-quality LoRA work needs measurable dataset quality, not vibes.

#### How

Manifest should include at least:

```json
{
  "build_version": "victor-hugo-lora-v1",
  "train_counts": {
    "chat": 0,
    "completion": 0
  },
  "valid_counts": {
    "chat": 0,
    "completion": 0
  },
  "language_mix": {
    "tr": 0,
    "en": 0,
    "cross": 0
  },
  "topic_mix": {},
  "source_coverage": {},
  "duplicate_prompt_ratio": 0.0,
  "seed": 0
}
```

## Assumptions & Decisions

### Locked decisions

- The new work includes both:
  - LoRA dataset generation
  - a corrective RAG pass
- The LoRA target must support both:
  - chat dataset format
  - completion dataset format
- Output languages should primarily support:
  - Turkish-heavy user prompting
  - meaningful English support
- Default assistant output language should follow the user’s language
- French remains the authenticity anchor in source material, but the delivered user-facing dataset should be Turkish-heavy plus English-capable

### Important non-goals for this pass

- no embedding-model swap
- no reranker
- no query-language detection subsystem
- no training hyperparameter experimentation yet
- no automatic web crawling beyond the already collected Hugo corpus

These are intentionally deferred until:

- dataset quality is validated
- bilingual retrieval failures are measured
- format support is confirmed working

### Safety / quality decision

The dataset should train “act like Victor Hugo” behavior as:

- historically and stylistically aligned assistant behavior
- not a deceptive identity claim that the model literally is Victor Hugo in reality

That means system prompts and outputs should encourage:

- Hugo-grounded tone
- Hugo-grounded knowledge and rhetoric
- source-aware humility where evidence is absent

not blind impersonation without grounding.

## Verification Steps

### Fine-tune format verification

Run targeted tests that prove:

- text rows still validate
- chat rows validate
- completion rows validate
- mixed rows are rejected
- malformed roles are rejected
- empty prompt/completion values are rejected

Relevant tests:

- `tests/test_finetune_dataset_validator.py`
- `tests/test_finetune_dataset_writer.py`

### Engine compatibility verification

Run tests showing:

- existing text-only datasets still work
- ChatML pre-splitting safety is preserved
- native chat/completion support does not break the current training entrypoint assumptions

Relevant tests:

- `tests/test_finetune_presplit_chatml.py`
- any existing preflight tests touching dataset assumptions

### Dataset quality verification

After generation, verify:

- `chat_train.jsonl` and `chat_valid.jsonl` exist
- `completion_train.jsonl` and `completion_valid.jsonl` exist
- all rows validate under the new validator
- Turkish/English/cross-lingual ratios match the intended policy approximately
- prompt duplication is below an explicitly chosen threshold
- examples cite or map back to source docs via `source_map.json`

### RAG bilingual verification

Run retrieval-focused tests for questions such as:

- `Victor Hugo neden sürgüne gitti?`
- `Sefiller ne anlatır?`
- `Victor Hugo idam cezasına neden karşıydı?`
- `What are the central themes of Les Misérables?`
- `Which Victor Hugo work is known in Turkish as Sefiller?`

Success condition:

- results are not merely non-empty
- they surface relevant Hugo sources, titles, or category metadata

### Manual review checklist

- Open `Big_DATA/VictorHugo/LoRa/README.md`
- Sample random rows from each JSONL output
- Confirm Turkish reads naturally and does not sound machine-translated
- Confirm English answers remain fluent and source-grounded
- Confirm RAG bridge files are easy to inspect and maintain

## Recommended Execution Order

1. Update dataset validator
2. Update dataset writer
3. Add tests for new formats
4. Add Victor Hugo builder script
5. Add RAG bridge metadata files
6. Add `.jsonl` ingest support and richer metadata preservation in `core/rag_engine.py`
7. Generate LoRA outputs
8. Run dataset validation and bilingual retrieval verification
9. Review output samples before any actual fine-tuning run

## Blockers That Must Stop Execution

Execution must stop immediately if any of these happen:

- format ambiguity between engine expectations and generated JSONL schemas
- chat rows require a conversion policy that is not yet defined
- Turkish output quality is obviously machine-garbled
- generated examples cannot be traced back to Hugo sources
- retrieval tests fail on obvious Turkish Hugo queries after bridge files are added

If a blocker appears, revise the plan before continuing.
