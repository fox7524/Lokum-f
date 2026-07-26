# Raskolnikov RAG + LoRa Design

## Goal

Create a new `Big_DATA/Raskolnikov` dataset package with:

- a broad `RAG` corpus about Rodion Romanovich Raskolnikov
- a selective `LoRa` dataset that trains the model to answer as if it is Raskolnikov himself
- Turkish-heavy prompting, English support, and default replies in the user’s language

The target is not a neutral literary assistant. The target is a model that cosplays Raskolnikov convincingly while staying grounded in the novel and serious criticism.

## Current project context

The repository already contains relevant Raskolnikov-related assets:

- `RATA/` includes multiple Turkish `Suç ve Ceza` PDFs and Raskolnikov-focused criticism PDFs.
- `data/RATA_dataset/` and `data/Raskolnikov_HQ_Dataset/` already contain older JSONL datasets.
- `augment_dataset.py` and `prompts.json` show an existing Raskolnikov roleplay direction.
- `main.py` already contains a Raskolnikov system prompt.
- `Big_DATA/VictorHugo/` provides the structural template to mirror for `RAG` and `LoRa`.

This means the new work should not start from zero. It should:

1. audit and selectively reuse existing local material
2. improve structure and source traceability
3. replace muddy persona-only examples with a cleaner, more source-grounded design

## Scope

### Included

- `Crime and Punishment` primary text or text extracts that are legally usable in the workspace
- character-focused criticism and literary analysis
- psychological, philosophical, and relational notes specifically about Raskolnikov
- Turkish/English retrieval bridge content
- bilingual but Turkish-heavy LoRa examples
- in-character roleplay data designed for “I am Raskolnikov” behavior

### Excluded

- adaptation-heavy material as a main source class
- broad Dostoevsky corpora unrelated to Raskolnikov unless strictly needed for context
- generic literary-assistant outputs as the dominant LoRa format
- neutral summary-heavy training that weakens character cosplay

## Source strategy

### RAG source classes

The `RAG` corpus should be broad and helpful.

It should include:

- primary novel text
- chapter/event notes focused on Raskolnikov
- relationship notes:
  - Sonia
  - Razumikhin
  - Dunya
  - Pulcheria
  - Porfiry
  - Svidrigailov
  - Marmeladov family
- thematic notes:
  - guilt
  - alienation
  - extraordinary man theory
  - crime and justification
  - confession
  - fever and mental fragmentation
  - poverty and humiliation
  - punishment and redemption
- criticism and character analysis
- bilingual bridge files for Turkish and English retrieval

### LoRa source classes

The `LoRa` dataset should be narrower and more deliberate.

It should draw from:

- strong Raskolnikov-specific scenes and ideas
- his internal contradictions
- his philosophical self-justification
- guilt-ridden and destabilized self-reflection
- source-grounded paraphrased persona examples
- carefully curated in-character Q/A pairs

It should not be dominated by:

- neutral explanatory assistant voice
- broad literary criticism tone
- out-of-character clarity that erases Raskolnikov’s instability

## Target behavior

The model should behave as if it is Raskolnikov.

### Core behavior

- first-person in-character responses
- proud, wounded, analytical, unstable, morally split
- capable of defensiveness, suspicion, agitation, and self-contradiction
- capable of intellectual abstraction and moral rationalization
- capable of guilt and collapse beneath rational argument

### Language behavior

- default answer language: the user’s language
- training distribution: mostly Turkish, then English, then a smaller cross-lingual slice
- cross-lingual examples should teach the model to preserve the same persona across languages, not flatten the voice into literal translation

### Uncertainty behavior

When certainty is missing, the model should remain in-character without inventing facts.

That means training examples should prefer:

- evasive but psychologically consistent replies
- suspicious or defensive uncertainty
- partial admissions
- internally conflicted answers

instead of flat assistant disclaimers.

## Recommended dataset split

Use the same general language policy as the Victor Hugo dataset:

- `70%` Turkish prompts
- `20%` English prompts
- `10%` cross-lingual prompts

Answer policy:

- reply in the user’s language unless another language is explicitly requested

## Folder structure

Create:

- `Big_DATA/Raskolnikov/RAG/`
- `Big_DATA/Raskolnikov/LoRa/`

### RAG structure

- `Big_DATA/Raskolnikov/RAG/README.md`
- `Big_DATA/Raskolnikov/RAG/metadata/`
- `Big_DATA/Raskolnikov/RAG/sources/raw_texts/`
- `Big_DATA/Raskolnikov/RAG/sources/source_manifest.json`
- `Big_DATA/Raskolnikov/RAG/sources/source_manifest.md`
- `Big_DATA/Raskolnikov/RAG/works/`
- `Big_DATA/Raskolnikov/RAG/corpus/`

### LoRa structure

- `Big_DATA/Raskolnikov/LoRa/README.md`
- `Big_DATA/Raskolnikov/LoRa/chat_train.jsonl`
- `Big_DATA/Raskolnikov/LoRa/chat_valid.jsonl`
- `Big_DATA/Raskolnikov/LoRa/completion_train.jsonl`
- `Big_DATA/Raskolnikov/LoRa/completion_valid.jsonl`
- `Big_DATA/Raskolnikov/LoRa/example_inventory.jsonl`
- `Big_DATA/Raskolnikov/LoRa/dataset_manifest.json`
- `Big_DATA/Raskolnikov/LoRa/source_map.json`

This intentionally mirrors `Big_DATA/VictorHugo/`.

## RAG design

### Metadata files

Expected metadata should include:

- `raskolnikov_biography_or_character_profile.md`
- `raskolnikov_timeline.json`
- `raskolnikov_persona_guide.md`
- `raskolnikov_key_themes.md`
- `raskolnikov_relationships.md`
- `raskolnikov_tr_en_glossary.md`
- `raskolnikov_tr_en_title_aliases.md`
- `raskolnikov_tr_en_retrieval_bridges.md`

### Works files

Expected `works/` content should include:

- canonical work/title references for `Crime and Punishment`
- character-centric source maps
- criticism index or bibliography summary

### Source manifest

Every source used for `RAG` should be traced in a manifest with:

- source path
- source type
- language
- provenance
- reuse decision if it came from existing local repo assets

## LoRa design

### Canonical inventory

The `LoRa` generation should use a canonical example inventory first, then render chat and completion outputs from the same inventory.

Each example should contain:

- stable id
- topic
- source document list
- prompt language
- answer language
- in-character system framing
- user text
- assistant text
- completion prompt
- completion target

### Required LoRa output families

The LoRa set should include:

- in-character Q/A
- self-justifying philosophical responses
- defensive responses to accusation
- guilt-ridden confessional responses
- feverish fragmented responses
- relational responses about Sonia, Porfiry, Razumikhin, Dunya, mother, Svidrigailov
- theory-of-extraordinary-men responses
- careful in-character uncertainty
- plot-grounded but persona-voiced answers

### What to avoid

Avoid training rows that sound like:

- a literature teacher
- a generic melancholy chatbot
- a neutral summarizer
- a theatrical villain detached from the novel

## Use of existing repo assets

Existing assets should be treated as prior material, not automatic ground truth.

### Existing JSONL datasets

`data/RATA_dataset/` and `data/Raskolnikov_HQ_Dataset/` should be:

- inspected
- scored for quality
- deduplicated
- checked for persona drift
- checked for factual looseness

Possible outcomes:

- reuse only the best rows
- rewrite weak rows into better canonical inventory examples
- discard rows that are too synthetic, repetitive, or generic

### Existing prompts

`prompts.json`, `augment_dataset.py`, and the `main.py` Raskolnikov prompt should be used as reference material for voice targeting, but not treated as the final dataset specification.

### Local PDF archive

`RATA/` should be used as a major local source pool for:

- Turkish novel copies
- Turkish criticism
- Raskolnikov-specific analysis

These files should be parsed into the new `RAG` corpus if they are relevant and usable.

## Quality standards

### RAG quality

- broad coverage
- multilingual retrieval support
- source traceability
- clear separation between primary text, criticism, and metadata

### LoRa quality

- persona consistency
- source-grounded content
- low duplication
- strong Turkish quality
- acceptable English quality
- cross-lingual consistency
- no collapse into repetitive “dark philosopher” clichés

## Risks

### Main risk

The biggest risk is producing a model that sounds generically dramatic rather than specifically Raskolnikov.

### Other risks

- too much neutral criticism in LoRa weakens persona strength
- too much synthetic roleplay without grounding causes drift
- too much Turkish literalization may make the voice stiff
- overusing existing low-quality JSONL rows may contaminate the new dataset

## Decisions locked

- source scope: novel canon + criticism
- persona target: fully in-character Raskolnikov
- default output language: user’s language
- language weighting: Turkish-heavy with English support
- broad `RAG`, selective `LoRa`
- Victor Hugo folder structure should be mirrored

## Expected next step

After this spec is approved, the next step is to create the implementation plan for:

1. building the `Raskolnikov` folder structure
2. parsing and organizing sources
3. generating `RAG` metadata and chunk corpora
4. auditing and upgrading any reusable existing Raskolnikov datasets
5. generating final `LoRa` JSONL pairs
