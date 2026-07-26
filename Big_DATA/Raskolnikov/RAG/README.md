# Raskolnikov RAG corpus

This folder is a source-backed starter corpus for retrieval-augmented generation about Rodion Romanovich Raskolnikov.

It is built to help an AI:
- retrieve grounded notes about Raskolnikov's psychology, relationships, and plot arc
- map Turkish literary queries to English retrieval anchors
- separate factual metadata from persona guidance
- trace every included local source through a manifest

## Structure
- `metadata/`: concise character, theme, relationship, glossary, and timeline files
- `sources/raw_texts/`: reserved for parsed or normalized local source text exports
- `sources/source_manifest.json`: machine-readable local source inventory
- `sources/source_manifest.md`: quick human-readable source summary
- `works/`: reuse-policy and work-level notes
- `corpus/`: chunked `jsonl` ready for indexing

## Limits
This is a compact bootstrap corpus, not a full scholarly edition of `Crime and Punishment`.
It is intended to provide reliable retrieval anchors for Raskolnikov-focused generation before any larger source parsing pipeline is run.
