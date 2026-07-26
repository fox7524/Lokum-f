# Victor Hugo RAG corpus

This folder is a source-backed starter corpus for retrieval-augmented generation about Victor Hugo.

It is built to help an AI:
- understand Hugo's life, politics, themes, and chronology
- retrieve lists of works and major titles
- access primary texts, correspondence, speeches, and criticism
- separate factual metadata from literary voice and from secondary commentary

## Structure
- `metadata/`: timeline, bibliography, persona notes, retrieval notes
- `sources/raw_texts/`: downloaded public-domain or reference source texts
- `works/`: curated notes and work-group summaries
- `corpus/`: chunked `jsonl` files ready for indexing

## Limits
This is broad, not literally exhaustive. Victor Hugo has thousands of cataloged manifestations, editions, adaptations, and studies. The bibliography in this folder aims to cover his core authored works and major posthumous publications, while the source manifest points to larger catalogs for deeper expansion.
