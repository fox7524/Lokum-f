---
title: "RAG + Fine-Tune Performance Architecture Refresh"
date: "2026-05-28"
status: draft
---

# Goal

Improve the internal architecture for RAG and fine-tuning so the system becomes:

- Faster during indexing, retrieval, dataset generation, and training preparation
- More memory-efficient for local execution on the user's machine
- Easier to maintain, profile, and tune without touching unrelated UI code
- Safer to extend without reintroducing prompt-quality or dataset-quality regressions

This work explicitly targets backend architecture only. The desktop UI should keep working with minimal or no visible behavior changes.

# Problems To Solve

## RAG

- `rag_engine.py` currently combines extraction, normalization, chunking, embedding, FAISS persistence, metadata persistence, state tracking, and query assembly in one large module.
- `file_ingest.py` overlaps with parts of the ingestion pipeline, increasing duplication and risk of drift.
- Persistence flows are difficult to optimize because data movement, state updates, and index writes are tightly coupled.
- Query assembly is mixed into retrieval logic, making latency optimization and context-quality tuning harder.

## Fine-tuning

- Dataset preparation responsibilities are split across runtime code and standalone scripts without a single shared contract.
- Validation, deduplication, content balancing, and artifact layout are not centralized enough.
- Current generation flow can overproduce low-signal patterns if a dataset script drifts in one direction.
- There is no clear preflight stage that validates dataset integrity, config assumptions, and artifact layout before launching MLX LoRA jobs.

# Non-Goals

- Full desktop UI redesign or large-scale `main.py` refactor in this phase
- Changing the external training file formats already expected by the app
- Replacing FAISS or the MLX LoRA pipeline
- Building distributed or server-based indexing/training infrastructure

# Recommended Approach

Adopt a safe modular backend refactor:

- Keep current external behavior and file formats stable
- Introduce focused internal modules with thin orchestration layers
- Migrate incrementally behind current interfaces
- Add validation and profiling seams first, then optimize the hot paths

This is preferred over a full rewrite because it provides meaningful performance wins without destabilizing the desktop application.

# Target Architecture

## RAG package structure

Create a new internal package layout:

`rag/`
- `extractors.py`
- `normalize.py`
- `chunker.py`
- `embedder.py`
- `index_store.py`
- `state_store.py`
- `query_service.py`
- `pipeline.py`

### Responsibilities

- `extractors.py`
  - File-type-specific extraction
  - Shared by both RAG and dataset-building code
- `normalize.py`
  - Text cleanup and canonicalization before chunking
- `chunker.py`
  - Chunk sizing, overlap, chunk hashing, optional chunk dedup helpers
- `embedder.py`
  - Embedding batch execution and batching policy
- `index_store.py`
  - FAISS writes/loads and metadata append/update flows
- `state_store.py`
  - File-level state, content hashes, chunk ranges, corruption-safe persistence
- `query_service.py`
  - Retrieval, optional rerank/scoring cleanup, prompt context assembly
- `pipeline.py`
  - High-level ingest orchestration using the focused modules above

## Fine-tune package structure

Create a new internal package layout:

`finetune/`
- `dataset_sources.py`
- `dataset_builder.py`
- `dataset_balance.py`
- `dataset_validator.py`
- `dataset_writer.py`
- `job_preflight.py`
- `job_runner.py`
- `artifacts.py`

### Responsibilities

- `dataset_sources.py`
  - Reads raw sources such as prompts, RAG-derived texts, and curated samples
- `dataset_builder.py`
  - Builds normalized examples with a single canonical schema
- `dataset_balance.py`
  - Controls ratios for code samples, RAG-grounded samples, and behavioral samples
- `dataset_validator.py`
  - Validates JSONL structure, duplicate rate, empty text, malformed turns, and split integrity
- `dataset_writer.py`
  - Streaming/batched writing to `train.jsonl` and `valid.jsonl`
- `job_preflight.py`
  - Verifies model path, dataset presence, config coherence, and output directory readiness
- `job_runner.py`
  - Runs MLX LoRA jobs and captures progress/state
- `artifacts.py`
  - Standardizes output directories for datasets, configs, logs, and adapters

# Data Flow

## RAG ingest flow

1. `extractors.py` reads source content
2. `normalize.py` cleans and canonicalizes text
3. `chunker.py` produces chunks and stable chunk hashes
4. `state_store.py` compares file and chunk hashes against prior state
5. `embedder.py` embeds only new or changed chunks
6. `index_store.py` appends vectors and metadata
7. `state_store.py` commits the new state atomically

### Expected benefits

- Avoid re-embedding unchanged content
- Make failures resumable per stage
- Reduce unnecessary full-store rewrites

## RAG query flow

1. `query_service.py` normalizes the query
2. Retrieve top-k chunk candidates from `index_store.py`
3. Apply lightweight cleanup or rerank heuristics
4. Assemble a compact context block with consistent boundaries
5. Return both context text and source metadata

### Expected benefits

- Lower latency
- Better control over noisy context injection
- Easier evaluation of retrieval quality

## Fine-tune dataset flow

1. `dataset_sources.py` gathers raw prompt/RAG/example inputs
2. `dataset_builder.py` emits canonical examples
3. `dataset_balance.py` enforces target composition
4. `dataset_validator.py` rejects malformed or low-signal items
5. `dataset_writer.py` writes train/valid JSONL in streaming mode
6. `job_preflight.py` validates the training run before launch
7. `job_runner.py` executes MLX LoRA training

# Performance Strategy

## RAG performance

- Add file hashes and chunk hashes to support incremental indexing
- Deduplicate repeated chunks before embedding
- Centralize embedding batch-size tuning in one module
- Reduce large full-array read/write behavior where possible
- Separate retrieval from context formatting to expose true query latency

## Fine-tune performance

- Stream dataset generation instead of building everything in large in-memory lists where avoidable
- Centralize quality gates so low-signal examples are filtered once
- Standardize artifact directories to reduce cleanup and resume complexity
- Add preflight checks so failed runs fail fast before expensive setup

# Backward Compatibility

- Keep the current app-level entrypoints working during migration
- Preserve existing JSONL training format: one JSON object per line with a `text` field
- Preserve current RAG persistence location and existing user data
- Introduce thin facades in the old modules first, then move internal logic behind them

# Error Handling

## RAG

- Each stage should fail with a stage-specific error message
- State writes must remain atomic
- Corrupted metadata or state should trigger recovery logic without destroying healthy data when possible
- Partial ingest should not mark a file as fully indexed until all stages complete successfully

## Fine-tune

- Invalid dataset rows should be counted and reported before training starts
- Empty or malformed splits should abort preflight
- Missing model paths or unwritable artifact directories should fail before MLX launch
- Output directories must be deterministic and collision-safe

# Testing Strategy

## RAG tests

- Incremental indexing skips unchanged files
- Changed files only re-embed affected chunks
- Query service returns consistent context boundaries and sources
- Persistence recovery still works after interrupted writes

## Fine-tune tests

- Dataset validation rejects malformed JSONL and empty examples
- Deduplication removes repeated conversations
- Balancing logic preserves minimum code-sample ratio
- Preflight fails on missing model paths or invalid dataset directories

# Migration Plan

## Phase 1: Extraction of boundaries

- Introduce the new backend modules without changing file formats
- Keep existing public classes as facades over the new internals
- Add tests around current behavior before moving logic

## Phase 2: Performance improvements

- Add incremental hashing and chunk-level deduplication
- Move dataset writing to a streaming/batched flow
- Add preflight validation and artifact standardization

## Phase 3: Cleanup

- Remove duplicated ingestion logic between `rag_engine.py` and `file_ingest.py`
- Reduce large orchestration responsibilities in legacy files
- Keep compatibility wrappers only where still needed by the app

# Success Criteria

- RAG re-indexing is materially faster for partially changed corpora
- Query latency and context assembly become independently measurable
- Dataset generation remains valid while becoming more memory-efficient
- Fine-tune runs fail less often due to preventable setup/data issues
- Existing app integrations continue to work with minimal changes

# Risks

- Moving persistence logic can introduce subtle compatibility bugs if not covered by tests
- Hash-based incremental indexing must be carefully aligned with current metadata layout
- Over-optimizing retrieval cleanup can accidentally remove useful context

# Decision Summary

- Use a safe modular refactor, not a rewrite
- Prioritize RAG ingest/query and fine-tune dataset/training prep
- Preserve current app behavior and on-disk formats
- Add performance seams and validation layers before deeper optimization
