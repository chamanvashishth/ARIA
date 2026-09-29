# ARIA Roadmap

This roadmap converts the project specification into implementation blocks. A milestone is complete only when its acceptance criteria are met.

## Phase 0 — Hardware and feasibility

**Goal:** establish what the actual machine can support.

Acceptance criteria:

- CPU architecture, cores, threads, instruction sets recorded
- RAM recorded
- GPU availability, backend, VRAM, and capabilities recorded
- NPU availability and runtime investigated
- available ML runtimes/toolchains recorded
- realistic SLM configurations estimated
- training/inference feasibility documented
- machine-readable hardware profile produced

**Status:** NOT IMPLEMENTED

## Phase 1 — Architecture

**Goal:** establish stable boundaries between model, runtime, memory, agent, API, and UI.

Acceptance criteria:

- contracts exist for major subsystem boundaries
- responsibilities do not overlap unnecessarily
- unsupported capabilities are explicit
- architecture decisions have measurable reasons

**Status:** FOUNDATION STARTED

## Phase 2 — Tokenizer

**Goal:** create a local tokenizer suitable for English, Hindi, Hinglish, programming, mathematics, and technical language.

Acceptance criteria:

- tokenizer trains locally
- tokenizer saves/loads locally
- tokenization benchmarks exist
- vocabulary and compression trade-offs are documented

**Status:** NOT IMPLEMENTED

## Phase 3 — SLM

**Goal:** implement a real decoder-only language model.

Acceptance criteria:

- forward propagation
- loss calculation
- backpropagation
- parameter updates
- checkpoint save/load
- autoregressive generation
- deterministic smoke tests

**Status:** NOT IMPLEMENTED

## Phase 4 — Training

**Goal:** reproducible local pretraining and instruction training.

Acceptance criteria:

- dataset cleaning/filtering
- deduplication
- tokenization/packing
- mixed precision where supported
- gradient accumulation
- gradient clipping
- learning-rate scheduling
- checkpoint recovery
- validation metrics
- reproducibility metadata

**Status:** NOT IMPLEMENTED

## Phase 5 — Inference runtime

**Goal:** make local generation practical.

Acceptance criteria:

- streaming
- KV cache
- efficient sampling
- memory-aware model loading
- quantization experiments
- TTFT and tokens/sec benchmarks

**Status:** NOT IMPLEMENTED

## Phase 6 — Hardware runtime

**Goal:** select execution hardware based on measured capability.

Acceptance criteria:

- hardware detector
- capability analyzer
- workload classification
- device scheduler
- memory checks
- transfer-cost awareness
- CPU/GPU/NPU benchmarks
- fallback paths

**Status:** NOT IMPLEMENTED

## Phase 7 — Memory and local knowledge

**Goal:** provide inspectable local state and retrieval.

Acceptance criteria:

- working/conversational/long-term memory separation
- searchable and editable memory
- local document ingestion
- local embeddings
- local index
- retrieval and context selection
- source-aware responses

**Status:** NOT IMPLEMENTED

## Phase 8 — Agent

**Goal:** allow controlled local actions.

Acceptance criteria:

- explicit tool schemas
- argument validation
- permissions
- filesystem sandbox
- command allowlist
- timeouts
- resource limits
- audit logging
- destructive-action protection

**Status:** NOT IMPLEMENTED

## Phase 9 — API and UI

**Goal:** expose the real local system through a usable interface.

Acceptance criteria:

- local health endpoint
- model/hardware/metrics endpoints
- chat/generation streaming
- memory and document operations
- polished local UI
- transparent runtime metrics
- no model execution logic in the frontend

**Status:** NOT IMPLEMENTED

## Phase 10 — Evaluation and security

**Goal:** make quality and safety measurable.

Acceptance criteria:

- language metrics
- instruction-following tests
- math exact-answer tests
- code execution tests
- conversation-quality tests
- hardware benchmarks
- security audit
- offline/network-isolation test

**Status:** NOT IMPLEMENTED

## Phase 11 — Optimization and research

Only after the stable system is measurable:

- quantization
- pruning
- distillation
- LoRA/QLoRA
- speculative decoding
- sparse attention
- MoE
- long-context methods
- multimodality
- controlled continual learning

These are research tracks, not foundation requirements.
