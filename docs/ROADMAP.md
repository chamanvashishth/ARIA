# ARIA Roadmap

This roadmap converts the project specification into implementation blocks. A milestone is complete only when its acceptance criteria are met.

## Phase 0 — Software feasibility and architecture

**Goal:** establish the software boundaries and constraints ARIA must satisfy.

**Status:** COMPLETE

## Phase 1 — ARIA virtual compute

**Goal:** build ARIA's own software-defined execution engines.

Acceptance criteria:

- Virtual CPU has executable general-purpose operations
- Virtual GPU has executable matrix operations
- Virtual NPU has executable neural-network operations
- engines have deterministic tests
- compute types and dispatch are explicit
- physical CPU/GPU/NPU are not required

**Status:** IN PROGRESS

## Phase 2 — Neural brain foundation

**Goal:** establish the first genuinely trainable neural language model owned by ARIA.

Acceptance criteria:

- model has learned parameters
- forward computation produces token logits
- softmax language-model loss is implemented
- backpropagation computes parameter updates
- training changes model parameters
- autoregressive next-token generation exists
- checkpoints save/load locally
- deterministic tests prove the learning loop

**Status:** FOUNDATION IMPLEMENTED**

Current implementation is intentionally tiny: token embedding + vocabulary projection. It is a real neural learning core, but it is not yet the final ARIA SLM.

## Phase 3 — Tokenizer

**Goal:** create a local tokenizer suitable for English, Hindi, Hinglish, programming, mathematics, and technical language.

Acceptance criteria:

- tokenizer trains locally
- tokenizer saves/loads locally
- tokenization benchmarks exist
- vocabulary and compression trade-offs are documented

**Status:** NOT IMPLEMENTED

## Phase 4 — Full SLM

**Goal:** evolve the neural foundation into an efficient decoder-only Transformer.

Acceptance criteria:

- Transformer blocks
- attention
- positional encoding such as RoPE
- normalization
- feed-forward network
- forward propagation
- loss calculation
- backpropagation
- checkpoint save/load
- autoregressive generation
- deterministic smoke tests

**Status:** NOT IMPLEMENTED

## Phase 5 — Training

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

## Phase 6 — Inference runtime

**Goal:** make local generation practical.

Acceptance criteria:

- streaming
- KV cache
- efficient sampling
- memory-aware model loading
- quantization experiments
- TTFT and tokens/sec benchmarks

**Status:** NOT IMPLEMENTED

## Phase 7 — Runtime orchestration

**Goal:** route model workloads through ARIA's virtual compute engines.

Acceptance criteria:

- compute capability registry
- workload classification
- scheduler
- execution dispatch
- memory accounting
- fallback paths
- reproducible benchmarks

**Status:** FOUNDATION PARTIALLY IMPLEMENTED

## Phase 8 — Memory and local knowledge

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

## Phase 9 — Agent

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

## Phase 10 — API and UI

**Goal:** expose the real local system through a usable interface.

Acceptance criteria:

- local health endpoint
- model/runtime/metrics endpoints
- chat/generation streaming
- memory and document operations
- polished local UI
- no model execution logic in the frontend

**Status:** NOT IMPLEMENTED

## Phase 11 — Evaluation and security

**Goal:** make quality and safety measurable.

Acceptance criteria:

- language metrics
- instruction-following tests
- math exact-answer tests
- code execution tests
- conversation-quality tests
- compute benchmarks
- security audit
- offline/network-isolation test

**Status:** NOT IMPLEMENTED

## Phase 12 — Optimization and research

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
