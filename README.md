# ARIA

**ARIA — Autonomous Research & Intelligence Architecture**

ARIA is a software-only, local-first AI system being built block-by-block around an ARIA-owned trainable neural brain. The project is intentionally separated into model, tokenizer, training, inference, runtime, memory, retrieval, tools, verification, API, UI, evaluation, and security layers.

> ARIA is not a chatbot wrapper. External hosted AI/model APIs are not part of normal runtime operation.

## Current status

**Block 0 — Repository & Engineering Foundation: IMPLEMENTED**

**Block 1 — Neural Tensor/Model Foundation: IMPLEMENTED**

**Block 2 — Neural Layer Foundation: IMPLEMENTED**

**Block 3 — Tokenizer Foundation: IMPLEMENTED**

**Block 4 — Trainable Language-Model Core: IMPLEMENTED**

**Block 5 — Decoder Transformer SLM Foundation: EXPERIMENTAL**

This block establishes the real project skeleton and development boundary. It does **not** claim that the neural brain, tokenizer, training engine, inference engine, RAG, agent, or runtime are implemented.

### Implemented in Block 0

- Python 3.11+ package foundation
- reproducible package metadata in `pyproject.toml`
- explicit ARIA versioning
- local configuration boundary
- application logging boundary
- CLI entry point
- package boundaries for future ARIA subsystems
- pytest foundation with initial tests
- basic local-data isolation via `.gitignore`

### Not implemented yet

- neural tensor/model backend — **IMPLEMENTED (Block 1 foundation)**
- ARIA tokenizer — **IMPLEMENTED (deterministic byte-level foundation)**
- trainable SLM
- training engine
- inference engine
- AI runtime
- virtual compute
- memory
- local RAG
- agent/tools
- verification
- local API
- UI
- evaluation suite
- security testing
- network-isolation validation

These will be implemented in later blocks only after their dependencies are established.

## Repository structure

```text
ARIA/
├── pyproject.toml
├── .python-version
├── .gitignore
├── README.md
├── src/
│   └── aria/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── logging.py
│       ├── brain/
│       ├── tokenizer/
│       ├── training/
│       ├── inference/
│       ├── runtime/
│       ├── memory/
│       ├── rag/
│       ├── agent/
│       └── verification/
└── tests/
    └── test_foundation.py
```

Block 1 currently provides a dependency-free correctness-first tensor primitive with shape tracking, elementwise addition/multiplication, scalar reduction, reverse-mode autodiff, gradient accumulation, trainable parameters, and recursive parameter discovery. It is intentionally not a high-performance tensor backend and is not yet the ARIA SLM.

The package boundaries are intentionally lightweight. A directory does not count as an implemented subsystem until it contains real behavior and tests.

## Architecture direction

```text
User
 │
 ▼
UI
 │
 ▼
Local API
 │
 ▼
AI Orchestrator
 ├──────────────► Context / Memory
 ├──────────────► Local RAG
 ├──────────────► Agent / Tools
 └──────────────► Verification
                    │
                    ▼
              ARIA AI Runtime
                    │
                    ▼
              Inference Engine
                    │
                    ▼
              ARIA Neural Brain
                    │
                    ▼
               Own Trainable SLM
```

The neural brain is intended to contain real learned parameters and real computation: embeddings, Transformer layers, attention, normalization, feed-forward blocks, vocabulary output, loss, backpropagation, parameter updates, checkpointing, and local inference.

The first Transformer is deliberately small enough to inspect and train locally. It currently uses a correctness-first single-head causal attention implementation and is **EXPERIMENTAL** until numerical, training, performance, and language-quality evaluation are established. More advanced architecture and optimization will be added only when implemented and measured.

## Independence requirements

Normal ARIA operation must not require:

- OpenAI, Anthropic, Gemini, or other hosted model APIs
- cloud inference
- cloud embeddings
- hosted vector databases
- remote agent services
- hidden model calls
- required internet access
- required physical GPU/NPU hardware

Development internet access may be used for dependencies, datasets, documentation, or optional artifacts. The finished system must eventually pass an explicit network-isolation test.

## Engineering truth

ARIA uses explicit capability states:

- **NOT IMPLEMENTED** — planned but absent
- **EXPERIMENTAL** — implemented but not sufficiently validated
- **IMPLEMENTED** — working behavior with relevant tests
- **VALIDATED** — working behavior supported by reproducible evidence

A class, mock, prompt, button, or API wrapper is not treated as intelligence.

## Development sequence

1. **Block 0 — Repository & engineering foundation** — package, configuration, logging, tests. **IMPLEMENTED**
2. **Block 1 — Neural tensor/model foundation** — real local numerical and trainable-model primitives. **IMPLEMENTED**
3. **Block 2 — Neural layer foundation** — trainable Linear, ReLU, Embedding, and Sequential components with gradient propagation. **IMPLEMENTED**
4. **Block 3 — Tokenizer foundation** — deterministic UTF-8 byte tokenizer with special tokens and round-trip tests. **IMPLEMENTED**
5. **Block 4 — Tokenizer expansion** — ARIA-owned tokenizer for natural language, Hindi/Hinglish, code, mathematics, and technical text.
5. **Block 4 — Trainable language-model core** — token embeddings, vocabulary logits, next-token cross-entropy, backpropagation, and local SGD. **IMPLEMENTED**
6. **Block 5 — Decoder Transformer SLM foundation** — causal self-attention, RMSNorm, positional embeddings, residual MLP blocks, and vocabulary head. **EXPERIMENTAL** — evolve the model into an efficient decoder-only language model.
7. **Block 6 — Training engine** — reproducible local pretraining/instruction-training pipeline.
8. **Block 7 — Inference engine** — autoregressive generation, sampling, streaming, and caching.
9. **Block 8 — AI runtime** — model loading, execution, scheduling, and runtime state.
10. **Block 9 — Context & memory** — inspectable local working, conversational, semantic, and episodic memory.
11. **Block 10 — Local RAG** — document parsing, chunking, local indexing, retrieval, and context selection.
12. **Block 11 — Agent & tools** — validated schemas, permissions, sandboxing, limits, and audit logs.
13. **Block 12 — Verification** — deterministic checks for math, code, retrieval, structured data, and tool results.
14. **Block 13 — Local API** — stable local boundary between UI and intelligence.
15. **Block 14 — UI** — interaction and runtime transparency.
16. **Block 15 — Evaluation & security** — quality, correctness, performance, regression, injection, filesystem, command, and resource tests.
17. **Block 16 — Research & optimization** — only after evidence supports the optimization.

## Block workflow

Every block follows:

```text
Inspect
  ↓
Define acceptance criteria
  ↓
Implement the smallest correct change
  ↓
Test
  ↓
Review
  ↓
Update documentation/status
  ↓
Commit
  ↓
Next block
```

No large subsystem should be built on an unverified foundation.

## Long-term target

ARIA is intended to become a locally executable AI system capable of natural conversation, coding, mathematics, technical explanation, local document understanding, persistent local memory, local retrieval, controlled tool use, verification, measurable model improvement, model versioning, efficient local inference, and offline operation.

**Build the brain. Build the runtime. Measure both. Keep the boundaries honest.**
