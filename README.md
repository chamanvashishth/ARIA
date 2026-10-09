# ARIA

<p align="center">
  <strong>Autonomous Research & Intelligence Architecture</strong>
</p>

<p align="center">
  A software-only, local-first AI system built from first principles around an ARIA-owned neural brain.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active%20development-0f766e?style=for-the-badge" alt="Active development">
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/inference-local-first-111827?style=for-the-badge" alt="Local first">
  <img src="https://img.shields.io/badge/external%20AI%20APIs-not%20required-7c3aed?style=for-the-badge" alt="No external AI APIs required">
</p>

---

## What is ARIA?

ARIA is being engineered as an **independent local AI stack**, rather than a wrapper around an existing hosted model.

The project is deliberately built in small, inspectable blocks:

**tensor engine → neural layers → tokenizer → trainable language model → Transformer SLM → training → inference → runtime → memory → RAG → tools → verification → API → UI → evaluation → security**

> **If a capability is not implemented, tested, and measurable, ARIA does not claim to have it.**

ARIA's normal runtime is designed to work without OpenAI, Anthropic, Gemini, hosted inference, hosted embeddings, or a hosted vector database.

---

## Project at a glance

| Area | Current state |
|---|---|
| Python package foundation | **Implemented** |
| Tensor + reverse-mode autodiff | **Implemented** |
| Trainable neural layers | **Implemented** |
| Deterministic byte tokenizer | **Implemented** |
| Trainable language-model core | **Implemented** |
| Decoder Transformer SLM | **Experimental** |
| Local training engine | **Implemented — foundation** |
| Local autoregressive inference | **Implemented — foundation** |
| Local AI runtime | **Implemented — foundation** |
| Local conversation memory | **Implemented — foundation** |
| Local RAG | **Implemented — foundation** |
| Agent / tools | **Implemented — controlled tool registry** |
| Verification | **Implemented — structural checks foundation** |
| Local API | **Implemented — loopback HTTP foundation** |
| Local UI | **Implemented — browser interface foundation** |
| Evaluation metrics | **Implemented — foundation** |
| Security hardening | **Implemented — local foundation** |

### Capability states

- **Implemented** — working behavior with relevant tests.
- **Experimental** — implemented, but not yet sufficiently validated.
- **Planned** — architecture/dependency exists, implementation has not started.
- **Validated** — reserved for capabilities backed by reproducible evidence.

---

## Architecture

### Target system

~~~mermaid
flowchart TD
    U[User] --> UI[ARIA UI]
    UI --> API[Local API]
    API --> ORCH[AI Orchestrator]

    ORCH --> MEM[Local Memory]
    ORCH --> RAG[Local RAG]
    ORCH --> AGENT[Controlled Tool Registry]
    ORCH --> VERIFY[Verification]

    ORCH --> RUNTIME[ARIA Runtime]
    RUNTIME --> INF[Inference Engine]
    INF --> BRAIN[ARIA Neural Brain]
    BRAIN --> SLM[Own Trainable SLM]
~~~

### Current neural pipeline

~~~mermaid
flowchart LR
    TEXT[Training Text] --> TOK[Byte Tokenizer]
    TOK --> DATA[Token Windows]
    DATA --> EMB[Token Embeddings]
    EMB --> TR[Transformer Blocks]
    TR --> HEAD[Vocabulary Head]
    HEAD --> LOSS[Next-Token Loss]
    LOSS --> BACK[Backpropagation]
    BACK --> OPT[SGD]
    OPT --> PARAMS[Updated Parameters]
~~~

The diagrams above show the **architecture direction** and the **currently implemented training path**. Components such as inference, memory, RAG, agent orchestration, API, and UI are not represented as completed just because they appear in the target architecture. Verification currently covers deterministic structural checks only; evaluation metrics now include next-token loss, perplexity, accuracy, and finite-value audits.

---

## Neural brain

The current brain is intentionally small and inspectable rather than optimized prematurely.

### Implemented foundations

- Tensor
  - shape tracking
  - rectangular validation
  - elementwise operations
  - reductions
  - reverse-mode autodiff
  - gradient accumulation
- Parameter
  - trainable model values
- Module
  - recursive parameter discovery
  - gradient reset
- Linear
- ReLU
- Embedding
- Sequential
- RMSNorm
- causal self-attention
- feed-forward blocks
- residual Transformer blocks
- vocabulary projection
- next-token cross-entropy
- local SGD

### Transformer status

The current Transformer is intentionally marked **EXPERIMENTAL**.

It currently provides:

- decoder-style causal attention
- single-head attention
- Q/K/V projections
- causal masking
- learned positional embeddings
- RMS normalization
- residual feed-forward blocks
- configurable depth and hidden/intermediate sizes
- vocabulary output

It is **not yet claimed** to be production-ready, performant, or language-quality validated.

---

## Tokenizer

ARIA currently uses a deterministic UTF-8 byte-level tokenizer.

### Vocabulary

| ID range | Meaning |
|---:|---|
| 0–255 | Raw UTF-8 byte values |
| 256 | PAD |
| 257 | BOS |
| 258 | EOS |
| 259 | UNK |

**Vocabulary size: 260**

This foundation works across English, Unicode text, Hindi/Hinglish, source code, symbols, and technical content without depending on an external tokenizer service.

The tokenizer is intentionally simple at this stage. More advanced tokenization can be introduced later when training and evaluation data justify it.

---

## Training engine

Block 6 establishes the first complete local training path around the Transformer foundation:

~~~text
Text
 ↓
Tokenization
 ↓
Sequence windows
 ↓
Input / next-token targets
 ↓
Transformer forward pass
 ↓
Cross-entropy loss
 ↓
Backpropagation
 ↓
Batched forward pass
 ↓
Mean batch loss
 ↓
Backpropagation through the batch
 ↓
One SGD parameter update
 ↓
Loss history
 ↓
Model-weight checkpoint
~~~

### Current capabilities

- deterministic token-window datasets
- deterministic train/validation token-stream splitting before window creation
- train/validation evaluation reports with loss change and generalization-gap metrics
- best-validation checkpointing with configurable patience and minimum improvement
- early stopping based on validation loss
- deterministic exponential and step learning-rate schedules
- configurable mini-batch training with averaged gradients
- batched TinyLanguageModel execution across equal-length examples
- Transformer execution with `[batch, time, hidden]` activations, flattened shared Linear projections, and independent causal attention per sample
- batched loss and gradient-equivalence checks against independent sequence execution
- deterministic batch sampler with seeded per-epoch shuffling
- optional `drop_last` behavior for fixed-size batches
- complete-dataset epoch semantics with a final partial batch when needed
- configurable sequence length
- configurable stride
- next-token target generation
- leakage-resistant train/validation window boundaries
- training step tracking
- resume-aware dataset ordering
- gradient reset
- forward/backward training
- SGD parameter updates
- loss history
- experiment configuration metadata
- versioned JSON model-weight checkpoints
- atomic checkpoint replacement
- named parameter values and shapes
- optimizer learning-rate state
- model restoration with parameter-name and shape validation
- trainer step restoration for continued training
- deterministic continuation from the saved dataset position
- checkpoint/trainer sequence-length compatibility validation

Checkpoint files remain human-readable JSON. Metadata-only checkpoints from the earlier format remain loadable, but they cannot restore model weights; attempting to resume from one fails explicitly.

---

## Inference engine

Block 7 establishes the first local autoregressive generation path.

~~~text
Prompt tokens
    ↓
Model forward pass
    ↓
Last-position logits
    ↓
Temperature scaling
    ↓
Optional top-k filtering
    ↓
Token sampling
    ↓
Append token
    ↓
Repeat until limit / EOS
~~~

### Current capabilities

- local autoregressive generation
- context-window enforcement
- temperature sampling
- optional top-k sampling
- deterministic generation with a seed
- EOS-aware early stopping
- reusable `generate()` and `sample_next_token()` APIs

The generator operates directly on an ARIA-compatible local model interface. It does **not** call an external model provider.

The inference layer is still a **foundation**: streaming, KV caching, efficient decoding, batching, advanced sampling, and production performance validation remain future work.

---

## AI runtime

Block 8 adds the first runtime boundary around ARIA's local model and inference stack.

~~~text
Application
    ↓
ARIA Runtime
    ├── Model lifecycle
    ├── Tokenizer
    ├── Inference configuration
    └── Generation metrics
            ↓
      Local Transformer
~~~

### Current capabilities

- explicit runtime start/stop lifecycle
- local model ownership
- tokenizer ownership
- text-to-token generation boundary
- runtime generation configuration
- generation timing metadata
- structured generation results
- no network dependency in the runtime path

The runtime is deliberately small. Scheduling, model loading, persistent runtime state, batching, resource management, streaming, and production observability remain future work.

---

## Context & memory

Block 9 adds an optional local conversation-memory store. It is deliberately separate from model weights and inference logic.

### Current capabilities

- append-only JSON Lines persistence
- in-memory operation when no file path is configured
- records for user, assistant, system, and tool roles
- UTC timestamps and string metadata
- recent-entry retrieval
- case-insensitive literal substring search
- explicit clear operation
- optional runtime integration to record prompts and generated responses

Example:

~~~python
from pathlib import Path
from aria.memory import LocalMemoryStore

memory = LocalMemoryStore(Path("data/conversations.jsonl"))
memory.add("user", "I am studying quantum computing")
print(memory.search("quantum"))
~~~

**Important limitation:** this is persistent conversation history, not semantic memory. Search is literal substring matching; it does not perform embeddings, semantic retrieval, summarization, automatic fact extraction, or long-term context injection into the model. Those capabilities require separate implementation and evaluation.

---

## Local RAG

Block 10 adds a local retrieval foundation for plain text supplied by the caller.

### Current capabilities

- deterministic character-window chunking with configurable overlap
- stable chunk IDs for identical source/index/content input
- source labels and caller-provided metadata carried into chunks
- dependency-free BM25-style lexical ranking
- top-k retrieval with deterministic tie ordering
- context assembly with source/chunk labels and a character cap
- no hosted embeddings, vector database, or model API dependency

Example:

~~~python
from aria.rag import LocalRAGPipeline, TextDocument

rag = LocalRAGPipeline(top_k=3)
rag.add_document(
    TextDocument(
        source="notes/quantum.txt",
        text="A qubit is the basic unit of quantum information.",
    )
)
print(rag.build_context("qubit quantum information"))
~~~

**Important limitations:** this is lexical retrieval, not semantic or vector search. It ingests already-extracted text; it does not parse PDFs, crawl websites, create embeddings, generate answers, validate citations, or guarantee that a downstream model uses retrieved context. Context assembly is exposed explicitly rather than automatically inserted into runtime prompts, because model context limits and grounding behavior still need evaluation.

---

## Agent & tools

Block 11 adds a controlled local tool registry. Tools are trusted callables registered explicitly by application code; ARIA does not dynamically import arbitrary functions, evaluate code, or run shell commands automatically.

### Current capabilities

- explicit tool registration with name and description
- object-shaped input contracts with required fields
- basic type checks and optional enum checks
- rejection of unexpected arguments by default
- deterministic tool listing
- structured success and error results
- handler exceptions converted to error results
- registered input schemas copied at registration and copied when exposed through registry accessors, preventing callers from mutating the stored validation contract

Example:

~~~python
from aria.agent import ToolRegistry

tools = ToolRegistry()
tools.register(
    "add",
    "Add two integers",
    lambda left, right: left + right,
    {
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
        "additionalProperties": False,
    },
)

result = tools.invoke("add", {"left": 2, "right": 3})
if result.success:
    print(result.output)
else:
    print(result.error)
~~~

**Important limitation:** this is a tool execution boundary, not yet an autonomous agent or planner. It does not decide which tool to call, loop over tool results, ask for user approval, sandbox untrusted code, or provide OS-level permission controls. Register only handlers that the application explicitly trusts.

---

## Verification

Block 12 adds a dependency-free verification layer for checking output contracts and collecting named check results. It can validate a tool result's required fields, success/error consistency, optional output type, and application-defined checks.

### Current capabilities

- structured `CheckResult` and aggregate `VerificationReport`
- aggregate pass/fail status, failed-check access, and concise summary
- reusable checks for Python types, required mapping keys, and non-empty strings
- `verify_tool_result()` for structural validation of tool invocation outcomes
- `VerificationSuite` for named checks, including exception-to-failure conversion
- tests covering passing checks, contract mismatches, malformed results, and failing checks

Example:

~~~python
from aria.agent import ToolRegistry
from aria.verification import verify_tool_result

tools = ToolRegistry()
tools.register("greet", "Return a greeting", lambda name: f"Hello, {name}")
result = tools.invoke("greet", {"name": "ARIA"})
report = verify_tool_result(result, expected_output_type=str)

if report.passed:
    print(result.output)
else:
    print(report.summary)
    for failure in report.failed_checks:
        print(failure.name, failure.message)
~~~

**Important limitation:** these checks validate structure and declared contracts, not whether a response is factually correct, complete, unbiased, safe, or grounded in source material. Custom predicates are only as reliable as their implementation. This is not a complete evaluation framework, a security guarantee, or a substitute for human review.

---

## Evaluation metrics

Block 15 adds a small, dependency-free evaluation foundation for local language models.

### Current capabilities

- aggregate next-token cross-entropy across evaluation examples
- perplexity derived from mean loss, with overflow handled as infinity
- exact next-token accuracy
- token and example counts in a structured result
- rejection of empty datasets, mismatched output dimensions, and non-finite logits/loss
- parameter and accumulated-gradient audits for non-finite values
- central finite-difference gradient checks against reverse-mode autodiff, with bounded checks for larger parameter sets
- forward-only evaluation: no backward pass, optimizer step, or gradient reset

Example:

~~~python
from aria.brain import TransformerLanguageModel
from aria.evaluation import evaluate_language_model, inspect_parameter_health

model = TransformerLanguageModel(
    vocab_size=260,
    hidden_size=16,
    intermediate_size=32,
    max_sequence_length=8,
    seed=7,
)

report = evaluate_language_model(model, [
    ([10, 11, 12], [11, 12, 13]),
])
print(report.mean_loss, report.perplexity, report.token_accuracy)

health = inspect_parameter_health(model)
print(health.healthy, health.non_finite_values, health.non_finite_gradients)
~~~

**Limitations:** these metrics measure next-token behavior on the examples supplied by the caller; they do not establish general language quality, factuality, safety, benchmark competitiveness, or training convergence. Gradient checks validate only the parameter values and deterministic loss path selected for a run; they do not establish correctness for every model operation. Reproducible performance benchmarks, representative held-out datasets, and broader quality evaluation are still needed. Checkpoint serialization and resume are now implemented as a foundation: weights, shapes, optimizer learning rate, and trainer step are persisted and restored. Optimizers with additional mutable state, distributed checkpoints, sharded formats, atomic checkpoint rotation, and production-scale binary formats remain future work.

---

## Research & optimization

Block 17 adds early diagnostics for validating learning behavior and collecting local performance measurements.

### Reproducible tiny-corpus check

A deterministic test trains the small foundation language model on a deliberately repetitive token pattern and compares next-token loss before and after training. This is a learning-pipeline sanity check only; it does not demonstrate language quality, generalization, or useful assistant behavior.

Run it with:

~~~bash
pytest tests/test_training.py
~~~

### Local inference benchmark

The dependency-free benchmark script creates a small, untrained Transformer and measures autoregressive forward-pass latency on the current machine. It also compares fixed-length batched forward execution with running the same sequences individually, checks logit equivalence, and reports median latency, throughput, and a median speedup ratio:

~~~bash
python scripts/benchmark_inference.py --warmup 2 --iterations 10 --prompt-length 8 --new-tokens 4 --batch-size 4 --batch-sequence-length 8 --sweep
~~~

The JSON output separates autoregressive inference measurements from the forward-only batch comparison. With `--sweep`, it also measures batch sizes `1, 2, 4, 8` at sequence lengths `4, 8, 16`, reporting logit-equivalence checks and throughput for each combination. A speedup ratio above `1` means the batched path was faster for that run; it is not a universal performance claim. Increase iterations for less noisy measurements and compare results only when the workload and environment are recorded consistently.

**Limitations:** the benchmark uses an untrained model, measures a small pure-Python implementation, and is not comparable to optimized inference engines. Run it on the target machine and retain the environment metadata before drawing performance conclusions. The comparison excludes backpropagation and optimizer updates. Numerical gradient checks cover only the selected values and loss paths supplied by the caller.


### Transformer component profiler

To compare where time is spent in isolated Linear and causal-attention forward/backward paths versus a full Transformer training step, run:

~~~bash
python scripts/profile_transformer.py --warmup 1 --iterations 5 --batch-size 2 --sequence-length 8
~~~

The JSON report separates forward-plus-loss time from backward time and includes Python/platform metadata. It does not include optimizer updates, and isolated component timings are diagnostic rather than a strict additive breakdown of full-model runtime. Run on the target machine and increase iterations before interpreting small differences.

---

## Local API

Block 13 adds a dependency-free HTTP interface around an existing `AriaRuntime`. It uses Python's standard library and does not download a model, create one automatically, or call an external AI provider.

### Endpoints

| Method | Route | Behavior |
|---|---|---|
| `GET` | `/health` | Returns service and runtime state |
| `POST` | `/generate` | Validates a prompt/config and returns local generation output |

Example request:

~~~http
POST /generate HTTP/1.1
Content-Type: application/json

{
  "prompt": "Explain a qubit in simple terms.",
  "config": {
    "max_new_tokens": 32,
    "temperature": 0.8,
    "top_k": 20,
    "seed": 7
  }
}
~~~

### Start the server from Python

Create and train/load your own compatible model first, then inject the runtime:

~~~python
from aria.api import create_api_server
from aria.runtime import AriaRuntime

# model must be an ARIA-compatible model instance
runtime = AriaRuntime(model)
runtime.start()

server = create_api_server(runtime, host="127.0.0.1", port=8765)
try:
    server.serve_forever()
finally:
    server.server_close()
    runtime.stop()
~~~

The server factory does not create or train a model for you. The caller owns the runtime lifecycle and should stop the server cleanly.

### Request protections and limitations

- loopback binding by default (`127.0.0.1`)
- bounded request body (64 KiB by default) and prompt length (16,384 characters)
- JSON content-type and UTF-8 JSON validation
- strict allowed fields and generation parameter checks
- duplicate JSON object keys and non-standard NaN/Infinity constants rejected
- `nosniff`, frame-denial, referrer, and permissions response headers
- structured JSON errors and no prompt text in standard access logs
- no CORS policy or authentication layer

**Security note:** this remains a local development API, not a production internet-facing service. The request checks and browser headers reduce common input and browser risks but do not provide authentication, authorization, rate limiting, TLS, connection/thread quotas, or protection against all denial-of-service conditions. Do not bind it to a public interface without deployment-specific hardening. Generation remains limited by the current experimental model and inference engine.

---

## Local browser UI

Block 14 adds a lightweight chat interface served from the local API root. Open `http://127.0.0.1:8765/` after starting the API server to use it.

### Included

- responsive chat layout with user and assistant messages
- connection status from `/health`
- generation controls for maximum new tokens, temperature, and top-k
- request/error states and generation timing/token metrics
- sample prompts, clear-visible-conversation action, and keyboard-friendly composer
- static HTML/CSS/JavaScript with no frontend build step or third-party browser dependencies
- packaged UI asset included through setuptools package data

The interface sends prompts to the same-origin `/generate` endpoint. It does not save chat history to disk, and clearing the visible conversation does not clear ARIA's optional memory store.

**Current limitation:** this is a UI foundation, not a complete desktop application. It does not provide authentication, user profiles, streaming output, persistent conversation history, or model management. Response quality depends on the model injected into the runtime; an untrained or minimally trained model may return incoherent text.

---

## Repository structure

~~~text
ARIA/
├── pyproject.toml
├── .python-version
├── .gitignore
├── README.md
│
├── src/
│   └── aria/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── logging.py
│       │
│       ├── brain/
│       │   ├── tensor.py
│       │   ├── parameter.py
│       │   ├── module.py
│       │   ├── layers.py
│       │   ├── optim.py
│       │   ├── language_model.py
│       │   └── transformer.py
│       │
│       ├── tokenizer/
│       │   └── core.py
│       │
│       ├── training/
│       │   ├── dataset.py
│       │   ├── trainer.py
│       │   └── checkpoint.py
│       │
│       ├── inference/
│       ├── api/
│       │   ├── __init__.py
│       │   └── server.py
│       ├── ui/
│       │   └── index.html
│       ├── runtime/
│       ├── memory/
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── documents.py
│       │   ├── chunker.py
│       │   ├── retriever.py
│       │   └── pipeline.py
│       ├── agent/
│       │   ├── __init__.py
│       │   └── tools.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── core.py
│       ├── verification/
│       │   ├── __init__.py
│       │   └── core.py
│
└── tests/
    ├── test_foundation.py
    ├── test_tensor.py
    ├── test_layers.py
    ├── test_tokenizer.py
    ├── test_language_model.py
    ├── test_transformer.py
    ├── test_training.py
    ├── test_checkpoint.py
    ├── test_inference.py
    ├── test_runtime.py
    ├── test_memory.py
    ├── test_rag.py
    ├── test_agent.py
    ├── test_evaluation.py
    ├── test_verification.py
    └── test_api.py
~~~

Empty subsystem directories are architectural boundaries, **not completed features**.

---

## Engineering principles

### 1. Build from the foundation upward

Every major subsystem depends on verified lower-level behavior.

### 2. Local-first by design

Normal operation should not depend on a hosted AI provider.

### 3. No fake intelligence

A prompt wrapper, mock response, API proxy, UI button, or placeholder class is not treated as an AI capability.

### 4. Measure before optimizing

Performance work comes after correctness, reproducibility, and evaluation.

### 5. Keep boundaries explicit

The model, tokenizer, training system, inference engine, runtime, memory, RAG, tools, verification, API, and UI are separate engineering concerns.

### 6. Reproducibility matters

Training configuration, deterministic datasets, loss history, and experiment metadata should remain inspectable.

---

## Independence requirements

ARIA's intended normal runtime must not require:

- OpenAI or other hosted model APIs
- cloud inference
- cloud embeddings
- hosted vector databases
- remote agent services
- hidden model calls
- mandatory internet access
- mandatory GPU/NPU hardware

Internet access may still be useful during development for package installation, documentation, datasets, or optional artifacts.

The finished system will require an explicit **network-isolation validation** before offline operation can be claimed.

---

## Development roadmap

~~~mermaid
timeline
    title ARIA Development Roadmap
    Block 0 : Engineering foundation : Implemented
    Block 1 : Tensor and autodiff : Implemented
    Block 2 : Neural layers : Implemented
    Block 3 : Tokenizer foundation : Implemented
    Block 4 : Trainable LM core : Implemented
    Block 5 : Transformer SLM : Experimental
    Block 6 : Training engine : Implemented foundation
    Block 7 : Inference engine : Implemented foundation
    Block 8 : AI runtime : Implemented foundation
    Block 9 : Context and memory : Implemented foundation
    Block 10 : Local RAG : Implemented foundation
    Block 11 : Agent and tools : Implemented foundation
    Block 12 : Verification : Implemented foundation
    Block 13 : Local API : Implemented foundation
    Block 14 : UI : Implemented foundation
    Block 15 : Evaluation metrics : Implemented foundation
    Block 16 : Security hardening : Implemented foundation
    Block 17 : Research and optimization : Implemented foundation
~~~

### Block workflow

~~~mermaid
flowchart LR
    A[Inspect] --> B[Acceptance Criteria]
    B --> C[Implement]
    C --> D[Test]
    D --> E[Review]
    E --> F[Update Docs]
    F --> G[Commit]
    G --> H[Next Block]
~~~

Every block should leave the repository in a more usable and more measurable state.

---

## Quick start

### Requirements

- Python 3.11+
- Git
- No external AI API key required

### Clone

~~~bash
git clone https://github.com/chamanvashishth/ARIA.git
cd ARIA
~~~

### Create an environment

Windows:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

Linux/macOS:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
~~~

### Install

~~~bash
python -m pip install -e .
python -m pip install pytest
~~~

### Run tests

~~~bash
pytest
~~~

The repository's test suite is intended to protect the low-level neural, tokenizer, Transformer, and training foundations as ARIA grows. GitHub Actions runs `python -m pytest -ra` on pushes to `main`, pull requests, and manual workflow dispatches.

---

## What ARIA is — and is not

### ARIA is

- a from-scratch AI engineering project
- local-first
- designed around its own trainable neural brain
- modular
- inspectable
- test-driven
- built incrementally
- intended to support offline operation

### ARIA is not yet

- a finished general-purpose AI assistant
- a production-grade SLM
- an autonomous agent with planning, approvals, or sandboxed tool execution
- a complete semantic or production-grade RAG system
- a complete persistent-memory system
- a production inference runtime
- a production UI
- a benchmark-proven replacement for established language models

Those claims require implementation and evidence.

---

## Contributing

ARIA is being built block-by-block so contributors can understand the system before changing it.

A useful contribution should:

1. preserve subsystem boundaries;
2. include tests for new behavior;
3. avoid unnecessary external AI dependencies;
4. document meaningful architectural changes;
5. avoid claiming capabilities that are not actually implemented;
6. keep the smallest correct implementation first.

For larger changes, explain the problem, proposed boundary, acceptance criteria, and validation approach before expanding the subsystem.

---

## Project philosophy

> **Build the brain. Build the runtime. Measure both. Keep the boundaries honest.**

ARIA is a long-term engineering project. The goal is not to make a convincing demo first; the goal is to build an AI system whose important behavior can be inspected, tested, measured, and improved.

---

<p align="center">
  <strong>ARIA — building intelligence from the foundation up.</strong>
</p>
