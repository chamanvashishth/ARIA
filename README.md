# ARIA

<div align="center">

# ARIA

**Adaptive • Reasoning • Intelligent Assistant**

*A local-first AI assistant platform built around modular application architecture and an independently evolving small language model (SLM).*

[![CI](https://github.com/chamanvashishth/ARIA/actions/workflows/ci.yml/badge.svg)](https://github.com/chamanvashishth/ARIA/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Modular-6f42c1)
![Execution](https://img.shields.io/badge/Execution-Local--First-2ea44f)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)

</div>

---

## 1. Project overview

ARIA is being engineered as a **complete, local-first AI assistant foundation**, not as a single chatbot script.

The project separates the assistant into independently testable boundaries:

- **Application core** — orchestration and runtime rules
- **Interfaces** — user-facing request/response contracts
- **SLM** — model architecture, data preparation, tokenization, and inference primitives
- **Providers** — replaceable intelligence backends
- **Memory** — contextual and long-term assistant state
- **Storage** — durable local persistence
- **Configuration** — explicit runtime configuration
- **UI** — interaction surfaces such as the CLI
- **Tests and CI** — automated quality gates

The repository is intentionally developed in small checkpoints. A new layer is not considered complete merely because its code exists; it must be validated locally and through GitHub Actions before the project moves forward.

> **Current focus:** establishing a clean, testable SLM and assistant foundation.  
> **Execution model:** local-first.  
> **Deployment:** not required for the core project.

---

## 2. Product vision

ARIA is designed around a simple architectural principle:

> **The assistant should not depend on one model provider, one interface, or one storage implementation.**

The application layer should communicate through stable contracts. Intelligence can then evolve independently, from deterministic test doubles to a native SLM and, where explicitly required, additional provider adapters.

Long term, the project is intended to support:

- natural-language interaction,
- local model inference,
- contextual memory,
- conversation history,
- extensible tools,
- deterministic testing,
- configurable runtime behavior,
- privacy-oriented local execution,
- and a maintainable developer ecosystem.

The repository is currently in the foundation stage, so planned capabilities must not be interpreted as already implemented.

---

## 3. Architecture

### High-level system

```text
                           ┌──────────────────────┐
                           │       ARIA UI        │
                           │   CLI / Interfaces   │
                           └──────────┬───────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │       Application Core       │
                    │ lifecycle • orchestration    │
                    │ policies • runtime state     │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
       ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
       │     SLM     │      │   Memory    │      │   Storage   │
       │ intelligence│      │  context    │      │ local state │
       └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │   Infrastructure    │
                         │ config • errors      │
                         │ logging • utilities  │
                         └──────────────────────┘
```

### Intelligence boundary

The core application should depend on a contract rather than a concrete model implementation:

```text
                       ┌─────────────────┐
                       │   ARIA Core     │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Model Contract  │
                       └────────┬────────┘
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
              ARIA SLM     Provider Adapter  Test Double
```

This boundary allows the model implementation to change without rewriting the assistant's orchestration layer.

---

## 4. Current SLM foundation

The repository currently contains the beginning of a dependency-light SLM layer.

The SLM foundation includes contracts and primitives for:

- model configuration,
- model output representation,
- training examples and datasets,
- text preprocessing,
- vocabulary management,
- tokenization,
- example validation,
- model interfaces,
- and a deterministic baseline forward-pass implementation.

The current baseline model is intentionally small. It provides a real numerical forward pass behind the model contract while the project's final neural architecture is established.

It is **not** presented as the final ARIA language model.

### Current SLM flow

```text
Raw text
   │
   ▼
Preprocessing
   │
   ▼
Vocabulary / Tokenizer
   │
   ▼
Token IDs
   │
   ▼
SLM Model Contract
   │
   ▼
Baseline / Future Neural Architecture
   │
   ▼
Vocabulary Logits
```

The model layer is being developed independently from the assistant runtime so that training and inference work can evolve without coupling the entire application to one implementation.

---

## 5. Repository structure

The intended structure is:

```text
ARIA/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   └── aria/
│       ├── __init__.py
│       ├── __main__.py
│       │
│       ├── config/
│       │   └── ...
│       │
│       ├── core/
│       │   └── ...
│       │
│       ├── interfaces/
│       │   └── ...
│       │
│       ├── memory/
│       │   └── ...
│       │
│       ├── providers/
│       │   └── ...
│       │
│       ├── slm/
│       │   ├── config.py
│       │   ├── dataset.py
│       │   ├── model.py
│       │   ├── model_impl.py
│       │   ├── preprocessing.py
│       │   ├── tokenizer.py
│       │   ├── validation.py
│       │   └── vocabulary.py
│       │
│       ├── storage/
│       │   └── ...
│       │
│       └── ui/
│           └── ...
│
├── tests/
│   ├── integration/
│   └── unit/
│
├── pyproject.toml
├── README.md
└── .gitignore
```

Directories represent architectural boundaries. Empty or placeholder modules should not become permanent dumping grounds; functionality should be introduced when its responsibility and contract are clear.

---

## 6. Development philosophy

### Small checkpoints

Every meaningful change should be independently understandable and testable.

### Contracts before complexity

Interfaces and data contracts should be stable before adding complex implementations.

### Dependency-light foundations

The core foundation should avoid unnecessary framework dependencies. A dependency should have a clear technical reason to exist.

### Deterministic tests first

Core behavior should be testable without a network connection, external model API, paid service, or secret credential.

### Explicit failure

Invalid input and invalid configuration should fail clearly and close to the source of the problem.

### Local ownership

The core project should remain useful without requiring a hosted runtime.

---

## 7. Local development

### Requirements

The foundation targets:

- Python **3.11 or newer**
- Git
- a local virtual environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

---

## 8. Quality gates

Before a checkpoint is considered complete, run the applicable checks:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

If static type checking is enabled for a component, run the configured type checker as part of that component's checkpoint.

The exact command set is intentionally defined by the project's `pyproject.toml` and CI workflow so local and CI validation remain aligned.

---

## 9. Testing strategy

ARIA uses multiple test layers.

### Unit tests

Unit tests validate isolated contracts and components:

- configuration,
- data models,
- tokenization,
- vocabulary behavior,
- preprocessing,
- model contracts,
- deterministic model behavior,
- core orchestration.

### Integration tests

Integration tests validate multiple project boundaries together, such as CLI/runtime behavior.

### External integrations

Network- or provider-dependent tests should not be part of the default deterministic suite unless explicitly required. When introduced, they should be clearly classified and documented.

### Testing rule

A feature is not considered stable until its expected behavior is represented by tests.

---

## 10. Configuration and secrets

Configuration must remain separate from implementation code.

### Rules

- Never commit API keys or credentials.
- Never place secrets in `README.md`, tests, fixtures, or source files.
- Prefer environment variables or local configuration for secrets.
- Validate required configuration at startup.
- Keep provider-specific configuration at the provider boundary.
- Core imports and unit tests should not require live credentials.

A local example configuration may be documented without containing real secrets.

---

## 11. Security and privacy

Because ARIA is intended for local assistant workloads, privacy is treated as an architectural concern.

The project should:

- avoid logging secrets,
- avoid exposing credentials in exceptions,
- validate external input,
- keep filesystem access scoped,
- treat conversation and memory data as potentially sensitive,
- review dependencies before introducing them,
- and keep provider credentials outside version control.

Security-sensitive behavior should be covered by tests where practical.

---

## 12. Continuous integration

GitHub Actions is the project's automated quality gate.

The CI workflow is intended to validate supported Python versions and run reproducible checks such as:

- dependency installation,
- formatting validation,
- linting,
- automated tests.

The repository should not require a deployment platform for CI to be useful.

A checkpoint should not be considered complete until the corresponding CI run is reviewed.

---

## 13. Development workflow

The project follows this sequence:

```text
Requirement
    │
    ▼
Define contract
    │
    ▼
Implement smallest useful unit
    │
    ▼
Add deterministic tests
    │
    ▼
Run local quality checks
    │
    ▼
Commit focused change
    │
    ▼
Verify GitHub Actions
    │
    ▼
Proceed to next layer
```

This workflow makes regressions easier to locate and prevents unrelated changes from becoming one large debugging problem.

---

## 14. Roadmap

### Phase 0 — Foundation

- [x] Clean repository baseline
- [x] Establish project documentation
- [x] Establish Python package structure
- [x] Establish initial SLM contracts
- [x] Establish SLM data and preprocessing primitives
- [x] Establish deterministic baseline model
- [ ] Stabilize CI and developer tooling

### Phase 1 — Assistant core

- [ ] Runtime lifecycle
- [ ] Request/response orchestration
- [ ] Assistant state
- [ ] Error-handling strategy
- [ ] Core integration tests

### Phase 2 — SLM development

- [ ] Final model architecture
- [ ] Training pipeline
- [ ] Loss and optimization
- [ ] Batching and data loading
- [ ] Evaluation metrics
- [ ] Checkpointing
- [ ] Inference pipeline
- [ ] Resource-aware local execution

### Phase 3 — Memory and storage

- [ ] Memory abstraction
- [ ] Conversation history
- [ ] Local persistence
- [ ] Retrieval/context assembly
- [ ] Data migration strategy

### Phase 4 — User interface

- [ ] Stable CLI
- [ ] Command system
- [ ] Interactive conversation loop
- [ ] Human-readable errors
- [ ] Configuration commands

### Phase 5 — Extensibility

- [ ] Tool contracts
- [ ] Tool execution boundary
- [ ] Additional provider adapters where required
- [ ] Observability
- [ ] Performance profiling

### Phase 6 — Stabilization

- [ ] Security review
- [ ] Dependency review
- [ ] Failure-mode testing
- [ ] Documentation review
- [ ] Performance review
- [ ] Release-readiness checklist

The roadmap is intentionally staged. Requirements may be refined when the authoritative project specification is incorporated.

---

## 15. Contributing

Contributions should preserve clear architectural boundaries.

Before submitting a change:

1. Keep the change focused.
2. Define or preserve the relevant contract.
3. Add tests for behavior that can regress.
4. Run formatting, linting, and tests locally.
5. Review the GitHub Actions result.
6. Update documentation when behavior or architecture changes.

Avoid committing generated files, local environments, credentials, or unrelated refactors.

---

## 16. Project status

ARIA is under active development.

The repository is currently focused on establishing the engineering foundation required for a reliable local assistant and native SLM implementation.

**Implemented foundation:**

- modular Python package layout,
- assistant/interface boundaries,
- initial SLM contracts,
- text preprocessing,
- vocabulary/tokenization primitives,
- training-data validation,
- deterministic baseline model,
- unit and integration test foundations.

**Still under development:**

- complete assistant runtime,
- production-quality SLM architecture,
- training and evaluation pipeline,
- persistent memory,
- full CLI experience,
- tool ecosystem,
- performance and security hardening.

---

## 17. License

No open-source license has been finalized for this repository yet.

Until a license is explicitly added, repository contents should not be assumed to be licensed for unrestricted redistribution or reuse.
