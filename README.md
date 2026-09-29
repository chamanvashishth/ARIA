<div align="center">

# ARIA

### **Adaptive • Reasoning • Intelligent Assistant**

*A local-first AI assistant engineered for modular intelligence, private execution, and extensible capabilities.*

[![CI](https://github.com/chamanvashishth/ARIA/actions/workflows/ci.yml/badge.svg)](https://github.com/chamanvashishth/ARIA/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Modular-6f42c1)
![Execution](https://img.shields.io/badge/Execution-Local--First-2ea44f)
![License](https://img.shields.io/badge/License-TBD-lightgrey)

**A production-oriented foundation for building a private, extensible AI assistant and its own SLM.**

</div>

---

## Product Vision

ARIA is designed as a complete assistant platform rather than a single chatbot script.

The system separates **conversation, orchestration, intelligence, memory, storage, configuration, and user interaction** so each part can be developed, tested, replaced, and improved independently.

The long-term architecture is centered around a locally runnable **Small Language Model (SLM)**, surrounded by a reliable application layer.

<div align="center">

### ARIA at a glance

| Layer | Responsibility |
|:---:|---|
| **UI** | Conversation and user interaction |
| **Core** | Assistant orchestration and application rules |
| **SLM** | Local language understanding and generation |
| **Memory** | Context and long-term assistant state |
| **Storage** | Durable application data |
| **Infrastructure** | Configuration, logging, errors, utilities |

</div>

---

# System Architecture

<div align="center">

```text
                         ┌──────────────────────┐
                         │        ARIA UI       │
                         │   CLI / Interfaces   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │       Application Core       │
                    │ orchestration • policies     │
                    │ state • request lifecycle    │
                    └──────────────┬───────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │                           │
                     ▼                           ▼
            ┌─────────────────┐         ┌─────────────────┐
            │  Intelligence   │         │     Memory      │
            │                 │         │                 │
            │   ARIA SLM      │         │ context/history │
            │   providers     │         │ retrieval       │
            └────────┬────────┘         └────────┬────────┘
                     │                           │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │     Persistence      │
                         │ storage / state      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Infrastructure     │
                         │ config • logging     │
                         │ errors • utilities   │
                         └──────────────────────┘
```

</div>

### Architectural principle

ARIA Core never needs to know how intelligence is implemented.

```text
                     ARIA Core
                         │
                         ▼
                ┌─────────────────┐
                │ Responder       │
                │ Contract        │
                └────────┬────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           ARIA SLM   Provider   Test Double
```

This boundary makes the application testable without a trained model and allows the intelligence layer to evolve independently.

---

# Core Capabilities

<div align="center">

| Capability | Purpose | State |
|:---|:---|:---:|
| **Conversational Core** | Message and response orchestration | Ready |
| **Modular Architecture** | Independent subsystem boundaries | Ready |
| **Local Execution** | Development without mandatory hosted infrastructure | Ready |
| **Provider Boundary** | Replaceable model/responder integration | Ready |
| **Automated Validation** | Tests, linting, formatting and CI | Ready |
| **SLM Layer** | Native model intelligence | Building |
| **Memory System** | Persistent context and memory | Building |
| **Interactive CLI** | Full conversational experience | Building |
| **Production Hardening** | Security, performance and reliability | Building |

</div>

> **Important:** “Building” describes planned product layers, not missing documentation. The repository is intentionally developed checkpoint-by-checkpoint so every layer can be validated before the next one is added.

---

# Intelligence Architecture

ARIA is being designed around its own SLM rather than permanently coupling the assistant to one external model vendor.

<div align="center">

```text
                    User Request
                         │
                         ▼
                ┌─────────────────┐
                │  ARIA Assistant │
                └────────┬────────┘
                         │
                         ▼
                 Responder Contract
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
       Local ARIA SLM          External Adapter
            │
            ▼
       ┌─────────────┐
       │ Tokenizer   │
       ├─────────────┤
       │ Embeddings  │
       ├─────────────┤
       │ Transformer │
       ├─────────────┤
       │ Decoder     │
       └──────┬──────┘
              │
              ▼
         Generated Text
```

</div>

The model subsystem will be independently testable and will not be required for basic application tests.

---

# Conversation Flow

<div align="center">

```text
┌──────────┐
│   User   │
└────┬─────┘
     │ message
     ▼
┌──────────────┐
│ ARIA UI      │
└────┬─────────┘
     │
     ▼
┌──────────────┐
│ ARIA Core    │
└────┬─────────┘
     │
     ├──────────────► Memory / Context
     │
     ▼
┌──────────────┐
│ Responder    │
└────┬─────────┘
     │
     ▼
┌──────────────┐
│ SLM / Model  │
└────┬─────────┘
     │ response
     ▼
┌──────────────┐
│ ARIA Core    │
└────┬─────────┘
     │
     ▼
┌──────────────┐
│     User     │
└──────────────┘
```

</div>

---

# Project Structure

```text
ARIA/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   └── aria/
│       ├── __init__.py
│       ├── __main__.py
│       │
│       ├── config/          # Runtime configuration boundary
│       ├── core/            # Assistant orchestration
│       ├── interfaces/      # Shared application contracts
│       ├── memory/          # Memory subsystem
│       ├── providers/       # Model/provider adapters
│       ├── storage/         # Persistence layer
│       └── ui/              # User-facing interfaces
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── pyproject.toml
├── .gitignore
└── README.md
```

The directory structure is intentionally simple. New complexity is introduced only when a feature requires it.

---

# Engineering Standards

ARIA follows a small set of non-negotiable engineering rules.

### 01 — Contract before implementation

Define the interface, expected behavior, invariants, and failure cases first.

### 02 — One feature per checkpoint

Each development step solves one coherent problem.

### 03 — Tests accompany behavior

A feature should receive appropriate automated tests in the same implementation cycle.

### 04 — Intelligence stays behind an interface

Model-specific implementation must not leak into ARIA Core.

### 05 — Local-first

The core development and testing loop must remain usable locally.

### 06 — CI is a gate

A commit is not considered validated until its GitHub Actions run has completed successfully.

### 07 — No artificial completeness

Documentation, tests, and status must reflect the actual implementation rather than claiming unfinished systems are complete.

---

# Local Development

## Requirements

- Git
- Python 3.11+
- pip
- virtual environment

CI currently validates:

- Python 3.11
- Python 3.12
- Python 3.13

## Installation

```bash
git clone https://github.com/chamanvashishth/ARIA.git
cd ARIA

python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

---

# Running ARIA

The current application entry point is intentionally minimal and validates the package foundation:

```bash
python -m aria
```

Expected:

```text
ARIA 0.1.0
```

The installed console command is also available:

```bash
aria
```

The interactive intelligence experience is introduced progressively as the SLM and assistant layers mature.

---

# Quality & CI

## Local checks

Run tests:

```bash
python -m pytest
```

Run linting:

```bash
python -m ruff check .
```

Check formatting:

```bash
python -m ruff format --check .
```

Format code:

```bash
python -m ruff format .
```

## CI pipeline

```text
             ┌───────────────┐
             │     Commit    │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ GitHub Actions│
             └───────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Python      Python      Python
        3.11        3.12        3.13
          │          │          │
          └──────────┼──────────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
           Ruff          Pytest
              │             │
              └──────┬──────┘
                     ▼
                 Validated
```

The repository's latest CI baseline is green before new feature work proceeds.

---

# Security

Security is part of the architecture, not a final checklist.

Never commit:

- API keys
- access tokens
- passwords
- private keys
- production credentials
- authentication cookies
- private user data

External integrations must use explicit configuration boundaries, and tests must not depend on production secrets.

---

# Roadmap

<div align="center">

### ARIA Development Path

```text
Foundation
    │
    ▼
Assistant Core
    │
    ▼
SLM Foundation
    │
    ▼
Tokenizer + Dataset
    │
    ▼
Model Architecture
    │
    ▼
Training + Checkpoints
    │
    ▼
Inference
    │
    ▼
Memory + Persistence
    │
    ▼
Interactive CLI
    │
    ▼
Advanced Assistant
    │
    ▼
Production Hardening
```

</div>

### Foundation

- [x] Repository reset
- [x] Project documentation
- [x] Python package structure
- [x] Project metadata
- [x] Development tooling
- [x] Initial tests
- [x] GitHub Actions CI

### Assistant Core

- [x] Message contract
- [x] Responder boundary
- [x] Assistant orchestration
- [x] Error handling tests
- [ ] Typed configuration
- [ ] Structured application errors
- [ ] Application lifecycle

### SLM

- [ ] Model configuration
- [ ] Tokenizer contract
- [ ] Vocabulary system
- [ ] Dataset contract
- [ ] Data preprocessing
- [ ] Training validation
- [ ] Model architecture
- [ ] Forward pass
- [ ] Loss
- [ ] Training loop
- [ ] Checkpointing
- [ ] Inference
- [ ] Evaluation

### Memory & Storage

- [ ] Storage interface
- [ ] Conversation persistence
- [ ] Memory records
- [ ] Memory lifecycle
- [ ] Retrieval/context

### Experience

- [ ] Interactive CLI
- [ ] Conversation sessions
- [ ] Command handling
- [ ] Configuration commands
- [ ] Terminal UX

### Production

- [ ] Security review
- [ ] Failure-mode testing
- [ ] Performance profiling
- [ ] Resource management
- [ ] Configuration hardening
- [ ] Release validation

---

# Project Status

<div align="center">

| Area | State |
|:---|:---:|
| Repository foundation | **READY** |
| Documentation | **READY** |
| Package architecture | **READY** |
| CI / quality gates | **READY** |
| Assistant contracts | **READY** |
| SLM | **NEXT DEVELOPMENT LAYER** |
| Memory | **PLANNED** |
| Storage | **PLANNED** |
| Interactive UI | **PLANNED** |
| Production hardening | **PLANNED** |

</div>

The repository is deliberately moving from a stable foundation toward the actual intelligence stack. The next major engineering layer is the SLM foundation.

---

# Contributing

Keep contributions focused and consistent with the architecture.

Before submitting a change:

1. understand the subsystem boundary
2. implement one coherent change
3. add or update tests
4. run local quality checks
5. update documentation if behavior changes
6. commit the change
7. verify GitHub Actions

Avoid unrelated refactoring in feature commits.

---

# License

The project license has not yet been finalized.

Until a license is explicitly added to the repository, no open-source licensing terms should be assumed.

---

<div align="center">

### ARIA

**A modular foundation for building a private, extensible AI assistant.**

[Repository](https://github.com/chamanvashishth/ARIA) · [Issues](https://github.com/chamanvashishth/ARIA/issues)

</div>
