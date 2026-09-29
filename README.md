# ARIA

**ARIA** is a local-first, modular AI assistant engineered as a production-quality Python system. Its architecture separates assistant orchestration from model providers, memory, persistence, configuration, and user interfaces so each subsystem can evolve independently.

> **Status:** Foundation and assistant orchestration are implemented. The SLM, training pipeline, persistent memory, and interactive assistant experience are planned and will be introduced incrementally.

## What ARIA Is

ARIA is being built as a real software system rather than a single AI script. The project prioritizes:

- clear interfaces between subsystems
- local development and execution
- provider-independent assistant logic
- deterministic application behavior around model output
- testable components
- explicit configuration and security boundaries
- incremental implementation with CI validation

The long-term system is intended to support conversational interaction, a locally runnable SLM, persistent conversation history, memory, configurable behavior, CLI interaction, and extensible assistant capabilities.

Only implemented functionality is described as complete in this document.

## Current Capabilities

### Implemented

- Python package structure using a src layout
- project metadata and development dependencies
- package entry point and version metadata
- conversational Message contract
- Assistant orchestration contract
- provider/responder decoupling
- unit tests
- CLI smoke test
- Ruff formatting and linting
- GitHub Actions CI for Python 3.11, 3.12, and 3.13

### Not implemented yet

- SLM architecture and weights
- tokenizer and vocabulary pipeline
- training dataset pipeline
- model training
- inference engine
- checkpoint management
- persistent memory
- conversation storage
- retrieval/context system
- production model-provider adapters
- interactive CLI
- tool execution
- observability and telemetry

This distinction is deliberate: the README is maintained as an accurate description of repository state.

## Architecture

ARIA uses a layered design:

```text
                         User Interface
                              |
                              v
                      Application Layer
                       orchestration/state
                              |
                              v
                         ARIA Core
                    assistant coordination
                         /          \
                        /            \
                       v              v
                Model Boundary    Memory/Storage
                 SLM/providers     persistence
                       \            /
                        \          /
                         v        v
                       Infrastructure
                  config/logging/utilities
```

The core assistant does not depend on a specific model implementation. A responder boundary allows the future ARIA SLM, another provider adapter, or a test double to be substituted without rewriting application orchestration.

## Core Contracts

The current conversational contract is:

```python
Message(role="user", content="Hello")
```

The assistant delegates response generation to a responder:

```text
User Message
     |
     v
 Assistant
     |
     v
 Responder
     |
     v
Assistant Message
```

This boundary is the foundation for integrating the SLM later without coupling model implementation to the application layer.

## Repository Layout

```text
ARIA/
├── .github/workflows/ci.yml
├── src/aria/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config/
│   ├── core/
│   │   └── assistant.py
│   ├── interfaces/
│   │   └── messages.py
│   ├── memory/
│   ├── providers/
│   ├── storage/
│   └── ui/
├── tests/
│   ├── unit/
│   │   ├── test_assistant.py
│   │   └── test_messages.py
│   └── integration/
│       └── test_cli.py
├── .gitignore
├── pyproject.toml
└── README.md
```

The subsystem directories are architectural boundaries. Implementation is added only when the corresponding feature is ready.

## Engineering Rules

### Contract first
Define inputs, outputs, invariants, and failure behavior before implementing a subsystem.

### One feature per checkpoint
Each implementation step should solve one coherent problem. Unrelated unfinished features should not be bundled together.

### Tests with behavior
New behavior receives tests in the same checkpoint whenever practical.

### Provider independence
ARIA Core must not know the internal implementation of a particular model.

### Deterministic boundaries
Application logic should remain deterministic wherever possible; probabilistic model behavior stays behind explicit interfaces.

### CI before progression
A commit is a checkpoint. Local validation and GitHub Actions are checked before advancing to the next feature.

### Avoid premature complexity
Introduce abstractions because the system needs them, not simply because they might be useful later.

## Requirements

Current requirements:

- Git
- Python 3.11+
- pip
- Python virtual environment

CI currently validates:

- Python 3.11
- Python 3.12
- Python 3.13

## Local Setup

Clone the repository:

```bash
git clone https://github.com/chamanvashishth/ARIA.git
cd ARIA
```

Create a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Running ARIA

The current entry point is intentionally minimal:

```bash
python -m aria
```

Expected output:

```text
ARIA 0.1.0
```

The console entry point is also available after installation:

```bash
aria
```

ARIA is not yet an interactive AI assistant. Interactive conversation will be introduced after the model and application contracts are ready.

## Testing and Quality

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

Apply formatting:

```bash
python -m ruff format .
```

A feature checkpoint should pass all configured local quality checks before commit.

## Continuous Integration

GitHub Actions is the automated quality gate.

The current workflow runs on pushes to main and pull requests. It installs the project and runs:

1. formatting verification
2. Ruff linting
3. pytest
4. the same checks across Python 3.11, 3.12, and 3.13

```text
Commit
  |
  v
GitHub Actions
  |
  +-- Python 3.11
  +-- Python 3.12
  +-- Python 3.13
       |
       +-- format
       +-- lint
       +-- tests
```

A checkpoint is considered validated only after its CI run has completed successfully.

## Configuration and Secrets

Runtime configuration must remain separate from source code.

Never commit:

- API keys
- access tokens
- passwords
- private keys
- authentication cookies
- personal credentials
- production secrets
- private user data

When integrations are added, credentials will be consumed through explicit provider/configuration boundaries. Tests should remain independent of production credentials.

## SLM Strategy

The SLM will be built as an independent subsystem behind the responder boundary.

Planned flow:

```text
ARIA Core
   |
Responder Contract
   |
   +---- ARIA SLM
   |       |
   |       +-- tokenizer
   |       +-- model
   |       +-- decoder/inference
   |
   +---- other future adapters
```

The SLM will be implemented one feature at a time:

1. model configuration contract
2. tokenizer contract
3. vocabulary representation
4. dataset contract
5. deterministic preprocessing
6. training-data validation
7. model interface
8. model architecture
9. forward pass and loss
10. training loop
11. checkpointing
12. inference
13. evaluation
14. ARIA Core integration

A trained model must not be required to execute the core unit-test suite.

## Development Workflow

Every feature follows the same sequence:

```text
Understand
   |
Define contract
   |
Implement one feature
   |
Add tests
   |
Local validation
   |
Commit
   |
Check GitHub Actions
   |
Fix failures if required
   |
Next feature
```

Example focused commits:

```text
feat: add tokenizer contract
feat: implement vocabulary builder
test: add model configuration coverage
fix: handle invalid model responses
docs: document local development workflow
```

## Roadmap

### Phase 0 — Foundation
- [x] Reset obsolete prototype
- [x] Establish project documentation
- [x] Establish Python package structure
- [x] Add project metadata
- [x] Add development tooling
- [x] Add initial tests
- [x] Add GitHub Actions CI

### Phase 1 — Assistant Core
- [x] Message contract
- [x] Responder boundary
- [x] Minimal assistant orchestration
- [x] Assistant error handling tests
- [ ] Typed application configuration
- [ ] Structured application errors
- [ ] Application lifecycle

### Phase 2 — SLM Foundation
- [ ] SLM configuration
- [ ] Tokenizer interface
- [ ] Vocabulary representation
- [ ] Dataset contract
- [ ] Deterministic preprocessing
- [ ] Training-data validation
- [ ] Model interface

### Phase 3 — SLM Implementation
- [ ] Tokenizer
- [ ] Vocabulary pipeline
- [ ] Model architecture
- [ ] Forward pass
- [ ] Loss calculation
- [ ] Training loop
- [ ] Checkpointing
- [ ] Inference
- [ ] Evaluation

### Phase 4 — Memory and Persistence
- [ ] Storage interface
- [ ] Conversation persistence
- [ ] Memory records
- [ ] Memory lifecycle
- [ ] Retrieval/context handling

### Phase 5 — User Experience
- [ ] Interactive CLI
- [ ] Command handling
- [ ] Conversation sessions
- [ ] Configuration commands
- [ ] Terminal UX

### Phase 6 — Assistant Capabilities
- [ ] Context-aware responses
- [ ] Memory-aware responses
- [ ] Tool/command boundaries
- [ ] Extensible assistant behavior
- [ ] Diagnostics and observability

### Phase 7 — Production Hardening
- [ ] Security review
- [ ] Failure-mode testing
- [ ] Performance profiling
- [ ] Resource management
- [ ] Configuration hardening
- [ ] Documentation review
- [ ] Release validation

## Project Status

| Component | Status |
|---|---|
| Repository foundation | Complete |
| Production-grade README | Complete |
| Python package structure | Complete |
| CI pipeline | Complete |
| Message contract | Complete |
| Assistant core contract | Complete |
| Configuration | Foundation only |
| Memory | Not implemented |
| Storage | Not implemented |
| Providers | Boundary only |
| SLM | Not implemented |
| Training | Not implemented |
| Inference | Not implemented |
| Interactive CLI | Not implemented |
| Production hardening | Not started |

## Security

Security is an architectural concern from the beginning.

Development must:

- keep secrets out of version control
- validate external inputs at subsystem boundaries
- avoid logging credentials or sensitive content
- isolate provider credentials
- use least-privilege access for external integrations
- fail explicitly on invalid configuration
- test security-sensitive boundaries

Security-sensitive changes should receive dedicated review.

## Contributing

Before submitting a change:

1. keep the change focused
2. preserve subsystem boundaries
3. add or update tests
4. run local quality checks
5. update documentation when behavior changes
6. verify GitHub Actions
7. avoid unrelated refactoring

Architectural changes should be introduced deliberately because the SLM, memory, storage, and provider layers will depend on stable contracts.

## License

The project license has not yet been finalized.

Until a license is explicitly added, no open-source licensing terms should be assumed.
