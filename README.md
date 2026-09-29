# ARIA

**ARIA** is a local-first, modular AI assistant designed to provide a maintainable foundation for conversational intelligence, memory, provider integration, and extensible user interaction.

The project is being rebuilt from a clean repository baseline. Development follows an incremental engineering process: establish a stable foundation, validate it locally and in CI, and only then introduce higher-level assistant capabilities.

> **Project status:** Foundation rebuild in progress. The current branch contains the project specification and documentation baseline; implementation is being introduced in small, validated stages.

---

## Table of Contents

- [Vision](#vision)
- [Engineering Principles](#engineering-principles)
- [Architecture Direction](#architecture-direction)
- [Repository Structure](#repository-structure)
- [Development Workflow](#development-workflow)
- [Quality Gates](#quality-gates)
- [Local Development](#local-development)
- [Configuration and Secrets](#configuration-and-secrets)
- [Testing Strategy](#testing-strategy)
- [CI/CD](#cicd)
- [Roadmap](#roadmap)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

---

## Vision

ARIA is intended to evolve into a practical AI assistant rather than a collection of tightly coupled scripts.

The architecture is being designed so that assistant behavior, model providers, memory, persistence, interfaces, and infrastructure can evolve independently. Local execution remains the primary development and validation target.

The long-term system is expected to support capabilities such as:

- conversational interaction
- pluggable model/provider integrations
- persistent and structured memory
- conversation and event history
- configurable assistant behavior
- deterministic application logic around model responses
- extensible command-line and other interfaces
- automated testing and validation
- clear boundaries between core logic and external services

Specific capabilities will be implemented only after their interfaces and responsibilities are defined and tested.

---

## Engineering Principles

### Local-first

The complete development loop should work on a local machine. External services are integrations, not prerequisites for the core architecture.

### Modular by responsibility

Core orchestration, configuration, providers, memory, persistence, interfaces, and utilities should have explicit boundaries.

### Test before expansion

Every meaningful implementation stage must have a corresponding validation step. New functionality should not be layered on top of an unstable foundation.

### Explicit dependencies

Dependencies, configuration, runtime requirements, and optional integrations should be declared rather than hidden inside scripts or environment-specific behavior.

### Provider independence

The assistant core should not be permanently coupled to a single model vendor. Provider-specific behavior belongs behind explicit interfaces.

### Safe persistence

User data, conversation history, memory, and credentials must be treated as separate concerns with clear storage and security boundaries.

### CI as a quality gate

GitHub Actions should validate the repository on every relevant change. A passing local check is useful; a passing automated check is required before advancing the implementation.

---

## Architecture Direction

The initial architecture is intentionally layered:

```text
┌───────────────────────────────────────────────┐
│                User Interfaces                │
│            CLI / future interfaces            │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│              Application Layer                │
│       commands • orchestration • state        │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│                 ARIA Core                     │
│       assistant logic • policies • flow       │
└───────────────┬───────────────────┬───────────┘
                │                   │
        ┌───────▼────────┐  ┌───────▼──────────┐
        │ Model Providers │  │ Memory / Storage │
        │ adapter boundary│  │ persistence      │
        └─────────────────┘  └──────────────────┘
                │                   │
        ┌───────▼───────────────────▼───────────┐
        │           Infrastructure              │
        │ configuration • logging • utilities   │
        └───────────────────────────────────────┘
```

This is an architectural direction, not a claim that every component already exists. Interfaces will be kept small and implementation details will remain replaceable.

---

## Repository Structure

The target foundation will follow a package-oriented layout:

```text
ARIA/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── aria/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config/
│       ├── core/
│       ├── interfaces/
│       ├── memory/
│       ├── providers/
│       ├── storage/
│       └── ui/
├── tests/
│   ├── unit/
│   └── integration/
├── pyproject.toml
├── .gitignore
└── README.md
```

Directories are introduced only when the corresponding responsibility becomes part of the implementation.

---

## Development Workflow

ARIA is developed through small, reviewable checkpoints.

### 1. Define

Establish the responsibility, interface, expected behavior, and failure cases before implementing a component.

### 2. Implement

Add the smallest complete implementation that satisfies the defined contract.

### 3. Validate locally

Run formatting, linting, type checks where applicable, tests, and basic runtime checks.

### 4. Validate through GitHub Actions

Push the checkpoint and inspect the associated CI run. A failing quality gate stops progression until the cause is resolved.

### 5. Commit

Use focused commits that describe one logical change.

### 6. Continue incrementally

Do not introduce the next architectural layer until the previous checkpoint is stable.

---

## Quality Gates

The project will use automated checks appropriate to the Python implementation, including:

- syntax/runtime validation
- unit tests
- integration tests where required
- linting
- formatting checks
- static/type analysis where justified
- import/package validation
- CI workflow validation

The exact toolchain will be pinned in `pyproject.toml` as the foundation is introduced.

---

## Local Development

### Prerequisites

The development environment will require:

- Git
- Python supported by the project configuration
- pip or another supported Python package manager
- a shell capable of running the project test and quality commands

### Setup

Once the project foundation is present:

```bash
git clone https://github.com/chamanvashishth/ARIA.git
cd ARIA

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Validation commands will be documented here as the tooling is finalized.

---

## Configuration and Secrets

Configuration must remain separate from source code.

Rules:

1. Never commit API keys, passwords, tokens, private credentials, or personal secrets.
2. Local configuration should use environment variables or explicitly ignored local configuration files.
3. Example configuration may be committed only with placeholder values.
4. Provider credentials should be consumed by provider adapters rather than scattered throughout application code.
5. Tests must not depend on production credentials.

The project will prefer safe defaults and explicit configuration failures over silently using incorrect settings.

---

## Testing Strategy

Testing will be layered:

### Unit tests

Validate individual components and business rules in isolation.

### Integration tests

Validate boundaries between components, such as application orchestration and persistence.

### Provider tests

Provider integrations should be tested through mocked/fake boundaries where possible so the core test suite remains deterministic.

### Runtime checks

The application entry point should have a basic executable smoke test once the CLI foundation exists.

The goal is not simply high test count; tests should protect interfaces, behavior, and failure handling.

---

## CI/CD

GitHub Actions is the project's automated validation layer.

The initial CI pipeline will:

1. install the supported Python runtime
2. install project and development dependencies
3. run formatting/lint checks
4. run tests
5. run additional static checks when configured

The core project does **not** require Vercel or another hosting platform. Deployment, if ever required, will be treated as a separate concern from local development and core validation.

---

## Roadmap

### Phase 0 — Repository reset

- [x] Remove the previous prototype implementation
- [x] Establish a clean repository baseline
- [x] Define engineering direction

### Phase 1 — Foundation

- [ ] Establish Python package structure
- [ ] Add project metadata and dependency management
- [ ] Add development tooling
- [ ] Add test layout and smoke tests
- [ ] Add GitHub Actions CI
- [ ] Verify the complete foundation locally and in CI

### Phase 2 — Application Core

- [ ] Define application lifecycle
- [ ] Define assistant/core interfaces
- [ ] Establish request/response models
- [ ] Add structured logging and error boundaries

### Phase 3 — Configuration and Persistence

- [ ] Implement typed configuration
- [ ] Establish storage abstraction
- [ ] Implement conversation/history persistence
- [ ] Define memory lifecycle and data boundaries

### Phase 4 — Model Providers

- [ ] Define provider abstraction
- [ ] Implement provider adapters
- [ ] Add provider-independent orchestration
- [ ] Add deterministic fallback/error handling

### Phase 5 — User Interface

- [ ] Implement the CLI interface
- [ ] Add command handling
- [ ] Add interactive conversation flow
- [ ] Improve terminal UX without coupling it to core logic

### Phase 6 — Assistant Capabilities

- [ ] Memory-aware conversations
- [ ] Context handling
- [ ] Tool/command execution boundaries
- [ ] Extensible assistant behaviors
- [ ] Observability and diagnostics

### Phase 7 — Hardening

- [ ] Expand unit and integration coverage
- [ ] Security review
- [ ] Configuration review
- [ ] Failure-mode testing
- [ ] Documentation completion
- [ ] Release-readiness validation

> Roadmap items are deliberately staged. A feature is not considered complete until its implementation, tests, documentation, and validation path are complete.

---

## Project Status

| Area | Status |
|---|---|
| Repository baseline | Complete |
| Project specification | Documented |
| Package foundation | In progress |
| Automated CI | Next |
| Core assistant | Planned |
| Memory | Planned |
| Provider layer | Planned |
| CLI | Planned |
| Production hardening | Planned |

---

## Contributing

Contributions should preserve the project's modular boundaries and validation workflow.

Before submitting a change:

1. keep the change focused
2. add or update tests where behavior changes
3. run the local quality checks
4. update documentation when interfaces or behavior change
5. ensure GitHub Actions passes

Large architectural changes should be discussed before implementation so responsibilities and interfaces remain coherent.

---

## Security

Please do not commit credentials, access tokens, private keys, personal data, or other sensitive information.

If a security issue is discovered, avoid publishing sensitive details in a public issue. Use an appropriate private disclosure channel when one is available.

---

## License

A project license will be added when the licensing decision is finalized.
