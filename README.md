# ARIA

ARIA is being rebuilt as a local-first, modular AI assistant.

This repository is intentionally being reconstructed from a clean baseline. The implementation will be developed incrementally, with each stage validated locally and through GitHub Actions before the next stage is added.

## Development principles

- Local-first development and execution
- Small, testable, maintainable components
- Explicit configuration and clear boundaries
- Automated validation through GitHub Actions
- Incremental implementation with a passing baseline at every stage
- No deployment platform is required for the core project

## Current status

The repository has been reset to a clean documentation baseline. The project skeleton and implementation will be introduced in subsequent stages.

## Planned direction

The architecture will be established around clearly separated concerns such as:

- application/runtime entry point
- configuration
- assistant/core orchestration
- model/provider integration
- memory and persistence
- conversation/history handling
- CLI/user interface
- tests
- CI and developer tooling

Exact components and interfaces will be finalized against the project specification before implementation.

## Local development

The project is designed to be developed and tested locally. Installation, testing, linting, type checking, and other validation commands will be documented here as the corresponding tooling is introduced.

## License

License and contribution guidance will be added with the project foundation.
