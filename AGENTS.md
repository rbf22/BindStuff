# Agent Instructions

This file provides instructions for AI agents working with this codebase.

## Development Setup

The development environment is managed by Poetry. To install all dependencies, including development dependencies, run:

```bash
make dev-install
```

## Running Tests

The project uses pytest for testing. To run the test suite, use the Makefile target:

```bash
make run-tests
```

This will run all files matching `tests/test_*.py` and generate a test coverage report.

## Linting

This project uses `ruff` for linting. To check the code for linting errors, use the Makefile target:

```bash
make run-linter
```

Please ensure all new code passes the linter before submitting.

### Pylint

This project also uses `pylint` for more in-depth static analysis. To run `pylint`, use the Makefile target:

```bash
make run-pylint
```

Please ensure all new code scores 10/10 with `pylint` before submitting.

## Dependency Checking

This project uses `deptry` to check for unused dependencies. To run the dependency checker, use the Makefile target:

```bash
make run-deptry
```

## Type Checking

This project uses `mypy` for static type checking. To run the type checker, use the Makefile target:

```bash
make run-mypy
```
