# Contributing

## Setup

```bash
git clone https://github.com/MSDLLCpapers/triton-sigmoid.git
cd triton-sigmoid
uv sync --extra cu128 --extra dev
source .venv/bin/activate
pre-commit install
```

## Workflow

```bash
git checkout -b feature/your-feature-name
```

Make changes, add tests, update docs.

### Run Tests

```bash
pytest tests/ -v
pre-commit run --all-files
```

## Testing

Tests should be reproducible, cover edge cases, and test both forward/backward passes against reference implementations.

## Code Style

- Line length: 120 characters
- Formatter: Black
- Linter: Ruff
- Use Google-style docstrings with type hints

## Pull Requests

Before submitting:
- Ensure tests pass: `pytest tests/ -v`
- Run formatters: `pre-commit run --all-files`
- Update docs if adding features

Use clear PR titles and describe what changed and why.
