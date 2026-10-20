# Test Suite Map (`tests/`)

- `conftest.py`: Session-scoped `_base_config` and function-scoped `config` Hydra fixtures. Always mutate `config` (via `with open_dict(config):`), never `_base_config`.
- `canned_responses.py`: Shared `SimpleNamespace` stub factories (`make_stub_openai_client`, `make_stub_zotero_client`, `make_stub_smtp`, `make_sample_paper`, `make_sample_corpus`, canned bioRxiv/chemRxiv payloads).
- `test_*.py`, `retriever/test_*.py`, `reranker/test_*.py`: 1:1 module test coverage.
- `test_watchdog.py`: Unit tests for `.github/workflows/` failure reporting and 24-hour schedule watchdog checks.

Use `pytest` `monkeypatch` and `types.SimpleNamespace` for all stubs (`unittest.mock` is banned by `ruff` rule `TID251`).
