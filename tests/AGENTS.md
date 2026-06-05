# OVERVIEW
Pytest-based test suite with custom mocking and Hydra configuration fixtures.

# STRUCTURE
- `conftest.py`: Root configuration, Hydra fixtures, and shared setup.
- `canned_responses.py`: Shared stub factories for tests.
- `test_*.py`: Unit and integration tests for specific modules.
- `utils/`: Mock servers (OpenAI, Zotero) for integration testing.
- `retriever/` & `reranker/`: Tests specific to retrieval and reranking logic.

# WHERE TO LOOK
- `conftest.py`: For `config` and `_base_config` fixtures.
- `canned_responses.py`: For `SimpleNamespace` stubs (e.g., `make_stub_openai_client`).
- `utils/mock_openai/`: Dockerized mock server for OpenAI API.
- `utils/mock_zotero/`: Dockerized mock server for Zotero API.

# CONVENTIONS
- **Framework**: Uses `pytest`. Run via `uv run pytest`.
- **Mocking**: Uses `pytest monkeypatch` + `types.SimpleNamespace`.
- **Configuration**: Uses `hydra` for config. Tests should use the function-scoped `config` fixture (a deep copy of the session config) to avoid polluting other tests.
- **Stubs**: Use `canned_responses.py` for shared stub factories instead of creating new ones in individual test files.
- **Grouping**: Tests within a file are grouped by the function/class they test using comment blocks (e.g., `# --- generate_tldr ---`).
- **Coverage**: CI enforces coverage reporting (`--cov=src/zotero_arxiv_daily --cov-report=term-missing`).

# ANTI-PATTERNS
- **No `unittest.mock`**: Do not use `unittest.mock` or `MagicMock`. Rely exclusively on `monkeypatch` and `SimpleNamespace`.
- **Mutating Session Config**: Never mutate the `_base_config` fixture directly. Always use the function-scoped `config` fixture.
- **External API Calls**: Tests must not make real network calls to OpenAI or Zotero. Use the provided stubs or mock servers.
- **Duplicating Stubs**: Do not create one-off stubs in test files if a generic version can be added to `canned_responses.py`.
