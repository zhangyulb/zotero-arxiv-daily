SPECIAL INSTRUCTION: think silently if needed. EFFORT LEVEL: 2.0
use rg over grep for faster content search

# OVERVIEW
Python application: recommends new arXiv/bioRxiv/medRxiv papers by embedding
similarity to the user's Zotero library, generates TLDRs via LLM, emails results.
Designed to run as a daily GitHub Actions workflow at zero cost.

# WHERE TO LOOK
- `src/zotero_arxiv_daily/`: All Python source.
- `tests/`: pytest suite (87 tests; `slow`-marked tests are deselected by default).
- `.github/workflows/`: `main.yml` (daily email), `watchdog.yml` (missed-run alarm),
  `ci.yml` (pytest on push/PR), `test.yml` (manual pipeline run), `keep-alive.yml`.
- `config/`: Hydra configs — `base.yaml` (defaults) + `custom.yaml` (user overrides).

# CODE MAP
- `src/zotero_arxiv_daily/main.py`: Entry point (`@hydra.main`).
- `src/zotero_arxiv_daily/executor.py`: `Executor.run()` orchestrates the pipeline:
  fetch Zotero corpus → filter → retrieve → rerank → TLDR/affiliations → email.
- `src/zotero_arxiv_daily/protocol.py`: `Paper`/`CorpusPaper` dataclasses.
- `src/zotero_arxiv_daily/retriever/`: source plugins (`@register_retriever`).
- `src/zotero_arxiv_daily/reranker/`: embedding plugins (`@register_reranker`);
  `local` (sentence-transformers) or `api` (OpenAI-compatible).

# CONVENTIONS
- **Structure**: `src/` layout package, not flat scripts.
- **Dependencies**: `uv` runner; sync with `uv sync`.
- **Plugins**: retrievers/rerankers register via decorators and are discovered by
  `get_retriever_cls()` / `get_reranker_cls()` — new sources need no orchestration changes.
- **Config**: Hydra composition; env vars via `${oc.env:VAR,default}`.
- **Formatting/Linting**: None. Match existing code style.

# ANTI-PATTERNS (THIS PROJECT)
- **LLM Output**: LLM must NOT return intermediate results.
- **Data Models**: Paper data shouldn't contain duplicated affiliations.

# COMMANDS
- Run: `uv run src/zotero_arxiv_daily/main.py` (requires env vars, see `config/base.yaml`)
- Test all: `uv run pytest -m ""` | fast only: `uv run pytest` | single: `uv run pytest tests/test_utils.py -v`
