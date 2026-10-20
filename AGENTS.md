SPECIAL INSTRUCTION: think silently if needed. EFFORT LEVEL: 2.0
use rg over grep for faster content search

# OVERVIEW
Python application recommending new arXiv/bioRxiv/medRxiv/chemRxiv papers by embedding similarity to the user's Zotero library, generating TLDRs/affiliations via LLM, and emailing results daily via GitHub Actions.

# NAVIGATION POINTERS
- **Coding & review standards**: Read [`CODING_STANDARDS.md`](CODING_STANDARDS.md) when modifying source, tests, or workflows, or when reviewing a diff.
- **Source map & module boundaries**: See [`src/zotero_arxiv_daily/AGENTS.md`](src/zotero_arxiv_daily/AGENTS.md).
- **Test fixtures & stub conventions**: See [`tests/AGENTS.md`](tests/AGENTS.md).
- **GitHub CLI (`gh`), issues & Action logs**: Read [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md) before running `gh` commands (`GH_HOST=github.com` and token extraction are required on this machine).
- **Triage & domain docs**: See [`docs/agents/triage-labels.md`](docs/agents/triage-labels.md) and [`docs/agents/domain.md`](docs/agents/domain.md).

# REPO GOTCHAS
- **Fork & branch model**: This repo (`zhangyulb/zotero-arxiv-daily`) is a fork of `upstream` (`TideDra/zotero-arxiv-daily`) that runs scheduled workflows directly from `main`. Preserve `${oc.env:VAR,default}` fallbacks in `config/base.yaml` when syncing from `upstream`.
- **Hydra struct mode, `oc.env` types & empty Action secrets**: Wrap new `DictConfig` keys in `with open_dict(cfg):`. OmegaConf's `oc.env` returns `str` for any non-`null` default (so `${oc.env:DEBUG,false}` resolves to truthy `'False'`); use `null` defaults for booleans (`${oc.env:DEBUG,null}`) and coerce via `normalize_bool()`. Unset GitHub Action secrets export `""`, overriding `${oc.env:VAR,default}` — normalize empty strings in `Executor.__init__`.

# COMMANDS
- Lint: `uv run ruff check src tests`
- Test fast: `uv run pytest` | all (including `@pytest.mark.slow`): `uv run pytest -m ""`
- Coverage: `uv run pytest --cov=src/zotero_arxiv_daily --cov-report=term-missing`
- Run pipeline: `uv run src/zotero_arxiv_daily/main.py`
