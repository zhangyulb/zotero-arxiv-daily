# Coding Standards

Architectural and review rules for `zotero-arxiv-daily`. Mechanical rules (`unittest.mock` ban, workflow YAML validity, test coverage) are enforced in CI via `ruff`, `actionlint`, and `pytest`.

## Pipeline Architecture & Seam Boundaries

- **Decoupled plugins**: Retrievers (`src/zotero_arxiv_daily/retriever/`) and rerankers (`src/zotero_arxiv_daily/reranker/`) interact only through `Executor` and the `Paper` / `CorpusPaper` data models in `protocol.py`. Register new implementations via `@register_retriever` or `@register_reranker`.
- **Entity-owned LLM prompts**: Keep prompt construction, `tiktoken` truncation, and response parsing inside `Paper` methods in `src/zotero_arxiv_daily/protocol.py` (`generate_tldr`, `generate_affiliations`).
- **Best-effort degradation**: External API failures (arXiv/bioRxiv/chemRxiv feeds, embedding batches, LLM calls) must log a warning/error, set `self.has_failures = True` on the retriever/reranker/paper, and fall back to safe defaults (such as `paper.abstract` for TLDR or `0.0` similarity for failed embedding batches). Only a 100% pipeline failure (`has_failures and len(executor.papers) == 0`) exits with code `1` in `main.py`.
- **Retry only transient errors**: Wrap network boundaries with `tenacity.retry`, and exclude permanent 4xx client errors (`400`, `401`, `403`, `404`, `422`) and `ValueError` from retry loops so misconfigured credentials fail fast instead of exhausting the GitHub Actions job timeout.

## Runtime & Environment Gotchas

- **Hydra struct mode**: `DictConfig` instances composed by Hydra run in `struct` mode. Wrap new key insertions in `with open_dict(cfg):`.
- **OmegaConf `oc.env` scalar coercion**: `oc.env` returns a `str` whenever the fallback is non-`null` (e.g., `${oc.env:DEBUG,false}` resolves to the truthy string `'False'`). Use `null` as the YAML fallback for boolean env vars (`${oc.env:DEBUG,null}`) and coerce with `utils.normalize_bool()`.
- **GitHub Actions empty secrets**: Unset `${{ secrets.X }}` or `${{ vars.X }}` expressions in `.github/workflows/*.yml` export `X=""` (empty string), which overrides `${oc.env:X,default}` in `config/base.yaml`. Normalize empty strings to defaults in `Executor.__init__`.
- **CPU embedding dtype**: Hugging Face models whose `config.json` specifies `"dtype": "bfloat16"` (such as `jinaai/jina-embeddings-v5-text-nano-retrieval`) must be cast to `torch.float32` when `encoder.device.type == "cpu"` to avoid 15x software-emulated BF16 slowdowns on non-AVX512-BF16 GitHub Actions runners.
- **SMTP transport fallback**: Email senders (`utils.send_email` and `watchdog.send_alert_email`) must attempt `smtplib.SMTP` + `starttls()` first, close the socket on failure, and fall back to `smtplib.SMTP_SSL` then plain `smtplib.SMTP`.
- **Workflow logic in Python modules**: Keep non-trivial Python logic out of `.github/workflows/*.yml` heredocs; place it in `src/zotero_arxiv_daily/` (e.g. `watchdog.py`) and cover it with `pytest`.
- **Logging**: Use `loguru.logger` across `src/zotero_arxiv_daily/`.

## Testing Conventions

- **Test doubles**: Use `pytest`'s `monkeypatch` fixture and `types.SimpleNamespace` (with shared factories in `tests/canned_responses.py`). `unittest.mock` is banned by `ruff` (`TID251`).
- **Config isolation**: Mutate the function-scoped `config` fixture (`with open_dict(config):`), never the session-scoped `_base_config` fixture.
