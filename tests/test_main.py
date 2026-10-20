"""Tests for main entry point.

The @hydra.main decorator makes main() hard to test directly in pytest
because config_path resolution depends on the calling context.
We test the inner logic by calling main's body with a composed config.
"""

import pytest
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(autouse=True)
def _clear_hydra():
    """Ensure GlobalHydra is clean before and after each test in this module."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_main_creates_executor_and_runs(config, monkeypatch):
    """Verify that the main function creates an Executor and calls run()."""
    calls = []

    class FakeExecutor:
        def __init__(self, cfg):
            calls.append(("init", cfg))

        def run(self):
            calls.append(("run",))

    monkeypatch.setattr("zotero_arxiv_daily.main.Executor", FakeExecutor)

    # Call main's body directly, bypassing @hydra.main
    from zotero_arxiv_daily import main as main_mod

    # Simulate what @hydra.main does: calls main(config)
    main_mod.main.__wrapped__(config)

    assert ("init", config) in calls
    assert ("run",) in calls


def test_main_debug_logging(config, monkeypatch):
    """Verify debug mode sets appropriate log level."""
    from omegaconf import open_dict

    with open_dict(config):
        config.executor.debug = True

    class FakeExecutor:
        def __init__(self, cfg):
            pass
        def run(self):
            pass

    monkeypatch.setattr("zotero_arxiv_daily.main.Executor", FakeExecutor)

    from zotero_arxiv_daily import main as main_mod

    main_mod.main.__wrapped__(config)
    # If we get here without error, the debug path executed successfully


def test_base_config_handles_empty_github_action_env_vars(monkeypatch):
    """Unset GitHub Action secrets are exported as empty strings ('').
    Verify Hydra config composition + Executor initialization handle them cleanly.
    """
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from zotero_arxiv_daily.executor import Executor

    monkeypatch.setenv("MODEL_NAME", "")
    monkeypatch.setenv("LLM_USER_AGENT", "")
    monkeypatch.setenv("LANGUAGE", "")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kwargs: object())

    config_dir = str(Path(__file__).resolve().parent.parent / "config")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="base")

    executor = Executor(cfg)
    assert executor.config.executor.debug is True
    assert executor.config.llm.generation_kwargs.model == "Qwen/Qwen3-8B"
    assert executor.config.llm.language == "English"


@pytest.mark.parametrize("debug_env", [None, "", "false", "False", "0"])
def test_base_config_debug_defaults_false_and_retrieves_all_papers(monkeypatch, debug_env):
    """Regression test: when CUSTOM_CONFIG is empty in main.yml, base.yaml is used directly.
    Unset or falsy DEBUG must resolve to False (not truthy string 'False') so ArxivRetriever
    does not truncate daily papers to the first 10 RSS entries before reranking.
    """
    from pathlib import Path
    from types import SimpleNamespace
    from hydra import compose, initialize_config_dir
    from zotero_arxiv_daily.executor import Executor
    from zotero_arxiv_daily.retriever import arxiv_retriever

    if debug_env is None:
        monkeypatch.delenv("DEBUG", raising=False)
    else:
        monkeypatch.setenv("DEBUG", debug_env)
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kwargs: object())

    entries = [
        SimpleNamespace(
            id=f"oai:arXiv.org:2610.{10000 + i}",
            title=f"Paper {i}",
            author="Author",
            summary="Abstract",
            link=f"https://arxiv.org/abs/2610.{10000 + i}",
            arxiv_announce_type="new",
            get=lambda k, d=None: "new" if k == "arxiv_announce_type" else d,
        )
        for i in range(25)
    ]
    monkeypatch.setattr(
        arxiv_retriever.feedparser,
        "parse",
        lambda url: SimpleNamespace(
            status=200,
            bozo=False,
            feed=SimpleNamespace(title="arXiv"),
            entries=entries,
        ),
    )

    config_dir = str(Path(__file__).resolve().parent.parent / "config")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="base")

    if debug_env in (None, ""):
        assert not bool(cfg.executor.debug)
    executor = Executor(cfg)
    assert executor.config.executor.debug is False
    raw_papers = executor.retrievers["arxiv"]._retrieve_raw_papers()
    assert len(raw_papers) == 25

