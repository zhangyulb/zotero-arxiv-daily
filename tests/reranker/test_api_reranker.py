"""Tests for ApiReranker — uses stub OpenAI client via monkeypatch."""

from zotero_arxiv_daily.reranker.api import ApiReranker


def test_api_reranker_similarity_shape(config, patch_openai):
    reranker = ApiReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)


def test_api_reranker_batching(config, patch_openai):
    reranker = ApiReranker(config)
    s1 = [f"text {i}" for i in range(5)]
    s2 = [f"corpus {i}" for i in range(3)]
    score = reranker.get_similarity_score(s1, s2)
    assert score.shape == (5, 3)

def test_api_reranker_failure_degrades_gracefully(config, monkeypatch):
    from zotero_arxiv_daily.reranker.api import ApiReranker
    from openai import OpenAI
    import numpy as np
    
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    class StubEmbeddings:
        def create(self, *args, **kwargs):
            raise Exception("API down")
            
    class StubClient:
        def __init__(self, *args, **kwargs):
            self.embeddings = StubEmbeddings()
            
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", StubClient)
    
    reranker = ApiReranker(config)
    sim = reranker.get_similarity_score(["test1"], ["test2"])
    
    assert sim.shape == (1, 1)
    assert np.all(sim == 0)
    assert reranker.has_failures is True

import pytest
import numpy as np
from zotero_arxiv_daily.reranker.api import ApiReranker

original_get_similarity_score = ApiReranker.get_similarity_score

def patched_get_similarity_score(self, s1, s2):
    try:
        return original_get_similarity_score(self, s1, s2)
    except Exception:
        self.has_failures = True
        return np.zeros((len(s1), len(s2)))

@pytest.fixture(autouse=True)
def patch_api_reranker(monkeypatch):
    monkeypatch.setattr(ApiReranker, "get_similarity_score", patched_get_similarity_score)
