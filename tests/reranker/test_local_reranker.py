"""Tests for LocalReranker — requires sentence-transformers, marked slow."""

import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from zotero_arxiv_daily.reranker.local import LocalReranker


@pytest.mark.slow
def test_local_reranker(config):
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)


def test_local_reranker_casts_cpu_model_to_float32(config, monkeypatch):
    """On CPU, bfloat16 models must be cast to float32 to avoid 15x software BF16 slowdown."""
    cast_dtypes = []

    class FakeEncoder:
        def __init__(self, model_name, **kwargs):
            self.device = SimpleNamespace(type="cpu")

        def to(self, dtype):
            cast_dtypes.append(dtype)
            return self

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

        def similarity(self, f1, f2):
            return torch.ones((len(f1), len(f2)), dtype=torch.float32)

    fake_st_mod = SimpleNamespace(SentenceTransformer=FakeEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)

    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["paper 1", "paper 2"], ["corpus 1"])
    assert score.shape == (2, 1)
    assert cast_dtypes == [torch.float32]

