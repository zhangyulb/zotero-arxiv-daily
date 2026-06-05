from .base import BaseReranker, register_reranker
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
import numpy as np
@register_reranker("api")
class ApiReranker(BaseReranker):

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3), reraise=True)
    def _fetch_embeddings_batch(self, client, texts_batch: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            input=texts_batch,
            model=self.config.reranker.api.model
        )
        return [res.embedding for res in response.data]

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        client = OpenAI(api_key=self.config.reranker.api.key, base_url=self.config.reranker.api.base_url)
        batch_size = self.config.reranker.api.get("batch_size") or 64
        all_texts = s1 + s2
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=self.config.reranker.api.model
            )
            all_embeddings.extend([r.embedding for r in response.data])
        s1_embeddings = np.array(all_embeddings[:len(s1)])           # [n_s1, d]
        s2_embeddings = np.array(all_embeddings[len(s1):])           # [n_s2, d]
        s1_embeddings_normalized = s1_embeddings / np.linalg.norm(s1_embeddings, axis=1, keepdims=True)
        s2_embeddings_normalized = s2_embeddings / np.linalg.norm(s2_embeddings, axis=1, keepdims=True)
        sim = np.dot(s1_embeddings_normalized, s2_embeddings_normalized.T) # [n_s1, n_s2]
        return sim
