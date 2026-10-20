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
        batch_sizes = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            batch_sizes.append(len(batch))
            try:
                embeddings = self._fetch_embeddings_batch(client, batch)
                all_embeddings.append(embeddings)
            except Exception:
                self.has_failures = True
                all_embeddings.append(None)
                
        dimension = next((len(b[0]) for b in all_embeddings if b is not None and len(b) > 0), None)
        if dimension is None:
            return np.zeros((len(s1), len(s2)))
            
        final_embeddings = []
        for batch, size in zip(all_embeddings, batch_sizes):
            if batch is None:
                final_embeddings.extend(np.zeros((size, dimension)).tolist())
            else:
                final_embeddings.extend(batch)
                
        s1_embeddings = np.array(final_embeddings[:len(s1)])           # [n_s1, d]
        s2_embeddings = np.array(final_embeddings[len(s1):])           # [n_s2, d]
        
        norm1 = np.linalg.norm(s1_embeddings, axis=1, keepdims=True)
        norm2 = np.linalg.norm(s2_embeddings, axis=1, keepdims=True)
        
        s1_embeddings_normalized = np.divide(s1_embeddings, norm1, out=np.zeros_like(s1_embeddings), where=norm1!=0)
        s2_embeddings_normalized = np.divide(s2_embeddings, norm2, out=np.zeros_like(s2_embeddings), where=norm2!=0)
        sim = np.dot(s1_embeddings_normalized, s2_embeddings_normalized.T) # [n_s1, n_s2]
        return sim
