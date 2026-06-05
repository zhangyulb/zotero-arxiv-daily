import re

with open('src/zotero_arxiv_daily/reranker/api.py', 'r') as f:
    content = f.read()

old_batch = '''        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=self.config.reranker.api.model
            )
            all_embeddings.extend([r.embedding for r in response.data])
        s1_embeddings = np.array(all_embeddings[:len(s1)])           # [n_s1, d]
        s2_embeddings = np.array(all_embeddings[len(s1):])           # [n_s2, d]
        s1_embeddings_normalized = s1_embeddings / np.linalg.norm(s1_embeddings, axis=1, keepdims=True)
        s2_embeddings_normalized = s2_embeddings / np.linalg.norm(s2_embeddings, axis=1, keepdims=True)'''

new_batch = '''        batch_sizes = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            batch_sizes.append(len(batch))
            try:
                embeddings = self._fetch_embeddings_batch(client, batch)
                all_embeddings.append(embeddings)
            except Exception as e:
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
        s2_embeddings_normalized = np.divide(s2_embeddings, norm2, out=np.zeros_like(s2_embeddings), where=norm2!=0)'''

content = content.replace(old_batch, new_batch)

with open('src/zotero_arxiv_daily/reranker/api.py', 'w') as f:
    f.write(content)
