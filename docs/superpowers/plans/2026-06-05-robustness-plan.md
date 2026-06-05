# Robustness via Retries & Best-Effort Degradation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement robust error handling across API boundaries (arXiv RSS, arXiv metadata fetch, OpenAI embeddings, OpenAI Chat) using the `tenacity` library, allowing the pipeline to skip failed items but crash via `sys.exit(1)` if total failure occurs.

**Architecture:** We will introduce `tenacity` to provide exponential backoff retry logic. We will wrap the RSS fetch, batch paper metadata fetch, embedding API, and chat completion APIs. If retries fail, exceptions are caught, logged, and either default values are returned (scores/TLDRs) or batches are skipped (retrieval). A safety net in `main.py` relies on a `has_failures` flag returned by `executor.py` to catch total pipeline failures.

**Tech Stack:** Python, `uv` (dependency management), `tenacity` (retries), `arxiv`, `openai`, `pytest`.

---

### Task 1: Add Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `tenacity` to dependencies**
Run: `uv add tenacity`
Expected: Adds `tenacity` to `pyproject.toml` and updates `uv.lock`.

- [ ] **Step 2: Commit**
```bash
git add pyproject.toml uv.lock
git commit -m "build: add tenacity dependency for retry logic"
```

---

### Task 2: Base Classes & Executor Safety Net

**Files:**
- Modify: `src/zotero_arxiv_daily/retriever/base.py`
- Modify: `src/zotero_arxiv_daily/reranker/base.py`
- Modify: `src/zotero_arxiv_daily/executor.py`
- Modify: `src/zotero_arxiv_daily/main.py`
- Modify: `tests/test_executor.py`

- [ ] **Step 1: Write the failing test for executor safety net**
Add this test to `tests/test_executor.py`:

```python
def test_run_returns_has_failures_flag(config, monkeypatch):
    from zotero_arxiv_daily.executor import Executor
    from tests.canned_responses import make_stub_zotero_client
    from zotero_arxiv_daily.retriever.base import registered_retrievers
    
    stub_zot = make_stub_zotero_client()
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)
    
    def mock_retrieve(self):
        self.has_failures = True
        return []
        
    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", mock_retrieve)
    
    executor = Executor(config)
    has_failures = executor.run()
    assert has_failures is True
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_executor.py::test_run_returns_has_failures_flag -v`
Expected: FAIL

- [ ] **Step 3: Add base flags**
In `src/zotero_arxiv_daily/retriever/base.py`, add the `has_failures` field below the `name` field in the `BaseRetriever` class:
```python
class BaseRetriever(ABC):
    name: str
    has_failures: bool = False
```

In `src/zotero_arxiv_daily/reranker/base.py`, add the `has_failures` field inside the `BaseReranker` class:
```python
class BaseReranker(ABC):
    has_failures: bool = False
```

- [ ] **Step 4: Track failures in `executor.py`**
In `src/zotero_arxiv_daily/executor.py`, update `run()` to track and return `has_failures`:

```python
    def run(self):
        has_failures = False
        self.papers = []
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return True
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            try:
                papers = retriever.retrieve_papers()
                if hasattr(retriever, 'has_failures') and retriever.has_failures:
                    has_failures = True
                if len(papers) == 0:
                    logger.info(f"No {source} papers found")
                    continue
                logger.info(f"Retrieved {len(papers)} {source} papers")
                all_papers.extend(papers)
            except Exception as e:
                logger.error(f"Retriever {source} failed completely: {e}")
                has_failures = True
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        self.papers = all_papers
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            try:
                reranked_papers = self.reranker.rerank(all_papers, corpus)
                if hasattr(self.reranker, 'has_failures') and self.reranker.has_failures:
                    has_failures = True
            except Exception as e:
                logger.error(f"Reranker failed completely: {e}")
                has_failures = True
                reranked_papers = []
            
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                try:
                    p.generate_tldr(self.openai_client, self.config.llm)
                    if getattr(p, 'has_failures', False):
                        has_failures = True
                except Exception as e:
                    logger.error(f"Failed to generate TLDR for {p.title}: {e}")
                    has_failures = True
                try:
                    p.generate_affiliations(self.openai_client, self.config.llm)
                    if getattr(p, 'has_failures', False):
                        has_failures = True
                except Exception as e:
                    logger.error(f"Failed to generate affiliations for {p.title}: {e}")
                    has_failures = True
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return has_failures
        
        try:
            logger.info("Sending email...")
            email_content = render_email(reranked_papers)
            send_email(self.config, email_content)
            logger.info("Email sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            has_failures = True
            
        return has_failures
```

- [ ] **Step 5: Implement hard fail in `main.py`**
In `src/zotero_arxiv_daily/main.py`, update `main()` to use the returned flag:

Find this block in `main`:
```python
    executor = Executor(config)
    executor.run()
```

Replace it with:
```python
    executor = Executor(config)
    has_failures = executor.run()
    if has_failures and len(executor.papers) == 0:
        sys.exit(1)
```

- [ ] **Step 6: Run test to verify it passes**
Run: `uv run pytest tests/test_executor.py::test_run_returns_has_failures_flag -v`
Expected: PASS

- [ ] **Step 7: Commit**
```bash
git add src/zotero_arxiv_daily/retriever/base.py src/zotero_arxiv_daily/reranker/base.py src/zotero_arxiv_daily/executor.py src/zotero_arxiv_daily/main.py tests/test_executor.py
git commit -m "feat: implement total failure safety net via executor flags"
```

---

### Task 3: Harden Biorxiv Retriever

**Files:**
- Modify: `src/zotero_arxiv_daily/retriever/biorxiv_retriever.py`
- Modify: `tests/retriever/test_biorxiv_retriever.py`

- [ ] **Step 1: Write the failing test**
Add this test to `tests/retriever/test_biorxiv_retriever.py`:

```python
def test_biorxiv_api_failure_degrades_gracefully(config, monkeypatch):
    import requests
    from zotero_arxiv_daily.retriever.biorxiv_retriever import BiorxivRetriever
    from omegaconf import open_dict
    
    def _patched_get(*args, **kwargs):
        raise requests.exceptions.RequestException("API down")
        
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(requests, "get", _patched_get)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    
    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
        
    retriever = BiorxivRetriever(config)
    papers = retriever._retrieve_raw_papers()
    
    assert papers == []
    assert retriever.has_failures is True
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/retriever/test_biorxiv_retriever.py::test_biorxiv_api_failure_degrades_gracefully -v`
Expected: FAIL

- [ ] **Step 3: Extract API Call with Tenacity**
In `src/zotero_arxiv_daily/retriever/biorxiv_retriever.py`, import `tenacity` and extract the `requests.get` call into a new method `_fetch_api_page` decorated with `@retry`.

```python
from tenacity import retry, wait_exponential, stop_after_attempt

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10), reraise=True)
    def _fetch_api_page(self, api_url: str) -> requests.Response:
        response = requests.get(api_url)
        response.raise_for_status()
        return response
```

- [ ] **Step 4: Update `_retrieve_raw_papers` to degrade gracefully**
In `src/zotero_arxiv_daily/retriever/biorxiv_retriever.py`, wrap the logic in a `try/except` block that returns `raw_papers` if it fails.

```python
    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        api_url = f"https://api.biorxiv.org/details/{self.server}/2d"
        raw_papers = []
        try:
            response = self._fetch_api_page(api_url)
            result = response.json()
            collection = result.get('collection', [])
            if len(collection) == 0:
                logger.warning(f"No paper found. API Message: {result.get('messages')}")
                return raw_papers
            all_dates = set(c['date'] for c in collection)
            latest_date = sorted(all_dates)[-1]
            collection = [c for c in collection if c['date'] == latest_date]
            categories = [c.lower() for c in self.retriever_config.category]
            raw_papers = [c for c in collection if c['category'] in categories]
            if self.config.executor.debug:
                raw_papers = raw_papers[:10]
        except Exception as e:
            logger.error(f"Failed to retrieve papers: {str(e)}")
            self.has_failures = True
            
        return raw_papers
```

- [ ] **Step 5: Run test to verify it passes**
Run: `uv run pytest tests/retriever/test_biorxiv_retriever.py::test_biorxiv_api_failure_degrades_gracefully -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add src/zotero_arxiv_daily/retriever/biorxiv_retriever.py tests/retriever/test_biorxiv_retriever.py
git commit -m "feat: use tenacity for biorxiv API retries"
```

---

### Task 4: Harden Arxiv Retrieval

**Files:**
- Modify: `src/zotero_arxiv_daily/retriever/arxiv_retriever.py`
- Modify: `tests/retriever/test_arxiv_retriever.py`

- [ ] **Step 1: Write the failing test**
Add this test to `tests/retriever/test_arxiv_retriever.py`:

```python
def test_arxiv_feed_failure_degrades_gracefully(config, monkeypatch):
    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
    import feedparser
    
    def _patched_parse(*args, **kwargs):
        raise Exception("Feed down")
        
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(feedparser, "parse", _patched_parse)
    
    retriever = ArxivRetriever(config)
    papers = retriever._retrieve_raw_papers()
    
    assert papers == []
    assert retriever.has_failures is True
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/retriever/test_arxiv_retriever.py::test_arxiv_feed_failure_degrades_gracefully -v`
Expected: FAIL

- [ ] **Step 3: Create `_fetch_feed` and `_fetch_arxiv_batch` helpers**
Extract the API calls into new methods decorated with `@retry`.

```python
from tenacity import retry, wait_exponential, stop_after_attempt

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3), reraise=True)
    def _fetch_feed(self, query: str):
        import feedparser
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if getattr(feed, 'status', None) not in (200, 301, 302):
            raise ValueError(f"RSS feed returned status {getattr(feed, 'status', None)}")
        if getattr(feed, "bozo", 0) == 1:
            raise ValueError("RSS feed is malformed (bozo=1)")
        return feed

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3), reraise=True)
    def _fetch_arxiv_batch(self, client: arxiv.Client, search: arxiv.Search):
        return list(client.results(search))
```

- [ ] **Step 4: Update `_retrieve_raw_papers` to use helpers and degrade gracefully**
In `src/zotero_arxiv_daily/retriever/arxiv_retriever.py`, use the new helpers and wrap them in `try/except` blocks.

```python
    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=0)
        
        if self.config.source.arxiv.category:
            query = '+'.join(self.config.source.arxiv.category)
        else:
            query = self.arxiv_query
            
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        try:
            feed = self._fetch_feed(query)
            if 'Feed error for query' in feed.feed.title:
                raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed: {e}")
            self.has_failures = True
            return []
            
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        import time
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i:i + 20])
            try:
                batch = self._fetch_arxiv_batch(client, search)
                bar.update(len(batch))
                raw_papers.extend(batch)
                time.sleep(3)
            except Exception as e:
                logger.error(f"Failed to fetch batch: {e}")
                self.has_failures = True
                bar.update(min(20, len(all_paper_ids) - i))
                continue
        bar.close()

        return raw_papers
```

- [ ] **Step 5: Run test to verify it passes**
Run: `uv run pytest tests/retriever/test_arxiv_retriever.py::test_arxiv_feed_failure_degrades_gracefully -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add src/zotero_arxiv_daily/retriever/arxiv_retriever.py tests/retriever/test_arxiv_retriever.py
git commit -m "feat: add retries and graceful degradation to arXiv fetching"
```

---

### Task 5: Harden LLM Embeddings (Reranker)

**Files:**
- Modify: `src/zotero_arxiv_daily/reranker/api.py`
- Modify: `tests/reranker/test_api_reranker.py`

- [ ] **Step 1: Write the failing test**
Add this test to `tests/reranker/test_api_reranker.py` :

```python
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
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/reranker/test_api_reranker.py::test_api_reranker_failure_degrades_gracefully -v`
Expected: FAIL

- [ ] **Step 3: Create `_fetch_embeddings_batch` helper with retries**
In `src/zotero_arxiv_daily/reranker/api.py`, extract the OpenAI API call into a decorated helper.

```python
from tenacity import retry, wait_exponential, stop_after_attempt

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3), reraise=True)
    def _fetch_embeddings_batch(self, client, texts_batch: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            input=texts_batch,
            model=self.config.reranker.api.model
        )
        return [res.embedding for res in response.data]
```

- [ ] **Step 4: Update `get_similarity_score`**
In `src/zotero_arxiv_daily/reranker/api.py`, use the helper in a `try/except`, appending `None` on failure. Afterwards, replace `None` with `np.zeros` matching the inferred dimension.

```python
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
        s2_embeddings_normalized = np.divide(s2_embeddings, norm2, out=np.zeros_like(s2_embeddings), where=norm2!=0)
        
        sim = np.dot(s1_embeddings_normalized, s2_embeddings_normalized.T) # [n_s1, n_s2]
        return sim
```

- [ ] **Step 5: Run test to verify it passes**
Run: `uv run pytest tests/reranker/test_api_reranker.py::test_api_reranker_failure_degrades_gracefully -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add src/zotero_arxiv_daily/reranker/api.py tests/reranker/test_api_reranker.py
git commit -m "feat: add retries and safety bounds to OpenAI embedding calls"
```

---

### Task 6: Harden LLM Chat (Protocol)

**Files:**
- Modify: `src/zotero_arxiv_daily/protocol.py`
- Modify: `tests/test_protocol.py`

- [ ] **Step 1: Add `has_failures` flag to `Paper` dataclass**
In `src/zotero_arxiv_daily/protocol.py`, add `has_failures: bool = False` as the final field of the `Paper` dataclass:

```python
    has_failures: bool = False
```

- [ ] **Step 2: Write the failing test**
Add this test to `tests/test_protocol.py`:

```python
def test_paper_tldr_failure_degrades_gracefully(monkeypatch):
    from zotero_arxiv_daily.protocol import Paper
    from openai import OpenAI
    
    paper = Paper(
        source="arxiv",
        title="Test Paper",
        authors=["Author"],
        abstract="Test abstract",
        url="http://test",
    )
    
    def _patched_create(*args, **kwargs):
        raise Exception("API down")
        
    client = OpenAI(api_key="test")
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(client.chat.completions, "create", _patched_create)
    
    tldr = paper.generate_tldr(client, {})
    
    assert tldr == "Test abstract"
    assert paper.has_failures is True
```

- [ ] **Step 3: Run test to verify it fails**
Run: `uv run pytest tests/test_protocol.py::test_paper_tldr_failure_degrades_gracefully -v`
Expected: FAIL

- [ ] **Step 4: Add retries to LLM calls**
In `src/zotero_arxiv_daily/protocol.py`, import `tenacity` and apply the `@retry` decorator directly to `_generate_tldr_with_llm` and `_generate_affiliations_with_llm`.

```python
from tenacity import retry, wait_exponential, stop_after_attempt

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3), reraise=True)
    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr
```

```python
    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3), reraise=True)
    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
```

- [ ] **Step 5: Update try/except blocks to set `has_failures`**
In `src/zotero_arxiv_daily/protocol.py`, ensure `generate_tldr` and `generate_affiliations` catch exceptions, set `has_failures`, and proceed with fallbacks.

```python
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            self.has_failures = True
            tldr = self.abstract
            self.tldr = tldr
            return tldr
```

```python
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.has_failures = True
            self.affiliations = None
            return None
```

- [ ] **Step 6: Run test to verify it passes**
Run: `uv run pytest tests/test_protocol.py::test_paper_tldr_failure_degrades_gracefully -v`
Expected: PASS

- [ ] **Step 7: Commit**
```bash
git add src/zotero_arxiv_daily/protocol.py tests/test_protocol.py
git commit -m "feat: add retries to OpenAI chat completion calls"
```
