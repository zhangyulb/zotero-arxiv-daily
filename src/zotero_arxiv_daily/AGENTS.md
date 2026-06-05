# OVERVIEW
Core package implementing the retrieval, ranking, and notification pipeline for academic papers.

# STRUCTURE
- `executor.py`: Main pipeline orchestrator connecting retrievers, rerankers, and email dispatch.
- `protocol.py`: Core data models (`Paper`, `CorpusPaper`) and LLM prompt logic for TLDRs and affiliations.
- `retriever/`: Source-specific fetchers (arXiv, bioRxiv, medRxiv) inheriting from `base.py`.
- `reranker/`: Scoring implementations (API-based or local) for evaluating paper relevance against the Zotero corpus.
- `construct_email.py`: HTML email rendering and templating logic.
- `utils.py`: Shared helpers for email dispatch and glob matching.
- `main.py`: Hydra-based entry point that configures logging and initializes the `Executor`.

# WHERE TO LOOK
- **Pipeline Logic**: `executor.py` defines the step-by-step execution flow (fetch -> score -> email).
- **Data Models & LLM Prompts**: `protocol.py` contains the `Paper` class which handles its own LLM-based TLDR and affiliation generation using `tiktoken` for truncation.
- **Adding Sources**: Look at `retriever/base.py` to implement new paper sources. Existing implementations use specific APIs (e.g., `arxiv_retriever.py`).
- **Adding Scoring**: Look at `reranker/base.py` to implement new similarity or LLM scoring methods.
- **Configuration**: `main.py` uses Hydra to load configurations from `../../config` and initialize the `Executor`.
- **Email Templates**: `construct_email.py` contains the HTML structure for the final output sent to users.
- **Utilities**: `utils.py` contains standalone functions like `send_email` and `glob_match` that don't depend on core models.

# ANTI-PATTERNS
- **Logging**: Do not use the standard `logging` module. Always use `loguru.logger`. The standard logger is explicitly suppressed in `main.py` for external libraries.
- **Configuration**: Avoid hardcoding parameters. Rely on the Hydra `DictConfig` passed through the `Executor`.
- **LLM Logic Placement**: Do not place prompt construction in the executor or retrievers; keep entity-specific LLM generation within the `Paper` class methods in `protocol.py`.
- **Token Limits**: Do not send unbounded text to the LLM. `protocol.py` explicitly truncates prompts using `tiktoken` before API calls to prevent context window overflow.
- **Error Handling**: Do not let LLM generation failures crash the pipeline. `protocol.py` catches exceptions and falls back to the abstract if TLDR generation fails.
- **Module Coupling**: Do not tightly couple retrievers and rerankers. They should remain independent and only interact through the `Executor` and `protocol.py` data models.
