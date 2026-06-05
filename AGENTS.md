SPECIAL INSTRUCTION: think silently if needed. EFFORT LEVEL: 2.0
use rg over grep for faster content search

# OVERVIEW
Flat Python script collection. 
Fetches arXiv papers, scores via LLM, emails recommendations based on Zotero library. 
Runs primarily via GitHub Actions.

# WHERE TO LOOK
- `/`: All Python logic.
- `.github/workflows/`: CI/CD and testing workflows (`test.yml`).

# CODE MAP
- `main.py`: Entry point. Orchestrates fetch, score, email.
- `paper.py`: Paper data models, parsing logic.
- `llm.py`: LLM integration for scoring, TL;DR generation.
- `recommender.py`: Recommendation, similarity scoring logic.
- `construct_email.py`: Email formatting, construction.

# CONVENTIONS
- **Architecture**: Flat structure. All source files in root.
- **Dependencies**: `uv` used as dependency runner. No traditional build system.
- **Testing**: No traditional testing framework. Relies on manual GitHub Actions test workflow (`test.yml`).
- **Formatting/Linting**: None. Match existing code style.

# ANTI-PATTERNS (THIS PROJECT)
- **LLM Output**: LLM must NOT return intermediate results.
- **Data Models**: Paper data shouldn't contain duplicated affiliations (per `paper.py`).

# COMMANDS
- Run locally: `uv run main.py` (requires env vars)
