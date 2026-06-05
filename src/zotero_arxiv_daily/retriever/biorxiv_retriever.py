import requests
from .base import BaseRetriever, register_retriever
from ..protocol import Paper
from loguru import logger
from typing import Any
from tenacity import retry, wait_exponential, stop_after_attempt

@register_retriever("biorxiv")
class BiorxivRetriever(BaseRetriever):
    server = "biorxiv"

    def __init__(self, config):
        super().__init__(config)
        if self.retriever_config.category is None:
            raise ValueError(f"category must be specified for {self.name}")

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10), reraise=True)
    def _fetch_api_page(self, api_url: str) -> requests.Response:
        response = requests.get(api_url)
        response.raise_for_status()
        return response

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


    def convert_to_paper(self, raw_paper:dict[str, Any]) -> Paper | None:
        title = raw_paper['title']
        authors = [a.strip() for a in raw_paper['authors'].split(';')]
        abstract = raw_paper['abstract']
        pdf_url = f"https://www.{self.server}.org/content/{raw_paper['doi']}v{raw_paper['version']}.full.pdf"
        full_text = None # biorxiv forbids scraping its pdf
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=pdf_url,
            pdf_url=pdf_url,
            full_text=full_text
        )