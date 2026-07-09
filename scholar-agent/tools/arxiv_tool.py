"""
arXiv search tool.

Wraps the `arxiv` package so the Search agent can look up papers for a
given query. Wrapped in retry_with_backoff since the arXiv API is a
shared public service that occasionally throttles or times out.
"""
from __future__ import annotations

import logging
from typing import List

from config import settings
from graph.state import Paper
from utils.retry import retry_with_backoff, RetryExhaustedError

logger = logging.getLogger("scholar")


@retry_with_backoff(
    max_attempts=settings.max_retries,
    base_delay=settings.retry_base_delay_s,
    label="arxiv_search",
)
def _fetch_arxiv(query: str, max_results: int) -> List[Paper]:
    import arxiv  # imported lazily so the whole package doesn't hard-require arxiv

    client = arxiv.Client(page_size=max_results, delay_seconds=2, num_retries=1)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: List[Paper] = []
    for result in client.results(search):
        papers.append(
            Paper(
                id=result.entry_id.split("/")[-1],
                title=result.title.strip().replace("\n", " "),
                authors=[a.name for a in result.authors],
                published=str(result.published.date()) if result.published else "",
                url=result.entry_id,
                abstract=result.summary.strip().replace("\n", " "),
                full_text=result.summary.strip().replace("\n", " "),
            )
        )
    return papers


def search_papers(query: str, max_results: int = 8) -> List[Paper]:
    """
    Public entrypoint used by the Search agent.
    Returns [] (never raises) on total failure -- the Supervisor decides
    what to do with an empty result set, rather than crashing the run.
    """
    try:
        return _fetch_arxiv(query, max_results)
    except RetryExhaustedError as exc:
        logger.error("arXiv search exhausted retries for query %r: %s", query, exc)
        return []
