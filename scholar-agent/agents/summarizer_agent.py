"""
Summarizer Agent.

Responsibility: produce one grounded summary per paper, using the
abstract plus any RAG-retrieved supporting context. Deliberately does
NOT critique or compare papers -- that's the Critique agent's job. This
separation is what makes the system genuinely multi-agent rather than
"one agent doing everything in one long prompt".
"""
from __future__ import annotations

import logging

from graph.state import ResearchState
from agents.rag_agent import get_context_for_paper
from utils.llm import invoke_text
from utils.prompts import SUMMARIZER_SYSTEM, SUMMARIZER_USER_TEMPLATE
from utils.retry import retry_with_backoff, RetryExhaustedError
from config import settings

logger = logging.getLogger("scholar")


@retry_with_backoff(
    max_attempts=settings.max_retries,
    base_delay=settings.retry_base_delay_s,
    label="summarizer_llm_call",
)
def _summarize_one(paper, context: str) -> str:
    prompt = SUMMARIZER_USER_TEMPLATE.format(
        title=paper["title"],
        authors=", ".join(paper.get("authors", [])) or "Unknown",
        published=paper.get("published", "n/a"),
        abstract=paper.get("abstract", ""),
        context=context or "(none retrieved)",
    )
    return invoke_text(
        prompt,
        system=SUMMARIZER_SYSTEM,
        max_tokens=settings.summarizer_max_tokens,
        temperature=0.2,
    )


def summarizer_node(state: ResearchState) -> ResearchState:
    state["current_agent"] = "summarizer"
    papers = state.get("papers", [])
    summaries: dict[str, str] = {}
    errors = state.get("errors", [])

    for paper in papers:
        try:
            context = get_context_for_paper(state, paper["id"], state.get("topic", ""))
            summaries[paper["id"]] = _summarize_one(paper, context)
            logger.info("[summarizer] summarized paper %s", paper["id"])
        except RetryExhaustedError as exc:
            msg = f"summarizer: failed to summarize paper {paper['id']} after retries: {exc}"
            logger.error(msg)
            errors.append(msg)
            # Graceful degradation: fall back to using the raw abstract
            # rather than dropping the paper entirely from the review.
            summaries[paper["id"]] = f"(auto-fallback, LLM summary failed) {paper.get('abstract', '')[:400]}"

    state["paper_summaries"] = summaries
    state["errors"] = errors
    state["log"] = state.get("log", []) + [f"summarizer: produced {len(summaries)} summaries"]
    return state
