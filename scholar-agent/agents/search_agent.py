"""
Search Agent.

Responsibility: given the topic (and the Supervisor's sub-questions),
query arXiv and populate `state["papers"]`. This agent owns *discovery*
only -- it does not summarize or judge quality, which keeps its
responsibility boundary clean (Week 4: division of responsibility).
"""
from __future__ import annotations

import logging

from graph.state import ResearchState
from tools.arxiv_tool import search_papers

logger = logging.getLogger("scholar")


def search_node(state: ResearchState) -> ResearchState:
    state["current_agent"] = "search"
    topic = state["topic"]
    max_papers = state.get("max_papers", 8)

    logger.info("[search] querying arXiv for topic=%r (max_papers=%d)", topic, max_papers)
    papers = search_papers(topic, max_results=max_papers)

    if not papers:
        msg = f"search: no papers found for topic {topic!r} (arXiv unavailable or no matches)"
        logger.warning(msg)
        state["errors"] = state.get("errors", []) + [msg]
    else:
        logger.info("[search] found %d papers", len(papers))

    state["papers"] = papers
    state["log"] = state.get("log", []) + [f"search: retrieved {len(papers)} papers"]
    return state
