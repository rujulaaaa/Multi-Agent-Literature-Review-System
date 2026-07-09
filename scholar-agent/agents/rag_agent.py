"""
RAG (Retrieval) Agent.

Responsibility: build a vector index over the fetched papers' abstracts
and, for each sub-question the Supervisor planned, retrieve the most
relevant chunks. This is the Week 2 (RAG) skill folded into the Week 4
multi-agent system, exactly as the brief suggests ("combine with RAG...
if it strengthens the project").

The retrieved context is later handed to the Summarizer agent so paper
summaries can be grounded not just in their own abstract but in how they
relate to the topic's sub-questions.
"""
from __future__ import annotations

import logging

from graph.state import ResearchState
from tools.vector_store import ScholarVectorStore
from config import settings

logger = logging.getLogger("scholar")

# Module-level store reused across a single process run. For a single
# literature-review request this is fine; a server deployment would key
# this per-request/session (see mcp_server/server.py for that handling).
#
# `_store_factory` is indirected through a module-level function (rather
# than instantiating ScholarVectorStore() inline) purely so tests can
# monkeypatch it with a lightweight fake and avoid pulling in the real
# sentence-transformers/FAISS dependency chain.
def _store_factory() -> ScholarVectorStore:
    return ScholarVectorStore()


_store = _store_factory()


def rag_index_node(state: ResearchState) -> ResearchState:
    state["current_agent"] = "rag_index"
    papers = state.get("papers", [])

    if not papers:
        logger.warning("[rag_index] no papers to index; skipping")
        state["index_ready"] = False
        state["log"] = state.get("log", []) + ["rag_index: skipped (no papers)"]
        return state

    global _store
    _store = _store_factory()  # fresh store per run
    total_chunks = 0
    for paper in papers:
        text = paper.get("full_text") or paper.get("abstract", "")
        added = _store.add_documents(paper["id"], text)
        total_chunks += added

    state["index_ready"] = _store.is_ready
    logger.info("[rag_index] indexed %d chunks from %d papers", total_chunks, len(papers))

    # Retrieve context per sub-question for later grounding
    retrieved: dict[str, list[str]] = {}
    for sq in state.get("sub_questions", []):
        hits = _store.query(sq, top_k=settings.top_k_retrieval)
        retrieved[sq] = [chunk for _, chunk, _ in hits]

    state["retrieved_context"] = retrieved
    state["log"] = state.get("log", []) + [
        f"rag_index: indexed {total_chunks} chunks, retrieved context for "
        f"{len(retrieved)} sub-questions"
    ]
    return state


def get_context_for_paper(state: ResearchState, paper_id: str, question_hint: str = "") -> str:
    """Helper used by the Summarizer agent to pull extra grounding context."""
    if not _store.is_ready:
        return ""
    query = question_hint or paper_id
    hits = _store.query(query, top_k=settings.top_k_retrieval)
    relevant = [chunk for pid, chunk, _ in hits if pid == paper_id]
    return "\n".join(relevant[:2])
