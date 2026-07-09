"""
Shared state definition for the Scholar multi-agent system.

This TypedDict is the single source of truth passed between every node
in the LangGraph StateGraph. Every agent reads what it needs from this
state and writes its results back into it -- this is how the agents
coordinate without calling each other directly (see docs/architecture.md).
"""
from __future__ import annotations

from typing import TypedDict, List, Dict, Optional, Literal

AgentName = Literal[
    "supervisor",
    "search",
    "rag_index",
    "summarizer",
    "critique",
    "writer",
    "end",
]


class Paper(TypedDict, total=False):
    id: str
    title: str
    authors: List[str]
    published: str
    url: str
    abstract: str
    full_text: str  # abstract-only in v1; extendable to full PDF text


class Critique(TypedDict, total=False):
    paper_id: str
    strengths: str
    weaknesses: str
    gaps: str


class ResearchState(TypedDict, total=False):
    # --- Input ---
    topic: str
    max_papers: int
    max_writer_revisions: int

    # --- Planning (Supervisor) ---
    sub_questions: List[str]

    # --- Search agent output ---
    papers: List[Paper]

    # --- RAG agent output ---
    index_ready: bool
    retrieved_context: Dict[str, List[str]]  # sub_question -> chunks

    # --- Summarizer agent output ---
    paper_summaries: Dict[str, str]  # paper_id -> summary

    # --- Critique agent output ---
    critiques: List[Critique]

    # --- Writer agent output ---
    final_report: str
    writer_revisions: int
    writer_feedback: Optional[str]

    # --- Control / bookkeeping (Supervisor + failure handling) ---
    current_agent: AgentName
    next_agent: AgentName
    retry_counts: Dict[str, int]
    errors: List[str]
    status: Literal["in_progress", "complete", "failed"]
    log: List[str]


def new_state(topic: str, max_papers: int = 8, max_writer_revisions: int = 2) -> ResearchState:
    """Factory for a fresh, well-formed initial state."""
    return ResearchState(
        topic=topic,
        max_papers=max_papers,
        max_writer_revisions=max_writer_revisions,
        sub_questions=[],
        papers=[],
        index_ready=False,
        retrieved_context={},
        paper_summaries={},
        critiques=[],
        final_report="",
        writer_revisions=0,
        writer_feedback=None,
        current_agent="supervisor",
        next_agent="supervisor",
        retry_counts={},
        errors=[],
        status="in_progress",
        log=[],
    )
