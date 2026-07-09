"""
Builds the LangGraph StateGraph for Scholar.

Orchestration pattern: **Supervisor wrapping a Pipeline**, with one
feedback loop.

    supervisor_plan
          |
        search  --(no papers)--> end
          |
      rag_index
          |
     summarizer --(no summaries)--> end
          |
      critique
          |
       writer  <---------------------+
          |                          |
  supervisor_quality_gate --(REVISE, revisions left)--+
          |
        (PASS or revisions exhausted)
          |
         end

Why this hybrid instead of a "pure" pattern: a pure Pipeline can't recover
from a bad draft, and a pure Supervisor-dispatch (LLM decides every next
step from scratch) is needlessly expensive and less predictable for a
workflow that is *mostly* linear. The Supervisor here does the two things
a supervisor is actually good for: (a) high-level planning up front, and
(b) judging output quality and deciding whether to loop -- while the
well-understood middle steps run as a deterministic pipeline. This is
documented in more depth in docs/architecture.md.

Termination is guaranteed by `max_writer_revisions` bounding the
writer <-> quality_gate loop, and by explicit "end" routes whenever a
stage produces nothing usable.
"""
from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from graph.state import ResearchState
from agents.supervisor import (
    supervisor_plan_node,
    supervisor_quality_gate_node,
    mark_search_failed_node,
    mark_summarizer_failed_node,
    route_after_search,
    route_after_rag,
    route_after_summarizer,
    route_after_critique,
    route_after_quality_gate,
)
from agents.search_agent import search_node
from agents.rag_agent import rag_index_node
from agents.summarizer_agent import summarizer_node
from agents.critique_agent import critique_node
from agents.writer_agent import writer_node

logger = logging.getLogger("scholar")


def build_graph():
    """Compiles and returns the LangGraph app."""
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor_plan", supervisor_plan_node)
    graph.add_node("search", search_node)
    graph.add_node("rag_index", rag_index_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("critique", critique_node)
    graph.add_node("writer", writer_node)
    graph.add_node("supervisor_quality_gate", supervisor_quality_gate_node)
    graph.add_node("mark_search_failed", mark_search_failed_node)
    graph.add_node("mark_summarizer_failed", mark_summarizer_failed_node)

    graph.set_entry_point("supervisor_plan")

    graph.add_edge("supervisor_plan", "search")

    graph.add_conditional_edges(
        "search", route_after_search, {"rag_index": "rag_index", "fail": "mark_search_failed"}
    )
    graph.add_edge("mark_search_failed", END)

    graph.add_conditional_edges(
        "rag_index", route_after_rag, {"summarizer": "summarizer"}
    )
    graph.add_conditional_edges(
        "summarizer", route_after_summarizer,
        {"critique": "critique", "fail": "mark_summarizer_failed"},
    )
    graph.add_edge("mark_summarizer_failed", END)

    graph.add_conditional_edges(
        "critique", route_after_critique, {"writer": "writer"}
    )

    graph.add_edge("writer", "supervisor_quality_gate")
    graph.add_conditional_edges(
        "supervisor_quality_gate",
        route_after_quality_gate,
        {"writer": "writer", "end": END},
    )

    return graph.compile()


_compiled_app = None


def get_app():
    """Lazily compiles and caches the graph app (avoids recompiling per call)."""
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = build_graph()
    return _compiled_app
