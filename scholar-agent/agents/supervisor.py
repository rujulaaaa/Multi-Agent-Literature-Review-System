"""
Supervisor Agent.

This is the orchestration core of Scholar (Week 4: agent orchestration,
task routing and handoff, failure handling). It has three jobs, each
implemented as its own node so the graph's control flow is explicit and
inspectable rather than hidden inside one giant function:

1. `supervisor_plan_node`   - LLM call: break the topic into sub-questions.
   Runs once, at the start.
2. `route_after_search`,
   `route_after_rag`,
   `route_after_summarizer`,
   `route_after_critique`    - deterministic, state-based routing
   functions (not LLM calls) used as LangGraph conditional edges. These
   also implement failure handling: if a stage produced nothing useful,
   the Supervisor decides whether to retry, skip ahead in degraded mode,
   or abort.
3. `supervisor_quality_gate_node` - LLM call: after the Writer produces a
   draft, judge whether it's good enough. If not (and revisions remain),
   route back to the Writer with concrete feedback -- otherwise finish.

Keeping routing decisions deterministic (rule-based) while planning and
quality-gating are LLM-driven is a deliberate design choice: it makes the
control flow reliable and testable while still using the LLM where
judgment genuinely helps. See docs/architecture.md for the full rationale.
"""
from __future__ import annotations

import logging

from graph.state import ResearchState
from utils.llm import invoke_text
from utils.prompts import (
    PLANNER_SYSTEM,
    PLANNER_USER_TEMPLATE,
    SUPERVISOR_QUALITY_SYSTEM,
    SUPERVISOR_QUALITY_USER_TEMPLATE,
)
from utils.retry import retry_with_backoff, RetryExhaustedError
from config import settings

logger = logging.getLogger("scholar")


# --------------------------------------------------------------------------
# 1. Planning
# --------------------------------------------------------------------------

@retry_with_backoff(
    max_attempts=settings.max_retries,
    base_delay=settings.retry_base_delay_s,
    label="planner_llm_call",
)
def _plan_sub_questions(topic: str) -> list[str]:
    prompt = PLANNER_USER_TEMPLATE.format(topic=topic)
    raw = invoke_text(
        prompt,
        system=PLANNER_SYSTEM,
        max_tokens=settings.planner_max_tokens,
        temperature=0.3,
    )
    questions = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip leading numbering like "1." or "1)" or "- "
        cleaned = line.lstrip("0123456789.-) ").strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:5] if questions else [topic]


def supervisor_plan_node(state: ResearchState) -> ResearchState:
    state["current_agent"] = "supervisor"
    try:
        sub_questions = _plan_sub_questions(state["topic"])
        logger.info("[supervisor] planned %d sub-questions", len(sub_questions))
    except RetryExhaustedError as exc:
        msg = f"supervisor_plan: failed after retries, falling back to topic-only: {exc}"
        logger.error(msg)
        state["errors"] = state.get("errors", []) + [msg]
        sub_questions = [state["topic"]]

    state["sub_questions"] = sub_questions
    state["log"] = state.get("log", []) + [f"supervisor: planned {len(sub_questions)} sub-questions"]
    return state


# --------------------------------------------------------------------------
# 2. Deterministic routing (conditional edges)
# --------------------------------------------------------------------------

def route_after_search(state: ResearchState) -> str:
    """If search found nothing, there is no point continuing the pipeline.

    NOTE: LangGraph conditional-edge functions are routing-only -- any
    mutation made to `state` here is NOT persisted back into the graph's
    channels. Terminal-failure bookkeeping is therefore done in the
    dedicated `mark_search_failed_node` / `mark_summarizer_failed_node`
    nodes below, which are routed to before the graph ends.
    """
    if not state.get("papers"):
        logger.error("[supervisor] no papers found -- aborting run")
        return "fail"
    return "rag_index"


def route_after_rag(state: ResearchState) -> str:
    """RAG indexing failing is non-fatal: summarizer can still work off raw abstracts."""
    if not state.get("index_ready"):
        logger.warning("[supervisor] RAG index not ready -- continuing without retrieval grounding")
    return "summarizer"


def route_after_summarizer(state: ResearchState) -> str:
    if not state.get("paper_summaries"):
        logger.error("[supervisor] no summaries produced -- aborting run")
        return "fail"
    return "critique"


def mark_search_failed_node(state: ResearchState) -> ResearchState:
    """Terminal node reached when Search found nothing. Persists status='failed'."""
    state["status"] = "failed"
    state["current_agent"] = "supervisor"
    state["log"] = state.get("log", []) + ["supervisor: aborted run (no papers found)"]
    return state


def mark_summarizer_failed_node(state: ResearchState) -> ResearchState:
    """Terminal node reached when Summarizer produced nothing. Persists status='failed'."""
    state["status"] = "failed"
    state["current_agent"] = "supervisor"
    state["log"] = state.get("log", []) + ["supervisor: aborted run (no summaries produced)"]
    return state


def route_after_critique(state: ResearchState) -> str:
    # Critique is a nice-to-have for depth; always proceed to writing even
    # if it came back empty (writer_agent handles that case).
    return "writer"


# --------------------------------------------------------------------------
# 3. Quality gate + revision loop
# --------------------------------------------------------------------------

@retry_with_backoff(
    max_attempts=settings.max_retries,
    base_delay=settings.retry_base_delay_s,
    label="quality_gate_llm_call",
)
def _judge_quality(summaries_block: str, draft: str) -> tuple[bool, str]:
    prompt = SUPERVISOR_QUALITY_USER_TEMPLATE.format(
        summaries_block=summaries_block, draft=draft
    )
    raw = invoke_text(
        prompt,
        system=SUPERVISOR_QUALITY_SYSTEM,
        max_tokens=400,
        temperature=0.0,
    )
    first_line = raw.strip().splitlines()[0].strip().upper() if raw.strip() else "PASS"
    passed = first_line.startswith("PASS")
    feedback = "\n".join(raw.strip().splitlines()[1:]).strip()
    return passed, feedback


def supervisor_quality_gate_node(state: ResearchState) -> ResearchState:
    """
    Reviews the Writer's draft. On REVISE (and revisions remaining), sets
    `writer_feedback` so the graph loops back to the Writer node. On PASS,
    or once `max_writer_revisions` is exhausted, marks the run complete.
    """
    state["current_agent"] = "supervisor"
    from agents.writer_agent import _build_summaries_block  # local import avoids cycle

    summaries_block = _build_summaries_block(state)
    draft = state.get("final_report", "")
    revisions = state.get("writer_revisions", 0)
    max_revisions = state.get("max_writer_revisions", settings.max_writer_revisions)

    if state.get("status") == "failed":
        # Writer already hit an unrecoverable error; don't loop further.
        return state

    if not draft:
        state["status"] = "failed"
        return state

    try:
        passed, feedback = _judge_quality(summaries_block, draft)
    except RetryExhaustedError as exc:
        msg = f"quality_gate: judge call failed after retries, accepting draft as-is: {exc}"
        logger.warning(msg)
        state["errors"] = state.get("errors", []) + [msg]
        passed, feedback = True, ""

    if passed or revisions >= max_revisions:
        logger.info(
            "[supervisor] quality gate: %s (revision %d/%d)",
            "PASS" if passed else "MAX REVISIONS REACHED, accepting",
            revisions, max_revisions,
        )
        state["status"] = "complete"
        state["writer_feedback"] = None
    else:
        logger.info("[supervisor] quality gate: REVISE (revision %d/%d) -- feedback: %s",
                    revisions, max_revisions, feedback)
        state["writer_feedback"] = feedback

    state["log"] = state.get("log", []) + [
        f"supervisor_quality_gate: {'PASS' if state['status'] == 'complete' else 'REVISE'}"
    ]
    return state


def route_after_quality_gate(state: ResearchState) -> str:
    if state.get("status") == "complete" or state.get("status") == "failed":
        return "end"
    return "writer"  # loop back for another revision
