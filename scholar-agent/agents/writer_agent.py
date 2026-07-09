"""
Writer Agent.

Responsibility: synthesize summaries + critiques into a single Markdown
literature review. Supports a revision loop: if the Supervisor's quality
gate rejects a draft, `writer_feedback` is populated and this node is
re-invoked with that feedback appended to its prompt (bounded by
`max_writer_revisions` to guarantee termination -- see graph/build_graph.py).
"""
from __future__ import annotations

import logging

from graph.state import ResearchState
from utils.llm import invoke_text
from utils.prompts import (
    WRITER_SYSTEM,
    WRITER_USER_TEMPLATE,
    WRITER_FEEDBACK_TEMPLATE,
)
from utils.retry import retry_with_backoff, RetryExhaustedError
from config import settings

logger = logging.getLogger("scholar")


def _build_summaries_block(state: ResearchState) -> str:
    papers = {p["id"]: p for p in state.get("papers", [])}
    lines = []
    for pid, summary in state.get("paper_summaries", {}).items():
        p = papers.get(pid, {})
        lines.append(
            f"[{pid}] {p.get('title', 'Unknown')} "
            f"({', '.join(p.get('authors', [])) or 'Unknown authors'}, "
            f"{p.get('published', 'n/a')}) - {p.get('url', '')}\n{summary}\n"
        )
    return "\n".join(lines)


def _build_critiques_block(state: ResearchState) -> str:
    if not state.get("critiques"):
        return "(no structured critique available for this run)"
    lines = []
    for c in state["critiques"]:
        lines.append(
            f"[{c.get('paper_id')}] Strength: {c.get('strengths', 'n/a')} | "
            f"Weakness: {c.get('weaknesses', 'n/a')} | "
            f"Relation/Gap: {c.get('gaps', 'n/a')}"
        )
    return "\n".join(lines)


@retry_with_backoff(
    max_attempts=settings.max_retries,
    base_delay=settings.retry_base_delay_s,
    label="writer_llm_call",
)
def _run_writer(prompt: str) -> str:
    return invoke_text(
        prompt,
        system=WRITER_SYSTEM,
        max_tokens=settings.writer_max_tokens,
        temperature=0.4,
    )


def writer_node(state: ResearchState) -> ResearchState:
    state["current_agent"] = "writer"

    sub_questions_block = "\n".join(f"- {q}" for q in state.get("sub_questions", [])) or "(none)"
    summaries_block = _build_summaries_block(state)
    critiques_block = _build_critiques_block(state)

    feedback_block = ""
    if state.get("writer_feedback"):
        feedback_block = WRITER_FEEDBACK_TEMPLATE.format(feedback=state["writer_feedback"])

    prompt = WRITER_USER_TEMPLATE.format(
        topic=state.get("topic", ""),
        sub_questions_block=sub_questions_block,
        summaries_block=summaries_block or "(no paper summaries available)",
        critiques_block=critiques_block,
        feedback_block=feedback_block,
    )

    try:
        report = _run_writer(prompt)
        state["final_report"] = report
        state["writer_revisions"] = state.get("writer_revisions", 0) + 1
        logger.info("[writer] draft produced (revision #%d)", state["writer_revisions"])
        state["log"] = state.get("log", []) + [
            f"writer: produced draft revision #{state['writer_revisions']}"
        ]
    except RetryExhaustedError as exc:
        msg = f"writer: LLM call failed after retries: {exc}"
        logger.error(msg)
        state["errors"] = state.get("errors", []) + [msg]
        if not state.get("final_report"):
            # Absolute last resort so the pipeline still produces *something*.
            state["final_report"] = (
                "# Literature Review (degraded output)\n\n"
                "The writing stage failed after repeated retries. Below are "
                "the raw paper summaries collected before the failure:\n\n"
                + summaries_block
            )
        state["status"] = "failed"

    return state
