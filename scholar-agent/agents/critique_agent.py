"""
Critique Agent.

Responsibility: given all paper summaries together (not one at a time),
identify strengths, weaknesses, and how each paper relates to the others
(agreement, contradiction, or an open gap). This is only possible because
it operates on the *aggregate* of the Summarizer's output -- a clean
example of a Parallel+Aggregator-style hand-off feeding into a later
pipeline stage, nested inside the overall Supervisor orchestration.
"""
from __future__ import annotations

import logging

from graph.state import ResearchState, Critique
from utils.llm import invoke_text
from utils.prompts import CRITIQUE_SYSTEM, CRITIQUE_USER_TEMPLATE
from utils.retry import retry_with_backoff, RetryExhaustedError
from config import settings

logger = logging.getLogger("scholar")


def _build_summaries_block(state: ResearchState) -> str:
    papers = {p["id"]: p for p in state.get("papers", [])}
    lines = []
    for pid, summary in state.get("paper_summaries", {}).items():
        title = papers.get(pid, {}).get("title", "Unknown title")
        lines.append(f"[{pid}] {title}\n{summary}\n")
    return "\n".join(lines)


@retry_with_backoff(
    max_attempts=settings.max_retries,
    base_delay=settings.retry_base_delay_s,
    label="critique_llm_call",
)
def _run_critique(topic: str, summaries_block: str) -> str:
    prompt = CRITIQUE_USER_TEMPLATE.format(topic=topic, summaries_block=summaries_block)
    return invoke_text(
        prompt,
        system=CRITIQUE_SYSTEM,
        max_tokens=settings.critique_max_tokens,
        temperature=0.3,
    )


def _parse_critiques(raw: str) -> list[Critique]:
    """Parses the pipe-delimited PAPER_ID/STRENGTH/WEAKNESS/GAP blocks."""
    critiques: list[Critique] = []
    blocks = [b.strip() for b in raw.split("---") if b.strip()]
    for block in blocks:
        entry: dict = {}
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("PAPER_ID:"):
                entry["paper_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("STRENGTH:"):
                entry["strengths"] = line.split(":", 1)[1].strip()
            elif line.startswith("WEAKNESS:"):
                entry["weaknesses"] = line.split(":", 1)[1].strip()
            elif line.startswith("GAP_OR_RELATION:"):
                entry["gaps"] = line.split(":", 1)[1].strip()
        if entry.get("paper_id"):
            critiques.append(Critique(**entry))
    return critiques


def critique_node(state: ResearchState) -> ResearchState:
    state["current_agent"] = "critique"
    summaries_block = _build_summaries_block(state)

    if not summaries_block:
        logger.warning("[critique] no summaries available; skipping critique")
        state["critiques"] = []
        state["log"] = state.get("log", []) + ["critique: skipped (no summaries)"]
        return state

    try:
        raw = _run_critique(state.get("topic", ""), summaries_block)
        critiques = _parse_critiques(raw)
        state["critiques"] = critiques
        logger.info("[critique] produced %d critique entries", len(critiques))
        state["log"] = state.get("log", []) + [f"critique: produced {len(critiques)} entries"]
    except RetryExhaustedError as exc:
        msg = f"critique: LLM call failed after retries: {exc}"
        logger.error(msg)
        state["errors"] = state.get("errors", []) + [msg]
        # Degrade gracefully: proceed with no structured critique rather
        # than aborting the whole run; the Writer agent handles empty
        # critiques fine (see writer_agent.py).
        state["critiques"] = []

    return state
