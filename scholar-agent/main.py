"""
Main entry point for running a full literature review synchronously.

Usable both as a library call (`run_review(...)`) and via the CLI
(see cli.py). Kept separate from cli.py so the MCP server can import
`run_review` directly without going through argparse.
"""
from __future__ import annotations

import logging

from config import settings, require_api_key
from utils.logging_config import setup_logging
from graph.state import new_state, ResearchState
from graph.build_graph import get_app

logger = setup_logging()


def run_review(
    topic: str,
    max_papers: int | None = None,
    max_writer_revisions: int | None = None,
) -> ResearchState:
    """
    Runs the full Scholar multi-agent pipeline for `topic` and returns the
    final state (including `final_report`, all intermediate artifacts,
    and any non-fatal errors encountered along the way).
    """
    require_api_key()

    initial_state = new_state(
        topic=topic,
        max_papers=max_papers or settings.max_papers_default,
        max_writer_revisions=max_writer_revisions or settings.max_writer_revisions,
    )

    logger.info("Starting Scholar run for topic=%r", topic)
    app = get_app()

    # recursion_limit guards against any unforeseen infinite loop in the
    # graph (belt-and-suspenders alongside the writer-revision bound).
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    logger.info(
        "Scholar run finished with status=%s (%d errors logged)",
        final_state.get("status"), len(final_state.get("errors", [])),
    )
    return final_state  # type: ignore[return-value]


if __name__ == "__main__":
    import sys

    topic_arg = " ".join(sys.argv[1:]) or "Retrieval-Augmented Generation for long-form question answering"
    result = run_review(topic_arg)
    print("\n" + "=" * 80)
    print(result.get("final_report", "(no report generated)"))
    print("=" * 80)
    if result.get("errors"):
        print("\nNon-fatal issues encountered during the run:")
        for e in result["errors"]:
            print(f"  - {e}")
