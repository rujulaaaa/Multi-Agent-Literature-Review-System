"""
MCP server wrapper for Scholar.

Exposes the whole multi-agent pipeline as a single MCP tool,
`generate_literature_review`, so it can be called from any MCP client
(Claude Desktop, Claude Code, etc.) instead of only the CLI. This lets
the Week 3 (MCP) material and the Week 4 (multi-agent) capstone compose
naturally: the multi-agent system *is* the tool implementation behind
one MCP endpoint.

Run with:
    python -m mcp_server.server

Then point an MCP client at this process over stdio (see README's
"Using as an MCP tool" section for a sample client config).
"""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from main import run_review
from config import settings
from utils.logging_config import setup_logging

logger = setup_logging()

mcp = FastMCP("scholar")


@mcp.tool()
def generate_literature_review(
    topic: str,
    max_papers: int = settings.max_papers_default,
    max_writer_revisions: int = settings.max_writer_revisions,
) -> str:
    """
    Generate a multi-agent literature review on a research topic.

    Runs the full Scholar pipeline: plans sub-questions, searches arXiv,
    builds a RAG index over the retrieved papers, summarizes each paper,
    critiques them against each other, and writes a synthesized Markdown
    literature review (with a bounded self-revision loop for quality).

    Args:
        topic: The research topic to review, e.g. "in-context learning in transformers".
        max_papers: Maximum number of arXiv papers to fetch (default from config).
        max_writer_revisions: Maximum writer self-revision loops (default from config).

    Returns:
        The final literature review as a Markdown string. If non-fatal
        errors occurred during the run (e.g. a paper failed to summarize),
        they are appended as a trailing note.
    """
    logger.info("[mcp] generate_literature_review called for topic=%r", topic)
    result = run_review(
        topic=topic,
        max_papers=max_papers,
        max_writer_revisions=max_writer_revisions,
    )

    report = result.get("final_report", "(no report generated)")
    if result.get("errors"):
        report += "\n\n---\n**Non-fatal issues during generation:**\n"
        report += "\n".join(f"- {e}" for e in result["errors"])
    return report


@mcp.tool()
def list_supported_config() -> dict:
    """Returns the current default configuration Scholar is running with."""
    return {
        "provider_chain": [
            {"name": p.name, "model": p.model} for p in settings.provider_chain
        ] or "NONE CONFIGURED -- set GEMINI_API_KEY, OPENROUTER_API_KEY, or GROQ_API_KEY",
        "max_papers_default": settings.max_papers_default,
        "max_writer_revisions": settings.max_writer_revisions,
        "embedding_model": settings.embedding_model,
        "top_k_retrieval": settings.top_k_retrieval,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
