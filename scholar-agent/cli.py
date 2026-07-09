"""
CLI for Scholar.

Usage:
    python cli.py "your research topic" --max-papers 6 --out report.md
    python cli.py --topic "diffusion models for audio" --json state.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from main import run_review
from config import settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scholar",
        description="Scholar: a multi-agent literature review assistant.",
    )
    parser.add_argument(
        "topic", nargs="*", help="Research topic to review (quote it, or pass multiple words)."
    )
    parser.add_argument(
        "--topic", dest="topic_flag", default=None,
        help="Alternative way to pass the topic as a single flag value.",
    )
    parser.add_argument(
        "--max-papers", type=int, default=settings.max_papers_default,
        help=f"Max papers to fetch from arXiv (default: {settings.max_papers_default}).",
    )
    parser.add_argument(
        "--max-revisions", type=int, default=settings.max_writer_revisions,
        help=f"Max writer revision loops (default: {settings.max_writer_revisions}).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Write the final Markdown report to this file.",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Write the full final state (all intermediate artifacts) to this JSON file.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger("scholar").setLevel(logging.DEBUG)

    topic = args.topic_flag or " ".join(args.topic)
    if not topic:
        parser.error("Please provide a topic, e.g. python cli.py \"large language model agents\"")

    result = run_review(
        topic=topic,
        max_papers=args.max_papers,
        max_writer_revisions=args.max_revisions,
    )

    report = result.get("final_report", "(no report generated)")
    print(report)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\n[scholar] Report written to {args.out}", file=sys.stderr)

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[scholar] Full state written to {args.json}", file=sys.stderr)

    if result.get("errors"):
        print(f"\n[scholar] {len(result['errors'])} non-fatal issue(s) occurred during the run; "
              f"see logs above for details.", file=sys.stderr)

    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
