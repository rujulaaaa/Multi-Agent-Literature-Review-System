# Example run

This shows exactly what running Scholar looks like end to end, including
the internal state trace. The excerpt below was captured from an actual
run of the compiled LangGraph app (with the LLM/arXiv calls substituted
for deterministic fakes, the same way `tests/test_graph_smoke.py` does it)
— the log lines, state shape, and control flow are all real; only the
paper content and LLM text are canned so this doc doesn't depend on a
live API key to reproduce.

For a run against a real LLM provider and real arXiv results, see
"Try it yourself" at the bottom.

## Command

```bash
python cli.py "retrieval augmented generation for long-form QA" --max-papers 3 --out report.md
```

## Internal execution trace (`state["log"]`)

```json
[
  "supervisor: planned 3 sub-questions",
  "search: retrieved 3 papers",
  "rag_index: indexed 3 chunks, retrieved context for 3 sub-questions",
  "summarizer: produced 3 summaries",
  "critique: produced 2 entries",
  "writer: produced draft revision #1",
  "supervisor_quality_gate: PASS"
]
```

Notice the graph went straight through with **one** writer revision here
because the quality gate passed the first draft. If the gate had returned
`REVISE`, you'd instead see:

```json
[
  "...",
  "writer: produced draft revision #1",
  "supervisor_quality_gate: REVISE",
  "writer: produced draft revision #2",
  "supervisor_quality_gate: PASS"
]
```

capped at `max_writer_revisions` (default 2) so the loop always terminates.

## Planned sub-questions (Supervisor output)

```json
[
  "What approaches exist?",
  "What are the tradeoffs?",
  "What gaps remain?"
]
```

## Per-paper critique (Critique agent output)

```json
[
  {
    "paper_id": "2501.00000",
    "strengths": "Good empirical results.",
    "weaknesses": "Small dataset.",
    "gaps": "Related to other papers in the set."
  },
  {
    "paper_id": "2501.00001",
    "strengths": "Clear writing.",
    "weaknesses": "Limited baselines.",
    "gaps": "Complements paper 2501.00000."
  }
]
```

## Final report (`report.md`, written to disk by the CLI)

```markdown
# Literature Review

## Introduction
This review covers the topic.

## Body
Papers [2501.00000] and [2501.00001] show consistent findings.

## Identified Research Gaps
More large-scale evaluation is needed.

## Conclusion
The field is progressing steadily.

## References
- [2501.00000] Fake Paper Title 0
- [2501.00001] Fake Paper Title 1
```

With a real `GEMINI_API_KEY` (or OpenRouter/Groq) and live arXiv results,
the same run produces a much richer report — real paper
titles/authors/abstracts, a genuinely synthesized multi-paragraph body
organized by theme (not a one-liner), and a substantive gaps section. The
structure above (headers, inline `[arXiv-id]` citations, References
section) is exactly what ships either way, since it's enforced by the
Writer's prompt template (`utils/prompts.py::WRITER_SYSTEM`).

## Try it yourself

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)

pip install -r requirements.txt
python cli.py "in-context learning in large language models" --max-papers 6 --out my_report.md -v
```

The `-v` flag turns on DEBUG logging so you can watch each agent hand off
to the next in real time, including any retries triggered by transient
arXiv/API hiccups.
