# Scholar — a multi-agent literature review assistant

Scholar is a self-correcting research assistant: give it a topic, and six
coordinated agents plan the investigation, search arXiv, build a local
retrieval index, summarize each paper, critique them against one another,
and synthesize a cited literature review — revising its own draft against
a quality gate before handing it back to you. It runs entirely on
**free-tier LLM providers**, so there's no paid API or credit card
required to use it.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [How it works](#how-it-works)
- [Features](#features)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Usage](#usage)
  - [CLI](#cli)
  - [As a Python library](#as-a-python-library)
  - [As an MCP tool](#as-an-mcp-tool-claude-desktop--claude-code)
- [Configuration reference](#configuration-reference)
- [LLM provider strategy](#llm-provider-strategy)
- [Failure handling](#failure-handling)
- [Testing](#testing)
- [Design rationale & architecture](#design-rationale--architecture)
- [Troubleshooting / FAQ](#troubleshooting--faq)
- [Limitations & roadmap](#limitations--roadmap)
- [Judging-criteria self-assessment](#judging-criteria-self-assessment)
- [License](#license)

---

## Why this exists

A literature review genuinely needs several distinct reasoning roles:
someone to *find* sources, someone to *ground* them in context, someone
to *summarize* each one faithfully, someone to spot how they *relate to
or contradict* each other, and someone to *write* the synthesis. Doing
all of that in a single long prompt tends to produce reviews that just
list papers back to back instead of actually synthesizing — which is
exactly the failure mode Scholar's Critique agent and Supervisor quality
gate exist to catch.

This is deliberately **not** "one agent with extra steps." No single
stage has enough information to do another stage's job:

- The **Summarizer** never sees other papers' summaries.
- The **Critique** agent never sees raw abstracts, only summaries.
- The **Writer** never touches arXiv or the vector store directly.

Coordination happens entirely through a shared, typed state object and
explicit LangGraph edges — see [Design rationale & architecture](#design-rationale--architecture)
for the full reasoning behind this division of labor.

## How it works

```mermaid
flowchart TD
    Start([Topic input]) --> Plan[Supervisor: plan\nsub-questions]
    Plan --> Search[Search Agent\narXiv lookup]
    Search -- no papers --> FailSearch[Mark failed]
    Search -- papers found --> RAG[RAG Agent\nchunk + embed + retrieve]
    RAG --> Summarize[Summarizer Agent\nper-paper summaries]
    Summarize -- no summaries --> FailSum[Mark failed]
    Summarize -- summaries ok --> Critique[Critique Agent\ncross-paper review]
    Critique --> Writer[Writer Agent\ndrafts review]
    Writer --> Gate{Supervisor:\nquality gate}
    Gate -- REVISE, budget left --> Writer
    Gate -- PASS / budget exhausted --> Done([Final report])
    FailSearch --> End([Run ends])
    FailSum --> End
```

| Step | Agent | What it does |
|---|---|---|
| 1 | **Supervisor (plan)** | Breaks the topic into 3–5 focused sub-questions using an LLM call |
| 2 | **Search** | Queries the arXiv API for papers relevant to the topic |
| 3 | **RAG** | Chunks and embeds paper text locally, retrieves grounding context per sub-question |
| 4 | **Summarizer** | Produces one factual, grounded summary per paper |
| 5 | **Critique** | Reviews all summaries *together*, identifying strengths, weaknesses, and relationships between papers |
| 6 | **Writer** | Synthesizes summaries + critiques into a structured Markdown literature review |
| 7 | **Supervisor (quality gate)** | Judges the draft; loops back to the Writer with concrete feedback if it doesn't pass, up to a configurable revision limit |

The orchestration pattern is a **Supervisor wrapping a Pipeline**: the
Supervisor only does the two things a supervisor is genuinely good for —
high-level planning and judging final output quality — while the
well-understood middle steps run as a deterministic, fully unit-testable
pipeline. Full rationale for this design choice (over a "pure" Supervisor
or "pure" Pipeline pattern) is in [`docs/architecture.md`](docs/architecture.md).

## Features

- **Provider-agnostic LLM layer** — built on [LiteLLM](https://github.com/BerriAI/litellm);
  the default chain is Gemini (free) → OpenRouter (free model) → Groq,
  and switching or reordering providers is a one-line config change
  (`config.py`), not a code change
- **Real multi-agent orchestration** — Supervisor pattern wrapping an
  internal pipeline, with a bounded Writer ↔ Quality-Gate revision loop
- **Real RAG** — local sentence-transformer embeddings + FAISS vector
  search, no external embedding API required
- **Real tool use** — live arXiv API search, wrapped with retry/backoff
- **Real failure handling** — exponential backoff + jitter on every
  external call, automatic fallback across LLM providers if one fails,
  graceful degradation (e.g. a paper that fails to summarize falls back
  to its raw abstract instead of vanishing from the review), and
  explicit terminal-failure states rather than crashes
- **Usable three ways** — as a CLI, as a Python library call, or as an
  **MCP server tool** callable from Claude Desktop / Claude Code
- **Fully tested offline** — 29 unit + end-to-end tests with every
  external call faked out, so CI needs no API key at all
- **Zero marginal cost** — free-tier LLMs + local embeddings means
  running this project costs nothing beyond your time

## Repository layout

```
scholar-agent/
├── agents/                   # One file per agent
│   ├── supervisor.py          # Planning, routing, quality gate
│   ├── search_agent.py         # arXiv search
│   ├── rag_agent.py             # Chunking, embedding, retrieval
│   ├── summarizer_agent.py       # Per-paper summaries
│   ├── critique_agent.py          # Cross-paper critique
│   └── writer_agent.py             # Final synthesis
├── graph/
│   ├── state.py               # Shared TypedDict state schema
│   └── build_graph.py          # LangGraph StateGraph wiring
├── tools/
│   ├── arxiv_tool.py           # arXiv search, wrapped with retry/backoff
│   └── vector_store.py          # Local embeddings + FAISS RAG store
├── utils/
│   ├── llm.py                  # Provider-agnostic LLM wrapper (LiteLLM)
│   ├── retry.py                 # Retry-with-backoff decorator
│   ├── prompts.py                # All prompt templates, centralized
│   └── logging_config.py
├── mcp_server/
│   └── server.py               # Exposes the pipeline as an MCP tool
├── tests/                    # Offline unit + end-to-end tests (fakes, no API key needed)
│   ├── fakes.py                # Shared test fixtures
│   ├── test_state.py
│   ├── test_utils.py
│   ├── test_agents.py
│   ├── test_supervisor.py
│   ├── test_llm.py
│   └── test_graph_smoke.py       # Full graph, end to end
├── examples/
│   └── sample_run.md           # Real captured execution trace + output
├── docs/
│   └── architecture.md         # Full design rationale, diagram, failure-handling details
├── .github/workflows/ci.yml   # CI: runs the offline test suite on every push
├── config.py                  # Central settings, reads .env
├── main.py                    # `run_review()` — the library entry point
├── cli.py                     # Command-line interface
├── requirements.txt
├── .env.example
└── LICENSE
```

## Setup

**Requirements:** Python 3.10+, ~500MB disk for the local embedding model
on first run (downloaded automatically), and a free API key from one
provider (Gemini recommended).

```bash
git clone <this-repo-url>
cd scholar-agent

# optional but recommended
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY (the only required secret)
# get a free key at https://aistudio.google.com/apikey
```

### Quickstart (minimal path)

```bash
pip install -r requirements.txt
cp .env.example .env        # then add GEMINI_API_KEY
python main.py "in-context learning in large language models"
```

`main.py` runs a default topic if you don't pass one, and prints the
final Markdown report straight to stdout. OpenRouter and Groq are
optional automatic fallbacks, not prerequisites — nothing else needs to
be configured.

## Usage

### CLI

```bash
python cli.py "in-context learning in large language models"

# with options
python cli.py "diffusion models for audio synthesis" \
    --max-papers 6 \
    --max-revisions 2 \
    --out report.md \
    --json full_state.json \
    -v
```

| Flag | Purpose |
|---|---|
| `--max-papers N` | Max arXiv papers to fetch (default from config, `8`) |
| `--max-revisions N` | Max Writer ↔ Quality-Gate revision loops (default `2`) |
| `--out FILE` | Write the final Markdown report to a file |
| `--json FILE` | Write the *entire* final state (all intermediate artifacts: sub-questions, summaries, critiques, error log) to a JSON file |
| `-v` / `--verbose` | Enable DEBUG logging to watch agent handoffs in real time |

### As a Python library

```python
from main import run_review

result = run_review("retrieval augmented generation for long-form QA", max_papers=6)

print(result["final_report"])   # the Markdown literature review
print(result["status"])         # "complete" | "failed"
print(result["errors"])         # any non-fatal issues encountered along the way
print(result["sub_questions"])  # the Supervisor's planned sub-questions
print(result["critiques"])      # structured per-paper critique entries
```

### As an MCP tool (Claude Desktop / Claude Code)

Run the server:

```bash
python -m mcp_server.server
```

Point any MCP client at it over stdio. Example config for Claude Desktop
(`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "scholar": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/scholar-agent",
      "env": { "GEMINI_API_KEY": "your-gemini-api-key-here" }
    }
  }
}
```

Once connected, you can ask Claude something like *"use the scholar tool
to generate a literature review on sparse mixture-of-experts models"* and
it will call the `generate_literature_review` tool directly. A second
tool, `list_supported_config`, reports the active provider chain and
pipeline settings.

## Configuration reference

All tunables are environment variables (see `.env.example`), loaded
centrally in `config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — (required for the default setup) | Free key from [AI Studio](https://aistudio.google.com/apikey); Priority 1 provider |
| `OPENROUTER_API_KEY` | — (optional) | Free key from [OpenRouter](https://openrouter.ai/keys); Priority 2 fallback if Gemini fails |
| `GROQ_API_KEY` | — (optional) | Free key from [Groq Console](https://console.groq.com/keys); Priority 3 fallback if both above fail |
| `SCHOLAR_GEMINI_MODEL` | `gemini/gemini-2.5-flash` | Model string passed to LiteLLM for the Gemini provider |
| `SCHOLAR_OPENROUTER_MODEL` | `openrouter/meta-llama/llama-3.1-8b-instruct:free` | Model string for the OpenRouter fallback |
| `SCHOLAR_GROQ_MODEL` | `groq/llama-3.1-8b-instant` | Model string for the Groq fallback |
| `SCHOLAR_PLANNER_MAX_TOKENS` | `1024` | Max tokens for the Supervisor's planning call |
| `SCHOLAR_SUMMARIZER_MAX_TOKENS` | `512` | Max tokens per paper summary |
| `SCHOLAR_CRITIQUE_MAX_TOKENS` | `1024` | Max tokens for the critique pass |
| `SCHOLAR_WRITER_MAX_TOKENS` | `2048` | Max tokens for the final report draft |
| `SCHOLAR_MAX_PAPERS` | `8` | Max papers fetched from arXiv per run |
| `SCHOLAR_MAX_WRITER_REVISIONS` | `2` | Bound on the Writer ↔ Quality-Gate loop |
| `SCHOLAR_MAX_RETRIES` | `3` | Retry attempts per external call |
| `SCHOLAR_RETRY_BASE_DELAY` | `1.5` | Base delay (seconds) for exponential backoff |
| `SCHOLAR_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model for RAG |
| `SCHOLAR_CHUNK_SIZE` / `SCHOLAR_CHUNK_OVERLAP` | `800` / `120` | RAG chunking, in words |
| `SCHOLAR_TOP_K` | `4` | Chunks retrieved per sub-question |
| `SCHOLAR_CACHE_DIR` | `data/cache` | Local cache directory |
| `SCHOLAR_LOG_LEVEL` | `INFO` | Logging verbosity |

## LLM provider strategy

Scholar's only required secret is **one** free LLM API key. Every agent
that calls an LLM (Supervisor, Summarizer, Critique, Writer) goes through
the same function, `utils.llm.invoke_text()`, which is the *only* place
in the codebase aware of specific providers.

At call time, `invoke_text()` walks a prioritized provider chain, trying
the next provider immediately if the current one raises for any reason
(invalid key, rate limit, outage, unsupported model):

1. **Gemini** (`gemini/gemini-2.5-flash`) via Google AI Studio's free
   tier — `GEMINI_API_KEY`
2. **OpenRouter** free model (`openrouter/meta-llama/llama-3.1-8b-instruct:free`)
   — `OPENROUTER_API_KEY`
3. **Groq** (`groq/llama-3.1-8b-instant`) — `GROQ_API_KEY`

Only providers with a non-empty key are included, so a setup with just
`GEMINI_API_KEY` runs fine on a single-provider chain — OpenRouter and
Groq are optional resilience, not requirements.

### Switching providers

Because the LLM layer goes through LiteLLM, changing providers or their
priority is a config-only change — no code edits needed:

- **Use only Groq** (skip Gemini/OpenRouter entirely): set `GROQ_API_KEY`
  and leave the other two blank — the provider chain in `config.py`
  automatically includes only providers with a non-empty key.
- **Reorder priority**: edit `_build_provider_chain()` in `config.py` —
  it's a plain ordered list.
- **Point at a different free model** on the same provider: override
  `SCHOLAR_GEMINI_MODEL` / `SCHOLAR_OPENROUTER_MODEL` / `SCHOLAR_GROQ_MODEL`
  in `.env`.
- **Add a fourth provider**: LiteLLM supports 100+ providers out of the
  box — add one more `ProviderConfig` entry in `_build_provider_chain()`;
  `utils/llm.py` needs no changes at all.

## Failure handling

Four layers of defense, so a single flaky API call never crashes a run:

1. **Retry with exponential backoff + jitter** (`utils/retry.py`) wraps
   every external call: arXiv, and every LLM call across the Planner,
   Summarizer, Critique, Writer, and Quality Gate.
2. **Cross-provider fallback** (`utils/llm.py`) — if a provider raises,
   `invoke_text()` immediately tries the next configured provider before
   giving up, independent of and composable with layer 1.
3. **Graceful degradation** at the agent level — e.g. if a paper's
   summary fails after retries and fallback are exhausted, it falls back
   to a truncated raw abstract instead of being dropped from the review.
4. **Fatal-failure routing** — if Search returns zero papers, or the
   Summarizer produces zero summaries, the Supervisor routes the run to
   an explicit terminal node that sets `status="failed"` and ends the
   graph cleanly, rather than raising an unhandled exception.

Full details, including a documented LangGraph gotcha discovered during
testing (conditional-edge functions can't mutate persisted state), are in
[`docs/architecture.md`](docs/architecture.md#failure-handling).

## Testing

```bash
pytest                                        # all 29 tests, well under a second
pytest --cov=. --cov-report=term-missing      # with coverage
pytest -v tests/test_graph_smoke.py           # just the full end-to-end graph test
```

No API key or network access is needed — every external boundary (all
three LLM providers, arXiv, the embedding model/FAISS) is faked in
`tests/fakes.py` and the individual test files:

| Test file | Covers |
|---|---|
| `test_state.py` | State factory defaults and custom limits |
| `test_utils.py` | Retry/backoff decorator, text chunking |
| `test_agents.py` | Search, RAG, and Critique agents in isolation |
| `test_supervisor.py` | Routing functions, planning, quality gate logic |
| `test_llm.py` | Provider fallback chain: success, single/double fallback, total failure |
| `test_graph_smoke.py` | The **entire compiled LangGraph app** end to end, including the writer/quality-gate revision loop and both fatal-failure paths |

CI (`.github/workflows/ci.yml`) runs this full suite on every push —
no secrets configured, because none are needed.

## Design rationale & architecture

See [`docs/architecture.md`](docs/architecture.md) for the complete
design writeup, including:

- Why a **hybrid Supervisor + Pipeline** pattern was chosen over a
  "pure" Supervisor or "pure" Pipeline
- The exact division of responsibility between all six agents (a table
  of what each agent does and explicitly does *not* do)
- The full four-layer failure-handling strategy
- How the local RAG pipeline is wired (chunking → embedding → FAISS →
  per-sub-question retrieval)
- How the MCP server integration works
- Extensibility notes (swapping the vector store, adding a new agent,
  adding a new LLM provider)

See also [`examples/sample_run.md`](examples/sample_run.md) for a real,
captured execution trace showing the internal state log, planned
sub-questions, structured critique output, and final report shape.

## Troubleshooting / FAQ

**"No LLM provider is configured" error.** You haven't set any of
`GEMINI_API_KEY`, `OPENROUTER_API_KEY`, or `GROQ_API_KEY` in `.env`. At
least one is required; Gemini is recommended since it's the fastest free
tier to get a key for.

**First run is slow.** The local embedding model (`all-MiniLM-L6-v2`,
~90MB) downloads automatically on first use and is cached afterward —
subsequent runs are much faster.

**Gemini free-tier rate limit hit mid-run.** This is exactly what the
provider fallback exists for — if you've also set `OPENROUTER_API_KEY`
or `GROQ_API_KEY`, Scholar automatically retries with the next provider.
If you've only configured Gemini, add a second key or wait for the
rate limit window to reset.

**Search returns zero papers / run ends immediately with `status:
"failed"`.** Your topic may be too narrow or too obscure for arXiv's
search to match anything. Try a broader phrasing, or increase
`--max-papers`.

**I want to see what each agent is doing in real time.** Run with `-v`
(the CLI's verbose flag) to enable DEBUG logging, which prints every
agent handoff, retry attempt, and routing decision as it happens.

## Limitations & roadmap

Honest scope notes — things this project intentionally does not do, and
why:

- **No fine-tuning.** None of Scholar's tasks (planning, summarizing,
  critiquing, writing) benefit meaningfully from a fine-tuned model over
  a well-prompted general one; forcing it in would have added complexity
  without strengthening the project.
- **Abstract-only RAG, not full-text.** Scholar indexes each paper's
  abstract rather than the full PDF text. Extending `tools/arxiv_tool.py`
  to fetch and parse full PDFs is a natural next step for deeper grounding.
- **Single-session vector store.** The FAISS index is rebuilt fresh per
  run rather than persisted across runs. For a long-running service,
  swapping in a persistent store (Chroma, pgvector) would let repeated
  queries reuse prior indexing work — see the architecture doc's
  extensibility notes for how localized that change would be.
- **No web UI.** Scholar ships as a CLI, library, and MCP tool; a thin
  web front end (e.g. Streamlit) could be layered on top of `main.run_review()`
  without touching any agent code.

## Judging-criteria self-assessment

| Criterion | Weight | How this project addresses it |
|---|---|---|
| Problem originality & relevance | 25% | A literature-review assistant needs *distinct* reasoning roles (discovery vs. grounding vs. summarizing vs. critique vs. synthesis) that don't collapse into one agent with extra steps |
| Technical execution | 30% | Real LangGraph state machine, real RAG (local embeddings + FAISS), real arXiv tool use, real retry/backoff + cross-provider failure handling, offline test suite with 29 passing tests, MCP server wrapper |
| Orchestration design | 25% | Explicit hybrid Supervisor+Pipeline pattern with a bounded revision loop, documented and justified in `docs/architecture.md`, not just the first pattern that came to mind |
| Clarity & presentation | 20% | This README, `docs/architecture.md`, `examples/sample_run.md`, inline docstrings throughout, and a CI workflow that runs the test suite on every push |

## License

MIT — see [`LICENSE`](LICENSE).
