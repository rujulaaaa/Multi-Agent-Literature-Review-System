"""
End-to-end smoke test.

Runs the *entire* compiled LangGraph app with every external dependency
(arXiv, the LLM provider, the embedding model/FAISS) faked out. This
verifies the graph wiring itself -- node ordering, conditional edges, and
the writer/quality-gate revision loop -- independent of any real network
or API key. This is the test to run in CI (see .github/workflows/ci.yml).
"""
from graph.state import new_state
from tests.fakes import fake_papers

import agents.search_agent as search_agent
import agents.rag_agent as rag_agent
import agents.supervisor as supervisor
import agents.summarizer_agent as summarizer_agent
import agents.critique_agent as critique_agent
import agents.writer_agent as writer_agent


class _FakeStore:
    """Stand-in for ScholarVectorStore that needs no ML deps installed."""

    def __init__(self):
        self._ready = False

    def add_documents(self, paper_id, text):
        self._ready = True
        return 1

    def query(self, question, top_k=4):
        return [("2501.00000", "fake retrieved context chunk", 0.9)] if self._ready else []

    @property
    def is_ready(self):
        return self._ready


def _fake_invoke_text(prompt, *, system=None, max_tokens=1024, temperature=0.2):
    """
    Single fake LLM entrypoint used by every agent (they all funnel through
    utils.llm.invoke_text). Returns a canned, well-formed response based on
    which system prompt is asking -- this keeps every downstream parser
    (critique parsing, etc.) exercised for real.
    """
    if system and "planning module" in system:
        return "1. What approaches exist?\n2. What are the tradeoffs?\n3. What gaps remain?"
    if system and "summarizer" in system:
        return "This paper proposes a method and evaluates it, finding improved results."
    if system and "peer reviewer" in system:
        return (
            "PAPER_ID: 2501.00000\n"
            "STRENGTH: Good empirical results.\n"
            "WEAKNESS: Small dataset.\n"
            "GAP_OR_RELATION: Related to other papers in the set.\n"
            "---\n"
            "PAPER_ID: 2501.00001\n"
            "STRENGTH: Clear writing.\n"
            "WEAKNESS: Limited baselines.\n"
            "GAP_OR_RELATION: Complements paper 2501.00000.\n"
            "---"
        )
    if system and "academic writer" in system:
        return (
            "# Literature Review\n\n## Introduction\nThis review covers the topic.\n\n"
            "## Body\nPapers [2501.00000] and [2501.00001] show consistent findings.\n\n"
            "## Identified Research Gaps\nMore large-scale evaluation is needed.\n\n"
            "## Conclusion\nThe field is progressing steadily.\n\n"
            "## References\n- [2501.00000] Fake Paper Title 0\n- [2501.00001] Fake Paper Title 1\n"
        )
    if system and "strict editor" in system:
        return "PASS"
    return "generic fake response"


def test_full_graph_smoke(monkeypatch):
    # Patch external boundaries in every agent module that imported them directly.
    monkeypatch.setattr(search_agent, "search_papers", lambda q, max_results: fake_papers(2))
    monkeypatch.setattr(rag_agent, "_store_factory", lambda: _FakeStore())
    monkeypatch.setattr(summarizer_agent, "get_context_for_paper", lambda state, paper_id, question_hint="": "fake retrieved context chunk")
    monkeypatch.setattr(supervisor, "invoke_text", _fake_invoke_text)
    monkeypatch.setattr(summarizer_agent, "invoke_text", _fake_invoke_text)
    monkeypatch.setattr(critique_agent, "invoke_text", _fake_invoke_text)
    monkeypatch.setattr(writer_agent, "invoke_text", _fake_invoke_text)

    from graph.build_graph import build_graph

    app = build_graph()
    initial_state = new_state("retrieval augmented generation", max_papers=2, max_writer_revisions=2)
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    assert final_state["status"] == "complete"
    assert "Literature Review" in final_state["final_report"]
    assert "2501.00000" in final_state["final_report"]
    assert final_state["writer_revisions"] >= 1
    assert len(final_state["paper_summaries"]) == 2
    assert len(final_state["critiques"]) == 2


def test_full_graph_smoke_aborts_on_empty_search(monkeypatch):
    monkeypatch.setattr(search_agent, "search_papers", lambda q, max_results: [])
    monkeypatch.setattr(rag_agent, "_store_factory", lambda: _FakeStore())
    monkeypatch.setattr(supervisor, "invoke_text", _fake_invoke_text)

    from graph.build_graph import build_graph

    app = build_graph()
    initial_state = new_state("an impossible nonsense query", max_papers=2)
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    assert final_state["status"] == "failed"
    assert final_state["papers"] == []
    assert final_state["final_report"] == ""
