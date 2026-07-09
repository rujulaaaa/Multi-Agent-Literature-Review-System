"""
Unit tests for individual agent nodes. Each agent is tested in isolation
by monkeypatching its external dependency (arXiv, the LLM, or the vector
store) so these tests run offline and deterministically.
"""
from graph.state import new_state
from tests.fakes import fake_papers

import agents.search_agent as search_agent
import agents.critique_agent as critique_agent
import agents.rag_agent as rag_agent


def test_search_node_populates_papers(monkeypatch):
    monkeypatch.setattr(search_agent, "search_papers", lambda q, max_results: fake_papers(3))
    state = new_state("retrieval augmented generation")
    out = search_agent.search_node(state)
    assert len(out["papers"]) == 3
    assert out["current_agent"] == "search"
    assert not out["errors"]


def test_search_node_handles_empty_results(monkeypatch):
    monkeypatch.setattr(search_agent, "search_papers", lambda q, max_results: [])
    state = new_state("an extremely obscure nonsense topic")
    out = search_agent.search_node(state)
    assert out["papers"] == []
    assert any("no papers found" in e for e in out["errors"])


def test_rag_index_node_skips_gracefully_with_no_papers():
    state = new_state("topic")
    state["papers"] = []
    out = rag_agent.rag_index_node(state)
    assert out["index_ready"] is False
    assert "skipped" in out["log"][-1]


def test_critique_parser_handles_well_formed_block():
    raw = (
        "PAPER_ID: 2501.00001\n"
        "STRENGTH: Strong empirical results.\n"
        "WEAKNESS: Small evaluation set.\n"
        "GAP_OR_RELATION: Contradicts paper 2501.00002's claims.\n"
        "---\n"
        "PAPER_ID: 2501.00002\n"
        "STRENGTH: Novel method.\n"
        "WEAKNESS: No ablation study.\n"
        "GAP_OR_RELATION: Complements paper 2501.00001.\n"
        "---"
    )
    parsed = critique_agent._parse_critiques(raw)
    assert len(parsed) == 2
    assert parsed[0]["paper_id"] == "2501.00001"
    assert "Contradicts" in parsed[0]["gaps"]


def test_critique_parser_handles_malformed_input_gracefully():
    parsed = critique_agent._parse_critiques("not a valid block at all")
    assert parsed == []
