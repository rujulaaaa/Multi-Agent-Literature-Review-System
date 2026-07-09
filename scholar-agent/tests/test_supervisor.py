"""Unit tests for the deterministic routing functions in agents/supervisor.py."""
from graph.state import new_state
from agents import supervisor


def test_route_after_search_routes_to_fail_on_no_papers():
    state = new_state("topic")
    state["papers"] = []
    assert supervisor.route_after_search(state) == "fail"


def test_mark_search_failed_node_persists_status():
    state = new_state("topic")
    out = supervisor.mark_search_failed_node(state)
    assert out["status"] == "failed"


def test_route_after_search_continues_with_papers():
    state = new_state("topic")
    state["papers"] = [{"id": "1"}]
    assert supervisor.route_after_search(state) == "rag_index"


def test_route_after_summarizer_routes_to_fail_on_no_summaries():
    state = new_state("topic")
    state["paper_summaries"] = {}
    assert supervisor.route_after_summarizer(state) == "fail"


def test_mark_summarizer_failed_node_persists_status():
    state = new_state("topic")
    out = supervisor.mark_summarizer_failed_node(state)
    assert out["status"] == "failed"


def test_route_after_summarizer_continues():
    state = new_state("topic")
    state["paper_summaries"] = {"1": "a summary"}
    assert supervisor.route_after_summarizer(state) == "critique"


def test_route_after_quality_gate_ends_when_complete():
    state = new_state("topic")
    state["status"] = "complete"
    assert supervisor.route_after_quality_gate(state) == "end"


def test_route_after_quality_gate_loops_when_in_progress():
    state = new_state("topic")
    state["status"] = "in_progress"
    assert supervisor.route_after_quality_gate(state) == "writer"


def test_plan_sub_questions_parses_numbered_list(monkeypatch):
    monkeypatch.setattr(
        supervisor, "invoke_text",
        lambda *a, **k: "1. What methods exist?\n2. What are the benchmarks?\n3. What are open gaps?",
    )
    questions = supervisor._plan_sub_questions("test topic")
    assert questions == [
        "What methods exist?",
        "What are the benchmarks?",
        "What are open gaps?",
    ]
