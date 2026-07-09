"""Unit tests for graph/state.py."""
from graph.state import new_state


def test_new_state_defaults():
    state = new_state("test topic")
    assert state["topic"] == "test topic"
    assert state["status"] == "in_progress"
    assert state["papers"] == []
    assert state["paper_summaries"] == {}
    assert state["writer_revisions"] == 0
    assert state["current_agent"] == "supervisor"


def test_new_state_custom_limits():
    state = new_state("topic", max_papers=3, max_writer_revisions=1)
    assert state["max_papers"] == 3
    assert state["max_writer_revisions"] == 1
