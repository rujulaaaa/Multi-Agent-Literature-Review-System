"""
Test fakes: deterministic stand-ins for the LLM provider and arXiv API
so the test suite runs offline, fast, and without needing real credentials.
"""
from __future__ import annotations

from graph.state import Paper


def fake_papers(n: int = 3) -> list[Paper]:
    return [
        Paper(
            id=f"2501.0000{i}",
            title=f"Fake Paper Title {i}",
            authors=[f"Author {i}A", f"Author {i}B"],
            published="2025-01-0{}".format(i + 1),
            url=f"https://arxiv.org/abs/2501.0000{i}",
            abstract=(
                f"This is a fake abstract number {i} discussing retrieval "
                f"augmented generation and its evaluation on long-context "
                f"benchmarks, with method {i} achieving improved recall."
            ),
            full_text=(
                f"This is a fake abstract number {i} discussing retrieval "
                f"augmented generation and its evaluation on long-context "
                f"benchmarks, with method {i} achieving improved recall."
            ),
        )
        for i in range(n)
    ]


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """
    Minimal stand-in for a LangChain chat model. `script` is a list of
    canned responses returned in order on successive `.invoke()` calls;
    if exhausted, the last response is repeated.
    """

    def __init__(self, script: list[str]):
        self.script = script
        self.calls = 0

    def invoke(self, messages):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return FakeLLMResponse(self.script[idx])
