"""Unit tests for utils/retry.py and tools/vector_store.py chunking."""
import pytest

from utils.retry import retry_with_backoff, RetryExhaustedError
from tools.vector_store import chunk_text


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=3, base_delay=0.01, label="test_op")
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_exhausts_and_raises():
    @retry_with_backoff(max_attempts=2, base_delay=0.01, label="test_op")
    def always_fails():
        raise ValueError("permanent")

    with pytest.raises(RetryExhaustedError):
        always_fails()


def test_retry_only_catches_specified_exceptions():
    @retry_with_backoff(max_attempts=2, base_delay=0.01, retry_on=(ValueError,), label="test_op")
    def raises_type_error():
        raise TypeError("not retried")

    with pytest.raises(TypeError):
        raises_type_error()


def test_chunk_text_basic():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    # every chunk should have at most 20 words
    for c in chunks:
        assert len(c.split()) <= 20


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=20, overlap=5) == []


def test_chunk_text_shorter_than_chunk_size():
    text = "just a few words here"
    chunks = chunk_text(text, chunk_size=50, overlap=5)
    assert chunks == [text]
