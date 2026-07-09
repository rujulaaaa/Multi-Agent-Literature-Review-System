"""
Reusable retry-with-backoff decorator.

This is the core building block of Scholar's failure handling strategy
(Week 4: Failure handling). Every external call an agent makes -- to any
configured LLM provider (Gemini, OpenRouter, Groq, via LiteLLM), the
arXiv API, or the embedding model -- is wrapped with this so transient
failures (rate limits, timeouts, flaky networks) don't crash the whole
pipeline.
"""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Tuple, Type, TypeVar

logger = logging.getLogger("scholar")

T = TypeVar("T")


class RetryExhaustedError(RuntimeError):
    """Raised when all retry attempts for an operation have failed."""


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.5,
    max_delay: float = 20.0,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    label: str = "operation",
):
    """
    Decorator: retries the wrapped function with exponential backoff + jitter.

    On final failure, raises RetryExhaustedError wrapping the last exception
    instead of silently swallowing it -- callers (agents) decide how to
    degrade gracefully (e.g. skip a paper, fall back to cached data, or
    surface the error to the Supervisor for a routing decision).
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:  # noqa: PERF203
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay += random.uniform(0, delay * 0.25)  # jitter
                    logger.warning(
                        "[%s] attempt %d/%d failed (%s). Retrying in %.1fs...",
                        label, attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
            raise RetryExhaustedError(
                f"{label} failed after {max_attempts} attempts"
            ) from last_exc

        return wrapper

    return decorator
