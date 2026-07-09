"""
Single point of contact for LLM calls -- provider-agnostic via LiteLLM.

All agents call `invoke_text()` rather than talking to any specific
provider's SDK. This module owns the only provider-specific logic in the
whole codebase: walking the configured provider chain (Gemini -> OpenRouter
-> Groq, by default) and falling back to the next provider if the current
one raises. Every agent (Supervisor, Search, RAG, Summarizer, Critique,
Writer) is unaffected by which provider actually served the request.

Messages are plain OpenAI-compatible chat dicts (`{"role", "content"}`),
which is what LiteLLM's `completion()` expects regardless of the
underlying provider -- so this module is the one and only place that
needs to change if a new provider is ever added.
"""
from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger("scholar")


def _build_messages(prompt: str, system: str | None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def invoke_text(prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """
    Sends a single-turn prompt to the first configured, working provider
    and returns the plain text response.

    Walks `settings.provider_chain` in priority order (Gemini -> OpenRouter
    -> Groq by default, restricted to whichever providers have an API key
    set). If a provider raises (rate limit, outage, invalid key, etc.),
    the next provider in the chain is tried immediately. This fallback is
    independent of -- and composes with -- the per-agent
    `retry_with_backoff` decorators in utils/retry.py: those retry the
    *whole* operation (including this entire fallback chain) on failure,
    while this function's job is only to pick the best available
    provider on any single attempt.
    """
    import litellm

    if not settings.provider_chain:
        raise RuntimeError(
            "No LLM provider configured. Set GEMINI_API_KEY, "
            "OPENROUTER_API_KEY, or GROQ_API_KEY in your .env file."
        )

    messages = _build_messages(prompt, system)
    last_exc: Exception | None = None

    for provider in settings.provider_chain:
        try:
            response = litellm.completion(
                model=provider.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=provider.api_key,
            )
            content = response.choices[0].message.content
            return content if isinstance(content, str) else str(content)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
            # provider failure (auth, rate limit, timeout, model not
            # found) should trigger fallback to the next provider rather
            # than crash the whole run.
            logger.warning(
                "[llm] provider %r (model=%s) failed: %s -- trying next provider",
                provider.name, provider.model, exc,
            )
            last_exc = exc
            continue

    raise RuntimeError(
        "All configured LLM providers failed. Last error: "
        f"{last_exc}"
    ) from last_exc
