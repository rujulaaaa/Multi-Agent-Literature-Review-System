"""
Unit tests for utils/llm.py's provider fallback chain.

litellm.completion itself is monkeypatched (not called for real), so
these tests run offline and don't need any real provider key.
"""
import types

import pytest

import utils.llm as llm_module
from config import ProviderConfig


def _fake_response(text: str):
    """Builds a minimal object shaped like litellm's completion response."""
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


def test_invoke_text_uses_first_provider_when_it_succeeds(monkeypatch):
    chain = [
        ProviderConfig(name="gemini", model="gemini/gemini-2.5-flash",
                       api_key_env="GEMINI_API_KEY", api_key="fake-gemini-key"),
        ProviderConfig(name="groq", model="groq/llama-3.1-8b-instant",
                       api_key_env="GROQ_API_KEY", api_key="fake-groq-key"),
    ]
    monkeypatch.setattr(llm_module, "settings", types.SimpleNamespace(provider_chain=chain))

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        return _fake_response("gemini says hi")

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)

    result = llm_module.invoke_text("hello", system="be nice")
    assert result == "gemini says hi"
    assert calls == ["gemini/gemini-2.5-flash"]  # never touched groq


def test_invoke_text_falls_back_to_next_provider_on_failure(monkeypatch):
    chain = [
        ProviderConfig(name="gemini", model="gemini/gemini-2.5-flash",
                       api_key_env="GEMINI_API_KEY", api_key="fake-gemini-key"),
        ProviderConfig(name="openrouter", model="openrouter/meta-llama/llama-3.1-8b-instruct:free",
                       api_key_env="OPENROUTER_API_KEY", api_key="fake-or-key"),
        ProviderConfig(name="groq", model="groq/llama-3.1-8b-instant",
                       api_key_env="GROQ_API_KEY", api_key="fake-groq-key"),
    ]
    monkeypatch.setattr(llm_module, "settings", types.SimpleNamespace(provider_chain=chain))

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "gemini/gemini-2.5-flash":
            raise RuntimeError("gemini rate limited")
        return _fake_response("openrouter says hi")

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)

    result = llm_module.invoke_text("hello")
    assert result == "openrouter says hi"
    assert calls == ["gemini/gemini-2.5-flash", "openrouter/meta-llama/llama-3.1-8b-instruct:free"]


def test_invoke_text_falls_back_through_all_three_providers(monkeypatch):
    chain = [
        ProviderConfig(name="gemini", model="gemini/gemini-2.5-flash",
                       api_key_env="GEMINI_API_KEY", api_key="fake-gemini-key"),
        ProviderConfig(name="openrouter", model="openrouter/meta-llama/llama-3.1-8b-instruct:free",
                       api_key_env="OPENROUTER_API_KEY", api_key="fake-or-key"),
        ProviderConfig(name="groq", model="groq/llama-3.1-8b-instant",
                       api_key_env="GROQ_API_KEY", api_key="fake-groq-key"),
    ]
    monkeypatch.setattr(llm_module, "settings", types.SimpleNamespace(provider_chain=chain))

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] != "groq/llama-3.1-8b-instant":
            raise RuntimeError(f"{kwargs['model']} unavailable")
        return _fake_response("groq says hi")

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)

    result = llm_module.invoke_text("hello")
    assert result == "groq says hi"
    assert len(calls) == 3


def test_invoke_text_raises_when_all_providers_fail(monkeypatch):
    chain = [
        ProviderConfig(name="gemini", model="gemini/gemini-2.5-flash",
                       api_key_env="GEMINI_API_KEY", api_key="fake-gemini-key"),
    ]
    monkeypatch.setattr(llm_module, "settings", types.SimpleNamespace(provider_chain=chain))

    def fake_completion(**kwargs):
        raise RuntimeError("gemini down")

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)

    with pytest.raises(RuntimeError, match="All configured LLM providers failed"):
        llm_module.invoke_text("hello")


def test_invoke_text_raises_clearly_when_no_provider_configured(monkeypatch):
    monkeypatch.setattr(llm_module, "settings", types.SimpleNamespace(provider_chain=[]))
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        llm_module.invoke_text("hello")
