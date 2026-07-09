"""
Central configuration for Scholar.

All tunables live here so agents never hardcode magic numbers. Reads
secrets from environment variables (see .env.example).

LLM provider strategy: Scholar is provider-agnostic via LiteLLM. Three
free-tier providers are configured in priority order -- Gemini (AI
Studio) first, then OpenRouter's free models, then Groq -- so the whole
system runs on $0 by default. `utils/llm.py` is what actually walks this
chain at call time; this module just declares it.
"""
import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is a soft dependency; env vars can also be set
    # directly in the shell / CI environment.
    pass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    api_key_env: str
    api_key: str


def _build_provider_chain() -> List[ProviderConfig]:
    """
    Builds the ordered list of usable providers based on which API keys
    are actually set. Only providers with a non-empty key are included,
    but they're always considered in this fixed priority order:

        1. Gemini (Google AI Studio free tier)   -- GEMINI_API_KEY
        2. OpenRouter (free model)                -- OPENROUTER_API_KEY
        3. Groq                                   -- GROQ_API_KEY
    """
    candidates = [
        ProviderConfig(
            name="gemini",
            model=os.getenv("SCHOLAR_GEMINI_MODEL", "gemini/gemini-2.5-flash"),
            api_key_env="GEMINI_API_KEY",
            api_key=os.getenv("GEMINI_API_KEY", ""),
        ),
        ProviderConfig(
            name="openrouter",
            model=os.getenv(
                "SCHOLAR_OPENROUTER_MODEL",
                "openrouter/meta-llama/llama-3.1-8b-instruct:free",
            ),
            api_key_env="OPENROUTER_API_KEY",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ),
        ProviderConfig(
            name="groq",
            model=os.getenv("SCHOLAR_GROQ_MODEL", "groq/llama-3.1-8b-instant"),
            api_key_env="GROQ_API_KEY",
            api_key=os.getenv("GROQ_API_KEY", ""),
        ),
    ]
    return [c for c in candidates if c.api_key]


@dataclass(frozen=True)
class Settings:
    provider_chain: List[ProviderConfig] = field(default_factory=_build_provider_chain)

    planner_max_tokens: int = int(os.getenv("SCHOLAR_PLANNER_MAX_TOKENS", "1024"))
    summarizer_max_tokens: int = int(os.getenv("SCHOLAR_SUMMARIZER_MAX_TOKENS", "512"))
    critique_max_tokens: int = int(os.getenv("SCHOLAR_CRITIQUE_MAX_TOKENS", "1024"))
    writer_max_tokens: int = int(os.getenv("SCHOLAR_WRITER_MAX_TOKENS", "2048"))

    embedding_model: str = os.getenv("SCHOLAR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("SCHOLAR_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("SCHOLAR_CHUNK_OVERLAP", "120"))
    top_k_retrieval: int = int(os.getenv("SCHOLAR_TOP_K", "4"))

    max_papers_default: int = int(os.getenv("SCHOLAR_MAX_PAPERS", "8"))
    max_writer_revisions: int = int(os.getenv("SCHOLAR_MAX_WRITER_REVISIONS", "2"))
    max_retries: int = int(os.getenv("SCHOLAR_MAX_RETRIES", "3"))
    retry_base_delay_s: float = float(os.getenv("SCHOLAR_RETRY_BASE_DELAY", "1.5"))

    cache_dir: str = os.getenv("SCHOLAR_CACHE_DIR", "data/cache")
    log_level: str = os.getenv("SCHOLAR_LOG_LEVEL", "INFO")


settings = Settings()


def require_api_key() -> None:
    if not settings.provider_chain:
        raise RuntimeError(
            "No LLM provider is configured. Copy .env.example to .env and set "
            "at least one of: GEMINI_API_KEY (recommended, free via Google AI "
            "Studio), OPENROUTER_API_KEY, or GROQ_API_KEY."
        )

