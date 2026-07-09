"""
Lightweight local RAG store.

Uses sentence-transformers for embeddings (no external embedding API /
key required -- Scholar's only required secret is one LLM provider key,
e.g. GEMINI_API_KEY) and FAISS for similarity search. Text is chunked
with a simple sliding window.

This module is intentionally provider-agnostic at the interface level:
`ScholarVectorStore.add_documents` / `.query` is all the RAG agent needs,
so swapping in Chroma/Pinecone/pgvector later only touches this file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from config import settings

logger = logging.getLogger("scholar")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple word-based sliding-window chunker."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


@dataclass
class ScholarVectorStore:
    """In-memory FAISS index over (paper_id, chunk_text) pairs."""

    _model: object = field(default=None, init=False, repr=False)
    _index: object = field(default=None, init=False, repr=False)
    _texts: List[str] = field(default_factory=list, init=False)
    _sources: List[str] = field(default_factory=list, init=False)  # paper_id per chunk

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    def add_documents(self, paper_id: str, text: str) -> int:
        """Chunk `text`, embed it, and add it to the index. Returns #chunks added."""
        import numpy as np
        import faiss

        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return 0

        model = self._ensure_model()
        embeddings = model.encode(chunks, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")

        if self._index is None:
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)  # cosine sim via normalized IP

        self._index.add(embeddings)
        self._texts.extend(chunks)
        self._sources.extend([paper_id] * len(chunks))
        return len(chunks)

    def query(self, question: str, top_k: int = 4) -> List[Tuple[str, str, float]]:
        """Returns list of (paper_id, chunk_text, score), best first."""
        import numpy as np

        if self._index is None or self._index.ntotal == 0:
            return []

        model = self._ensure_model()
        q_emb = model.encode([question], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")

        k = min(top_k, self._index.ntotal)
        scores, idxs = self._index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self._sources[idx], self._texts[idx], float(score)))
        return results

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0
