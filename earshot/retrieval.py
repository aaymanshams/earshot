"""Per-user note search: FAISS vector search followed by an optional re-ranker."""
import math
import re
import threading
from functools import lru_cache
from typing import Protocol

import faiss

from .chunking import STRATEGIES
from .embeddings import Embedder
from .schemas import Chunk, Hit, Note

STOPWORDS = frozenset(
    "a an and are am at be can did do does for from has have how i in is it me my of on or "
    "please the their there this to was what when where which who will with would you your "
    "any tell we us should could about".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> set[str]:
    """Words cut to five letters, so "allergic" matches "allergy"."""
    return {w[:5] for w in _WORD.findall(text.lower()) if w not in STOPWORDS}


def searchable(chunk: Chunk) -> str:
    return f"{chunk.title}: {chunk.text}"


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class NoReranker:
    name = "none"

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        return hits


class LexicalReranker:
    """Blend the vector score with how much of the query the chunk really covers."""

    name = "lexical"

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        wanted = content_words(query)
        rescored = []
        for hit in hits:
            covered = len(wanted & content_words(searchable(hit.chunk))) / max(len(wanted), 1)
            rescored.append(Hit(chunk=hit.chunk, score=0.5 * hit.score + 0.5 * covered))
        return sorted(rescored, key=lambda h: h.score, reverse=True)


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, searchable(h.chunk)) for h in hits])
        # Logistic scaling gives a 0-1 score, comparable to a threshold.
        rescored = [
            Hit(chunk=h.chunk, score=1 / (1 + math.exp(-float(s))))
            for h, s in zip(hits, scores, strict=True)
        ]
        return sorted(rescored, key=lambda h: h.score, reverse=True)


@lru_cache
def get_reranker(name: str) -> Reranker:
    if name == "cross-encoder":
        return CrossEncoderReranker()
    return LexicalReranker() if name == "lexical" else NoReranker()


class NoteIndex:
    """One user's notes. An index only ever holds a single user's chunks."""

    def __init__(self, embedder: Embedder, chunking: str = "sentence"):
        self.embedder = embedder
        self.chunker = STRATEGIES[chunking]
        self.chunks: list[Chunk] = []
        self.notes = 0
        self._index = None
        self._lock = threading.Lock()

    def add(self, notes: list[Note]) -> int:
        fresh: list[Chunk] = []
        for note in notes:
            fresh.extend(self.chunker(note, f"n{self.notes}"))
            self.notes += 1
        if not fresh:
            return 0
        vectors = self.embedder.encode([searchable(c) for c in fresh])
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors)
            self.chunks.extend(fresh)
        return len(fresh)

    def search(self, query: str, k: int) -> list[Hit]:
        if self._index is None:
            return []
        vector = self.embedder.encode([query])
        with self._lock:
            scores, ids = self._index.search(vector, min(k, len(self.chunks)))
        return [
            Hit(chunk=self.chunks[i], score=float(s))
            for s, i in zip(scores[0], ids[0], strict=True)
            if i >= 0
        ]


class NoteStore:
    """Keeps each user's index separate, which is what keeps notes private."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._indexes: dict[str, NoteIndex] = {}

    def for_user(self, user_id: str) -> NoteIndex:
        return self._indexes.setdefault(user_id, NoteIndex(self.embedder))
