"""Embedders. Both return L2-normalised float32 vectors."""
import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Word and character-trigram hashing. No model download, lexical only.

    It exists so the project runs anywhere, and as the baseline the real
    embedding model has to beat in the evaluation.
    """

    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = TOKEN.findall(text.lower())
        grams = [w[i : i + 3] for w in words if len(w) > 3 for i in range(len(w) - 2)]
        return words + grams

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                sign = 1.0 if value >> 63 else -1.0
                matrix[row, value % self.dim] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-9)


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


@lru_cache
def get_embedder(name: str) -> Embedder:
    return MiniLMEmbedder() if name == "minilm" else HashingEmbedder()
