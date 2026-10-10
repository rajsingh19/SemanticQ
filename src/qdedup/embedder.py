"""Embedding generation and vector indexing for question-dedup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from qdedup.normalize import hash_text, normalize

if TYPE_CHECKING:
    from qdedup.models import Question


class QuestionEmbedder:
    """Wraps SentenceTransformers (all-MiniLM-L6-v2) to generate L2-normalized vectors.

    Embeds ONLY the normalized question text (never tags or other metadata).
    Caches computed embeddings in-memory keyed by the question text's SHA-256 hash.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        cache: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._cache: dict[str, np.ndarray] = dict(cache or {})
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazy loader for the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _extract_normalized_and_hash(self, text_or_q: str | Question) -> tuple[str, str]:
        """Extract only the normalized text and text_hash, ignoring tags."""
        if hasattr(text_or_q, "normalized") and hasattr(text_or_q, "text_hash"):
            norm = text_or_q.normalized or normalize(text_or_q.text)
            thash = text_or_q.text_hash or hash_text(norm)
            return norm, thash

        # Plain string input
        norm = normalize(str(text_or_q))
        thash = hash_text(norm)
        return norm, thash

    def encode(self, item: str | Question) -> np.ndarray:
        """Generate an L2-normalized embedding for a single question.

        Embeds ONLY the normalized question text. Checks cache by text_hash first.
        """
        norm_text, thash = self._extract_normalized_and_hash(item)

        # Check cache
        if thash in self._cache:
            return self._cache[thash].copy()

        model = self._get_model()
        vec = model.encode(norm_text, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32).flatten()

        # Ensure strict L2 normalization
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

        self._cache[thash] = vec
        return vec.copy()

    def encode_batch(self, items: Sequence[str | Question]) -> np.ndarray:
        """Generate L2-normalized embeddings for multiple questions with batching.

        Utilizes cached vectors where available and batch-encodes uncached texts.
        """
        if not items:
            return np.empty((0, 384), dtype=np.float32)

        pairs = [self._extract_normalized_and_hash(it) for it in items]
        results: list[np.ndarray | None] = [None] * len(items)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, (norm_text, thash) in enumerate(pairs):
            if thash in self._cache:
                results[idx] = self._cache[thash].copy()
            else:
                uncached_indices.append(idx)
                uncached_texts.append(norm_text)

        if uncached_texts:
            model = self._get_model()
            raw_vecs = model.encode(uncached_texts, normalize_embeddings=True)
            for local_idx, orig_idx in enumerate(uncached_indices):
                _, thash = pairs[orig_idx]
                v = np.asarray(raw_vecs[local_idx], dtype=np.float32).flatten()
                norm = float(np.linalg.norm(v))
                if norm > 0:
                    v = v / norm
                self._cache[thash] = v
                results[orig_idx] = v.copy()

        return np.vstack([r for r in results if r is not None])

    @property
    def cache_size(self) -> int:
        """Return the number of cached question embeddings."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear cached embeddings."""
        self._cache.clear()


# Alias for convenience
Embedder = QuestionEmbedder


class VectorIndex:
    """In-memory vector index utilizing NumPy brute-force cosine similarity.

    Architecture & Swappability:
    ----------------------------
    The public methods (`add(id, vec)` and `search(vec, k) -> list[(id, cosine)]`)
    define a standard vector index interface. This allows future drop-in replacement
    with Approximate Nearest Neighbor (ANN) libraries such as FAISS (e.g., `IndexFlatIP`
    or `IndexHNSWFlat`) or HNSWlib (`hnswlib.Index(space='cosine')`) without breaking
    or modifying downstream components (such as duplicate detectors or rankers).
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def add(self, id: str, vec: np.ndarray) -> None:
        """Add or update an item's embedding in the index.

        Vectors are automatically converted to float32 and L2-normalized.
        """
        v = np.asarray(vec, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm

        if id in self._id_to_idx:
            idx = self._id_to_idx[id]
            self._vectors[idx] = v
        else:
            self._id_to_idx[id] = len(self._ids)
            self._ids.append(id)
            self._vectors.append(v)

        # Invalidate compiled matrix cache
        self._matrix = None

    def search(self, vec: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Search the index for the top-k nearest neighbors using cosine similarity.

        Returns a list of tuples: (id, cosine_score), ranked from highest similarity
        to lowest.
        """
        if not self._ids or k <= 0:
            return []

        q = np.asarray(vec, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm

        if self._matrix is None:
            self._matrix = np.vstack(self._vectors)

        # Cosine similarity is the dot product of two L2-normalized vectors
        scores = self._matrix @ q

        k_eff = min(k, len(self._ids))
        # Top-k indices sorted descending
        top_indices = np.argsort(-scores)[:k_eff]

        return [(self._ids[i], float(scores[i])) for i in top_indices]

    def get(self, id: str) -> np.ndarray | None:
        """Retrieve vector for an ID if present."""
        if id in self._id_to_idx:
            return self._vectors[self._id_to_idx[id]].copy()
        return None

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id: str) -> bool:
        return id in self._id_to_idx
