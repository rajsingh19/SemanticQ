"""Context selection using candidate pool pruning, KMeans clustering, and MMR diversification."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans

from qdedup.embedder import QuestionEmbedder, VectorIndex
from qdedup.models import Question
from qdedup.store import QuestionStore


@dataclass
class SelectionResult:
    """Result of context selection for a topic query."""

    questions: list[Question]
    covered_concepts: list[str]
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions": [q.to_dict() for q in self.questions],
            "covered_concepts": list(self.covered_concepts),
            "token_estimate": self.token_estimate,
        }


class ContextSelector:
    """Selects a diverse, concept-representative set of questions for a topic query.

    Algorithm Overview:
    -------------------
    1. Candidate Pool: Filters questions by tags if specified, else retrieves all.
       If pool size > neighbour_pool, prunes to the top neighbour_pool questions
       nearest to the topic query by cosine similarity.
    2. KMeans Clustering: Clusters candidate embeddings into min(n_clusters, len(pool))
       groups. From each cluster, picks the question nearest to the centroid as
       representative, guaranteeing every concept cluster is covered.
    3. MMR Diversification: Fills remaining budget slots from the rest of the pool using
       Maximal Marginal Relevance (MMR).
    4. Budget Guardrails: Halts when max_questions or max_tokens is reached.

    Complexity Analysis:
    --------------------
    - The candidate pool is strictly capped at `neighbour_pool` (default 100), so the
      KMeans clustering and MMR diversification cost is O(neighbour_pool * k * d)
      regardless of corpus size, where k is the number of clusters and d is the embedding
      dimension.
    - Prompt size is bounded by `max_questions` and `max_tokens`.
    - With an Approximate Nearest Neighbor (ANN) index (e.g. FAISS or HNSW), the
      initial nearest-neighbour retrieval step scales as O(log N) where N is total corpus size.

    Constraints:
    ------------
    - Does NOT import from `detector.py` or `generator.py`.
    - Tags are used ONLY to filter candidate pool membership and NEVER influence
      embeddings, similarity calculations, clustering, or MMR scores.
    """

    def __init__(
        self,
        store: QuestionStore,
        embedder: QuestionEmbedder,
        index: VectorIndex | None = None,
        max_questions: int = 30,
        max_tokens: int = 1500,
        n_clusters: int = 8,
        mmr_lambda: float = 0.6,
        neighbour_pool: int = 100,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.index = index if index is not None else VectorIndex()
        self.max_questions = max_questions
        self.max_tokens = max_tokens
        self.n_clusters = n_clusters
        self.mmr_lambda = mmr_lambda
        self.neighbour_pool = neighbour_pool

        # Cache cluster summaries keyed by (pool ids hash + k)
        self._concept_cache: dict[str, list[str]] = {}

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count: len(text) / 4 + 4 overhead."""
        return int(len(text) / 4) + 4

    def _hash_pool(self, pool: Sequence[Question], k: int) -> str:
        """Compute deterministic hash of pool ids and cluster count k."""
        ids = sorted(q.id for q in pool)
        content = f"k={k}:" + ",".join(ids)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def summarize_covered_concepts(
        self,
        pool: Sequence[Question] | None = None,
        n_clusters: int | None = None,
    ) -> list[str]:
        """Summarize covered concept clusters in the pool.

        Returns short bullets: "<representative question> (N similar questions)",
        size-sorted descending. Results are cached by pool hash.
        """
        cand_pool = list(pool if pool is not None else self.store.all())
        if not cand_pool:
            return []

        k = min(n_clusters or self.n_clusters, len(cand_pool))
        if k <= 0:
            return []

        pool_hash = self._hash_pool(cand_pool, k)
        if pool_hash in self._concept_cache:
            return list(self._concept_cache[pool_hash])

        # Get embeddings
        embeddings = self.embedder.encode_batch(cand_pool)

        # Run KMeans with ConvergenceWarning suppressed for edge-case duplicate points
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            centroids = kmeans.cluster_centers_

        cluster_members: dict[int, list[int]] = {c: [] for c in range(k)}
        for idx, lbl in enumerate(labels):
            cluster_members[lbl].append(idx)

        bullets: list[tuple[int, str]] = []

        for c_idx, member_indices in cluster_members.items():
            if not member_indices:
                continue

            centroid = centroids[c_idx]
            member_vecs = embeddings[member_indices]

            # Find member nearest to centroid
            dists = np.linalg.norm(member_vecs - centroid, axis=1)
            best_local_idx = int(np.argmin(dists))
            rep_q = cand_pool[member_indices[best_local_idx]]

            cluster_size = len(member_indices)
            bullet = f"{rep_q.text} ({cluster_size} similar questions)"
            bullets.append((cluster_size, bullet))

        # Size-sorted descending
        bullets.sort(key=lambda x: x[0], reverse=True)
        result = [b[1] for b in bullets]

        self._concept_cache[pool_hash] = result
        return list(result)

    def select(
        self,
        topic_query: str,
        tags_filter: Mapping[str, str] | None = None,
    ) -> SelectionResult:
        """Select questions for the topic query using clustering and MMR diversification."""
        # a) Candidate pool selection
        if tags_filter:
            pool = self.store.filter_by_tags(tags_filter)
        else:
            pool = self.store.all()

        if not pool:
            return SelectionResult(questions=[], covered_concepts=[], token_estimate=0)

        # Encode query
        query_vec = self.embedder.encode(topic_query)

        # Prune pool if larger than neighbour_pool
        if len(pool) > self.neighbour_pool:
            pool_vecs = self.embedder.encode_batch(pool)
            sims = pool_vecs @ query_vec
            top_indices = np.argsort(-sims)[: self.neighbour_pool]
            pool = [pool[i] for i in top_indices]

        # b) Pool embeddings
        pool_vecs = self.embedder.encode_batch(pool)

        # c) Concept clusters
        k = min(self.n_clusters, len(pool))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans.fit_predict(pool_vecs)
            centroids = kmeans.cluster_centers_

        cluster_members: dict[int, list[int]] = {c: [] for c in range(k)}
        for idx, lbl in enumerate(labels):
            cluster_members[lbl].append(idx)

        # Identify representative for each cluster
        cluster_reps: list[tuple[int, int]] = []  # (cluster_size, pool_idx)
        for c_idx, member_indices in cluster_members.items():
            if not member_indices:
                continue
            centroid = centroids[c_idx]
            member_vecs = pool_vecs[member_indices]
            dists = np.linalg.norm(member_vecs - centroid, axis=1)
            rep_pool_idx = member_indices[int(np.argmin(dists))]
            cluster_reps.append((len(member_indices), rep_pool_idx))

        # Order representatives by cluster size descending
        cluster_reps.sort(key=lambda x: x[0], reverse=True)

        selected_indices: list[int] = []
        selected_ids: set[str] = set()
        current_tokens = 0

        # Add representatives while budget permits
        for _, rep_idx in cluster_reps:
            q = pool[rep_idx]
            if q.id in selected_ids:
                continue
            cost = self.estimate_tokens(q.text)
            if len(selected_indices) >= self.max_questions:
                break
            if current_tokens + cost > self.max_tokens:
                break

            selected_indices.append(rep_idx)
            selected_ids.add(q.id)
            current_tokens += cost

        # d) Fill remaining slots with MMR
        remaining_indices = [i for i in range(len(pool)) if pool[i].id not in selected_ids]

        while remaining_indices and len(selected_indices) < self.max_questions:
            best_mmr = -float("inf")
            best_idx = None
            best_cost = 0

            for cand_idx in remaining_indices:
                cand_vec = pool_vecs[cand_idx]
                sim_to_query = float(cand_vec @ query_vec)

                # Similarity to already selected
                if selected_indices:
                    sim_to_selected = max(
                        float(cand_vec @ pool_vecs[sel_idx]) for sel_idx in selected_indices
                    )
                else:
                    sim_to_selected = 0.0

                mmr_score = (
                    self.mmr_lambda * sim_to_query
                    - (1.0 - self.mmr_lambda) * sim_to_selected
                )

                if mmr_score > best_mmr:
                    cost = self.estimate_tokens(pool[cand_idx].text)
                    if current_tokens + cost <= self.max_tokens:
                        best_mmr = mmr_score
                        best_idx = cand_idx
                        best_cost = cost

            if best_idx is None:
                # No candidates fit within token budget
                break

            selected_indices.append(best_idx)
            selected_ids.add(pool[best_idx].id)
            current_tokens += best_cost
            remaining_indices.remove(best_idx)

        selected_questions = [pool[i] for i in selected_indices]
        covered_concepts = self.summarize_covered_concepts(pool, n_clusters=k)

        return SelectionResult(
            questions=selected_questions,
            covered_concepts=covered_concepts,
            token_estimate=current_tokens,
        )
