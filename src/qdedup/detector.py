"""Duplicate question detector with multi-band routing and LLM judge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from qdedup.embedder import QuestionEmbedder, VectorIndex
from qdedup.llm import LLMClient, MockLLM
from qdedup.models import Question
from qdedup.normalize import extract_numbers, hash_text, normalize
from qdedup.store import QuestionStore


class Decision(str, Enum):
    """Categorical duplicate decision bands."""

    EXACT = "EXACT"
    DUPLICATE = "DUPLICATE"
    REVIEW = "REVIEW"
    UNIQUE = "UNIQUE"


@dataclass
class DetectionResult:
    """Detailed result of duplicate question detection."""

    decision: Decision
    matched_id: str | None = None
    score: float = 0.0
    explanation: str = ""
    evaluated_candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "matched_id": self.matched_id,
            "score": self.score,
            "explanation": self.explanation,
            "evaluated_candidates": self.evaluated_candidates,
        }


@dataclass
class DetectorConfig:
    """Configuration for duplicate detector thresholds and LLM judge."""

    dup_threshold: float = 0.88
    review_threshold: float = 0.72
    judge_min_confidence: float = 0.70
    use_llm_judge: bool = True


class DuplicateDetector:
    """Multi-band duplicate question detector.

    Decision Bands:
    1. EXACT: Identical normalized text hash match (score = 1.0).
    2. DUPLICATE: Top neighbor cosine similarity >= duplicate_threshold (e.g. 0.95).
    3. REVIEW Band: Cosine similarity >= review_threshold (top <= 3 candidates).
       - Calls the LLM judge on each neighbor whose cosine >= review_threshold (max 3).
       - If ANY neighbor is judged same with confidence >= judge_min_confidence:
         -> DUPLICATE (matched_id = that neighbor).
       - If ALL evaluated neighbors are confidently different:
         -> UNIQUE.
       - Otherwise:
         -> REVIEW.
    4. UNIQUE: No candidates meeting review_threshold or exact match.
    """

    def __init__(
        self,
        store: QuestionStore,
        embedder: QuestionEmbedder,
        index: VectorIndex | None = None,
        llm: LLMClient | None = None,
        duplicate_threshold: float = 0.88,
        review_threshold: float = 0.72,
        judge_min_confidence: float = 0.70,
        config: DetectorConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.index = index if index is not None else VectorIndex()
        self.llm = llm if llm is not None else MockLLM()

        if config is None:
            config = DetectorConfig(
                dup_threshold=duplicate_threshold,
                review_threshold=review_threshold,
                judge_min_confidence=judge_min_confidence,
                use_llm_judge=True,
            )
        self.config = config

    @property
    def config(self) -> DetectorConfig:
        return self._config

    @config.setter
    def config(self, cfg: DetectorConfig) -> None:
        self._config = cfg
        self.duplicate_threshold = cfg.dup_threshold
        self.review_threshold = cfg.review_threshold
        self.judge_min_confidence = cfg.judge_min_confidence
        self.use_llm_judge = cfg.use_llm_judge

    def _resolve_question(self, query: str | Question) -> Question:
        if isinstance(query, Question):
            return query
        norm = normalize(query)
        thash = hash_text(norm)
        return Question(id="", text=query, normalized=norm, text_hash=thash)

    def detect(self, query: str | Question) -> DetectionResult:
        """Detect whether a query question is a duplicate of any existing question."""
        q = self._resolve_question(query)

        # 1. Exact match check using text hash
        for existing in self.store.all():
            if existing.text_hash == q.text_hash:
                return DetectionResult(
                    decision=Decision.EXACT,
                    matched_id=existing.id,
                    score=1.0,
                    explanation=f"Exact normalized text match with question {existing.id}.",
                )

        # 2. Vector search over existing index
        if len(self.index) == 0:
            return DetectionResult(
                decision=Decision.UNIQUE,
                matched_id=None,
                score=0.0,
                explanation="No existing questions in store/index.",
            )

        query_vec = self.embedder.encode(q)
        neighbors = self.index.search(query_vec, k=5)

        if not neighbors:
            return DetectionResult(
                decision=Decision.UNIQUE,
                matched_id=None,
                score=0.0,
                explanation="No vector neighbors returned.",
            )

        top_id, top_score = neighbors[0]

        # Number guard: extract numbers from candidate and top-matched question
        top_q = self.store.get(top_id)
        cand_numbers = extract_numbers(q.text)
        top_numbers = extract_numbers(top_q.text) if top_q else set()
        numbers_match = (cand_numbers == top_numbers)

        # 3. High cosine similarity -> DUPLICATE (guarded: number sets must not differ)
        if top_score >= self.duplicate_threshold and numbers_match:
            return DetectionResult(
                decision=Decision.DUPLICATE,
                matched_id=top_id,
                score=round(top_score, 4),
                explanation=f"Cosine similarity ({top_score:.4f}) >= duplicate threshold ({self.duplicate_threshold}).",
            )

        # 4. REVIEW Band
        # Filter neighbors whose cosine is >= review_threshold (max top 3)
        # If numbers differed on high cosine, ensure that top pair is sent to the LLM judge
        review_candidates = [n for n in neighbors if n[1] >= self.review_threshold][:3]
        if top_score >= self.duplicate_threshold and not any(n[0] == top_id for n in review_candidates):
            review_candidates.insert(0, (top_id, top_score))

        if not review_candidates:
            # Below review threshold -> UNIQUE
            return DetectionResult(
                decision=Decision.UNIQUE,
                matched_id=None,
                score=round(top_score, 4),
                explanation=f"Top cosine similarity ({top_score:.4f}) is below review threshold ({self.review_threshold}).",
            )

        if not getattr(self, "use_llm_judge", True):
            return DetectionResult(
                decision=Decision.REVIEW,
                matched_id=top_id,
                score=round(top_score, 4),
                explanation=(
                    f"Cosine similarity ({top_score:.4f}) >= duplicate threshold ({self.duplicate_threshold}) "
                    f"but differing number sets ({cand_numbers} vs {top_numbers}); LLM judge disabled."
                    if top_score >= self.duplicate_threshold
                    else f"Borderline similarity in review band [{self.review_threshold}, {self.duplicate_threshold}); LLM judge disabled."
                ),
                evaluated_candidates=[],
            )

        evaluated: list[dict[str, Any]] = []
        judged_same: list[tuple[str, float, Any]] = []
        confidently_different: list[tuple[str, float, Any]] = []

        for cand_id, cand_score in review_candidates:
            cand_q = self.store.get(cand_id)
            cand_text = cand_q.text if cand_q else cand_id
            judge_res = self.llm.judge(q.text, cand_text)

            evaluated.append({
                "candidate_id": cand_id,
                "cosine": round(cand_score, 4),
                "same": judge_res.same,
                "confidence": judge_res.confidence,
                "explanation": judge_res.explanation,
            })

            if judge_res.same and judge_res.confidence >= self.judge_min_confidence:
                judged_same.append((cand_id, cand_score, judge_res))
            elif (not judge_res.same) and judge_res.confidence >= self.judge_min_confidence:
                confidently_different.append((cand_id, cand_score, judge_res))

        # If ANY neighbor is judged same with confidence >= judge_min_confidence -> DUPLICATE
        if judged_same:
            best_match = max(judged_same, key=lambda x: (x[2].confidence, x[1]))
            return DetectionResult(
                decision=Decision.DUPLICATE,
                matched_id=best_match[0],
                score=round(best_match[1], 4),
                explanation=f"Judged duplicate of {best_match[0]} (conf={best_match[2].confidence:.2f}): {best_match[2].explanation}",
                evaluated_candidates=evaluated,
            )

        # If ALL evaluated candidates are confidently different -> UNIQUE
        if len(confidently_different) == len(review_candidates):
            return DetectionResult(
                decision=Decision.UNIQUE,
                matched_id=None,
                score=round(top_score, 4),
                explanation=f"All {len(review_candidates)} review candidate(s) confidently judged different.",
                evaluated_candidates=evaluated,
            )

        # Otherwise -> REVIEW
        return DetectionResult(
            decision=Decision.REVIEW,
            matched_id=top_id,
            score=round(top_score, 4),
            explanation=f"Borderline similarity in review band; required human or second-pass review.",
            evaluated_candidates=evaluated,
        )

    def ingest(
        self,
        question: Question,
        embedding: np.ndarray | None = None,
    ) -> DetectionResult:
        """Check duplicate status against existing questions, then store and index."""
        result = self.detect(question)

        vec = embedding if embedding is not None else self.embedder.encode(question)
        self.store.add(question, embedding=vec)
        self.index.add(question.id, vec)

        return result

    # Alias for non-ingesting duplicate check
    check = detect
