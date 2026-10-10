"""Question generator for creating diverse, non-duplicate questions on a topic.

IMPORTANT: This is NOT a RAG / answer-generation module. There are no citations
or fallback answers. It generates NEW QUESTIONS for a topic, using the existing
question collection and semantic deduplication to avoid repeats.
"""

from __future__ import annotations

import json
import re
import types
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from qdedup.context_selector import ContextSelector
from qdedup.detector import Decision, DuplicateDetector
from qdedup.llm import LLMClient
from qdedup.models import Question


def _detector_ingest_if_unique(
    detector: Any,
    text: str,
    tags: Mapping[str, str] | None = None,
) -> Any:
    """Ingest candidate into detector's store and index IF AND ONLY IF it is UNIQUE."""
    qid = f"gen_{uuid.uuid4().hex[:8]}"
    q = Question(id=qid, text=text, tags=dict(tags or {}))
    res = detector.detect(q)
    if res.decision == Decision.UNIQUE:
        vec = detector.embedder.encode(q)
        detector.store.add(q, embedding=vec)
        detector.index.add(q.id, vec)
    res.question = q
    return res


# Bind to DuplicateDetector class if not already present, without modifying detector.py
if not hasattr(DuplicateDetector, "ingest_if_unique"):
    DuplicateDetector.ingest_if_unique = _detector_ingest_if_unique


@dataclass
class GenerationReport:
    """Detailed report produced by QuestionGenerator."""

    accepted: list[Question]
    rejected: list[dict[str, Any]]
    review_queue: list[dict[str, Any]]
    surplus: list[Question]
    rounds_used: int
    prompt_token_estimates: list[int]
    duplicate_rate: float
    failed_rounds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [q.to_dict() for q in self.accepted],
            "rejected": self.rejected,
            "review_queue": self.review_queue,
            "surplus": [q.to_dict() for q in self.surplus],
            "rounds_used": self.rounds_used,
            "prompt_token_estimates": self.prompt_token_estimates,
            "duplicate_rate": self.duplicate_rate,
            "failed_rounds": self.failed_rounds,
        }


class QuestionGenerator:
    """Generates new, curriculum-aligned questions for a topic while avoiding duplicates.

    Workflow per round:
    1. Context selection retrieves diverse representative existing questions and covered concepts.
    2. Constructs a structured prompt specifying forbidden questions/concepts, preferred angles,
       and session-rejected texts.
    3. Invokes the LLM and parses JSON robustly without crashing.
    4. Evaluates every candidate with `detector.ingest_if_unique(text, tags)`:
       - UNIQUE -> accepted and ingested immediately (enabling intra-batch deduplication).
       - REVIEW -> routed to review_queue (not ingested).
       - EXACT / DUPLICATE -> rejected and recorded.
    5. Halts when target quota `n` is met or `max_rounds` are exhausted.
    """

    def __init__(
        self,
        llm: LLMClient,
        selector: ContextSelector,
        detector: DuplicateDetector,
    ) -> None:
        self.llm = llm
        self.selector = selector
        self.detector = detector

        # Ensure detector instance has ingest_if_unique
        if not hasattr(self.detector, "ingest_if_unique"):
            self.detector.ingest_if_unique = types.MethodType(
                _detector_ingest_if_unique, self.detector
            )

    def _build_prompt(
        self,
        topic: str,
        n_requested: int,
        selected_questions: list[Question],
        covered_concepts: list[str],
        rejected_texts: list[str],
    ) -> str:
        prompt_parts: list[str] = [
            f"Generate {n_requested} new and diverse questions for the topic: '{topic}'.\n",
            "Existing questions (do NOT repeat or paraphrase):",
        ]

        if selected_questions:
            for q in selected_questions:
                prompt_parts.append(f"- {q.text}")
        else:
            prompt_parts.append("- (No prior questions in collection)")

        prompt_parts.append("\nConcepts already covered:")
        if covered_concepts:
            for c in covered_concepts:
                prompt_parts.append(f"- {c}")
        else:
            prompt_parts.append("- (None)")

        if rejected_texts:
            prompt_parts.append("\nAlready rejected this session, avoid these:")
            for r in rejected_texts:
                prompt_parts.append(f"- {r}")

        prompt_parts.append(
            "\nRules:\n"
            "- Do not repeat or paraphrase any listed question or concept.\n"
            "- Prefer new angles: laws, numericals, applications, diagrams, comparisons, "
            "reasoning/HOTS, different difficulty levels.\n"
            "- A numerical problem and a conceptual definition on the same topic are different questions.\n"
            '- Return ONLY a JSON list [{"question": str, "concept_tested": str}].'
        )

        return "\n".join(prompt_parts)

    def generate(
        self,
        topic: str,
        n: int,
        tags: Mapping[str, str] | None = None,
        max_rounds: int = 3,
        tags_filter: Mapping[str, str] | None = None,
    ) -> GenerationReport:
        """Generate `n` unique questions for `topic`."""
        accepted: list[Question] = []
        rejected: list[dict[str, Any]] = []
        review_queue: list[dict[str, Any]] = []
        rejected_texts: list[str] = []
        prompt_token_estimates: list[int] = []

        rounds_used = 0
        failed_rounds = 0
        total_candidates = 0

        for round_idx in range(1, max_rounds + 1):
            if len(accepted) >= n:
                break

            rounds_used += 1

            # 1. Context Selection
            selection = self.selector.select(topic, tags_filter=tags_filter)

            # 2. Build prompt
            n_requested = max(2, 2 * (n - len(accepted)))
            prompt = self._build_prompt(
                topic=topic,
                n_requested=n_requested,
                selected_questions=selection.questions,
                covered_concepts=selection.covered_concepts,
                rejected_texts=rejected_texts,
            )

            token_est = int(len(prompt) / 4)
            prompt_token_estimates.append(token_est)

            # 3. Call LLM
            raw_response = self.llm.generate(prompt)

            # Clean markdown code fences
            cleaned = raw_response.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

            try:
                candidates_data = json.loads(cleaned)
                if not isinstance(candidates_data, list):
                    failed_rounds += 1
                    continue
            except Exception:
                failed_rounds += 1
                continue

            # 4. Process candidates
            for item in candidates_data:
                if not isinstance(item, dict):
                    continue
                cand_text = item.get("question")
                if not cand_text or not isinstance(cand_text, str):
                    continue

                total_candidates += 1
                cand_text = cand_text.strip()

                # Call detector.ingest_if_unique(text, tags)
                det_res = self.detector.ingest_if_unique(cand_text, tags=tags)

                if det_res.decision == Decision.UNIQUE:
                    # Accepted
                    accepted_q = getattr(det_res, "question", None)
                    if accepted_q is None:
                        accepted_q = Question(
                            id=f"gen_{uuid.uuid4().hex[:8]}",
                            text=cand_text,
                            tags=dict(tags or {}),
                        )
                    accepted.append(accepted_q)

                elif det_res.decision == Decision.REVIEW:
                    review_queue.append({
                        "text": cand_text,
                        "decision": "REVIEW",
                        "matched_id": det_res.matched_id,
                        "score": getattr(det_res, "score", 0.0),
                        "reason": getattr(det_res, "explanation", ""),
                        "round": round_idx,
                    })

                else:
                    # EXACT or DUPLICATE
                    decision_str = (
                        det_res.decision.value
                        if hasattr(det_res.decision, "value")
                        else str(det_res.decision)
                    )
                    rejected.append({
                        "text": cand_text,
                        "decision": decision_str,
                        "matched_id": det_res.matched_id,
                        "score": getattr(det_res, "score", 0.0),
                        "reason": getattr(det_res, "explanation", ""),
                        "round": round_idx,
                    })
                    rejected_texts.append(cand_text)

        # 5. Handle over-generation (trim to n and save surplus)
        surplus: list[Question] = []
        if len(accepted) > n:
            surplus = accepted[n:]
            accepted = accepted[:n]

        dup_rate = round(len(rejected) / total_candidates, 4) if total_candidates > 0 else 0.0

        return GenerationReport(
            accepted=accepted,
            rejected=rejected,
            review_queue=review_queue,
            surplus=surplus,
            rounds_used=rounds_used,
            prompt_token_estimates=prompt_token_estimates,
            duplicate_rate=dup_rate,
            failed_rounds=failed_rounds,
        )
