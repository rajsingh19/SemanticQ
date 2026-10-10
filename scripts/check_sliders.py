#!/usr/bin/env python3
"""scripts/check_sliders.py: Verify that DetectorConfig is dynamically honoured."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure src is on Python search path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from qdedup.detector import Decision, DetectorConfig, DuplicateDetector
from qdedup.embedder import QuestionEmbedder, VectorIndex
from qdedup.llm import MockLLM
from qdedup.models import Question
from qdedup.store import QuestionStore


def get_method(res: Any, det: DuplicateDetector) -> str:
    if res.decision == Decision.EXACT:
        return "hash"
    elif res.decision == Decision.DUPLICATE:
        return "embedding" if res.score >= det.duplicate_threshold else "llm_judge"
    elif res.decision == Decision.UNIQUE:
        if res.evaluated_candidates and det.config.use_llm_judge:
            return "llm_judge"
        return "embedding" if res.score > 0 else "none"
    elif res.decision == Decision.REVIEW:
        return "llm_judge" if (res.evaluated_candidates and det.config.use_llm_judge) else "none"
    return "none"


def main() -> None:
    corpus_file = Path(__file__).resolve().parent.parent / "data" / "corpus.json"
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    store = QuestionStore(":memory:")
    embedder = QuestionEmbedder()
    index = VectorIndex()
    llm = MockLLM()
    detector = DuplicateDetector(store, embedder, index, llm)

    for item in corpus_data:
        detector.ingest(Question.from_dict(item))

    query = "Tell me what light reflection means."

    # Run 1: dup_threshold = 0.88, use_llm_judge = True
    detector.config = DetectorConfig(
        dup_threshold=0.88,
        review_threshold=0.72,
        judge_min_confidence=0.7,
        use_llm_judge=True,
    )
    res1 = detector.check(query)
    m1 = get_method(res1, detector)

    # Run 2: dup_threshold = 0.95, use_llm_judge = True
    detector.config = DetectorConfig(
        dup_threshold=0.95,
        review_threshold=0.72,
        judge_min_confidence=0.7,
        use_llm_judge=True,
    )
    res2 = detector.check(query)
    m2 = get_method(res2, detector)

    # Run 3: dup_threshold = 0.95, use_llm_judge = False
    detector.config = DetectorConfig(
        dup_threshold=0.95,
        review_threshold=0.72,
        judge_min_confidence=0.7,
        use_llm_judge=False,
    )
    res3 = detector.check(query)
    m3 = get_method(res3, detector)

    print(f"Query: '{query}'\n")
    print(f"1. dup_threshold=0.88, judge=True  -> Label: {res1.decision.value:9} | Method: {m1:9} | Score: {res1.score:.4f}")
    print(f"2. dup_threshold=0.95, judge=True  -> Label: {res2.decision.value:9} | Method: {m2:9} | Score: {res2.score:.4f}")
    print(f"3. dup_threshold=0.95, judge=False -> Label: {res3.decision.value:9} | Method: {m3:9} | Score: {res3.score:.4f}")


if __name__ == "__main__":
    main()
