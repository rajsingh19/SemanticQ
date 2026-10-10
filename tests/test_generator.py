"""Tests for QuestionGenerator and duplicate-free question generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qdedup.context_selector import ContextSelector
from qdedup.detector import DuplicateDetector
from qdedup.embedder import QuestionEmbedder, VectorIndex
from qdedup.generator import QuestionGenerator
from qdedup.llm import MockLLM
from qdedup.models import Question
from qdedup.store import QuestionStore


@pytest.fixture(scope="session")
def real_embedder() -> QuestionEmbedder:
    """Session-scoped real embedder so model is loaded only once across tests."""
    return QuestionEmbedder()


@pytest.fixture
def test_setup(real_embedder: QuestionEmbedder):
    """Preload Q1..Q10 corpus into fresh store and index with mock LLM."""
    corpus_path = Path(__file__).resolve().parent.parent / "data" / "corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    store = QuestionStore(":memory:")
    index = VectorIndex()
    llm = MockLLM()

    detector = DuplicateDetector(store, real_embedder, index, llm)
    for d in corpus_data:
        q = Question.from_dict(d)
        detector.ingest(q)

    selector = ContextSelector(store, real_embedder, index)
    generator = QuestionGenerator(llm, selector, detector)

    return {
        "generator": generator,
        "store": store,
        "index": index,
        "detector": detector,
        "selector": selector,
        "llm": llm,
    }


def test_generation_round1_duplicates_and_acceptance(test_setup: dict[str, Any]):
    """Tests 1, 2, 3, 4, 9a:

    1. exact duplicate rejected with label EXACT
    2. semantic duplicate rejected with label DUPLICATE
    3. new questions accepted and present in store + index afterwards
    4. intra-batch duplicate (#6) rejected, assert label in {EXACT, DUPLICATE}
    9a. n=3 works with scripted 3 new questions in round 1.
    """
    generator: QuestionGenerator = test_setup["generator"]
    store: QuestionStore = test_setup["store"]
    index: VectorIndex = test_setup["index"]

    report = generator.generate("reflection of light", n=3)

    # 9a. n=3 satisfied in Round 1
    assert report.rounds_used == 1
    assert len(report.accepted) == 3

    # 1. Exact duplicate rejected with label EXACT (#1: "Define the reflection of light.")
    exact_rejections = [
        r for r in report.rejected
        if r["text"] == "Define the reflection of light."
    ]
    assert len(exact_rejections) == 1
    assert exact_rejections[0]["decision"] == "EXACT"

    # 2. Semantic duplicate rejected with label DUPLICATE (#2: "Describe what is meant by light reflection.")
    sem_rejections = [
        r for r in report.rejected
        if r["text"] == "Describe what is meant by light reflection."
    ]
    assert len(sem_rejections) >= 1
    assert sem_rejections[0]["decision"] == "DUPLICATE"

    # 4. Intra-batch duplicate (#6) rejected with label in {EXACT, DUPLICATE}
    assert len(sem_rejections) == 2  # Candidate #2 and Candidate #6
    for r in sem_rejections:
        assert r["decision"] in {"EXACT", "DUPLICATE"}

    # 3. New questions accepted and present in store + index afterwards
    for q in report.accepted:
        stored = store.get(q.id)
        assert stored is not None
        assert stored.text == q.text
        assert q.id in index


def test_n5_forces_round2_and_rejected_texts_appear_in_prompt(test_setup: dict[str, Any]):
    """Tests 6 and 9b:

    6. rejected texts appear in the round-2 prompt
    9b. n=5 request forces round 2.
    """
    generator: QuestionGenerator = test_setup["generator"]
    llm: MockLLM = test_setup["llm"]

    report = generator.generate("reflection of light", n=5)

    # 9b. n=5 forces Round 2
    assert report.rounds_used == 2
    assert len(report.accepted) == 5

    # 6. Rejected texts appear in round-2 prompt
    assert len(llm.call_history) >= 2
    round2_prompt = llm.call_history[1]

    assert "Already rejected this session, avoid these:" in round2_prompt
    assert "Define the reflection of light." in round2_prompt
    assert "Describe what is meant by light reflection." in round2_prompt


def test_malformed_json_handling(test_setup: dict[str, Any]):
    """Test 5: malformed JSON from the LLM -> failed round, no crash, retry happens."""
    generator: QuestionGenerator = test_setup["generator"]
    llm: MockLLM = test_setup["llm"]

    # First call returns invalid JSON, second call returns valid JSON
    original_generate = llm.generate
    call_count = [0]

    def faulty_generate(prompt: str, **kwargs: Any) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return "This is malformed JSON without code fences or brackets"
        return original_generate(prompt, **kwargs)

    llm.generate = faulty_generate

    report = generator.generate("reflection of light", n=3, max_rounds=2)

    assert report.failed_rounds == 1
    assert report.rounds_used == 2
    assert len(report.accepted) == 3


def test_tags_stored_on_accepted_questions(test_setup: dict[str, Any]):
    """Test 7: tags stored on accepted questions but do not change accept/reject outcome."""
    generator: QuestionGenerator = test_setup["generator"]

    custom_tags = {"subject": "physics", "curriculum": "CBSE", "grade": "10"}
    report = generator.generate("reflection of light", n=3, tags=custom_tags)

    assert len(report.accepted) == 3
    for q in report.accepted:
        assert q.tags == custom_tags

    # Rejection decisions remain identical
    rejected_texts = {r["text"] for r in report.rejected}
    assert "Define the reflection of light." in rejected_texts
    assert "Describe what is meant by light reflection." in rejected_texts


def test_prompt_token_estimate_bounded(test_setup: dict[str, Any]):
    """Test 8: prompt token estimate <= selector budget + fixed overhead."""
    generator: QuestionGenerator = test_setup["generator"]
    selector: ContextSelector = test_setup["selector"]

    report = generator.generate("reflection of light", n=3)

    assert len(report.prompt_token_estimates) > 0
    # Fixed prompt text overhead is well under 500 tokens
    for est in report.prompt_token_estimates:
        assert est <= selector.max_tokens + 500
