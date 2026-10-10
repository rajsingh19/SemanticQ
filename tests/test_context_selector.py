import json
import re
from pathlib import Path
import numpy as np
import pytest

from qdedup.context_selector import ContextSelector, SelectionResult
from qdedup.embedder import QuestionEmbedder, VectorIndex
from qdedup.models import Question
from qdedup.store import QuestionStore


class DummyEmbedder:
    """Fast deterministic embedder for non-model unit tests."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, item: str | Question) -> np.ndarray:
        text = item.text if isinstance(item, Question) else str(item)
        # Seeded deterministic pseudo-embedding based on text characters
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, ch in enumerate(text[: self.dim]):
            vec[i] = (ord(ch) % 17) + 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode_batch(self, items: list[str | Question]) -> np.ndarray:
        return np.vstack([self.encode(it) for it in items])


def test_respects_max_questions_and_max_tokens():
    """Test 1: respects max_questions and max_tokens."""
    store = QuestionStore(":memory:")
    embedder = DummyEmbedder()

    for i in range(20):
        q = Question(
            id=f"q_{i}",
            text=f"This is sample question number {i} asking about fundamental concepts.",
        )
        store.add(q)

    # 1. Test max_questions constraint
    selector_q = ContextSelector(store, embedder, max_questions=3, max_tokens=5000)
    res_q = selector_q.select("sample query")
    assert len(res_q.questions) <= 3
    assert res_q.token_estimate == sum(
        ContextSelector.estimate_tokens(q.text) for q in res_q.questions
    )

    # 2. Test max_tokens constraint
    selector_tok = ContextSelector(store, embedder, max_questions=50, max_tokens=40)
    res_tok = selector_tok.select("sample query")
    assert res_tok.token_estimate <= 40
    assert len(res_tok.questions) > 0


def test_tags_filter_narrows_pool_and_tags_do_not_affect_embedding():
    """Test 2:

    tags_filter narrows the pool but a different tag value with the same text gives the
    same embedding/similarity.
    """
    store = QuestionStore(":memory:")
    embedder = QuestionEmbedder()  # Real or mock

    q_physics1 = Question(
        id="P1",
        text="What is reflection of light?",
        tags={"subject": "physics", "topic": "optics"},
    )
    q_physics2 = Question(
        id="P2",
        text="Define the reflection of light.",
        tags={"subject": "physics", "topic": "optics"},
    )
    q_bio = Question(
        id="B1",
        text="What is photosynthesis?",
        tags={"subject": "biology", "topic": "plants"},
    )
    # Question with same text as P1 but different tags
    q_physics_diff_tag = Question(
        id="P1_diff",
        text="What is reflection of light?",
        tags={"subject": "general_science", "difficulty": "hard"},
    )

    store.add(q_physics1)
    store.add(q_physics2)
    store.add(q_bio)

    selector = ContextSelector(store, embedder)

    # Filtering by subject="physics" excludes biology
    res_physics = selector.select("reflection", tags_filter={"subject": "physics"})
    selected_ids = {q.id for q in res_physics.questions}
    assert "B1" not in selected_ids
    assert len(selected_ids) <= 2

    # Verify that different tags with same text produce exact same embedding vector
    vec1 = embedder.encode(q_physics1)
    vec2 = embedder.encode(q_physics_diff_tag)
    assert np.allclose(vec1, vec2)
    # Cosine similarity is 1.0
    assert pytest.approx(float(vec1 @ vec2), abs=1e-5) == 1.0


def test_every_cluster_has_at_least_one_representative():
    """Test 3: every cluster has at least one representative when budget allows."""
    store = QuestionStore(":memory:")
    embedder = DummyEmbedder(dim=16)

    # Create 4 distinct groups of questions
    for group in range(4):
        for member in range(3):
            q = Question(
                id=f"g{group}_m{member}",
                text=f"Group {group} distinct keyword topic query {member}",
            )
            store.add(q)

    n_clusters = 4
    selector = ContextSelector(
        store,
        embedder,
        n_clusters=n_clusters,
        max_questions=10,
        max_tokens=2000,
    )
    res = selector.select("topic query")
    # All 4 clusters must be represented
    assert len(res.questions) >= n_clusters
    assert len(res.covered_concepts) == n_clusters


def test_empty_store_and_store_smaller_than_n_clusters():
    """Test 4: empty store, and store smaller than n_clusters."""
    store_empty = QuestionStore(":memory:")
    embedder = DummyEmbedder()

    selector_empty = ContextSelector(store_empty, embedder, n_clusters=8)
    res_empty = selector_empty.select("test query")
    assert res_empty.questions == []
    assert res_empty.covered_concepts == []
    assert res_empty.token_estimate == 0

    # Store with only 2 items, n_clusters = 8
    store_small = QuestionStore(":memory:")
    q1 = Question(id="s1", text="First question")
    q2 = Question(id="s2", text="Second question")
    store_small.add(q1)
    store_small.add(q2)

    selector_small = ContextSelector(store_small, embedder, n_clusters=8)
    res_small = selector_small.select("test query")
    assert len(res_small.questions) == 2
    assert len(res_small.covered_concepts) == 2
    assert {q.id for q in res_small.questions} == {"s1", "s2"}


@pytest.mark.slow
def test_covered_concepts_for_corpus_definition_cluster():
    """Test 5:

    covered_concepts for the Q1..Q10 corpus (real embedder, mark slow):
    result includes the definition cluster (Q1-Q4, Q10) with a count >= 4.
    """
    corpus_path = Path(__file__).resolve().parent.parent / "data" / "corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    store = QuestionStore(":memory:")
    embedder = QuestionEmbedder()

    for d in corpus_data:
        q = Question.from_dict(d)
        store.add(q, embedding=embedder.encode(q))

    selector = ContextSelector(store, embedder, n_clusters=5)
    res = selector.select("What is reflection of light?")
    bullets = res.covered_concepts

    assert len(bullets) > 0
    # Find cluster bullet with definition keywords and count >= 4
    found_definition_cluster = False
    for bullet in bullets:
        match = re.search(r"\((\d+) similar questions\)", bullet)
        if match:
            count = int(match.group(1))
            if count >= 4 and ("reflection" in bullet.lower() or "light" in bullet.lower()):
                found_definition_cluster = True
                break

    assert found_definition_cluster, f"Definition cluster not found in bullets: {bullets}"


def test_prompt_size_stays_constant_with_synthetic_5000_questions():
    """Test 6:

    prompt size stays constant: synthetic 5000 questions still returns <= max_questions.
    """
    store = QuestionStore(":memory:")
    embedder = DummyEmbedder(dim=8)

    # Insert 5000 synthetic questions
    questions = [
        Question(id=f"syn_{i}", text=f"Synthetic question {i} regarding science")
        for i in range(5000)
    ]
    for q in questions:
        store.add(q)

    max_q = 12
    selector = ContextSelector(
        store,
        embedder,
        neighbour_pool=100,
        max_questions=max_q,
        max_tokens=1500,
    )

    res = selector.select("science query")
    assert len(res.questions) <= max_q
    assert res.token_estimate <= 1500
