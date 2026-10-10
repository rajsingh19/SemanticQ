from pathlib import Path
import numpy as np
import pytest

from qdedup.models import Question
from qdedup.store import QuestionStore


def test_store_add_get_all():
    store = QuestionStore(":memory:")
    q = Question(
        id="Q1",
        text="What is reflection of light?",
        tags={"subject": "physics", "topic": "optics", "grade": "10"},
    )
    store.add(q)

    retrieved = store.get("Q1")
    assert retrieved is not None
    assert retrieved.id == "Q1"
    assert retrieved.text == q.text
    assert retrieved.tags == q.tags
    assert retrieved.normalized == q.normalized
    assert retrieved.text_hash == q.text_hash

    all_questions = store.all()
    assert len(all_questions) == 1
    assert all_questions[0].id == "Q1"


def test_store_filter_by_arbitrary_tags():
    store = QuestionStore(":memory:")
    store.add(
        Question(
            id="Q1",
            text="Question 1",
            tags={"subject": "physics", "level": "advanced", "author": "Alice"},
        )
    )
    store.add(
        Question(
            id="Q2",
            text="Question 2",
            tags={"subject": "physics", "level": "basic", "custom_flag": "yes"},
        )
    )
    store.add(
        Question(
            id="Q3",
            text="Question 3",
            tags={"subject": "chemistry", "level": "basic"},
        )
    )

    # Filter by single arbitrary tag
    res_physics = store.filter_by_tags({"subject": "physics"})
    assert {q.id for q in res_physics} == {"Q1", "Q2"}

    # Filter by multiple arbitrary tags
    res_multi = store.filter_by_tags({"subject": "physics", "level": "basic"})
    assert [q.id for q in res_multi] == ["Q2"]

    # Filter by a completely novel tag key
    res_custom = store.filter_by_tags({"custom_flag": "yes"})
    assert [q.id for q in res_custom] == ["Q2"]

    # Filter by non-matching tags
    res_none = store.filter_by_tags({"non_existent_key": "val"})
    assert res_none == []


def test_store_embeddings_blob():
    store = QuestionStore(":memory:")
    q = Question(id="Q1", text="Sample question")
    emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    store.add(q, embedding=emb)
    retrieved_emb = store.get_embedding("Q1")
    assert retrieved_emb is not None
    assert np.allclose(retrieved_emb, emb)


def test_store_embeddings_npy(tmp_path):
    store = QuestionStore(
        db_path=tmp_path / "test.db",
        npy_dir=tmp_path / "embeddings",
    )
    q = Question(id="Q1", text="Sample question")
    emb = np.array([0.5, 0.6, 0.7], dtype=np.float32)

    store.add(q, embedding=emb)
    assert (tmp_path / "embeddings" / "Q1.npy").exists()

    retrieved_emb = store.get_embedding("Q1")
    assert retrieved_emb is not None
    assert np.allclose(retrieved_emb, emb)


def test_load_corpus_file():
    corpus_path = Path(__file__).resolve().parent.parent / "data" / "corpus.json"
    assert corpus_path.exists()

    store = QuestionStore(":memory:")
    loaded = store.load_corpus(corpus_path)
    assert len(loaded) == 10
    assert [q.id for q in loaded] == [f"Q{i}" for i in range(1, 11)]

    # Q2 and Q10 have the same text
    q2 = store.get("Q2")
    q10 = store.get("Q10")
    assert q2 is not None and q10 is not None
    assert q2.text == q10.text
    assert q2.text_hash == q10.text_hash
