import numpy as np
import pytest

from qdedup.embedder import QuestionEmbedder, VectorIndex
from qdedup.models import Question


def test_vector_index_add_and_search():
    index = VectorIndex()
    # 3 orthogonal and diagonal vectors in 3D
    v_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v_xy = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # will be normalized

    index.add("x", v_x)
    index.add("y", v_y)
    index.add("xy", v_xy)

    assert len(index) == 3
    assert "x" in index

    # Search with query aligned with v_x
    results = index.search(v_x, k=2)
    assert len(results) == 2
    assert results[0][0] == "x"
    assert pytest.approx(results[0][1], rel=1e-5) == 1.0
    assert results[1][0] == "xy"
    assert pytest.approx(results[1][1], rel=1e-4) == 0.7071


def test_vector_index_empty_and_bounds():
    index = VectorIndex()
    assert index.search(np.array([1.0, 0.0]), k=3) == []

    index.add("single", np.array([1.0, 0.0]))
    # Asking for k > len(index)
    results = index.search(np.array([1.0, 0.0]), k=5)
    assert len(results) == 1
    assert results[0][0] == "single"


def test_vector_index_swappability_documented():
    doc = VectorIndex.__doc__ or ""
    assert "FAISS" in doc or "faiss" in doc.lower()
    assert "HNSW" in doc or "hnsw" in doc.lower()


def test_question_embedder_cache_and_only_normalizes_text(monkeypatch):
    """Test QuestionEmbedder caching behavior and guarantee tags are never embedded."""
    embedder = QuestionEmbedder()

    # Pre-populate dummy cache for known hash
    dummy_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    q1 = Question(
        id="Q1",
        text="Define the reflection of light.",
        tags={"subject": "physics", "author": "Alice"},
    )
    # Inject into cache
    embedder._cache[q1.text_hash] = dummy_vec

    # Encode q1 - should hit cache
    res1 = embedder.encode(q1)
    assert np.allclose(res1, dummy_vec)

    # Encode another question with the exact same text but different tags
    q2 = Question(
        id="Q2",
        text="define  the Reflection of light ?!",
        tags={"subject": "completely_different_subject", "special_key": "999"},
    )
    # Because tags are excluded and text normalizes to same, this MUST hit cache!
    res2 = embedder.encode(q2)
    assert np.allclose(res2, dummy_vec)
    assert embedder.cache_size == 1


def test_embedder_live_encode():
    """Verify live SentenceTransformer encodes properly to L2-normalized vectors."""
    embedder = QuestionEmbedder()
    vec = embedder.encode("What is the reflection of light?")

    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape == (384,)
    # Verify L2 norm is 1.0
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-5) == 1.0

    # Ensure cached
    assert embedder.cache_size == 1

    # Second call should return cached vector without re-invoking model
    cached_vec = embedder.encode("What is the reflection of light?")
    assert np.allclose(vec, cached_vec)
