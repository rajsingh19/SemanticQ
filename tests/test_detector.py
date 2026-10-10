import json
from pathlib import Path
import pytest

from qdedup.detector import Decision, DuplicateDetector
from qdedup.embedder import QuestionEmbedder, VectorIndex
from qdedup.llm import JudgeResult, MockLLM
from qdedup.models import Question
from qdedup.store import QuestionStore


def test_mock_llm_judge_jaccard_rule():
    """Requirement 1:

    MockLLM.judge word-overlap rule must use Jaccard similarity (|A∩B| / |A∪B|)
    over content words, NOT overlap relative to the smaller set.
    Threshold 0.6. Expected: 'State the laws of reflection of light' vs
    'Define the reflection of light' -> Jaccard 0.5 -> same=False.
    """
    mock_llm = MockLLM(judge_threshold=0.6)
    q1 = "State the laws of reflection of light"
    q2 = "Define the reflection of light"

    # Verify exact Jaccard similarity calculation
    jaccard = mock_llm.jaccard_similarity(q1, q2)
    assert pytest.approx(jaccard, abs=1e-4) == 0.5

    result = mock_llm.judge(q1, q2)
    assert isinstance(result, tuple)
    assert result.same is False
    assert result.confidence >= 0.7


def test_review_band_any_neighbour_duplicate():
    """Requirement 2a:

    If ANY neighbour is judged same with confidence >= judge_min_confidence -> DUPLICATE.
    """
    store = QuestionStore(":memory:")
    q_existing = Question(id="Q1", text="What is reflection of light?")
    store.add(q_existing)

    embedder = QuestionEmbedder()
    index = VectorIndex()
    # Dummy vector
    dummy_vec = embedder.encode(q_existing)
    index.add("Q1", dummy_vec)

    # MockLLM configured to return same=True with high confidence
    mock_llm = MockLLM()
    mock_llm.judge = lambda q1, q2, **kw: JudgeResult(same=True, confidence=0.85, explanation="duplicate")

    detector = DuplicateDetector(
        store=store,
        embedder=embedder,
        index=index,
        llm=mock_llm,
        duplicate_threshold=0.99,  # Force into review band
        review_threshold=0.50,
        judge_min_confidence=0.70,
    )

    query = Question(id="Q_new", text="Can you explain reflection of light?")
    res = detector.detect(query)
    assert res.decision == Decision.DUPLICATE
    assert res.matched_id == "Q1"


def test_review_band_all_confidently_different():
    """Requirement 2b:

    If ALL neighbours are confidently different -> UNIQUE.
    """
    store = QuestionStore(":memory:")
    q_existing = Question(id="Q1", text="Define the reflection of light.")
    store.add(q_existing)

    embedder = QuestionEmbedder()
    index = VectorIndex()
    index.add("Q1", embedder.encode(q_existing))

    # MockLLM returns same=False with high confidence
    mock_llm = MockLLM()
    mock_llm.judge = lambda q1, q2, **kw: JudgeResult(same=False, confidence=0.90, explanation="distinct")

    detector = DuplicateDetector(
        store=store,
        embedder=embedder,
        index=index,
        llm=mock_llm,
        duplicate_threshold=0.99,
        review_threshold=0.50,
        judge_min_confidence=0.70,
    )

    query = Question(id="Q_new", text="State the laws of reflection of light.")
    res = detector.detect(query)
    assert res.decision == Decision.UNIQUE
    assert res.matched_id is None


def test_review_band_borderline_review():
    """Requirement 2c:

    Otherwise -> REVIEW when judge is unconfident.
    """
    store = QuestionStore(":memory:")
    q_existing = Question(id="Q1", text="Define the reflection of light.")
    store.add(q_existing)

    embedder = QuestionEmbedder()
    index = VectorIndex()
    index.add("Q1", embedder.encode(q_existing))

    # MockLLM returns unconfident verdict (confidence < judge_min_confidence)
    mock_llm = MockLLM()
    mock_llm.judge = lambda q1, q2, **kw: JudgeResult(same=False, confidence=0.55, explanation="uncertain")

    detector = DuplicateDetector(
        store=store,
        embedder=embedder,
        index=index,
        llm=mock_llm,
        duplicate_threshold=0.99,
        review_threshold=0.50,
        judge_min_confidence=0.70,
    )

    query = Question(id="Q_new", text="What is reflection?")
    res = detector.detect(query)
    assert res.decision == Decision.REVIEW
    assert res.matched_id == "Q1"


@pytest.mark.slow
def test_real_embedder_corpus_sequential_ingestion():
    """Requirement 3:

    Add tests using the REAL embedder on the Q1..Q10 corpus (mark with @pytest.mark.slow):
    assert Q2,Q3,Q4 -> DUPLICATE, Q10 -> EXACT,
    Q5,Q6,Q7,Q9 -> UNIQUE, Q8 in {REVIEW, UNIQUE}.
    Ingest in order Q1..Q10, each checked against earlier ones only.
    """
    corpus_path = Path(__file__).resolve().parent.parent / "data" / "corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    questions = [Question.from_dict(d) for d in corpus_data]
    assert len(questions) == 10

    # Ingest in order Q1..Q10 into fresh store and index
    store = QuestionStore(":memory:")
    embedder = QuestionEmbedder()
    index = VectorIndex()
    llm = MockLLM(judge_threshold=0.6)

    detector = DuplicateDetector(
        store=store,
        embedder=embedder,
        index=index,
        llm=llm,
        duplicate_threshold=0.95,
        review_threshold=0.78,
        judge_min_confidence=0.70,
    )

    decisions: dict[str, Decision] = {}

    for q in questions:
        # Check against earlier ones only via detect(), then ingest
        detection = detector.ingest(q)
        decisions[q.id] = detection.decision

    # Assertions according to Requirement 3:
    assert decisions["Q2"] == Decision.DUPLICATE
    assert decisions["Q3"] == Decision.DUPLICATE
    assert decisions["Q4"] == Decision.DUPLICATE
    assert decisions["Q10"] == Decision.EXACT
    assert decisions["Q5"] == Decision.UNIQUE
    assert decisions["Q6"] == Decision.UNIQUE
    assert decisions["Q7"] == Decision.UNIQUE
    assert decisions["Q9"] == Decision.UNIQUE
    assert decisions["Q8"] in {Decision.REVIEW, Decision.UNIQUE}
