from qdedup.models import Question
from qdedup.normalize import hash_text, normalize


def test_question_model_auto_computation():
    q = Question(
        id="Q1",
        text="Define the reflection of light.",
        tags={"subject": "physics", "topic": "optics"},
    )
    assert q.normalized == "define the reflection of light"
    assert q.text_hash == hash_text("Define the reflection of light.")


def test_question_model_hash_does_not_depend_on_tags():
    q1 = Question(
        id="Q1",
        text="Define the reflection of light.",
        tags={"subject": "physics", "grade": "10"},
    )
    q2 = Question(
        id="Q2",
        text="Define the reflection of light.",
        tags={"arbitrary_key": "arbitrary_val", "another": "123"},
    )
    assert q1.text_hash == q2.text_hash


def test_question_to_dict_and_from_dict():
    q = Question(
        id="Q1",
        text="What is reflection?",
        tags={"subject": "physics"},
    )
    d = q.to_dict()
    q_copy = Question.from_dict(d)
    assert q_copy.id == q.id
    assert q_copy.text == q.text
    assert q_copy.tags == q.tags
    assert q_copy.normalized == q.normalized
    assert q_copy.text_hash == q.text_hash
