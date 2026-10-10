import pytest
from qdedup.normalize import hash_text, normalize


def test_normalization_and_hash_equivalence():
    """Verify that 'Define the reflection of light.' and 'define  the Reflection of light ?!'

    give the exact same hash and normalized text.
    """
    s1 = "Define the reflection of light."
    s2 = "define  the Reflection of light ?!"

    norm1 = normalize(s1)
    norm2 = normalize(s2)

    assert norm1 == "define the reflection of light"
    assert norm2 == "define the reflection of light"
    assert norm1 == norm2

    h1 = hash_text(s1)
    h2 = hash_text(s2)

    assert h1 == h2
    assert len(h1) == 64  # SHA-256 length in hex


def test_keep_digits_and_degree_sign():
    text = "The angle of incidence is 30°. What is the angle of reflection?"
    norm = normalize(text)
    assert "30°" in norm
    assert norm == "the angle of incidence is 30° what is the angle of reflection"


def test_strip_trailing_question_marks_and_dots():
    assert normalize("What is reflection????") == "what is reflection"
    assert normalize("What is reflection...") == "what is reflection"
    assert normalize("What is reflection? . ?") == "what is reflection"


def test_empty_and_whitespace():
    assert normalize("") == ""
    assert normalize("   ") == ""
