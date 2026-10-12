"""Question normalization and hashing utilities."""

from __future__ import annotations

import hashlib
import re
import string
import unicodedata


def normalize(text: str) -> str:
    """Normalize question text.

    Applies:
    - Unicode NFKC normalization
    - Lowercasing
    - Trailing '?' and '.' (and '!') removed
    - Punctuation stripped, extra whitespace collapsed
    - Digits and the degree sign ('°') are preserved.
    """
    if not text:
        return ""

    # 1. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Explicitly remove trailing '?' and '.' (and '!')
    text = re.sub(r"[\?\.!\s]+$", "", text)

    # 4. Strip punctuation, preserving alphanumeric, whitespace, and degree sign ('°')
    cleaned: list[str] = []
    for ch in text:
        if ch.isalnum() or ch.isspace() or ch == "°":
            cleaned.append(ch)
        elif unicodedata.category(ch).startswith("P") or ch in string.punctuation:
            cleaned.append(" ")
        else:
            cleaned.append(ch)

    text = "".join(cleaned)

    # 5. Collapse extra whitespace and strip edges
    return " ".join(text.split())


def hash_text(text: str) -> str:
    """Compute SHA-256 hash of normalized text.

    Tags are intentionally NOT included in this hash.
    """
    normalized = normalize(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?(?:\s*(?:°|deg(?:ree)?s?))?", re.IGNORECASE)


def extract_numbers(text: str) -> set[str]:
    """Extract numbers using regex for digits, with or without degree sign or word 'degree'.

    Canonicalizes degree representations so that e.g. '60°', '60 degrees', and '60 degree'
    normalize to '60°'.
    """
    if not text:
        return set()
    matches = NUMBER_PATTERN.findall(text)
    numbers = set()
    for m in matches:
        m = m.strip().lower()
        if not m:
            continue
        canon = re.sub(r"\s*deg(?:ree)?s?", "°", m, flags=re.IGNORECASE)
        numbers.add(canon)
    return numbers
