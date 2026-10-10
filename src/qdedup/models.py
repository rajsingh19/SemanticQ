"""Data models for question-dedup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdedup.normalize import hash_text, normalize


@dataclass
class Question:
    """Represents a question entity with arbitrary tags and normalization metadata."""

    id: str
    text: str
    tags: dict[str, str] = field(default_factory=dict)
    normalized: str = ""
    text_hash: str = ""

    def __post_init__(self) -> None:
        if not self.normalized and self.text:
            self.normalized = normalize(self.text)
        if not self.text_hash and self.normalized:
            self.text_hash = hash_text(self.normalized)

    def to_dict(self) -> dict[str, Any]:
        """Convert Question instance to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "tags": dict(self.tags),
            "normalized": self.normalized,
            "text_hash": self.text_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        """Create Question instance from dictionary."""
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            tags=dict(data.get("tags") or {}),
            normalized=str(data.get("normalized", "")),
            text_hash=str(data.get("text_hash", "")),
        )
