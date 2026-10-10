"""SQLite storage for questions and embeddings with arbitrary JSON tags."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qdedup.models import Question


class QuestionStore:
    """SQLite-backed question store.

    Stores arbitrary tag key-value pairs in a JSON column so schema changes are never
    required when adding or modifying tag keys.
    Embeddings can be persisted as BLOBs within SQLite or in a separate directory of .npy files.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        npy_dir: str | Path | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.npy_dir = Path(npy_dir) if npy_dir is not None else None

        if self.npy_dir is not None:
            self.npy_dir.mkdir(parents=True, exist_ok=True)

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding BLOB
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_questions_text_hash
                ON questions (text_hash);
                """
            )

    def add(
        self,
        question: Question,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Add or update a question in the store with optional embedding."""
        tags_json = json.dumps(question.tags or {})
        blob: bytes | None = None

        if embedding is not None:
            arr = np.asarray(embedding, dtype=np.float32)
            blob = arr.tobytes()

            if self.npy_dir is not None:
                np.save(self.npy_dir / f"{question.id}.npy", arr)

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO questions (id, text, tags, normalized, text_hash, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text = excluded.text,
                    tags = excluded.tags,
                    normalized = excluded.normalized,
                    text_hash = excluded.text_hash,
                    embedding = COALESCE(excluded.embedding, questions.embedding);
                """,
                (
                    question.id,
                    question.text,
                    tags_json,
                    question.normalized,
                    question.text_hash,
                    blob,
                ),
            )

    def get(self, id: str) -> Question | None:
        """Retrieve a question by its id."""
        cursor = self._conn.execute(
            "SELECT id, text, tags, normalized, text_hash FROM questions WHERE id = ?",
            (id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        tags_dict = json.loads(row[2]) if row[2] else {}
        return Question(
            id=row[0],
            text=row[1],
            tags=tags_dict,
            normalized=row[3],
            text_hash=row[4],
        )

    def get_embedding(self, id: str) -> np.ndarray | None:
        """Retrieve the embedding vector for a given question id."""
        # 1. Check separate .npy file if directory was provided
        if self.npy_dir is not None:
            npy_path = self.npy_dir / f"{id}.npy"
            if npy_path.exists():
                return np.load(npy_path)

        # 2. Check BLOB in SQLite
        cursor = self._conn.execute(
            "SELECT embedding FROM questions WHERE id = ?",
            (id,),
        )
        row = cursor.fetchone()
        if row and row[0]:
            return np.frombuffer(row[0], dtype=np.float32)

        return None

    def set_embedding(self, id: str, embedding: np.ndarray) -> None:
        """Update the embedding for an existing question."""
        arr = np.asarray(embedding, dtype=np.float32)
        blob = arr.tobytes()

        if self.npy_dir is not None:
            np.save(self.npy_dir / f"{id}.npy", arr)

        with self._conn:
            self._conn.execute(
                "UPDATE questions SET embedding = ? WHERE id = ?",
                (blob, id),
            )

    def all(self) -> list[Question]:
        """Retrieve all questions stored in the database."""
        cursor = self._conn.execute(
            "SELECT id, text, tags, normalized, text_hash FROM questions ORDER BY id"
        )
        results: list[Question] = []
        for row in cursor.fetchall():
            tags_dict = json.loads(row[2]) if row[2] else {}
            results.append(
                Question(
                    id=row[0],
                    text=row[1],
                    tags=tags_dict,
                    normalized=row[3],
                    text_hash=row[4],
                )
            )
        return results

    def filter_by_tags(self, tags: Mapping[str, str]) -> list[Question]:
        """Filter questions where all provided key-value tags match."""
        if not tags:
            return self.all()

        clauses: list[str] = []
        params: list[Any] = []
        for key, value in tags.items():
            clauses.append("json_extract(tags, ?) = ?")
            params.extend([f'$."{key}"', str(value)])

        query = f"""
            SELECT id, text, tags, normalized, text_hash
            FROM questions
            WHERE {' AND '.join(clauses)}
            ORDER BY id
        """

        cursor = self._conn.execute(query, params)
        results: list[Question] = []
        for row in cursor.fetchall():
            tags_dict = json.loads(row[2]) if row[2] else {}
            # Verify exact tag match in Python
            if all(tags_dict.get(k) == str(v) for k, v in tags.items()):
                results.append(
                    Question(
                        id=row[0],
                        text=row[1],
                        tags=tags_dict,
                        normalized=row[3],
                        text_hash=row[4],
                    )
                )
        return results

    def load_corpus(self, corpus_path: str | Path) -> list[Question]:
        """Convenience method to load and add questions from a JSON corpus file."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        questions: list[Question] = []
        for item in items:
            q = Question.from_dict(item)
            self.add(q)
            questions.append(q)
        return questions

    def close(self) -> None:
        """Close SQLite database connection."""
        self._conn.close()

    def __enter__(self) -> QuestionStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# Alias for explicit naming
SQLiteStore = QuestionStore
