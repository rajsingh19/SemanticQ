#!/usr/bin/env python3
"""scripts/sim_check.py: Compute 10x10 cosine similarity matrix for Q1..Q10 and best earlier matches."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table

from qdedup.embedder import QuestionEmbedder
from qdedup.models import Question


def main() -> None:
    console = Console(width=120)
    corpus_path = root_dir / "data" / "corpus.json"

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    questions = [Question.from_dict(d) for d in corpus_data]
    q_ids = [q.id for q in questions]

    embedder = QuestionEmbedder()
    # Batch encode all 10 questions (normalized text only)
    embeddings = embedder.encode_batch(questions)

    # 10x10 cosine similarity matrix (dot product of L2-normalized vectors)
    sim_matrix = embeddings @ embeddings.T

    # 1. Print full 10x10 similarity matrix table
    matrix_table = Table(
        title="10x10 Cosine Similarity Matrix (all-MiniLM-L6-v2)",
        show_header=True,
        header_style="bold cyan",
    )
    matrix_table.add_column("Question", style="bold yellow")
    for q_id in q_ids:
        matrix_table.add_column(q_id, justify="right")

    for i, q_id in enumerate(q_ids):
        row_vals = [f"{sim_matrix[i, j]:.2f}" for j in range(len(q_ids))]
        matrix_table.add_row(q_id, *row_vals)

    console.print()
    console.print(matrix_table)
    console.print()

    # 2. Print best-matching earlier question for each of Q2..Q10
    match_table = Table(
        title="Best-Matching Earlier Question for Q2..Q10",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    match_table.add_column("Target", style="bold yellow", no_wrap=True)
    match_table.add_column("Target Question Text", style="white")
    match_table.add_column("Best Earlier", style="bold green", no_wrap=True)
    match_table.add_column("Score", justify="right", style="bold cyan", no_wrap=True)
    match_table.add_column("Earlier Question Text", style="dim white")

    for i in range(1, len(questions)):
        target = questions[i]
        # Search among all earlier questions j < i
        earlier_scores = [(j, float(sim_matrix[i, j])) for j in range(i)]
        best_j, best_score = max(earlier_scores, key=lambda x: x[1])
        best_earlier = questions[best_j]

        match_table.add_row(
            target.id,
            target.text,
            best_earlier.id,
            f"{best_score:.2f}",
            best_earlier.text,
        )

    console.print(match_table)
    console.print()


if __name__ == "__main__":
    main()
