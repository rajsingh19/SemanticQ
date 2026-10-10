#!/usr/bin/env python3
"""demo.py: End-to-end demonstration of question-dedup system."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure 'src' is available on python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from qdedup.context_selector import ContextSelector
from qdedup.detector import DuplicateDetector
from qdedup.embedder import QuestionEmbedder, VectorIndex
from qdedup.generator import QuestionGenerator
from qdedup.llm import MockLLM, get_llm_client
from qdedup.models import Question
from qdedup.normalize import hash_text, normalize
from qdedup.store import QuestionStore

load_dotenv()
console = Console(width=110)


def main() -> None:
    console.print(
        Panel(
            "[bold green]question-dedup: End-to-End System Demo[/bold green]\n"
            "Semantic deduplication, context selection, and non-duplicate question generation.",
            expand=False,
        )
    )

    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    console.print(f"GEMINI_API_KEY detected: [bold]{api_key_set}[/bold] (Running offline with deterministic MockLLM)\n")

    # 1. Normalization & Hashing
    console.print("[bold cyan]1. Normalization & Text Hashing:[/bold cyan]")
    raw1 = "Define the reflection of light."
    raw2 = "define  the Reflection of light ?!"
    norm1, norm2 = normalize(raw1), normalize(raw2)
    h1, h2 = hash_text(raw1), hash_text(raw2)
    console.print(f"Raw 1: '{raw1}' -> Norm: '[green]{norm1}[/green]' (Hash: {h1[:12]}...)")
    console.print(f"Raw 2: '{raw2}' -> Norm: '[green]{norm2}[/green]' (Hash: {h2[:12]}...)")
    status_style = "bold green" if h1 == h2 else "bold red"
    console.print(f"Exact Hash Match: [{status_style}]{h1 == h2}[/{status_style}]\n")

    # 2. Loading Corpus into Store & Index
    console.print("[bold cyan]2. Ingesting Question Corpus (Q1..Q10):[/bold cyan]")
    corpus_file = Path(__file__).resolve().parent / "data" / "corpus.json"
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    store = QuestionStore(":memory:")
    embedder = QuestionEmbedder()
    index = VectorIndex()
    llm = MockLLM()
    detector = DuplicateDetector(store, embedder, index, llm)

    for item in corpus_data:
        q = Question.from_dict(item)
        detector.ingest(q)

    console.print(f"Ingested [green]{len(store.all())}[/green] questions into SQLite store and VectorIndex.\n")

    # 3. Duplicate Detection on Sample Queries
    console.print("[bold cyan]3. Multi-Band Duplicate Detection Demo:[/bold cyan]")
    test_queries = [
        "what is reflection of light",
        "Describe what is meant by light reflection.",
        "Draw a ray diagram showing reflection from a plane mirror.",
        "The angle of incidence is 45°. Find the angle of reflection.",
        "Why do we see our image in a plane mirror?",
    ]

    label_descriptions = {
        "EXACT": "Exact match (normalized text hash match)",
        "DUPLICATE": "Semantic duplicate (high similarity or judge confirmed)",
        "REVIEW": "Ambiguous / borderline case (flagged for manual review)",
        "UNIQUE": "Novel / distinct question (below review band or judge verified)",
    }

    det_table = Table(title="Duplicate Detection Results", header_style="bold magenta")
    det_table.add_column("Query Text", style="white")
    det_table.add_column("Decision", style="bold green")
    det_table.add_column("Score", justify="right")
    det_table.add_column("Matched ID")
    det_table.add_column("Description", style="dim white")

    for q_text in test_queries:
        res = detector.detect(q_text)
        desc = label_descriptions.get(res.decision.value, res.decision.value)
        det_table.add_row(
            q_text,
            res.decision.value,
            f"{res.score:.4f}",
            res.matched_id or "—",
            desc,
        )

    console.print(det_table)
    console.print()

    # 4. Context Selection with Clustering & MMR
    console.print("[bold cyan]4. Context Selection (KMeans + MMR Diversification):[/bold cyan]")
    selector = ContextSelector(store, embedder, index, max_questions=5, max_tokens=500, n_clusters=4)
    sel_res = selector.select("reflection of light")
    console.print(f"Selected [bold]{len(sel_res.questions)}[/bold] questions, Token estimate: [bold]{sel_res.token_estimate}[/bold] tokens.")
    console.print("[bold yellow]Covered Concept Clusters:[/bold yellow]")
    for bullet in sel_res.covered_concepts:
        console.print(f" - {bullet}")
    console.print()

    # 5. Question Generation with Duplicate Prevention
    console.print("[bold cyan]5. Question Generation with Intra-Batch Duplicate Prevention:[/bold cyan]")
    generator = QuestionGenerator(llm, selector, detector)
    gen_report = generator.generate("reflection of light", n=3)

    console.print(f"Target: 3 questions | Rounds used: [bold]{gen_report.rounds_used}[/bold] | Failed rounds: [bold]{gen_report.failed_rounds}[/bold]")
    console.print(f"Accepted: [bold green]{len(gen_report.accepted)}[/bold green] | Rejected: [bold red]{len(gen_report.rejected)}[/bold red] | Duplicate rate: [bold]{gen_report.duplicate_rate:.1%}[/bold]\n")

    accepted_table = Table(title="Generated & Accepted Questions", header_style="bold green")
    accepted_table.add_column("ID", style="bold yellow", width=12)
    accepted_table.add_column("Question Text", style="white")

    for q in gen_report.accepted:
        accepted_table.add_row(q.id, q.text)

    console.print(accepted_table)
    console.print()

    if gen_report.rejected:
        rejected_table = Table(title="Prevented Duplicates (Filtered Out)", header_style="bold red")
        rejected_table.add_column("Candidate Text", style="white", width=48)
        rejected_table.add_column("Decision", style="bold red", width=12)
        rejected_table.add_column("Matched ID", width=10)
        rejected_table.add_column("Score", justify="right", width=8)

        for r in gen_report.rejected:
            rejected_table.add_row(r["text"], r["decision"], r["matched_id"] or "—", f"{r['score']:.4f}")

        console.print(rejected_table)
        console.print()

    console.print(Panel("[bold green]All systems operational![/bold green]", expand=False))


if __name__ == "__main__":
    main()
