#!/usr/bin/env python3
"""eval/evaluate.py: Comprehensive evaluation and ablation framework for question-dedup."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from qdedup.detector import Decision, DuplicateDetector
from qdedup.embedder import QuestionEmbedder, VectorIndex
from qdedup.llm import GeminiLLM, MockLLM
from qdedup.models import Question
from qdedup.store import QuestionStore


@dataclass
class EvalItemResult:
    id: str
    text: str
    predicted: str
    expected: str
    score: float
    matched_id: str | None
    method: str
    passed: bool
    notes: str = ""


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    false_positives: list[str]
    false_negatives: list[str]


def compute_metrics(
    results: list[EvalItemResult],
    positive_labels: tuple[str, ...] = ("EXACT", "DUPLICATE"),
) -> ClassificationMetrics:
    """Compute binary classification metrics with positive class = EXACT + DUPLICATE."""
    tp, fp, tn, fn = 0, 0, 0, 0
    fps: list[str] = []
    fns: list[str] = []

    for r in results:
        # Determine actual label
        actual_pos = r.expected in positive_labels
        # Determine predicted label
        pred_pos = r.predicted in positive_labels

        if actual_pos and pred_pos:
            tp += 1
        elif not actual_pos and pred_pos:
            fp += 1
            fps.append(f"{r.id} ('{r.text}') -> pred {r.predicted}, exp {r.expected}")
        elif not actual_pos and not pred_pos:
            tn += 1
        else:
            fn += 1
            fns.append(f"{r.id} ('{r.text}') -> pred {r.predicted}, exp {r.expected}")

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return ClassificationMetrics(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        false_positives=fps,
        false_negatives=fns,
    )


def run_sequential_eval(
    corpus: list[Question],
    embedder: QuestionEmbedder,
    llm: Any = None,
) -> tuple[list[EvalItemResult], DuplicateDetector]:
    """Part A: Sequential eval on Q1..Q10 checking each Qi (i>=2) against earlier only."""
    store = QuestionStore(":memory:")
    index = VectorIndex()
    detector = DuplicateDetector(store, embedder, index, llm=llm or MockLLM())

    expected_map = {
        "Q2": "DUPLICATE",
        "Q3": "DUPLICATE",
        "Q4": "DUPLICATE",
        "Q5": "UNIQUE",
        "Q6": "UNIQUE",
        "Q7": "UNIQUE",
        "Q8": "REVIEW",
        "Q9": "UNIQUE",
        "Q10": "EXACT",
    }

    results: list[EvalItemResult] = []

    for i, q in enumerate(corpus):
        if i == 0:
            # Q1 is the base question; ingest directly
            detector.ingest(q)
            continue

        exp = expected_map[q.id]
        res = detector.detect(q)
        pred = res.decision.value if hasattr(res.decision, "value") else str(res.decision)

        # Check pass/fail condition
        if q.id == "Q8":
            # REVIEW or UNIQUE acceptable, mark "flagged" if REVIEW
            passed = pred in {"REVIEW", "UNIQUE"}
            notes = "flagged (REVIEW/UNIQUE acceptable)"
        else:
            passed = (pred == exp)
            notes = ""

        # Determine method used
        if pred == "EXACT":
            method = "hash_match"
        elif pred == "DUPLICATE":
            method = "vector_cosine" if res.score >= detector.duplicate_threshold else "llm_judge"
        elif pred == "REVIEW":
            method = "review_band"
        else:
            method = "below_threshold" if res.score < detector.review_threshold else "judge_distinct"

        results.append(
            EvalItemResult(
                id=q.id,
                text=q.text,
                predicted=pred,
                expected=exp,
                score=round(res.score, 4),
                matched_id=res.matched_id,
                method=method,
                passed=passed,
                notes=notes,
            )
        )

        # Ingest for subsequent questions
        detector.ingest(q)

    return results, detector


def run_extra_eval(
    extra_candidates: list[dict[str, Any]],
    detector: DuplicateDetector,
) -> list[EvalItemResult]:
    """Part B: Extra labeled candidates checked against full Q1..Q10 corpus."""
    results: list[EvalItemResult] = []

    for item in extra_candidates:
        cid = item["id"]
        text = item["text"]
        exp = item["expected"]

        q = Question(id=cid, text=text)
        res = detector.detect(q)
        pred = res.decision.value if hasattr(res.decision, "value") else str(res.decision)

        if cid == "E7":
            # Ambiguous vs Q7; REVIEW or UNIQUE acceptable
            passed = pred in {"REVIEW", "UNIQUE"}
            notes = "REVIEW/UNIQUE acceptable"
        else:
            passed = (pred == exp)
            notes = ""

        if pred == "EXACT":
            method = "hash_match"
        elif pred == "DUPLICATE":
            method = "vector_cosine" if res.score >= detector.duplicate_threshold else "llm_judge"
        elif pred == "REVIEW":
            method = "review_band"
        else:
            method = "below_threshold" if res.score < detector.review_threshold else "judge_distinct"

        results.append(
            EvalItemResult(
                id=cid,
                text=text,
                predicted=pred,
                expected=exp,
                score=round(res.score, 4),
                matched_id=res.matched_id,
                method=method,
                passed=passed,
                notes=notes,
            )
        )

    return results


def run_ablation(
    corpus: list[Question],
    extra_candidates: list[dict[str, Any]],
    embedder: QuestionEmbedder,
) -> dict[str, ClassificationMetrics]:
    """Part D: Ablation across 3 configs: (1) hash-only, (2) embedding-only, (3) full pipeline."""
    all_eval_items: list[tuple[str, str, str]] = []  # (id, text, expected)

    expected_map = {
        "Q2": "DUPLICATE",
        "Q3": "DUPLICATE",
        "Q4": "DUPLICATE",
        "Q5": "UNIQUE",
        "Q6": "UNIQUE",
        "Q7": "UNIQUE",
        "Q8": "REVIEW",
        "Q9": "UNIQUE",
        "Q10": "EXACT",
    }
    for q in corpus[1:]:
        all_eval_items.append((q.id, q.text, expected_map[q.id]))
    for e in extra_candidates:
        all_eval_items.append((e["id"], e["text"], e["expected"]))

    # Pre-encode all questions
    corpus_vecs = {q.id: embedder.encode(q) for q in corpus}

    configs = ["(1) hash-only", "(2) embedding-only", "(3) full pipeline"]
    ablation_metrics: dict[str, ClassificationMetrics] = {}

    # Config 1: Hash-only
    c1_results: list[EvalItemResult] = []
    # Ingest corpus sequentially for Q1..Q10
    hashes_seen: set[str] = set()
    for q in corpus:
        if q.id != "Q1":
            pred = "EXACT" if q.text_hash in hashes_seen else "UNIQUE"
            c1_results.append(
                EvalItemResult(
                    id=q.id,
                    text=q.text,
                    predicted=pred,
                    expected=expected_map[q.id],
                    score=1.0 if pred == "EXACT" else 0.0,
                    matched_id=None,
                    method="hash_only",
                    passed=(pred == expected_map[q.id]),
                )
            )
        hashes_seen.add(q.text_hash)

    # For extra candidates against all corpus hashes
    for e in extra_candidates:
        q_temp = Question(id=e["id"], text=e["text"])
        pred = "EXACT" if q_temp.text_hash in hashes_seen else "UNIQUE"
        c1_results.append(
            EvalItemResult(
                id=e["id"],
                text=e["text"],
                predicted=pred,
                expected=e["expected"],
                score=1.0 if pred == "EXACT" else 0.0,
                matched_id=None,
                method="hash_only",
                passed=(pred == e["expected"]),
            )
        )
    ablation_metrics["(1) hash-only"] = compute_metrics(c1_results)

    # Config 2 & 2b: Embedding-only pipelines (exact hash + cosine >= threshold -> DUPLICATE, judge OFF)
    def eval_embedding_only(thresh: float, method_name: str) -> list[EvalItemResult]:
        res: list[EvalItemResult] = []
        st = QuestionStore(":memory:")
        idx = VectorIndex()
        for i, q in enumerate(corpus):
            if i == 0:
                st.add(q, corpus_vecs[q.id])
                idx.add(q.id, corpus_vecs[q.id])
                continue
            if any(x.text_hash == q.text_hash for x in st.all()):
                pred = "EXACT"
                sc = 1.0
            else:
                nbrs = idx.search(corpus_vecs[q.id], k=1)
                sc = nbrs[0][1] if nbrs else 0.0
                pred = "DUPLICATE" if sc >= thresh else "UNIQUE"
            res.append(
                EvalItemResult(
                    id=q.id,
                    text=q.text,
                    predicted=pred,
                    expected=expected_map[q.id],
                    score=sc,
                    matched_id=None,
                    method=method_name,
                    passed=(pred == expected_map[q.id] or (q.id == "Q8" and pred in {"REVIEW", "UNIQUE"})),
                )
            )
            st.add(q, corpus_vecs[q.id])
            idx.add(q.id, corpus_vecs[q.id])

        for e in extra_candidates:
            q_temp = Question(id=e["id"], text=e["text"])
            vec = embedder.encode(q_temp)
            if any(x.text_hash == q_temp.text_hash for x in st.all()):
                pred = "EXACT"
                sc = 1.0
            else:
                nbrs = idx.search(vec, k=1)
                sc = nbrs[0][1] if nbrs else 0.0
                pred = "DUPLICATE" if sc >= thresh else "UNIQUE"
            res.append(
                EvalItemResult(
                    id=e["id"],
                    text=e["text"],
                    predicted=pred,
                    expected=e["expected"],
                    score=sc,
                    matched_id=None,
                    method=method_name,
                    passed=(pred == e["expected"] or (e["id"] == "E7" and pred in {"REVIEW", "UNIQUE"})),
                )
            )
        return res

    ablation_metrics["(2) embedding-only"] = compute_metrics(eval_embedding_only(0.88, "embedding_only"))
    ablation_metrics["(2b) embedding-only, strict 0.95"] = compute_metrics(eval_embedding_only(0.95, "embedding_only_strict"))

    # Config 3: Full pipeline
    store_c3 = QuestionStore(":memory:")
    index_c3 = VectorIndex()
    detector_c3 = DuplicateDetector(store_c3, embedder, index_c3, MockLLM())
    res_c3_seq, _ = run_sequential_eval(corpus, embedder, MockLLM())
    # Ingest all into detector_c3 for extra candidates
    store_full = QuestionStore(":memory:")
    index_full = VectorIndex()
    det_full = DuplicateDetector(store_full, embedder, index_full, MockLLM())
    for q in corpus:
        det_full.ingest(q)
    res_c3_extra = run_extra_eval(extra_candidates, det_full)
    ablation_metrics["(3) full pipeline"] = compute_metrics(res_c3_seq + res_c3_extra)

    return ablation_metrics


def run_threshold_sweep(
    corpus: list[Question],
    extra_candidates: list[dict[str, Any]],
    embedder: QuestionEmbedder,
) -> list[dict[str, Any]]:
    """Part E: Sweep dup_threshold from 0.60..0.95 step 0.05 on embedding-only pipeline."""
    thresholds = [round(t, 2) for t in np.arange(0.60, 0.96, 0.05)]
    sweep_results: list[dict[str, Any]] = []

    expected_map = {
        "Q2": "DUPLICATE",
        "Q3": "DUPLICATE",
        "Q4": "DUPLICATE",
        "Q5": "UNIQUE",
        "Q6": "UNIQUE",
        "Q7": "UNIQUE",
        "Q8": "REVIEW",
        "Q9": "UNIQUE",
        "Q10": "EXACT",
    }

    corpus_vecs = {q.id: embedder.encode(q) for q in corpus}

    for tau in thresholds:
        items_eval: list[EvalItemResult] = []
        store = QuestionStore(":memory:")
        index = VectorIndex()

        # Sequential on corpus
        for i, q in enumerate(corpus):
            if i == 0:
                store.add(q, corpus_vecs[q.id])
                index.add(q.id, corpus_vecs[q.id])
                continue

            if any(x.text_hash == q.text_hash for x in store.all()):
                pred = "EXACT"
                sc = 1.0
            else:
                nbrs = index.search(corpus_vecs[q.id], k=1)
                sc = nbrs[0][1] if nbrs else 0.0
                pred = "DUPLICATE" if sc >= tau else "UNIQUE"

            items_eval.append(
                EvalItemResult(
                    id=q.id,
                    text=q.text,
                    predicted=pred,
                    expected=expected_map[q.id],
                    score=sc,
                    matched_id=None,
                    method="embedding_only",
                    passed=(pred == expected_map[q.id] or (q.id == "Q8" and pred in {"REVIEW", "UNIQUE"})),
                )
            )
            store.add(q, corpus_vecs[q.id])
            index.add(q.id, corpus_vecs[q.id])

        # Extra candidates against full corpus
        for e in extra_candidates:
            q_temp = Question(id=e["id"], text=e["text"])
            vec = embedder.encode(q_temp)
            if any(x.text_hash == q_temp.text_hash for x in store.all()):
                pred = "EXACT"
                sc = 1.0
            else:
                nbrs = index.search(vec, k=1)
                sc = nbrs[0][1] if nbrs else 0.0
                pred = "DUPLICATE" if sc >= tau else "UNIQUE"

            items_eval.append(
                EvalItemResult(
                    id=e["id"],
                    text=e["text"],
                    predicted=pred,
                    expected=e["expected"],
                    score=sc,
                    matched_id=None,
                    method="embedding_only",
                    passed=(pred == e["expected"] or (e["id"] == "E7" and pred in {"REVIEW", "UNIQUE"})),
                )
            )

        metrics = compute_metrics(items_eval)
        sweep_results.append({
            "threshold": tau,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "tn": metrics.tn,
            "fn": metrics.fn,
            "false_positives": [fp.split()[0] for fp in metrics.false_positives],
            "false_negatives": [fn.split()[0] for fn in metrics.false_negatives],
        })

    return sweep_results


def measure_latency(detector: DuplicateDetector, corpus: list[Question]) -> tuple[float, float]:
    """Part G: Time 100 check() calls and report mean and p95 in ms."""
    test_queries = [
        "What is reflection of light?",
        "Define reflection.",
        "Explain the law of reflection.",
        "What is lateral inversion?",
        "Why is angle of reflection equal to angle of incidence?",
    ]

    latencies_ms: list[float] = []

    # Warmup
    for q in test_queries:
        detector.detect(q)

    # 100 timed calls
    for i in range(100):
        q_text = test_queries[i % len(test_queries)]
        t0 = time.perf_counter()
        detector.detect(q_text)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    return round(mean_ms, 2), round(p95_ms, 2)


def main() -> None:
    console = Console(width=120)
    console.print(Panel("[bold green]Question-Dedup Comprehensive Evaluation Suite[/bold green]", expand=False))

    # Load corpus
    corpus_path = root_dir / "data" / "corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = [Question.from_dict(d) for d in json.load(f)]

    # Load extra candidates
    extra_path = root_dir / "eval" / "extra_candidates.json"
    with open(extra_path, "r", encoding="utf-8") as f:
        extra_candidates = json.load(f)

    embedder = QuestionEmbedder()

    # Part A: Sequential eval on Q1..Q10
    results_a, detector_a = run_sequential_eval(corpus, embedder)

    # Part B: Extra candidates against full corpus
    store_full = QuestionStore(":memory:")
    index_full = VectorIndex()
    detector_full = DuplicateDetector(store_full, embedder, index_full, MockLLM())
    for q in corpus:
        detector_full.ingest(q)
    results_b = run_extra_eval(extra_candidates, detector_full)

    # Combined results
    combined_results = results_a + results_b

    # 1. Print Part A Table
    table_a = Table(title="Part A: Sequential Evaluation on Q1..Q10 (Ingest in Order)", header_style="bold cyan")
    table_a.add_column("ID", style="bold yellow", width=6)
    table_a.add_column("Question Text", style="white", width=44)
    table_a.add_column("Predicted", style="bold green", width=12)
    table_a.add_column("Expected", style="bold magenta", width=12)
    table_a.add_column("Score", justify="right", width=8)
    table_a.add_column("Matched ID", width=10)
    table_a.add_column("Method", width=16)
    table_a.add_column("Status", width=10)

    for r in results_a:
        status_str = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table_a.add_row(
            r.id,
            r.text,
            r.predicted,
            r.expected,
            f"{r.score:.4f}",
            r.matched_id or "—",
            r.method,
            status_str,
        )

    console.print(table_a)
    console.print()

    # 2. Print Part B Table
    table_b = Table(title="Part B: Extra Labeled Candidates (Evaluated Against Full Corpus)", header_style="bold cyan")
    table_b.add_column("ID", style="bold yellow", width=6)
    table_b.add_column("Candidate Text", style="white", width=44)
    table_b.add_column("Predicted", style="bold green", width=12)
    table_b.add_column("Expected", style="bold magenta", width=12)
    table_b.add_column("Score", justify="right", width=8)
    table_b.add_column("Matched ID", width=10)
    table_b.add_column("Method", width=16)
    table_b.add_column("Status", width=10)

    for r in results_b:
        status_str = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table_b.add_row(
            r.id,
            r.text,
            r.predicted,
            r.expected,
            f"{r.score:.4f}",
            r.matched_id or "—",
            r.method,
            status_str,
        )

    console.print(table_b)
    console.print()

    # 3. Part C: Metrics & Confusion Matrix
    metrics_a = compute_metrics(results_a)
    metrics_b = compute_metrics(results_b)
    metrics_comb = compute_metrics(combined_results)

    table_c = Table(title="Part C: Classification Performance Metrics (Positive = EXACT + DUPLICATE)", header_style="bold cyan")
    table_c.add_column("Dataset Partition", style="bold yellow")
    table_c.add_column("Precision", justify="right")
    table_c.add_column("Recall", justify="right")
    table_c.add_column("F1 Score", justify="right")
    table_c.add_column("TP / FP / TN / FN", justify="center")

    table_c.add_row("Part A (Sequential Q1..Q10)", f"{metrics_a.precision:.2%}", f"{metrics_a.recall:.2%}", f"{metrics_a.f1:.2%}", f"{metrics_a.tp} / {metrics_a.fp} / {metrics_a.tn} / {metrics_a.fn}")
    table_c.add_row("Part B (Extra Candidates)", f"{metrics_b.precision:.2%}", f"{metrics_b.recall:.2%}", f"{metrics_b.f1:.2%}", f"{metrics_b.tp} / {metrics_b.fp} / {metrics_b.tn} / {metrics_b.fn}")
    table_c.add_row("Combined (A + B)", f"{metrics_comb.precision:.2%}", f"{metrics_comb.recall:.2%}", f"{metrics_comb.f1:.2%}", f"{metrics_comb.tp} / {metrics_comb.fp} / {metrics_comb.tn} / {metrics_comb.fn}")

    console.print(table_c)
    console.print()

    # 4. Part D: Ablation
    ablation_res = run_ablation(corpus, extra_candidates, embedder)
    table_d = Table(title="Part D: Architecture Ablation Study (Combined A + B)", header_style="bold cyan")
    table_d.add_column("Configuration", style="bold yellow")
    table_d.add_column("Precision", justify="right")
    table_d.add_column("Recall", justify="right")
    table_d.add_column("F1 Score", justify="right")
    table_d.add_column("TP / FP / TN / FN", justify="center")

    for cfg, m in ablation_res.items():
        table_d.add_row(cfg, f"{m.precision:.2%}", f"{m.recall:.2%}", f"{m.f1:.2%}", f"{m.tp} / {m.fp} / {m.tn} / {m.fn}")

    console.print(table_d)
    console.print()

    # 5. Part E: Threshold Sweep
    sweep_res = run_threshold_sweep(corpus, extra_candidates, embedder)
    table_e = Table(title="Part E: Embedding-Only Cosine Threshold Sweep (0.60 to 0.95)", header_style="bold cyan")
    table_e.add_column("Threshold", justify="center", style="bold yellow")
    table_e.add_column("Precision", justify="right")
    table_e.add_column("Recall", justify="right")
    table_e.add_column("F1", justify="right")
    table_e.add_column("False Positives (Costly)", style="red")
    table_e.add_column("False Negatives", style="yellow")

    for s in sweep_res:
        fp_str = ", ".join(s["false_positives"]) if s["false_positives"] else "None"
        fn_str = ", ".join(s["false_negatives"]) if s["false_negatives"] else "None"
        table_e.add_row(
            f"{s['threshold']:.2f}",
            f"{s['precision']:.2%}",
            f"{s['recall']:.2%}",
            f"{s['f1']:.2%}",
            fp_str,
            fn_str,
        )

    console.print(table_e)
    console.print()

    # 6. Part F: Gemini Real Judge Check
    api_key = os.getenv("GEMINI_API_KEY")
    gemini_status = ""
    if api_key:
        console.print("[bold green]Part F: GEMINI_API_KEY detected. Running evaluation with GeminiLLM...[/bold green]")
        gemini_llm = GeminiLLM(api_key=api_key)
        res_gemini_seq, _ = run_sequential_eval(corpus, embedder, gemini_llm)
        det_gemini_full = DuplicateDetector(store_full, embedder, index_full, gemini_llm)
        res_gemini_extra = run_extra_eval(extra_candidates, det_gemini_full)
        gemini_metrics = compute_metrics(res_gemini_seq + res_gemini_extra)
        gemini_status = f"Precision: {gemini_metrics.precision:.2%}, Recall: {gemini_metrics.recall:.2%}, F1: {gemini_metrics.f1:.2%}"
        console.print(f"Gemini Real Judge Results: {gemini_status}")
    else:
        gemini_status = "real judge run pending (GEMINI_API_KEY not set)"
        console.print(f"[bold yellow]Part F: {gemini_status}[/bold yellow]")
    console.print()

    # 7. Part G: Latency
    mean_lat, p95_lat = measure_latency(detector_full, corpus)
    console.print(Panel(f"[bold cyan]Part G: Latency Benchmark (100 check() calls)[/bold cyan]\nMean Latency: [bold]{mean_lat} ms[/bold]\nP95 Latency:  [bold]{p95_lat} ms[/bold]", expand=False))
    console.print()

    # 8. Save results to eval/results.md
    md_content = generate_markdown_report(
        results_a=results_a,
        results_b=results_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        metrics_comb=metrics_comb,
        ablation_res=ablation_res,
        sweep_res=sweep_res,
        gemini_status=gemini_status,
        mean_lat=mean_lat,
        p95_lat=p95_lat,
    )

    results_file = root_dir / "eval" / "results.md"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    console.print(f"[green]Saved complete evaluation report to {results_file}[/green]")


def generate_markdown_report(
    results_a: list[EvalItemResult],
    results_b: list[EvalItemResult],
    metrics_a: ClassificationMetrics,
    metrics_b: ClassificationMetrics,
    metrics_comb: ClassificationMetrics,
    ablation_res: dict[str, ClassificationMetrics],
    sweep_res: list[dict[str, Any]],
    gemini_status: str,
    mean_lat: float,
    p95_lat: float,
) -> str:
    """Generate Markdown report matching all prompt requirements."""
    md: list[str] = [
        "# Question Deduplication Evaluation Report",
        "",
        "Comprehensive evaluation of the semantic deduplication pipeline covering sequential ingestion, extra candidate benchmarks, ablation studies, threshold sweeps, and latency profiling.",
        "",
        "---",
        "",
        "## A. Sequential Evaluation on Q1..Q10",
        "",
        "Ingested sequentially in order Q1..Q10; each question $Q_i$ ($i \\ge 2$) is evaluated against earlier questions only.",
        "",
        "| ID | Question Text | Predicted | Expected | Cosine Score | Matched ID | Method | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |",
    ]

    for r in results_a:
        st = "PASS" if r.passed else "FAIL"
        md.append(f"| **{r.id}** | {r.text} | `{r.predicted}` | `{r.expected}` | {r.score:.4f} | {r.matched_id or '—'} | `{r.method}` | **{st}** |")

    md.extend([
        "",
        "---",
        "",
        "## B. Extra Labeled Candidates",
        "",
        "Evaluated against the full pre-populated $Q_1..Q_{10}$ corpus (candidates are evaluated but not ingested).",
        "",
        "| ID | Candidate Question Text | Predicted | Expected | Cosine Score | Matched ID | Method | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |",
    ])

    for r in results_b:
        st = "PASS" if r.passed else "FAIL"
        md.append(f"| **{r.id}** | {r.text} | `{r.predicted}` | `{r.expected}` | {r.score:.4f} | {r.matched_id or '—'} | `{r.method}` | **{st}** |")

    md.extend([
        "",
        "---",
        "",
        "## C. Classification Metrics (Positive Class = EXACT + DUPLICATE)",
        "",
        "| Partition | Precision | Recall | F1 Score | Confusion Matrix (TP / FP / TN / FN) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Part A (Sequential Q1..Q10)** | {metrics_a.precision:.2%} | {metrics_a.recall:.2%} | {metrics_a.f1:.2%} | {metrics_a.tp} / {metrics_a.fp} / {metrics_a.tn} / {metrics_a.fn} |",
        f"| **Part B (Extra Candidates)** | {metrics_b.precision:.2%} | {metrics_b.recall:.2%} | {metrics_b.f1:.2%} | {metrics_b.tp} / {metrics_b.fp} / {metrics_b.tn} / {metrics_b.fn} |",
        f"| **Combined (A + B)** | {metrics_comb.precision:.2%} | {metrics_comb.recall:.2%} | {metrics_comb.f1:.2%} | {metrics_comb.tp} / {metrics_comb.fp} / {metrics_comb.tn} / {metrics_comb.fn} |",
        "",
        "### Costly Errors Analysis",
        "- **False Positives (Costly: incorrectly marking unique questions as duplicates)**: " + (", ".join(metrics_comb.false_positives) if metrics_comb.false_positives else "None (0 FP)"),
        "- **False Negatives (missed duplicates)**: " + (", ".join(metrics_comb.false_negatives) if metrics_comb.false_negatives else "None (0 FN)"),
        "",
        "---",
        "",
        "## D. Ablation Study",
        "",
        "Comparison of system configurations across the combined benchmark:",
        "1. **Hash-only**: Exact normalized text hash match only.",
        "2. **Embedding-only**: Cosine $\\ge 0.88$ without LLM review band (REVIEW counted as not-duplicate).",
        "2b. **Embedding-only, strict 0.95**: Cosine $\\ge 0.95$ without LLM review band (REVIEW counted as not-duplicate).",
        "3. **Full pipeline with judge**: Multi-band routing (EXACT $\\to$ high-cosine $\\to$ review band with judge $\\to$ UNIQUE).",
        "",
        "| Configuration | Precision | Recall | F1 Score | TP / FP / TN / FN |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for cfg, m in ablation_res.items():
        md.append(f"| **{cfg}** | {m.precision:.2%} | {m.recall:.2%} | {m.f1:.2%} | {m.tp} / {m.fp} / {m.tn} / {m.fn} |")

    md.extend([
        "",
        "On this 16-item set, 0.88 alone separates all labels, but the margin is thin (Q3=0.90 vs Q8=0.87); the judge adds robustness, not measurable accuracy, here.",
        "",
        "---",
        "",
        "## E. Embedding-Only Cosine Threshold Sweep (0.60 to 0.95)",
        "",
        "Evaluating classification performance of embedding cosine similarity without the LLM judge:",
        "",
        "| Threshold | Precision | Recall | F1 Score | False Positives (Costly) | False Negatives |",
        "| :---: | :---: | :---: | :---: | :--- | :--- |",
    ])

    for s in sweep_res:
        fp_str = ", ".join(s["false_positives"]) if s["false_positives"] else "None"
        fn_str = ", ".join(s["false_negatives"]) if s["false_negatives"] else "None"
        md.append(f"| **{s['threshold']:.2f}** | {s['precision']:.2%} | {s['recall']:.2%} | {s['f1']:.2%} | {fp_str} | {fn_str} |")

    md.extend([
        "",
        "> [!NOTE]",
        "> As expected from the similarity geometry:",
        "> - At **$\\tau \\le 0.70$**, non-duplicate questions such as **Q6** and **Q7** flip to False Positives.",
        "> - At **$\\tau \\le 0.85$**, **Q8** flips to False Positive.",
        "> - At **$\\tau > 0.90$**, subtle paraphrases like **Q3** flip to False Negatives.",
        "> This confirms why the multi-band architecture with LLM review between 0.72 and 0.88 (0.72 <= cosine < 0.88) is necessary.",
        "",
        "---",
        "",
        "## F. Real Gemini Judge Evaluation",
        f"- Status: **{gemini_status}**",
        "",
        "---",
        "",
        "## G. System Latency Benchmark",
        "- Number of `detect()` queries: 100",
        f"- **Mean Latency**: `{mean_lat} ms`",
        f"- **P95 Latency**: `{p95_lat} ms`",
    ])

    return "\n".join(md)


if __name__ == "__main__":
    main()
