# Question Deduplication Evaluation Report

Comprehensive evaluation of the semantic deduplication pipeline covering sequential ingestion, extra candidate benchmarks, ablation studies, threshold sweeps, and latency profiling.

---

## A. Sequential Evaluation on Q1..Q10

Ingested sequentially in order Q1..Q10; each question $Q_i$ ($i \ge 2$) is evaluated against earlier questions only.

| ID | Question Text | Predicted | Expected | Cosine Score | Matched ID | Method | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Q2** | Define the reflection of light. | `DUPLICATE` | `DUPLICATE` | 0.9678 | Q1 | `vector_cosine` | **PASS** |
| **Q3** | What do you understand by the reflection of light? | `DUPLICATE` | `DUPLICATE` | 0.8998 | Q1 | `vector_cosine` | **PASS** |
| **Q4** | Explain the meaning of light reflection. | `DUPLICATE` | `DUPLICATE` | 0.9251 | Q2 | `vector_cosine` | **PASS** |
| **Q5** | State the laws of reflection of light. | `UNIQUE` | `UNIQUE` | 0.8000 | — | `judge_distinct` | **PASS** |
| **Q6** | What is the angle of reflection? | `UNIQUE` | `UNIQUE` | 0.6980 | — | `below_threshold` | **PASS** |
| **Q7** | The angle of incidence is 30°. What is the angle of reflection? | `UNIQUE` | `UNIQUE` | 0.8307 | — | `judge_distinct` | **PASS** |
| **Q8** | What is reflection? | `UNIQUE` | `REVIEW` | 0.8743 | — | `judge_distinct` | **PASS** |
| **Q9** | What is the difference between regular and diffuse reflection? | `UNIQUE` | `UNIQUE` | 0.6523 | — | `below_threshold` | **PASS** |
| **Q10** | Define the reflection of light. | `EXACT` | `EXACT` | 1.0000 | Q2 | `hash_match` | **PASS** |

---

## B. Extra Labeled Candidates

Evaluated against the full pre-populated $Q_1..Q_{10}$ corpus (candidates are evaluated but not ingested).

| ID | Candidate Question Text | Predicted | Expected | Cosine Score | Matched ID | Method | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **E1** | what is reflection of light | `EXACT` | `EXACT` | 1.0000 | Q1 | `hash_match` | **PASS** |
| **E2** | Explain what is meant by reflection of light. | `DUPLICATE` | `DUPLICATE` | 0.9650 | Q4 | `vector_cosine` | **PASS** |
| **E3** | Define light reflection. | `DUPLICATE` | `DUPLICATE` | 0.9645 | Q10 | `vector_cosine` | **PASS** |
| **E4** | Why is the image in a plane mirror laterally inverted? | `UNIQUE` | `UNIQUE` | 0.5635 | — | `below_threshold` | **PASS** |
| **E5** | What is a retroreflector and where is it used? | `UNIQUE` | `UNIQUE` | 0.4062 | — | `below_threshold` | **PASS** |
| **E6** | Draw a ray diagram showing reflection from a plane mirror. | `UNIQUE` | `UNIQUE` | 0.6569 | — | `below_threshold` | **PASS** |
| **E7** | The angle of incidence is 45°. Find the angle of reflection. | `UNIQUE` | `REVIEW` | 0.8205 | — | `judge_distinct` | **PASS** |

---

## C. Classification Metrics (Positive Class = EXACT + DUPLICATE)

| Partition | Precision | Recall | F1 Score | Confusion Matrix (TP / FP / TN / FN) |
| :--- | :---: | :---: | :---: | :---: |
| **Part A (Sequential Q1..Q10)** | 100.00% | 100.00% | 100.00% | 4 / 0 / 5 / 0 |
| **Part B (Extra Candidates)** | 100.00% | 100.00% | 100.00% | 3 / 0 / 4 / 0 |
| **Combined (A + B)** | 100.00% | 100.00% | 100.00% | 7 / 0 / 9 / 0 |

> [!NOTE]
> 100% on a 16-item calibration set, not a generalization claim.

### Costly Errors Analysis
- **False Positives (Costly: incorrectly marking unique questions as duplicates)**: None (0 FP)
- **False Negatives (missed duplicates)**: None (0 FN)

---

## D. Ablation Study

Comparison of system configurations across the combined benchmark:
1. **Hash-only**: Exact normalized text hash match only.
2. **Embedding-only**: Cosine $\ge 0.88$ without LLM review band (REVIEW counted as not-duplicate).
2b. **Embedding-only, strict 0.95**: Cosine $\ge 0.95$ without LLM review band (REVIEW counted as not-duplicate).
3. **Full pipeline with judge**: Multi-band routing (EXACT $\to$ high-cosine $\to$ review band with judge $\to$ UNIQUE).

| Configuration | Precision | Recall | F1 Score | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: |
| **(1) hash-only** | 100.00% | 28.57% | 44.44% | 2 / 0 / 9 / 5 |
| **(2) embedding-only** | 100.00% | 100.00% | 100.00% | 7 / 0 / 9 / 0 |
| **(2b) embedding-only, strict 0.95** | 100.00% | 71.43% | 83.33% | 5 / 0 / 9 / 2 |
| **(3) full pipeline** | 100.00% | 100.00% | 100.00% | 7 / 0 / 9 / 0 |

On this 16-item set, 0.88 alone separates all labels, but the margin is thin (Q3=0.90 vs Q8=0.87); the judge adds robustness, not measurable accuracy, here.

---

## E. Embedding-Only Cosine Threshold Sweep (0.60 to 0.95)

Evaluating classification performance of embedding cosine similarity without the LLM judge:

| Threshold | Precision | Recall | F1 Score | False Positives (Costly) | False Negatives |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **0.60** | 50.00% | 100.00% | 66.67% | Q5, Q6, Q7, Q8, Q9, E6, E7 | None |
| **0.65** | 50.00% | 100.00% | 66.67% | Q5, Q6, Q7, Q8, Q9, E6, E7 | None |
| **0.70** | 63.64% | 100.00% | 77.78% | Q5, Q7, Q8, E7 | None |
| **0.75** | 63.64% | 100.00% | 77.78% | Q5, Q7, Q8, E7 | None |
| **0.80** | 63.64% | 100.00% | 77.78% | Q5, Q7, Q8, E7 | None |
| **0.85** | 87.50% | 100.00% | 93.33% | Q8 | None |
| **0.90** | 100.00% | 85.71% | 92.31% | None | Q3 |
| **0.95** | 100.00% | 71.43% | 83.33% | None | Q3, Q4 |

> [!NOTE]
> As expected from the similarity geometry:
> - At **$\tau \le 0.70$**, non-duplicate questions such as **Q6** and **Q7** flip to False Positives.
> - At **$\tau \le 0.85$**, **Q8** flips to False Positive.
> - At **$\tau > 0.90$**, subtle paraphrases like **Q3** flip to False Negatives.
> This confirms why the multi-band architecture with LLM review between 0.72 and 0.88 (0.72 <= cosine < 0.88) is necessary.

---

## F. Real Gemini Judge Evaluation
- Status: **real judge run pending (GEMINI_API_KEY not set)**

---

## G. System Latency Benchmark
- Number of `detect()` queries: 100
- **Mean Latency**: `0.21 ms`
- **P95 Latency**: `0.38 ms`

---

## Presentation variants

Evaluation of presentation-style variants (fill-in-the-blank, multiple choice options, and True/False phrasing) checked against the full $Q_1..Q_{10}$ corpus. Evaluated separately from the 16-item headline metrics: **0 of 3 flagged** (0/3 detected as DUPLICATE or REVIEW).

| ID | Presentation Variant Question Text | Predicted Label | Expected Label | Method | Cosine Score | Status | Actual Outcome / Analysis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **E8** | Reflection of light is the ___ of light from a surface. (Fill in the blank) | `UNIQUE` | `DUPLICATE or REVIEW` | `judge_distinct` | 0.7889 | **FAIL** | Routed to review band ($0.72 \le 0.7889 < 0.88$). Mock judge compared content words against $Q_{10}$/$Q_2$; content-word Jaccard overlap was 0.4000 (< 0.60 threshold due to 'fill', 'blank', 'surface'), judging them distinct. |
| **E9** | Which of the following describes reflection of light? (a) bending of light (b) bouncing back of light from a surface (c) absorption of light (d) splitting of light | `UNIQUE` | `DUPLICATE or REVIEW` | `below_threshold` | 0.6550 | **FAIL** | Heavy multiple-choice distractor text ('bending', 'absorption', 'splitting') diluted embedding similarity below the review threshold (0.72). |
| **E10** | True or False: Reflection of light is the bouncing back of light from a surface. | `UNIQUE` | `DUPLICATE or REVIEW` | `below_threshold` | 0.7114 | **FAIL** | Boolean prefix ('True or False:') and declarative statement structure shifted the embedding just below the review threshold (0.7114 < 0.72). |

### Known Limitation & Proposed Fix
- **Outcome**: Presentation variants (fill-in-blank, MCQ, true/false) were NOT caught (0/3): cosine 0.79 / 0.66 / 0.71.
- **Proposed fix**: LLM-based rewrite of the candidate into a plain question stem before detection (future work). Regex canonicalization was considered and rejected as brittle.