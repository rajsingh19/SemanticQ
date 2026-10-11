#!/usr/bin/env python3
"""app.py: Streamlit frontend for question deduplication, selection, and generation."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Disable SQLite check_same_thread so connections can be safely accessed
# across Streamlit worker/rerun threads.
_orig_sqlite_connect = sqlite3.connect


def _threadsafe_sqlite_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
    kwargs["check_same_thread"] = False
    return _orig_sqlite_connect(*args, **kwargs)


sqlite3.connect = _threadsafe_sqlite_connect

import pandas as pd
import streamlit as st

# Ensure src is on Python search path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

st.set_page_config(
    page_title="SemanticQ — Question Deduplication Engine",
    page_icon="🔍",
    layout="wide",
)

try:
    from qdedup.context_selector import ContextSelector
    from qdedup.detector import Decision, DetectorConfig, DuplicateDetector
    from qdedup.embedder import QuestionEmbedder, VectorIndex
    from qdedup.generator import QuestionGenerator
    from qdedup.llm import JudgeResult, MockLLM, get_llm_client
    from qdedup.models import Question
    from qdedup.store import QuestionStore
except ModuleNotFoundError as e:
    st.error(
        f"**Missing dependency: `{e.name}`**\n\n"
        "Streamlit is currently running in an external Python environment instead of the project virtualenv (`.venv`).\n\n"
        "Please start Streamlit using the virtual environment:\n"
        "```bash\n"
        "./.venv/bin/streamlit run app.py\n"
        "# or\n"
        "source .venv/bin/activate && streamlit run app.py\n"
        "```"
    )
    st.stop()


@st.cache_resource
def get_shared_embedder() -> QuestionEmbedder:
    """Load and cache the sentence-transformers embedding model once."""
    return QuestionEmbedder()


@st.cache_resource
def load_base_corpus() -> list[dict[str, Any]]:
    """Load base Q1..Q10 corpus from data/corpus.json."""
    corpus_file = Path(__file__).resolve().parent / "data" / "corpus.json"
    with open(corpus_file, "r", encoding="utf-8") as f:
        return json.load(f)


def is_store_thread_safe() -> bool:
    if "store" not in st.session_state:
        return False
    try:
        st.session_state.store._conn.execute("SELECT 1")
        return True
    except (sqlite3.ProgrammingError, sqlite3.OperationalError):
        return False


def init_session_state(force_reset: bool = False) -> None:
    """Initialize or reset working store, index, detector, and generator."""
    if force_reset or not is_store_thread_safe():
        corpus_data = load_base_corpus()
        embedder = get_shared_embedder()
        store = QuestionStore(":memory:")
        index = VectorIndex()
        base_llm = get_llm_client()

        detector = DuplicateDetector(store, embedder, index, base_llm)
        for item in corpus_data:
            q = Question.from_dict(item)
            detector.ingest(q)

        selector = ContextSelector(
            store, embedder, index, max_questions=5, max_tokens=500, n_clusters=4
        )
        generator = QuestionGenerator(base_llm, selector, detector)

        st.session_state.store = store
        st.session_state.index = index
        st.session_state.detector = detector
        st.session_state.selector = selector
        st.session_state.generator = generator
        st.session_state.base_llm = base_llm
        st.session_state.last_report = None


init_session_state()

store: QuestionStore = st.session_state.store
index: VectorIndex = st.session_state.index
detector: DuplicateDetector = st.session_state.detector
selector: ContextSelector = st.session_state.selector
generator: QuestionGenerator = st.session_state.generator
base_llm = st.session_state.base_llm

# Sidebar Configuration
st.sidebar.title("⚙️ Engine Controls")

llm_name = "Gemini" if os.getenv("GEMINI_API_KEY") else "MockLLM"
if llm_name == "Gemini":
    st.sidebar.success("Running with Gemini")
else:
    st.sidebar.info("Running with MockLLM")

st.sidebar.markdown("---")
st.sidebar.subheader("Threshold Controls")
dup_threshold = st.sidebar.slider(
    "Duplicate Threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.88,
    step=0.01,
    help="Cosine similarity >= duplicate threshold immediately flags DUPLICATE.",
)
review_threshold = st.sidebar.slider(
    "Review Threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.72,
    step=0.01,
    help="Cosine similarity between review and duplicate threshold triggers LLM judge review.",
)
use_judge = st.sidebar.checkbox(
    "Use LLM judge",
    value=True,
    help="Evaluate candidates in the review band with the LLM judge.",
)

# Build DetectorConfig using exact field names and assign ONLY via detector.config = config
config = DetectorConfig(
    dup_threshold=dup_threshold,
    review_threshold=review_threshold,
    judge_min_confidence=0.7,
    use_llm_judge=use_judge,
)
detector.config = config

# Display active thresholds caption in sidebar
st.sidebar.caption(
    f"Active thresholds: `dup ≥ {detector.config.dup_threshold:.2f}`, "
    f"`review ≥ {detector.config.review_threshold:.2f}`, "
    f"judge: {'enabled' if detector.config.use_llm_judge else 'disabled'}"
)


# Title and Global Header
st.title("SemanticQ: Question Deduplication & Generation")
st.markdown("##### *Detection is independent of generation.*")
st.markdown(
    "High-precision multi-band semantic deduplication, KMeans + MMR diversity selection, "
    "and intra-batch duplicate prevention."
)
st.write("")

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["🔍 Tab 1: Check a Question", "✨ Tab 2: Generate Questions", "📚 Tab 3: Corpus & Tags"]
)


# Helper function to get badge styling
def render_decision_badge(decision: str) -> str:
    color_map = {
        "EXACT": ("#ef4444", "#fee2e2", "#991b1b"),
        "DUPLICATE": ("#ef4444", "#fee2e2", "#991b1b"),
        "REVIEW": ("#f59e0b", "#fef3c7", "#92400e"),
        "UNIQUE": ("#10b981", "#d1fae5", "#065f46"),
    }
    border, bg, text = color_map.get(decision, ("#6b7280", "#f3f4f6", "#1f2937"))
    return (
        f'<span style="background-color: {bg}; color: {text}; border: 1px solid {border}; '
        f'padding: 6px 16px; border-radius: 9999px; font-weight: 700; font-size: 1.05rem; '
        f'letter-spacing: 0.05em; display: inline-block;">{decision}</span>'
    )


def determine_method(res: Any, det: DuplicateDetector, judge_enabled: bool) -> str:
    if res.decision == Decision.EXACT:
        return "hash"
    elif res.decision == Decision.DUPLICATE:
        return "embedding" if res.score >= det.duplicate_threshold else "llm_judge"
    elif res.decision == Decision.UNIQUE:
        if res.evaluated_candidates and judge_enabled:
            return "llm_judge"
        return "embedding" if res.score > 0 else "none"
    elif res.decision == Decision.REVIEW:
        return "llm_judge" if (res.evaluated_candidates and judge_enabled) else "embedding"
    return "none"


# -------------------------------------------------------------
# TAB 1: Check a Question
# -------------------------------------------------------------
with tab1:
    st.subheader("Duplicate Detection Check")
    st.write("Inspect any query against the corpus without modifying or ingesting it.")

    st.markdown("**Example queries (click to populate):**")
    ex_cols = st.columns(4)
    examples = [
        "DEFINE THE REFLECTION OF LIGHT!!!",
        "Tell me what light reflection means.",
        "Why is the sky blue?",
        "The angle of incidence is 60°. What is the angle of reflection.",
    ]
    for col, ex in zip(ex_cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["check_query_text"] = ex
            st.rerun()

    query_val = st.session_state.get(
        "check_query_text", "Explain what is meant by reflection of light."
    )
    user_query = st.text_input("Enter Question Text:", value=query_val, key="check_input")

    if st.button("Check Question", type="primary", use_container_width=False):
        if not user_query.strip():
            st.warning("Please enter a question to check.")
        else:
            # Call detector.check() WITHOUT ingesting
            res = detector.check(user_query.strip())
            method = determine_method(res, detector, use_judge)

            st.write("")
            col_b1, col_b2, col_b3, col_b4 = st.columns([1.5, 1.2, 1.2, 3])
            with col_b1:
                st.markdown("**Decision Badge:**")
                st.markdown(
                    render_decision_badge(res.decision.value), unsafe_allow_html=True
                )
            with col_b2:
                st.metric("Method", method)
            with col_b3:
                st.metric("Top Score", f"{res.score:.4f}")
            with col_b4:
                matched_q = store.get(res.matched_id) if res.matched_id else None
                matched_display = (
                    f"{res.matched_id}: {matched_q.text}" if matched_q else "None"
                )
                st.metric("Matched Question", matched_display)

            if res.explanation:
                st.info(f"**Explanation:** {res.explanation}")

            # Top-3 Vector Neighbors
            st.markdown("#### Top-3 Corpus Neighbors")
            q_resolved = detector._resolve_question(user_query.strip())
            q_vec = detector.embedder.encode(q_resolved)
            nbrs = detector.index.search(q_vec, k=3)

            nbr_records = []
            for rank, (nid, nscore) in enumerate(nbrs, 1):
                nq = store.get(nid)
                nbr_records.append(
                    {
                        "Rank": rank,
                        "Neighbor ID": nid,
                        "Cosine Similarity": f"{nscore:.4f}",
                        "Question Text": nq.text if nq else "—",
                        "Tags": str(nq.tags) if nq and nq.tags else "{}",
                    }
                )

            st.dataframe(pd.DataFrame(nbr_records), use_container_width=True)


# -------------------------------------------------------------
# TAB 2: Generate Questions
# -------------------------------------------------------------
with tab2:
    st.subheader("Non-Duplicate Question Generation")
    st.write(
        "Generate curriculum-aligned questions for a topic while preventing repeats using "
        "concept clustering, MMR diversification, and intra-batch deduplication."
    )

    g_col1, g_col2 = st.columns([2, 1])
    with g_col1:
        gen_topic = st.text_input("Topic Query:", value="reflection of light")
    with g_col2:
        gen_n = st.slider("Target Questions (n):", min_value=1, max_value=10, value=3)

    tf_col1, tf_col2 = st.columns(2)
    with tf_col1:
        tag_key = st.text_input("Optional Tag Filter Key:", placeholder="e.g. subject")
    with tf_col2:
        tag_val = st.text_input("Optional Tag Filter Value:", placeholder="e.g. optics")

    tags_filter = None
    if tag_key.strip() and tag_val.strip():
        tags_filter = {tag_key.strip(): tag_val.strip()}

    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        run_gen = st.button("Generate Questions", type="primary")
    with btn_col2:
        if st.button("🔄 Reset session", type="secondary"):
            init_session_state(force_reset=True)
            st.success("Session reset! Store restored to original corpus.")
            st.rerun()

    if run_gen:
        with st.spinner("Generating questions with intra-batch duplicate prevention..."):
            report = generator.generate(
                topic=gen_topic.strip(),
                n=gen_n,
                tags_filter=tags_filter,
            )
            st.session_state.last_report = report

    report = st.session_state.last_report
    if report is not None:
        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rounds Used", report.rounds_used)
        m2.metric("Duplicate Rate", f"{report.duplicate_rate:.1%}")
        m3.metric("Accepted Count", len(report.accepted))
        m4.metric("Rejected Count", len(report.rejected))

        st.markdown("#### Prompt Token Estimates per Round")
        tok_str = " | ".join(
            [
                f"**Round {i+1}:** {tok} tokens"
                for i, tok in enumerate(report.prompt_token_estimates)
            ]
        )
        st.write(tok_str if tok_str else "No rounds executed.")

        # Tables: Accepted, Rejected, Review Queue
        st.markdown("### 1. Accepted Questions")
        if report.accepted:
            df_acc = pd.DataFrame(
                [
                    {
                        "ID": q.id,
                        "Question Text": q.text,
                        "Normalized": q.normalized,
                        "Tags": str(q.tags),
                    }
                    for q in report.accepted
                ]
            )
            st.dataframe(df_acc, use_container_width=True)
        else:
            st.info("No questions accepted.")

        st.markdown("### 2. Rejected Candidates (Filtered Out)")
        if report.rejected:
            df_rej = pd.DataFrame(
                [
                    {
                        "Candidate Text": r.get("text", ""),
                        "Decision": r.get("decision", ""),
                        "Matched ID": r.get("matched_id") or "—",
                        "Score": f"{float(r.get('score', 0.0)):.4f}",
                    }
                    for r in report.rejected
                ]
            )
            st.dataframe(df_rej, use_container_width=True)
        else:
            st.success("No duplicates detected; zero candidates rejected.")

        st.markdown("### 3. Review Queue")
        if report.review_queue:
            df_rev = pd.DataFrame(
                [
                    {
                        "Candidate Text": r.get("text", ""),
                        "Decision": r.get("decision", "REVIEW"),
                        "Score": f"{float(r.get('score', 0.0)):.4f}",
                    }
                    for r in report.review_queue
                ]
            )
            st.dataframe(df_rev, use_container_width=True)
        else:
            st.caption("Review queue is empty.")


# -------------------------------------------------------------
# TAB 3: Corpus & Tags
# -------------------------------------------------------------
with tab3:
    st.subheader("Question Corpus & Arbitrary Tag Filtering")
    st.caption("Tags filter the pool only. They never affect similarity or duplicate decisions.")

    all_qs = store.all()
    # Find all arbitrary tag keys across the corpus
    all_keys = sorted(list({k for q in all_qs for k in (q.tags or {}).keys()}))

    c_filter1, c_filter2 = st.columns(2)
    with c_filter1:
        chosen_key = st.selectbox(
            "Filter by Tag Key:",
            options=["(All)"] + all_keys,
            help="Select an arbitrary tag key to filter.",
        )
    with c_filter2:
        if chosen_key != "(All)":
            unique_vals = sorted(
                list(
                    {
                        str(q.tags[chosen_key])
                        for q in all_qs
                        if chosen_key in (q.tags or {})
                    }
                )
            )
            chosen_vals = st.multiselect("Filter by Tag Values:", options=unique_vals)
        else:
            chosen_vals = []

    # Apply tag filter if selected
    if chosen_key != "(All)" and chosen_vals:
        filtered_ids: set[str] = set()
        for val in chosen_vals:
            for q_match in store.filter_by_tags({chosen_key: val}):
                filtered_ids.add(q_match.id)
        display_qs = [q for q in all_qs if q.id in filtered_ids]
    else:
        display_qs = all_qs

    st.write(f"Displaying **{len(display_qs)}** of **{len(all_qs)}** questions.")

    rows = []
    for q in display_qs:
        item = {"ID": q.id, "Question Text": q.text}
        for k in all_keys:
            item[k] = (q.tags or {}).get(k, "—")
        rows.append(item)

    df_corpus = pd.DataFrame(rows)
    st.dataframe(df_corpus, use_container_width=True)
