"""LLM Client interface and implementations for question-dedup."""

from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Mapping, NamedTuple

from qdedup.normalize import extract_numbers, normalize


class JudgeResult(NamedTuple):
    """Result of an LLM semantic equivalence judgment."""

    same: bool
    confidence: float
    explanation: str = ""


# Default question-framing / functional stopwords for content word extraction
QUESTION_STOP_WORDS = {
    "what", "is", "are", "do", "does", "did", "you", "understand", "by",
    "the", "of", "a", "an", "in", "to", "for", "with", "from",
    "explain", "meaning", "define", "describe", "tell", "give", "me",
}


def extract_content_words(text: str, stop_words: set[str] | None = None) -> set[str]:
    """Extract content words from text by stripping punctuation and stop words."""
    stops = QUESTION_STOP_WORDS if stop_words is None else stop_words
    norm = normalize(text)
    words = norm.split()
    content = set(w for w in words if w not in stops)
    return content or set(words)


def compute_jaccard_similarity(text1: str, text2: str, stop_words: set[str] | None = None) -> float:
    """Compute Jaccard similarity (|A ∩ B| / |A ∪ B|) over content words."""
    w1 = extract_content_words(text1, stop_words)
    w2 = extract_content_words(text2, stop_words)
    inter = len(w1 & w2)
    union = len(w1 | w2)
    return inter / union if union > 0 else 0.0


class LLMClient(ABC):
    """Abstract base interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response text for a given prompt."""
        raise NotImplementedError

    @abstractmethod
    def judge(self, q1: str, q2: str, **kwargs: Any) -> JudgeResult:
        """Judge whether two questions are semantic duplicates."""
        raise NotImplementedError


SCRIPTED_ROUND1_CANDIDATES = [
    {"question": "Define the reflection of light.", "concept_tested": "definition of light reflection"},
    {"question": "Describe what is meant by light reflection.", "concept_tested": "meaning of light reflection"},
    {"question": "Why do we see our image in a plane mirror?", "concept_tested": "image formation in plane mirror"},
    {"question": "Name two applications of diffuse reflection.", "concept_tested": "applications of diffuse reflection"},
    {"question": "A ray strikes a mirror at 45°. Find the angle between the incident and reflected ray.", "concept_tested": "law of reflection numerical angle calculation"},
    {"question": "Describe what is meant by light reflection.", "concept_tested": "meaning of light reflection"},
]

SCRIPTED_ROUND2_CANDIDATES = [
    {"question": "State two differences between a plane mirror and a concave mirror image.", "concept_tested": "plane vs concave mirror image comparison"},
    {"question": "Why does a rough wall not form a clear image of an object?", "concept_tested": "diffuse reflection explanation"},
]


class MockLLM(LLMClient):
    """Deterministic mock LLM for testing and running without an API key.

    Implements:
    - Word-overlap judge rule using Jaccard similarity (|A ∩ B| / |A ∪ B|) over content words.
    - Scripted generation mode for multi-round question generation testing.
    """

    def __init__(
        self,
        default_response: str | None = None,
        responses: Mapping[str, str] | None = None,
        judge_threshold: float = 0.6,
        stop_words: set[str] | None = None,
    ) -> None:
        self.default_response = default_response
        self.responses: dict[str, str] = dict(responses or {})
        self.judge_threshold = judge_threshold
        self.stop_words = set(stop_words) if stop_words is not None else QUESTION_STOP_WORDS
        self.call_history: list[str] = []
        self.scripted_round_counter: int = 0

    def reset_scripted_rounds(self) -> None:
        """Reset scripted round counter."""
        self.scripted_round_counter = 0

    def register_response(self, prompt_substring: str, response: str) -> None:
        """Register a canned response triggered when prompt contains substring."""
        self.responses[prompt_substring] = response

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return a deterministic response for the prompt."""
        self.call_history.append(prompt)

        # 1. Match specific registered substrings
        for key, value in self.responses.items():
            if key in prompt:
                return value

        # 2. Scripted generation mode if prompt requests question generation JSON list
        if "JSON list" in prompt or "Existing questions" in prompt or "concept_tested" in prompt:
            if "Already rejected" in prompt or self.scripted_round_counter >= 1:
                self.scripted_round_counter += 1
                return json.dumps(SCRIPTED_ROUND2_CANDIDATES, indent=2)
            self.scripted_round_counter += 1
            return json.dumps(SCRIPTED_ROUND1_CANDIDATES, indent=2)

        # 3. Use explicit default if provided
        if self.default_response is not None:
            return self.default_response

        # 4. Deterministic output generated from prompt digest
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return f"[MockLLM-{digest}] Deterministic response for prompt length {len(prompt)}."

    def jaccard_similarity(self, q1: str, q2: str) -> float:
        """Calculate Jaccard similarity (|A ∩ B| / |A ∪ B|) over content words."""
        return compute_jaccard_similarity(q1, q2, self.stop_words)

    def judge(self, q1: str, q2: str, **kwargs: Any) -> JudgeResult:
        """Judge whether two questions are semantic duplicates using Jaccard similarity.

        Uses Jaccard similarity (|A ∩ B| / |A ∪ B|) over content words (not overlap relative
        to the smaller set). Threshold defaults to 0.6.
        If questions contain differing numbers, they are distinct questions.
        """
        # If numbers differ, questions represent distinct numerical problems
        nums1 = extract_numbers(q1)
        nums2 = extract_numbers(q2)
        if nums1 != nums2:
            return JudgeResult(
                same=False,
                confidence=0.85,
                explanation=f"Questions contain differing numerical values ({nums1} vs {nums2})",
            )

        threshold = kwargs.get("threshold", self.judge_threshold)
        sim = self.jaccard_similarity(q1, q2)
        same = sim >= threshold

        if same:
            # Scaled confidence between 0.70 and 1.00
            conf = 0.70 + 0.30 * ((sim - threshold) / (1.0 - threshold) if threshold < 1.0 else 1.0)
        else:
            # Scaled confidence that they are different
            conf = 0.70 + 0.30 * ((threshold - sim) / threshold if threshold > 0 else 1.0)

        conf = round(min(max(conf, 0.0), 1.0), 4)
        return JudgeResult(
            same=same,
            confidence=conf,
            explanation=f"Content-word Jaccard similarity={sim:.4f} (threshold={threshold})",
        )


class GeminiLLM(LLMClient):
    """Google Gemini LLM implementation using google-generativeai."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Provide api_key argument or set the "
                "GEMINI_API_KEY environment variable. Alternatively, use MockLLM."
            )

        try:
            import google.generativeai as genai
        except ImportError as err:
            raise ImportError(
                "google-generativeai package is required to use GeminiLLM."
            ) from err

        genai.configure(api_key=self.api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(model_name=self.model_name)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using Gemini."""
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            **{k: v for k, v in kwargs.items() if k != "temperature"},
        }
        response = self._model.generate_content(
            prompt,
            generation_config=generation_config if generation_config else None,
        )
        return response.text or ""

    def judge(self, q1: str, q2: str, **kwargs: Any) -> JudgeResult:
        """Judge whether two questions are semantic duplicates using Gemini."""
        prompt = (
            "You are an expert curriculum evaluator. Compare these two questions:\n"
            f"Question A: {q1}\n"
            f"Question B: {q2}\n\n"
            "Do both questions ask for the exact same core concept/answer, or are they different?\n"
            "Respond strictly in valid JSON format with keys:\n"
            '{"same": true/false, "confidence": float between 0.0 and 1.0, "explanation": "short string"}'
        )
        raw = self.generate(prompt, temperature=0.0)
        try:
            # Extract JSON block if surrounded by markdown code blocks
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return JudgeResult(
                    same=bool(data.get("same", False)),
                    confidence=float(data.get("confidence", 0.5)),
                    explanation=str(data.get("explanation", "")),
                )
        except Exception:
            pass

        return JudgeResult(same=False, confidence=0.5, explanation=raw)


def get_llm_client(
    use_mock: bool = False,
    api_key: str | None = None,
    model_name: str = "gemini-1.5-flash",
    mock_default: str | None = None,
) -> LLMClient:
    """Convenience factory to obtain either GeminiLLM or MockLLM."""
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if use_mock or not resolved_key:
        return MockLLM(default_response=mock_default)
    return GeminiLLM(api_key=resolved_key, model_name=model_name)
