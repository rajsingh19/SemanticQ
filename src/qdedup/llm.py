"""LLM Client interface and implementations for question-dedup."""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Any, Mapping


class LLMClient(ABC):
    """Abstract base interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response text for a given prompt."""
        raise NotImplementedError


class MockLLM(LLMClient):
    """Deterministic mock LLM for testing and running without an API key.

    Produces predictable, reproducible outputs based on configured responses
    or a deterministic hash of the prompt.
    """

    def __init__(
        self,
        default_response: str | None = None,
        responses: Mapping[str, str] | None = None,
    ) -> None:
        self.default_response = default_response
        self.responses: dict[str, str] = dict(responses or {})
        self.call_history: list[str] = []

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

        # 2. Use explicit default if provided
        if self.default_response is not None:
            return self.default_response

        # 3. Deterministic output generated from prompt digest
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return f"[MockLLM-{digest}] Deterministic response for prompt length {len(prompt)}."


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
