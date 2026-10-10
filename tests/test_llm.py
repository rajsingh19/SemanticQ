import pytest
from qdedup.llm import LLMClient, MockLLM, GeminiLLM, get_llm_client


def test_mock_llm_is_llmclient_instance():
    client = MockLLM()
    assert isinstance(client, LLMClient)


def test_mock_llm_deterministic_output():
    client = MockLLM()
    prompt = "What is the capital of France?"
    res1 = client.generate(prompt)
    res2 = client.generate(prompt)
    assert res1 == res2
    assert "MockLLM" in res1
    assert len(client.call_history) == 2


def test_mock_llm_custom_responses():
    client = MockLLM(default_response="Default answer")
    assert client.generate("unknown prompt") == "Default answer"

    client.register_response("capital of France", "Paris")
    assert client.generate("What is the capital of France?") == "Paris"


def test_gemini_llm_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        GeminiLLM(api_key=None)


def test_get_llm_client_fallback_to_mock(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = get_llm_client()
    assert isinstance(client, MockLLM)
