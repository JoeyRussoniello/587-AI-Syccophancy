import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import models
from prompts import SystemPrompt


class DummyClient(models.LLM_Client):
    """A mock LLM client for testing retry behavior"""

    def __init__(self, results: list[str | None], max_retries: int = 3):
        self.cfg = SimpleNamespace(max_retries=max_retries, retry_base_delay=0)
        self._results = iter(results)
        self.seen_prompts: list[str] = []

    async def _call_model_once(self, prompt: str) -> str | None:
        self.seen_prompts.append(prompt)
        return next(self._results)


class TrackingSemaphore:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited += 1
        return False


class FakeAPIStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status={status_code}")
        self.status_code = status_code


class FakeGenAIError(Exception):
    def __init__(self, code: int):
        super().__init__(f"code={code}")
        self.code = code


def test_model_config_requires_environment_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TEST_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="TEST_API_KEY"):
        models.ModelConfig(
            required_key="TEST_API_KEY",
            system_prompt=SystemPrompt.BASE,
        )


def test_model_config_allows_present_environment_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_API_KEY", "present")

    cfg = models.ModelConfig(
        required_key="TEST_API_KEY",
        system_prompt=SystemPrompt.BASE,
    )

    assert cfg.required_key == "TEST_API_KEY"


def test_call_model_retries_until_success():
    client = DummyClient([None, None, "NTA. Final answer"], max_retries=4)

    result = asyncio.run(models.LLM_Client._call_model(client, "prompt text"))

    assert result == "NTA. Final answer"
    assert client.seen_prompts == ["prompt text", "prompt text", "prompt text"]


def test_call_model_returns_error_after_retry_exhaustion():
    client = DummyClient([None, None, None], max_retries=3)

    result = asyncio.run(models.LLM_Client._call_model(client, "prompt text"))

    assert result == "ERROR"


def test_call_model_uses_semaphore_wrapper():
    client = DummyClient(["YTA. Because"], max_retries=1)
    semaphore = TrackingSemaphore()

    result = asyncio.run(models.LLM_Client.call_model(client, "prompt text", semaphore))

    assert result == "YTA. Because"
    assert semaphore.entered == 1
    assert semaphore.exited == 1


def test_anthropic_client_returns_stripped_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    message = SimpleNamespace(content=[SimpleNamespace(text="  NTA. Reasoning  ")])
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=message))
    )

    anthropic_client = models.get_anthropic_client(SystemPrompt.BASE)
    anthropic_client.client = client

    result = asyncio.run(anthropic_client._call_model_once("prompt text"))

    assert result == "NTA. Reasoning"


def test_anthropic_client_returns_error_when_no_text_block(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    message = SimpleNamespace(content=[SimpleNamespace(id="tool-use")])
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=message))
    )

    anthropic_client = models.get_anthropic_client(SystemPrompt.BASE)
    anthropic_client.client = client

    result = asyncio.run(anthropic_client._call_model_once("prompt text"))

    assert result == "ERROR"


def test_anthropic_client_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")

    class FakeRateLimitError(Exception):
        pass

    monkeypatch.setattr(
        models,
        "anthropic",
        SimpleNamespace(
            RateLimitError=FakeRateLimitError,
            APIStatusError=FakeAPIStatusError,
        ),
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=FakeRateLimitError()))
    )

    anthropic_client = models.get_anthropic_client(SystemPrompt.BASE)
    anthropic_client.client = client

    result = asyncio.run(anthropic_client._call_model_once("prompt text"))

    assert result is None


def test_openai_client_returns_error_when_message_content_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "present")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )

    openai_client = models.get_openai_client(SystemPrompt.BASE)
    openai_client.client = client

    result = asyncio.run(openai_client._call_model_once("prompt text"))

    assert result == "ERROR"


def test_openai_client_retries_on_retryable_status(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    class FakeRateLimitError(Exception):
        pass

    monkeypatch.setattr(
        models,
        "openai",
        SimpleNamespace(
            RateLimitError=FakeRateLimitError,
            APIStatusError=FakeAPIStatusError,
        ),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=FakeAPIStatusError(429))
            )
        )
    )

    openai_client = models.get_openai_client(SystemPrompt.BASE)
    openai_client.client = client

    result = asyncio.run(openai_client._call_model_once("prompt text"))

    assert result is None


def test_gemini_client_returns_error_when_text_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "present")
    response = SimpleNamespace(text=None)
    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(return_value=response))
        )
    )

    gemini_client = models.get_gemini_client(SystemPrompt.BASE)
    gemini_client.client = client

    result = asyncio.run(gemini_client._call_model_once("prompt text"))

    assert result == "ERROR"


def test_gemini_client_retries_on_retryable_status(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "present")
    monkeypatch.setattr(
        models, "genai_errors", SimpleNamespace(APIError=FakeGenAIError)
    )
    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=AsyncMock(side_effect=FakeGenAIError(503))
            )
        )
    )

    gemini_client = models.get_gemini_client(SystemPrompt.BASE)
    gemini_client.client = client

    result = asyncio.run(gemini_client._call_model_once("prompt text"))

    assert result is None
