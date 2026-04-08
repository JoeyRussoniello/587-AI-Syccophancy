"""An entry point for all LLM wrappers. Abstracts method class, retry strategies, and async processing"""

import asyncio
import logging
import os
from asyncio import Semaphore
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Protocol

import anthropic
import openai
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import AsyncOpenAI

from prompts import SystemPrompt

load_dotenv()
logger = logging.getLogger(__name__)


class ModelProvider(StrEnum):
    """Supported model identifiers for each provider used in the project."""

    CLAUDE = "claude-haiku-4-5-20251001"
    OPEN_AI = "gpt-4.1-mini"
    GEMINI = "gemini-2.5-flash"


DEFAULT_MAX_TOKENS = 150
DEFAULT_NUM_RESPONSES = 15
DEFAULT_MAX_WORKERS = 3
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY = 2.0


def _strip_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    return stripped or None


def _first_anthropic_text_block(content: object) -> str | None:
    if not isinstance(content, list):
        return None

    for block in content:
        text = _strip_text(getattr(block, "text", None))
        if text is not None:
            return text

    return None


@dataclass
class ModelConfig:
    """Runtime configuration shared by all model client wrappers."""

    required_key: str
    system_prompt: SystemPrompt
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_rows: int | None = DEFAULT_NUM_RESPONSES
    max_workers: int = DEFAULT_MAX_WORKERS
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY

    def ensure_key(self) -> None:
        """Validate that the required environment variable is present."""

        key = os.getenv(self.required_key)
        if key is None or key.strip() == "":
            raise EnvironmentError(
                f"Missing Required Environment Variable to initialize client {self.required_key}"
            )

    def __post_init__(self) -> None:
        self.ensure_key()


class LLM_Client(Protocol):
    """Protocol for LLM clients. Subclasses implement only _call_model_once.

    _call_model_once returns:
        str: on success (response text) or non-retryable failure ("ERROR")
        None: signal that the call should be retried
    """

    cfg: ModelConfig

    def __init__(self, client: Any, configuration: ModelConfig, **kwargs): ...

    async def _call_model_once(self, prompt: str) -> str | None: ...

    async def _call_model(self, prompt: str) -> str:
        delay = self.cfg.retry_base_delay
        for attempt in range(self.cfg.max_retries):
            if attempt >= 1:
                logging.debug("Attempt %d to _call_model_once.", attempt + 1)
            result = await self._call_model_once(prompt)
            if result is not None:
                return result
            if attempt < self.cfg.max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
        return "ERROR"

    async def call_model(self, prompt: str, semaphore: Semaphore) -> str:
        """Run a prompt through the model with concurrency control and retries.

        This is the main public entry point for model execution in the module.
        It acquires the provided semaphore before calling the internal retry
        wrapper, ensuring the caller can bound concurrent in-flight requests
        across many prompts.

        Args:
            prompt: The user prompt text to send to the configured model.
            semaphore: An asyncio semaphore used to limit how many model calls
                can run concurrently.

        Returns:
            The final model response text. Returns "ERROR" when all retry
            attempts fail or a non-retryable API error occurs.
        """

        async with semaphore:
            msg = await self._call_model(prompt)
            logger.debug("Got response from model:\n%s", msg)
            return msg


class AnthropicClient(LLM_Client):
    """Anthropic implementation of the shared LLM client protocol."""

    def __init__(self, client: AsyncAnthropic, configuration: ModelConfig):
        self.client = client
        self.cfg = configuration

    async def _call_model_once(self, prompt: str) -> str | None:
        try:
            msg = await self.client.messages.create(
                model=ModelProvider.CLAUDE,
                max_tokens=len(prompt) + self.cfg.max_tokens,
                system=self.cfg.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            text = _first_anthropic_text_block(msg.content)
            if text is None:
                logger.error("Claude response did not include a text block")
                return "ERROR"
            return text
        except anthropic.RateLimitError:
            return None
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529):
                return None
            logger.error(f"Claude API error {e.status_code}: {e}")
            return "ERROR"
        except Exception as e:
            logger.error(f"Unexpected Claude error: {e}")
            return "ERROR"


class OpenAIClient(LLM_Client):
    """OpenAI implementation of the shared LLM client protocol."""

    def __init__(
        self,
        client: AsyncOpenAI,
        configuration: ModelConfig,
        provider: ModelProvider,
    ):
        self.client = client
        self.cfg = configuration
        self.provider = provider

    async def _call_model_once(self, prompt: str) -> str | None:
        try:
            resp = await self.client.chat.completions.create(
                model=self.provider,
                max_tokens=len(prompt) + self.cfg.max_tokens,
                messages=[
                    {"role": "system", "content": self.cfg.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            content = _strip_text(resp.choices[0].message.content)
            if content is None:
                logger.error("OpenAI response did not include message content")
                return "ERROR"
            return content
        except openai.RateLimitError:
            return None
        except openai.APIStatusError as e:
            if e.status_code in (429, 529):
                return None
            logger.error(f"OpenAI API error {e.status_code}: {e}")
            return "ERROR"
        except Exception as e:
            logger.error(f"Unexpected OpenAI error: {e}")
            return "ERROR"


class GeminiClient(LLM_Client):
    """Google GenAI implementation of the shared LLM client protocol."""

    def __init__(self, client: genai.Client, configuration: ModelConfig):
        self.client = client
        self.cfg = configuration

    async def _call_model_once(self, prompt: str) -> str | None:
        try:
            resp = await self.client.aio.models.generate_content(
                model=ModelProvider.GEMINI,
                contents=str(prompt),
                config=genai_types.GenerateContentConfig(
                    system_instruction=str(self.cfg.system_prompt),
                    max_output_tokens=len(prompt) + self.cfg.max_tokens,
                ),
            )
            text = _strip_text(resp.text)
            if text is None:
                logger.error("Gemini response did not include text output")
                return "ERROR"
            return text
        except genai_errors.APIError as e:
            if e.code in (429, 503):
                return None
            logger.error(f"Gemini API error {e.code}: {e}")
            return "ERROR"
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return "ERROR"


def get_anthropic_client(system_prompt: SystemPrompt, **kwargs) -> AnthropicClient:
    """Build the default Anthropic client wrapper for a given system prompt."""

    cfg = ModelConfig(
        required_key="ANTHROPIC_API_KEY", system_prompt=system_prompt, **kwargs
    )
    return AnthropicClient(AsyncAnthropic(), cfg)


def get_openai_client(
    system_prompt: SystemPrompt,
    provider: ModelProvider = ModelProvider.OPEN_AI,
    **kwargs,
) -> OpenAIClient:
    """Build the default OpenAI client wrapper for a given system prompt."""

    cfg = ModelConfig(
        required_key="OPENAI_API_KEY", system_prompt=system_prompt, **kwargs
    )
    return OpenAIClient(AsyncOpenAI(), cfg, provider)


def get_gemini_client(system_prompt: SystemPrompt, **kwargs) -> GeminiClient:
    """Build the default Gemini client wrapper for a given system prompt."""

    cfg = ModelConfig(
        required_key="GOOGLE_API_KEY", system_prompt=system_prompt, **kwargs
    )
    client = genai.Client()
    return GeminiClient(client, cfg)


ClientFn = Callable[..., LLM_Client]

CLIENT_FUNCTIONS: dict[ModelProvider, ClientFn] = {
    ModelProvider.CLAUDE: get_anthropic_client,
    ModelProvider.OPEN_AI: get_openai_client,
    ModelProvider.GEMINI: get_gemini_client,
}
