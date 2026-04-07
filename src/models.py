"""An entry point for all LLM wrappers. Abstracts method class, retry strategies, and async processing"""

import asyncio
import logging
from asyncio import Semaphore
from dataclasses import dataclass
from typing import Protocol

import anthropic
import openai
from anthropic import AsyncAnthropic
from google.api_core.exceptions import (
    DeadlineExceeded,
    ResourceExhausted,
    ServiceUnavailable,
)
from google.generativeai import GenerativeModel
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MODELS = {
    "claude": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
}

@dataclass
class ModelConfig:
    model: str
    system_prompt: str 
    max_tokens: int = 150
    max_rows: int | None  = 15
    max_workers: int = 3 
    max_retries: int = 5
    retry_base_delay: float = 2.0


class LLM_Client(Protocol):
    """Protocol for LLM clients. Subclasses implement only _call_model_once.
    
    _call_model_once returns:
        str: on success (response text) or non-retryable failure ("ERROR")
        None: signal that the call should be retried
    """
    cfg: ModelConfig

    def __init__(self, client: any, configuration: ModelConfig): 
        ...
    
    async def _call_model_once(self, prompt: str) -> str | None:
        ...
        
    async def _call_model(self, prompt: str) -> str: 
        delay = self.cfg.retry_base_delay
        for attempt in range(self.cfg.max_retries):
            result = await self._call_model_once(prompt)
            if result is not None:
                return result
            if attempt < self.cfg.max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
        return "ERROR"
    
    async def call_model(self, prompt: str, semaphore: Semaphore) -> str:
        async with semaphore:
            return await self._call_model(prompt)


class AnthropicClient(LLM_Client):
    def __init__(self, client: AsyncAnthropic, configuration: ModelConfig):
        self.client = client
        self.cfg = configuration

    async def _call_model_once(self, prompt: str) -> str | None:
        try:
            msg = await self.client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=self.cfg.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
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
    def __init__(self, client: AsyncOpenAI, configuration: ModelConfig):
        self.client = client
        self.cfg = configuration
    
    async def _call_model_once(self, prompt: str) -> str | None:
        try: 
            resp = await self.client.chat.completions.create(
                model = self.cfg.model,
                max_tokens = self.cfg.max_tokens,
                messages = [
                    {'role': 'system', 'content': self.cfg.system_prompt},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
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
    def __init__(self, client: GenerativeModel, configuration: ModelConfig):
        self.client = client
        self.cfg = configuration

    async def _call_model_once(self, prompt: str) -> str | None:
        try:
            resp = await self.client.generate_content_async(
                f"{self.cfg.system_prompt}\n\n{prompt}",
                generation_config={"max_output_tokens": self.cfg.max_tokens},
            )
            return resp.text.strip()
        except (ResourceExhausted, ServiceUnavailable, DeadlineExceeded):
            return None
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return "ERROR"
        