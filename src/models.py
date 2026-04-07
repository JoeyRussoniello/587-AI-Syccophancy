import asyncio
import logging
from asyncio import Semaphore
from dataclasses import dataclass
from typing import Protocol

import anthropic
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

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
    ...

class GeminiClient(LLM_Client):
    ...
        