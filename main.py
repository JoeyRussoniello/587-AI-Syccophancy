"""
collect_responses.py

Queries Claude, OpenAI, and Gemini on the consolidated AITA dataset and saves
responses for comparison.  Output is normalized to three columns per model:

    source | prompt | response_claude | response_openai | response_gemini

Token-efficient: prompts are truncated to MAX_PROMPT_CHARS characters and
responses are capped at MAX_TOKENS tokens.

Checkpointing: if the output file already exists, rows that already have a
response are skipped so the run can be resumed after an interruption.

API keys are loaded from the .env file at the repo root:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
"""

import asyncio
import logging
from pathlib import Path

from db.crud import (
    ensure_system_prompt,
    get_pending_prompts,
    save_response,
    seed_prompts,
)
from db.database import get_session, init_db
from models import CLIENT_FUNCTIONS, LLM_Client, ModelProvider
from prompts import SystemPrompt

# CONFIG - Change variables to change the running model
SYSTEM_PROMPT = SystemPrompt.BASE
PROVIDER = ModelProvider.GEMINI

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level = logging.DEBUG)
client_fn = CLIENT_FUNCTIONS[PROVIDER]
llm = client_fn(SYSTEM_PROMPT)
REPO_ROOT = Path(__file__).parent
DATASETS_DIR = REPO_ROOT / "datasets"


async def get_responses_for_model(
    llm: LLM_Client, system_prompt: SystemPrompt, semaphore: asyncio.Semaphore
):
    num_calls = 0
    max_calls = llm.cfg.max_rows
    with get_session() as session:
        seed_prompts(session, DATASETS_DIR)
        system_prompt_db_object = ensure_system_prompt(session, system_prompt)

        pending = get_pending_prompts(session, PROVIDER, system_prompt_db_object)
        for prompt in pending:
            response = await llm.call_model(prompt.prompt, semaphore)
            num_calls += 1
            if response != "ERROR":
                save_response(session, prompt, system_prompt_db_object, PROVIDER, response)
            
            if max_calls is not None and num_calls >= max_calls:
                return


async def main():
    sem = asyncio.Semaphore()
    await get_responses_for_model(llm, SYSTEM_PROMPT, sem)


if __name__ == "__main__":
    init_db()
    asyncio.run(main())
