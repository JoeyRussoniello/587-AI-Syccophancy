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

#########################################################
# CONFIG - Change these variables to change the experiment settings
SYSTEM_PROMPT = SystemPrompt.BASE
PROVIDER = ModelProvider.GEMINI
MAX_RETRIES = 3
NUM_RESPONSES = 1  # Or None to pull all. By default will ONLY generate responses for prompts that haven't been processed already
MAX_WORKERS = 1
DRY_RUN = True  # Set to True to only make API calls and not append response records to database - used for testing AI connections
LOGGING_LEVEL = logging.DEBUG
#########################################################

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=LOGGING_LEVEL
)
client_fn = CLIENT_FUNCTIONS[PROVIDER]
llm = client_fn(
    SYSTEM_PROMPT,
    max_retries=MAX_RETRIES,
    max_rows=NUM_RESPONSES,
    max_workers=MAX_WORKERS,
)
REPO_ROOT = Path(__file__).parent
DATASETS_DIR = REPO_ROOT / "datasets"


async def get_responses_for_model(
    llm: LLM_Client,
    system_prompt: SystemPrompt,
    semaphore: asyncio.Semaphore,
    dry_run: bool = False,
):
    with get_session() as session:
        seed_prompts(session, DATASETS_DIR)
        system_prompt_db_object = ensure_system_prompt(session, system_prompt)
        pending = get_pending_prompts(session, PROVIDER, system_prompt_db_object)

        if llm.cfg.max_rows is not None:
            pending = pending[: llm.cfg.max_rows]

        async def process(prompt):
            response = await llm.call_model(prompt.prompt, semaphore)
            if response != "ERROR" and not dry_run:
                save_response(
                    session, prompt, system_prompt_db_object, PROVIDER, response
                )

        await asyncio.gather(*[process(p) for p in pending])


async def main():
    sem = asyncio.Semaphore(llm.cfg.max_workers)
    await get_responses_for_model(llm, SYSTEM_PROMPT, sem, DRY_RUN)


if __name__ == "__main__":
    init_db()
    asyncio.run(main())
