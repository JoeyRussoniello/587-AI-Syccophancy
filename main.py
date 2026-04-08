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

from tqdm import tqdm

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
MAX_WORKERS = 1
LOGGING_LEVEL = logging.INFO

# Or None to pull all. By default will ONLY generate responses for prompts that haven't been processed already
MAX_RESPONSES = 3

# Set to True to only make API calls and not append response records to database - used for testing AI connections
DRY_RUN = True

# Set to True to only get responses for 'YTA' prompts to get non-control group sycophancy rates.
YTA_ONLY = False
#########################################################

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).parent
DATASETS_DIR = REPO_ROOT / "datasets"
LOGS_DIR = REPO_ROOT / "logs"
LOG_FILE = LOGS_DIR / "app.log"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=LOGGING_LEVEL,
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
client_fn = CLIENT_FUNCTIONS[PROVIDER]
llm = client_fn(
    SYSTEM_PROMPT,
    max_retries=MAX_RETRIES,
    max_rows=MAX_RESPONSES,
    max_workers=MAX_WORKERS,
)


async def run_tasks_with_progress(
    tasks: list[asyncio.Task[None]], description: str
) -> None:
    """Await a batch of tasks while displaying tqdm progress."""

    with tqdm(total=len(tasks), desc=description, unit="prompt") as progress:
        for task in asyncio.as_completed(tasks):
            await task
            progress.update(1)


async def get_responses_for_model(
    llm: LLM_Client,
    system_prompt: SystemPrompt,
    semaphore: asyncio.Semaphore,
    dry_run: bool = False,
) -> None:
    """Fetch and persist model responses for all pending prompts."""

    with get_session() as session:
        seed_prompts(session, DATASETS_DIR)
        system_prompt_db_object = ensure_system_prompt(session, system_prompt)
        pending = get_pending_prompts(
            session, PROVIDER, system_prompt_db_object, yta_only=YTA_ONLY
        )

        if llm.cfg.max_rows is not None:
            pending = pending[: llm.cfg.max_rows]

        async def process(prompt) -> None:
            response = await llm.call_model(prompt.prompt, semaphore)
            if response != "ERROR" and not dry_run:
                save_response(
                    session, prompt, system_prompt_db_object, PROVIDER, response
                )

        tasks = [asyncio.create_task(process(prompt)) for prompt in pending]
        await run_tasks_with_progress(tasks, f"{PROVIDER} prompts")


async def main() -> None:
    """Run the configured model collection job."""

    sem = asyncio.Semaphore(llm.cfg.max_workers)
    await get_responses_for_model(llm, SYSTEM_PROMPT, sem, DRY_RUN)


if __name__ == "__main__":
    init_db()
    asyncio.run(main())
