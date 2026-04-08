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
SYSTEM_PROMPT = SystemPrompt.HONEST_ASSISTANT
PROVIDERS = [ModelProvider.OPEN_AI, ModelProvider.GEMINI]
MAX_RETRIES = 3
MAX_WORKERS_PER_MODEL = 2
LOGGING_LEVEL = logging.INFO

# Or None to pull all. By default will ONLY generate responses for prompts that haven't been processed already
MAX_RESPONSES = 15

# Set to True to only make API calls and not append response records to database - used for testing AI connections
DRY_RUN = False

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


def build_llm(provider: ModelProvider) -> LLM_Client:
    """Construct the configured client for a provider."""

    client_fn = CLIENT_FUNCTIONS[provider]
    return client_fn(
        SYSTEM_PROMPT,
        max_retries=MAX_RETRIES,
        max_rows=MAX_RESPONSES,
        max_workers=MAX_WORKERS_PER_MODEL,
    )


async def run_tasks_with_progress(
    tasks: list[asyncio.Task[None]], description: str, position: int = 0
) -> None:
    """Await a batch of tasks while displaying tqdm progress."""

    with tqdm(
        total=len(tasks),
        desc=description,
        unit="prompt",
        position=position,
    ) as progress:
        for task in asyncio.as_completed(tasks):
            await task
            progress.update(1)


async def get_responses_for_model(
    provider: ModelProvider,
    llm: LLM_Client,
    system_prompt: SystemPrompt,
    semaphore: asyncio.Semaphore,
    dry_run: bool = False,
    progress_position: int = 0,
) -> None:
    """Fetch and persist model responses for all pending prompts."""

    with get_session() as session:
        system_prompt_db_object = ensure_system_prompt(session, system_prompt)
        pending = get_pending_prompts(
            session, provider, system_prompt_db_object, yta_only=YTA_ONLY
        )

        if llm.cfg.max_rows is not None:
            pending = pending[: llm.cfg.max_rows]

        async def process(prompt) -> None:
            response = await llm.call_model(prompt.prompt, semaphore)
            if response != "ERROR" and not dry_run:
                save_response(
                    session, prompt, system_prompt_db_object, provider, response
                )

        tasks = [asyncio.create_task(process(prompt)) for prompt in pending]
        await run_tasks_with_progress(
            tasks,
            f"{provider} prompts",
            position=progress_position,
        )


async def get_responses_for_models(
    providers: list[ModelProvider],
    system_prompt: SystemPrompt,
    dry_run: bool = False,
) -> None:
    """Run response collection for multiple providers concurrently."""

    with get_session() as session:
        seed_prompts(session, DATASETS_DIR)
        ensure_system_prompt(session, system_prompt)

    tasks = [
        asyncio.create_task(
            get_responses_for_model(
                provider,
                build_llm(provider),
                system_prompt,
                asyncio.Semaphore(MAX_WORKERS_PER_MODEL),
                dry_run=dry_run,
                progress_position=index,
            )
        )
        for index, provider in enumerate(providers)
    ]
    await asyncio.gather(*tasks)


async def main() -> None:
    """Run the configured model collection job."""

    await get_responses_for_models(PROVIDERS, SYSTEM_PROMPT, DRY_RUN)


if __name__ == "__main__":
    init_db()
    asyncio.run(main())
