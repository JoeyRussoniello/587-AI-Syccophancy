import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from db.crud import (
    ensure_system_prompt,
    get_pending_prompts,
    save_response,
    seed_prompts,
)
from db.database import get_session
from models import LLM_Client, ModelName, build_llm
from prompts import SystemPrompt


@dataclass(frozen=True)
class ResponseCollectionConfig:
    """Shared runtime settings for multi-model response collection."""

    system_prompt: SystemPrompt
    datasets_dir: Path
    max_retries: int
    max_rows: int | None
    max_workers_per_model: int
    yta_only: bool
    dry_run: bool = False


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
    model: ModelName,
    llm: LLM_Client,
    config: ResponseCollectionConfig,
    semaphore: asyncio.Semaphore,
    progress_position: int = 0,
) -> None:
    """Fetch and persist model responses for all pending prompts."""

    with get_session() as session:
        system_prompt_db_object = ensure_system_prompt(session, config.system_prompt)
        pending = get_pending_prompts(
            session, str(model), system_prompt_db_object, yta_only=config.yta_only
        )

        if llm.cfg.max_rows is not None:
            pending = pending[: llm.cfg.max_rows]

        async def process(prompt) -> None:
            response = await llm.call_model(prompt.prompt, semaphore)
            if response != "ERROR" and not config.dry_run:
                save_response(
                    session, prompt, system_prompt_db_object, str(model), response
                )

        tasks = [asyncio.create_task(process(prompt)) for prompt in pending]
        await run_tasks_with_progress(
            tasks,
            f"{model} prompts",
            position=progress_position,
        )


async def get_responses_for_models(
    models: Sequence[ModelName],
    config: ResponseCollectionConfig,
) -> None:
    """Run response collection for multiple models concurrently."""

    with get_session() as session:
        seed_prompts(session, config.datasets_dir)
        ensure_system_prompt(session, config.system_prompt)

    tasks = [
        asyncio.create_task(
            get_responses_for_model(
                model,
                build_llm(
                    config.system_prompt,
                    model,
                    max_retries=config.max_retries,
                    max_rows=config.max_rows,
                    max_workers=config.max_workers_per_model,
                ),
                config,
                asyncio.Semaphore(config.max_workers_per_model),
                progress_position=index,
            )
        )
        for index, model in enumerate(models)
    ]
    await asyncio.gather(*tasks)
