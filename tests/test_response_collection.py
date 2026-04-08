import asyncio
from pathlib import Path

import response_collection
from models import Model
from prompts import SystemPrompt


class DummySessionContext:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


def test_get_responses_for_models_uses_one_semaphore_per_provider(
    monkeypatch,
):
    models = [
        Model.GPT_4_1_MINI,
        Model.GEMINI,
    ]
    llm_by_provider = {str(model): object() for model in models}
    calls: list[
        tuple[
            Model,
            object,
            response_collection.ResponseCollectionConfig,
            asyncio.Semaphore,
            int,
        ]
    ] = []
    config = response_collection.ResponseCollectionConfig(
        system_prompt=SystemPrompt.BASE,
        datasets_dir=Path("datasets"),
        max_retries=3,
        max_rows=None,
        max_workers_per_model=5,
        yta_only=True,
        dry_run=True,
    )

    monkeypatch.setattr(
        response_collection, "get_session", lambda: DummySessionContext()
    )
    monkeypatch.setattr(
        response_collection, "seed_prompts", lambda session, datasets_dir: 0
    )
    monkeypatch.setattr(
        response_collection,
        "ensure_system_prompt",
        lambda session, system_prompt: object(),
    )
    monkeypatch.setattr(
        response_collection,
        "build_llm",
        lambda system_prompt, model, max_retries, max_rows, max_workers: (
            llm_by_provider[str(model)]
        ),
    )

    async def fake_get_responses_for_model(
        model,
        llm,
        config,
        semaphore,
        progress_position=0,
    ):
        calls.append((model, llm, config, semaphore, progress_position))

    monkeypatch.setattr(
        response_collection,
        "get_responses_for_model",
        fake_get_responses_for_model,
    )

    asyncio.run(
        response_collection.get_responses_for_models(
            models,
            config,
        )
    )

    assert len(calls) == 2

    calls_by_provider = {str(model): call for model, *call in calls}
    assert set(calls_by_provider) == {str(model) for model in models}

    openai_call = calls_by_provider[str(Model.GPT_4_1_MINI)]
    gemini_call = calls_by_provider[str(Model.GEMINI)]

    assert openai_call[0] is llm_by_provider[str(Model.GPT_4_1_MINI)]
    assert gemini_call[0] is llm_by_provider[str(Model.GEMINI)]
    assert openai_call[1] == config
    assert gemini_call[1] == config
    assert isinstance(openai_call[2], asyncio.Semaphore)
    assert isinstance(gemini_call[2], asyncio.Semaphore)
    assert openai_call[2] is not gemini_call[2]
    assert openai_call[3] == 0
    assert gemini_call[3] == 1
