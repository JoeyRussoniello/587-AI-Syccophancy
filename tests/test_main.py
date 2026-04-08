import asyncio

import main
from models import ModelProvider
from prompts import SystemPrompt


class DummySessionContext:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


def test_get_responses_for_models_uses_one_semaphore_per_provider(
    monkeypatch,
):
    providers = [ModelProvider.OPEN_AI, ModelProvider.GEMINI]
    llm_by_provider = {provider: object() for provider in providers}
    calls: list[tuple[ModelProvider, object, asyncio.Semaphore, bool, int]] = []

    monkeypatch.setattr(main, "get_session", lambda: DummySessionContext())
    monkeypatch.setattr(main, "seed_prompts", lambda session, datasets_dir: 0)
    monkeypatch.setattr(
        main, "ensure_system_prompt", lambda session, system_prompt: object()
    )
    monkeypatch.setattr(main, "build_llm", lambda provider: llm_by_provider[provider])

    async def fake_get_responses_for_model(
        provider,
        llm,
        system_prompt,
        semaphore,
        dry_run=False,
        progress_position=0,
    ):
        calls.append((provider, llm, semaphore, dry_run, progress_position))

    monkeypatch.setattr(main, "get_responses_for_model", fake_get_responses_for_model)

    asyncio.run(
        main.get_responses_for_models(
            providers,
            SystemPrompt.BASE,
            dry_run=True,
        )
    )

    assert len(calls) == 2

    calls_by_provider = {provider: call for provider, *call in calls}
    assert set(calls_by_provider) == set(providers)

    openai_call = calls_by_provider[ModelProvider.OPEN_AI]
    gemini_call = calls_by_provider[ModelProvider.GEMINI]

    assert openai_call[0] is llm_by_provider[ModelProvider.OPEN_AI]
    assert gemini_call[0] is llm_by_provider[ModelProvider.GEMINI]
    assert isinstance(openai_call[1], asyncio.Semaphore)
    assert isinstance(gemini_call[1], asyncio.Semaphore)
    assert openai_call[1] is not gemini_call[1]
    assert openai_call[2] is True
    assert gemini_call[2] is True
    assert openai_call[3] == 0
    assert gemini_call[3] == 1
