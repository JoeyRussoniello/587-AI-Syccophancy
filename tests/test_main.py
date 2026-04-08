import asyncio

import main
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
        Model.OPEN_AI,
        Model.GEMINI,
    ]
    llm_by_provider = {str(model): object() for model in models}
    calls: list[tuple[Model, object, asyncio.Semaphore, bool, int]] = []

    monkeypatch.setattr(main, "get_session", lambda: DummySessionContext())
    monkeypatch.setattr(main, "seed_prompts", lambda session, datasets_dir: 0)
    monkeypatch.setattr(
        main, "ensure_system_prompt", lambda session, system_prompt: object()
    )
    monkeypatch.setattr(main, "build_llm", lambda model: llm_by_provider[str(model)])

    async def fake_get_responses_for_model(
        model,
        llm,
        system_prompt,
        semaphore,
        dry_run=False,
        progress_position=0,
    ):
        calls.append((model, llm, semaphore, dry_run, progress_position))

    monkeypatch.setattr(main, "get_responses_for_model", fake_get_responses_for_model)

    asyncio.run(
        main.get_responses_for_models(
            models,
            SystemPrompt.BASE,
            dry_run=True,
        )
    )

    assert len(calls) == 2

    calls_by_provider = {str(model): call for model, *call in calls}
    assert set(calls_by_provider) == {str(model) for model in models}

    openai_call = calls_by_provider[str(main.Model.OPEN_AI)]
    gemini_call = calls_by_provider[str(main.Model.GEMINI)]

    assert openai_call[0] is llm_by_provider[str(main.Model.OPEN_AI)]
    assert gemini_call[0] is llm_by_provider[str(main.Model.GEMINI)]
    assert isinstance(openai_call[1], asyncio.Semaphore)
    assert isinstance(gemini_call[1], asyncio.Semaphore)
    assert openai_call[1] is not gemini_call[1]
    assert openai_call[2] is True
    assert gemini_call[2] is True
    assert openai_call[3] == 0
    assert gemini_call[3] == 1
