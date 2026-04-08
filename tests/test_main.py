import asyncio

import main


def test_main_delegates_to_response_collection(
    monkeypatch,
):
    captured = {}

    async def fake_get_responses_for_models(
        models,
        config,
    ):
        captured["models"] = models
        captured["config"] = config

    monkeypatch.setattr(main, "get_responses_for_models", fake_get_responses_for_models)

    asyncio.run(main.main())

    assert captured["models"] == main.MODELS
    assert captured["config"] == main.COLLECTION_CONFIG
