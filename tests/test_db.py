from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import inspect

from db import crud, database
from db.models import LLMResponse, Prompt, SystemPrompt
from models import ModelProvider
from prompts import SystemPrompt as SystemPromptEnum


def _write_csv(
    datasets_dir: Path, filename: str, column: str, values: list[str | None]
) -> None:
    datasets_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({column: values})
    frame.to_csv(datasets_dir / filename)


def test_init_db_creates_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "database" / "test.db"
    engine = database.create_engine(f"sqlite:///{db_path}", echo=False)

    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", database.sessionmaker(bind=engine))

    database.init_db()

    tables = set(inspect(engine).get_table_names())

    assert db_path.exists()
    assert tables == {"llm_responses", "prompts", "system_prompts"}


def test_get_session_commits_changes(session_factory, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(database, "SessionLocal", session_factory)

    with database.get_session() as session:
        session.add(
            Prompt(
                prompt="Test prompt",
                YTA_NTA="YTA",
                Flipped=False,
                Validation=False,
            )
        )

    with database.get_session() as verification_session:
        assert verification_session.query(Prompt).count() == 1


def test_get_session_rolls_back_on_exception(
    session_factory, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(database, "SessionLocal", session_factory)

    with pytest.raises(RuntimeError, match="boom"):
        with database.get_session() as session:
            session.add(
                Prompt(
                    prompt="Test prompt",
                    YTA_NTA="YTA",
                    Flipped=False,
                    Validation=False,
                )
            )
            raise RuntimeError("boom")

    with database.get_session() as verification_session:
        assert verification_session.query(Prompt).count() == 0


def test_seed_prompts_loads_csv_rows_once(session, datasets_dir: Path):
    _write_csv(datasets_dir, "AITA-YTA.csv", "prompt", ["yta-1", None, "yta-2"])
    _write_csv(datasets_dir, "AITA-NTA-OG.csv", "original_post", ["nta-1"])
    _write_csv(datasets_dir, "AITA-NTA-FLIP.csv", "flipped_story", ["flip-1", "flip-2"])

    inserted = crud.seed_prompts(session, datasets_dir)
    prompts = session.query(Prompt).order_by(Prompt.prompt_id).all()

    assert inserted == 5
    assert [prompt.prompt for prompt in prompts] == [
        "yta-1",
        "yta-2",
        "nta-1",
        "flip-1",
        "flip-2",
    ]
    assert [(prompt.YTA_NTA, prompt.Flipped) for prompt in prompts] == [
        ("YTA", False),
        ("YTA", False),
        ("NTA", False),
        ("NTA", True),
        ("NTA", True),
    ]

    assert crud.seed_prompts(session, datasets_dir) == 0
    assert session.query(Prompt).count() == 5


def test_ensure_system_prompt_is_idempotent(session):
    first = crud.ensure_system_prompt(session, SystemPromptEnum.BASE)
    second = crud.ensure_system_prompt(session, SystemPromptEnum.BASE)

    assert first.system_prompt_id == second.system_prompt_id
    assert session.query(SystemPrompt).count() == 1


def test_get_pending_prompts_filters_by_model_and_system_prompt(session):
    prompt_one = Prompt(
        prompt="Prompt one",
        YTA_NTA="YTA",
        Flipped=False,
        Validation=False,
    )
    prompt_two = Prompt(
        prompt="Prompt two",
        YTA_NTA="NTA",
        Flipped=False,
        Validation=False,
    )
    prompt_three = Prompt(
        prompt="Prompt three",
        YTA_NTA="NTA",
        Flipped=True,
        Validation=False,
    )
    base_prompt = SystemPrompt(system_prompt=SystemPromptEnum.BASE)
    alternate_prompt = SystemPrompt(system_prompt="Different system prompt")

    session.add_all(
        [prompt_one, prompt_two, prompt_three, base_prompt, alternate_prompt]
    )
    session.flush()

    session.add_all(
        [
            LLMResponse(
                prompt_id=prompt_one.prompt_id,
                system_prompt_id=base_prompt.system_prompt_id,
                model=ModelProvider.GEMINI,
                llm_label="YTA",
                response="YTA. done",
            ),
            LLMResponse(
                prompt_id=prompt_two.prompt_id,
                system_prompt_id=base_prompt.system_prompt_id,
                model=ModelProvider.OPEN_AI,
                llm_label="NTA",
                response="NTA. different model",
            ),
            LLMResponse(
                prompt_id=prompt_three.prompt_id,
                system_prompt_id=alternate_prompt.system_prompt_id,
                model=ModelProvider.GEMINI,
                llm_label="NTA",
                response="NTA. different system prompt",
            ),
        ]
    )
    session.flush()

    pending = crud.get_pending_prompts(session, ModelProvider.GEMINI, base_prompt)

    assert [prompt.prompt for prompt in pending] == ["Prompt two", "Prompt three"]


def test_extract_label_handles_case_and_missing_values():
    assert crud._extract_label("yta. because") == "YTA"
    assert crud._extract_label("The answer is NTA here") == "NTA"
    assert crud._extract_label("No verdict present") is None


def test_save_response_persists_row_and_normalized_label(session):
    prompt = Prompt(
        prompt="Prompt text",
        YTA_NTA="YTA",
        Flipped=False,
        Validation=False,
    )
    system_prompt = SystemPrompt(system_prompt=SystemPromptEnum.BASE)
    session.add_all([prompt, system_prompt])
    session.flush()

    row = crud.save_response(
        session,
        prompt,
        system_prompt,
        ModelProvider.CLAUDE,
        "nta. brief explanation",
    )

    session.expire_all()
    stored = session.query(LLMResponse).one()

    assert row.id == stored.id
    assert stored.prompt_id == prompt.prompt_id
    assert stored.system_prompt_id == system_prompt.system_prompt_id
    assert stored.model == ModelProvider.CLAUDE
    assert stored.llm_label == "NTA"
    assert stored.response == "nta. brief explanation"
