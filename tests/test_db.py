from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from db import database
from db.crud import ensure_system_prompt
from db.models import Base
from prompts import BASE_SYSTEM_PROMPT, HONEST_ASSISTANT_PROMPT, SystemPrompt


def test_ensure_system_prompt_updates_existing_named_prompt(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'crud.db'}")
    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        session.execute(
            text(
                """
                INSERT INTO system_prompts (system_prompt_name, system_prompt)
                VALUES ('HONEST_ASSISTANT', :prompt_text)
                """
            ),
            {"prompt_text": "outdated prompt text"},
        )
        session.commit()

        row = ensure_system_prompt(session, SystemPrompt.HONEST_ASSISTANT)
        session.commit()

        assert row.system_prompt_name == "HONEST_ASSISTANT"
        assert row.system_prompt == HONEST_ASSISTANT_PROMPT


def test_migrate_system_prompt_names_backfills_existing_rows(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE system_prompts (
                    system_prompt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_prompt TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO system_prompts (system_prompt) VALUES (:prompt_text)"
            ),
            {"prompt_text": BASE_SYSTEM_PROMPT},
        )

    database._migrate_system_prompt_names(engine)

    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT system_prompt_name, system_prompt FROM system_prompts LIMIT 1"
            )
        ).one()

    assert row.system_prompt_name == "BASE"
    assert row.system_prompt == BASE_SYSTEM_PROMPT