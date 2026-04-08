from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from db.crud import ensure_system_prompt
from db.models import Base
from prompts import HONEST_ASSISTANT_PROMPT, SystemPrompt


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
