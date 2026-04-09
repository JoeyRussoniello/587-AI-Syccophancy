"""CRUD helpers for the sycophancy database."""

import logging
import re
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from db.models import LLMResponse, Prompt, SystemPrompt
from prompts import SystemPrompt as SystemPromptEnum

logger = logging.getLogger(__name__)
# Maps each CSV file to (prompt_column, YTA_NTA value, Flipped flag)
_CSV_SPEC: list[tuple[str, str, str, bool]] = [
    ("AITA-YTA.csv", "prompt", "YTA", False),
    ("AITA-NTA-OG.csv", "original_post", "NTA", False),
    ("AITA-NTA-FLIP.csv", "flipped_story", "NTA", True),
]


def seed_prompts(session: Session, datasets_dir: Path) -> int:
    """Load prompts from CSVs into the prompts table (skips if already seeded).

    Returns the number of new rows inserted.
    """
    if session.query(Prompt).first() is not None:
        return 0

    count = 0
    for filename, col, yta_nta, flipped in _CSV_SPEC:
        df = pd.read_csv(datasets_dir / filename, index_col=0)
        for text in df[col].dropna():
            session.add(
                Prompt(
                    prompt=text,
                    YTA_NTA=yta_nta,
                    Flipped=flipped,
                    Validation=False,
                )
            )
            count += 1
    logger.info("Seeded %d new prompts from CSVs in %s", count, datasets_dir)
    session.flush()
    return count


def ensure_system_prompt(session: Session, text: SystemPromptEnum) -> SystemPrompt:
    """Return the SystemPrompt for `text`, creating it if necessary."""
    row = session.query(SystemPrompt).filter_by(system_prompt_name=text.name).first()
    if row is not None:
        if row.system_prompt != str(text):
            setattr(row, "system_prompt", str(text))
            session.flush()
        return row

    row = SystemPrompt(system_prompt_name=text.name, system_prompt=str(text))
    session.add(row)
    session.flush()
    logger.info("Ensured system prompt exists for %s", text.name)
    return row


def get_pending_prompts(
    session: Session,
    model: str,
    system_prompt: SystemPrompt,
    yta_only: bool = False,
) -> list[Prompt]:
    """Return prompts that do NOT yet have a response for this model + system prompt."""
    already_done = (
        session.query(LLMResponse.prompt_id)
        .filter(
            LLMResponse.model == model,
            LLMResponse.system_prompt_id == system_prompt.system_prompt_id,
        )
        .scalar_subquery()
    )
    unprocessed = session.query(Prompt).filter(~Prompt.prompt_id.in_(already_done))
    if yta_only:
        unprocessed = unprocessed.filter(Prompt.YTA_NTA == "YTA")
    result = unprocessed.all()
    logger.info(
        "Found %d pending prompts for model %s and system prompt %s (YTA only: %s)",
        len(result),
        model,
        system_prompt.system_prompt_name,
        yta_only,
    )
    return result


def _extract_label(response_text: str) -> str | None:
    """Extract YTA/NTA label from the response text."""
    match = re.search(r"\b(YTA|NTA)\b", response_text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def save_response(
    session: Session,
    prompt_id: int,
    system_prompt_id: int,
    model: str,
    response_text: str,
) -> LLMResponse:
    """Insert a single LLM response row and flush it to the database."""
    row = LLMResponse(
        prompt_id=prompt_id,
        system_prompt_id=system_prompt_id,
        model=model,
        llm_label=_extract_label(response_text),
        response=response_text,
    )
    session.add(row)
    session.flush()
    return row


def save_responses_bulk(
    session: Session,
    entries: list[tuple[Prompt, str]],
    system_prompt: SystemPrompt,
    model: str,
) -> list[LLMResponse]:
    """Insert multiple LLM response rows in a single flush.

    `entries` is a list of (prompt, response_text) pairs.
    Returns the list of inserted LLMResponse objects.
    """
    rows = [
        LLMResponse(
            prompt_id=prompt.prompt_id,
            system_prompt_id=system_prompt.system_prompt_id,
            model=model,
            llm_label=_extract_label(response_text),
            response=response_text,
        )
        for prompt, response_text in entries
    ]
    session.add_all(rows)
    session.flush()
    logger.info(
        "Saved %d responses for model %s and system prompt %s",
        len(rows),
        model,
        system_prompt.system_prompt_name,
    )
    return rows
