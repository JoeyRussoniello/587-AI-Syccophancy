"""CRUD helpers for the sycophancy database."""

import logging
import re
from functools import wraps
from pathlib import Path
from typing import Callable, Concatenate, ParamSpec, TypeVar

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from db.database import get_session
from db.models import LLMResponse, Prompt, SystemPrompt
from prompts import SystemPrompt as SystemPromptEnum

logger = logging.getLogger(__name__)
# Maps each CSV file to (prompt_column, YTA_NTA value, Flipped flag, top_comment_column|None)
_CSV_SPEC: list[tuple[str, str, str, bool, str | None]] = [
    ("AITA-YTA.csv", "prompt", "YTA", False, "top_comment"),
    ("AITA-NTA-OG.csv", "original_post", "NTA", False, None),
    ("AITA-NTA-FLIP.csv", "flipped_story", "NTA", True, None),
]


def migrate_add_top_comment(session: Session, datasets_dir: Path) -> None:
    """Add the top_comment column to prompts and backfill from AITA-YTA.csv.

    Safe to call repeatedly -- skips rows that already have a top_comment.
    """
    columns = [c["name"] for c in inspect(session.bind).get_columns("prompts")]
    if "top_comment" not in columns:
        session.execute(text("ALTER TABLE prompts ADD COLUMN top_comment TEXT"))
        logger.info("Added top_comment column to prompts table")

    # Build a lookup from prompt text -> top_comment (normalize line endings)
    df = pd.read_csv(datasets_dir / "AITA-YTA.csv", index_col=0)
    comment_by_prompt = {
        p.replace("\r\n", "\n"): c for p, c in zip(df["prompt"], df["top_comment"])
    }

    # Backfill YTA rows that still have a NULL top_comment
    rows = (
        session.query(Prompt)
        .filter(Prompt.YTA_NTA == "YTA", Prompt.top_comment.is_(None))
        .all()
    )
    count = 0
    for prompt in rows:
        comment = comment_by_prompt.get(prompt.prompt.replace("\r\n", "\n"))
        if comment is not None:
            prompt.top_comment = comment
            count += 1
    session.flush()
    logger.info("Backfilled top_comment for %d YTA prompts", count)


def seed_prompts(session: Session, datasets_dir: Path) -> int:
    """Load prompts from CSVs into the prompts table (skips if already seeded).

    Returns the number of new rows inserted.
    """
    if session.query(Prompt).first() is not None:
        return 0

    count = 0
    for filename, col, yta_nta, flipped, top_comment_col in _CSV_SPEC:
        df = pd.read_csv(datasets_dir / filename, index_col=0)
        for _, row in df.dropna(subset=[col]).iterrows():
            session.add(
                Prompt(
                    prompt=row[col],
                    top_comment=row.get(top_comment_col) if top_comment_col else None,
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
    entries: list[tuple[int, str]],
    system_prompt_id: int,
    model: str,
) -> list[LLMResponse]:
    """Insert multiple LLM response rows in a single flush.

    `entries` is a list of (prompt_id, response_text) pairs.
    Returns the list of inserted LLMResponse objects.
    """
    rows = [
        LLMResponse(
            prompt_id=prompt_id,
            system_prompt_id=system_prompt_id,
            model=model,
            llm_label=_extract_label(response_text),
            response=response_text,
        )
        for prompt_id, response_text in entries
    ]
    session.add_all(rows)
    session.flush()
    logger.info(
        "Saved %d responses for model %s and system prompt %s",
        len(rows),
        model,
        system_prompt_id,
    )
    return rows


# Arbitrary input and output parameter types for the decorator
R = TypeVar("R")
P = ParamSpec("P")


def contained_db_function(func: Callable[Concatenate[Session, P], R]) -> Callable[P, R]:
    """Decorator to wrap functions that need a short-lived database session"""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with get_session() as session:
            return func(session, *args, **kwargs)

    return wrapper


@contained_db_function
def get_all_prompts(session: Session) -> pd.DataFrame:
    prompts = session.query(Prompt).all()
    return pd.DataFrame(
        [
            {
                "prompt_id": p.prompt_id,
                "prompt": p.prompt,
                "top_comment": p.top_comment,
                "YTA_NTA": p.YTA_NTA,
                "Flipped": p.Flipped,
                "Validation": p.Validation,
            }
            for p in prompts
        ]
    )


@contained_db_function
def get_all_responses(session: Session) -> pd.DataFrame:
    responses = session.query(LLMResponse).all()
    return pd.DataFrame(
        [
            {
                "response_id": r.id,
                "prompt_id": r.prompt_id,
                "system_prompt_id": r.system_prompt_id,
                "model": r.model,
                "llm_label": r.llm_label,
                "response": r.response,
            }
            for r in responses
        ]
    )


@contained_db_function
def get_all_system_prompts(session: Session) -> pd.DataFrame:
    system_prompts = session.query(SystemPrompt).all()
    return pd.DataFrame(
        [
            {
                "system_prompt_id": sp.system_prompt_id,
                "system_prompt_name": sp.system_prompt_name,
                "system_prompt": sp.system_prompt,
            }
            for sp in system_prompts
        ]
    )


if __name__ == "__main__":
    sys_prompts = get_all_system_prompts()
    prompts = get_all_prompts()
    responses = get_all_responses()

    print(
        f"Got {len(sys_prompts)} system prompts, {len(prompts)} prompts, and {len(responses)} responses"
    )
