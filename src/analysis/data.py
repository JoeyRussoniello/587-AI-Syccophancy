"""Data loading helpers for BERTScore analysis."""

import re
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from db.models import LLMResponse, Prompt


def load_llm_crowd_pairs(
    session: Session,
    datasets_dir: Path,
    model: str,
    system_prompt_id: int,
    limit: int | None = None,
) -> list[dict]:
    """Load (llm_reasoning, crowd_comment) pairs for a model × system_prompt.

    Only YTA posts are included — NTA CSVs do not carry top_comment.

    Each returned dict has keys:
        prompt_id, yta_nta, llm_label, llm_reasoning, crowd_reasoning
    """
    yta_df = pd.read_csv(datasets_dir / "AITA-YTA.csv", index_col=0)
    crowd_map: dict[str, str] = dict(zip(yta_df["prompt"], yta_df["top_comment"]))

    rows = (
        session.query(LLMResponse, Prompt)
        .join(Prompt, LLMResponse.prompt_id == Prompt.prompt_id)
        .filter(
            LLMResponse.model == model,
            LLMResponse.system_prompt_id == system_prompt_id,
            Prompt.YTA_NTA == "YTA",
        )
        .all()
    )

    pairs = []
    for response, prompt in rows:
        crowd_comment = crowd_map.get(prompt.prompt)
        if not crowd_comment or pd.isna(crowd_comment):
            continue
        pairs.append(
            {
                "prompt_id": prompt.prompt_id,
                "yta_nta": prompt.YTA_NTA,
                "llm_label": response.llm_label,
                "llm_reasoning": _extract_reasoning(response.response),
                "crowd_reasoning": str(crowd_comment),
            }
        )

    return pairs[:limit] if limit is not None else pairs


def _extract_reasoning(response_text: str) -> str:
    """Strip the leading YTA/NTA verdict token, returning just the explanation."""
    match = re.match(
        r"\b(?:YTA|NTA)\b[.\s,]+(.+)", response_text, re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else response_text
