"""
Run BERTScore F1 comparisons for each model and system_prompt combination.

Loads up to PAIR_LIMIT (LLM response, crowd top_comment) pairs per group,
scores them with BERTScore, and prints a summary table.

Usage:
    python bert_main.py
"""

import logging
from pathlib import Path

import pandas as pd
import transformers

from analysis.bert_score import BERTScoreScorer
from analysis.data import load_llm_crowd_pairs
from db.crud import ensure_system_prompt
from db.database import get_session
from models import Model
from prompts import SystemPrompt

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

######################################################################################################################
# CONFIG - Change these variables to customise the BERTScore run
MODELS = [Model.CLAUDE]
SYSTEM_PROMPTS = [SystemPrompt.BASE]
PAIR_LIMIT = 20  # None = all (takes a while)
######################################################################################################################

DATASETS_DIR = Path(__file__).parent / "datasets"


def main() -> None:
    scorer = BERTScoreScorer()
    rows: list[dict] = []

    for system_prompt in SYSTEM_PROMPTS:
        with get_session() as session:
            sp_row = ensure_system_prompt(session, system_prompt)

            for model in MODELS:
                pairs = load_llm_crowd_pairs(
                    session,
                    DATASETS_DIR,
                    str(model),
                    sp_row.system_prompt_id,
                    limit=PAIR_LIMIT,
                )
                if not pairs:
                    print(f"  [skip] {model} / {system_prompt.name} — no pairs found")
                    continue

                f1_scores = scorer.score_pairs(
                    references=[p["crowd_reasoning"] for p in pairs],
                    candidates=[p["llm_reasoning"] for p in pairs],
                )

                mean_f1 = float(f1_scores.mean())
                rows.append(
                    {
                        "model": str(model),
                        "system_prompt": system_prompt.name,
                        "n_pairs": len(pairs),
                        "mean_f1": round(mean_f1, 4),
                    }
                )
                print(
                    f"  scored {len(pairs):>3} pairs  {str(model):<30} {system_prompt.name}"
                )

    if not rows:
        print("No data. Run response collection first.")
        return

    df = pd.DataFrame(rows).pivot(
        index="model", columns="system_prompt", values="mean_f1"
    )
    df.columns.name = None
    df.index.name = "model"

    print("\n=== Mean BERTScore F1 by model and system_prompt ===\n")
    print(df.to_string())
    print()


if __name__ == "__main__":
    main()
