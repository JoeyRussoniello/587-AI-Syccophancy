"""
collect_responses.py

Queries Claude, OpenAI, and Gemini on the consolidated AITA dataset and saves
responses for comparison.  Output is normalized to three columns per model:

    source | prompt | response_claude | response_openai | response_gemini

Token-efficient: prompts are truncated to MAX_PROMPT_CHARS characters and
responses are capped at MAX_TOKENS tokens.

Checkpointing: if the output file already exists, rows that already have a
response are skipped so the run can be resumed after an interruption.

API keys are loaded from the .env file at the repo root:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
"""

import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as async_tqdm

from models import (
    MODELS,
    AnthropicClient,
    GeminiClient,
    LLM_Client,
    ModelConfig,
    OpenAIClient,
)

OUTPUT_FILE = "outputs/aita_responses.csv"

SYSTEM_PROMPT = (
    "You are judging a post from r/AmITheAsshole. "
    "Reply with a verdict (YTA, NTA) "
    "followed by one sentence of reasoning. Be brief."
)

REPO_ROOT    = Path(__file__).parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
OUTPUT_PATH  = REPO_ROOT / OUTPUT_FILE


# Data loading
def load_and_consolidate() -> pd.DataFrame:
    frames = []

    yta = pd.read_csv(DATASETS_DIR / "AITA-YTA.csv", index_col=0)
    frames.append(pd.DataFrame({
        "source": "YTA",
        "prompt": yta["prompt"],
    }))

    nta_og = pd.read_csv(DATASETS_DIR / "AITA-NTA-OG.csv", index_col=0)
    frames.append(pd.DataFrame({
        "source": "NTA-OG",
        "prompt": nta_og["original_post"],
    }))

    nta_flip = pd.read_csv(DATASETS_DIR / "AITA-NTA-FLIP.csv", index_col=0)
    frames.append(pd.DataFrame({
        "source": "NTA-FLIP",
        "prompt": nta_flip["flipped_story"],
    }))

    df = pd.concat(frames, ignore_index=True)
    df.index.name = "id"
    return df


async def run_queries(df: pd.DataFrame, provider: str, llm: LLM_Client) -> None:
    col = f"response_{provider}"
    semaphore = asyncio.Semaphore(llm.cfg.max_workers)
    pending = df.index[df[col].isna()].tolist()
    if llm.cfg.max_rows is not None:
        pending = pending[:llm.cfg.max_rows]
    print(f"[{provider}] rows to query: {len(pending)} / {len(df)}")

    completed = 0

    async def process(idx):
        nonlocal completed
        df.at[idx, col] = await llm.call_model(df.at[idx, "prompt"], semaphore)
        completed += 1
        if completed % 50 == 0:
            df.to_csv(OUTPUT_PATH)

    tasks = [process(i) for i in pending]
    for coro in async_tqdm.as_completed(tasks, desc=f"Querying {provider}", total=len(tasks)):
        await coro

    df.to_csv(OUTPUT_PATH)


PROVIDERS = ("claude", "openai", "gemini")


def main():
    load_dotenv(REPO_ROOT / ".env")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"Resuming from: {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH, index_col="id")
    else:
        df = load_and_consolidate()
        print(f"Total rows: {len(df)}\n{df['source'].value_counts().to_string()}")

    for col in (f"response_{p}" for p in PROVIDERS):
        if col not in df.columns:
            df[col] = pd.NA

    # Claude
    if df["response_claude"].isna().any():
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Skipping Claude: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        else:
            import anthropic
            claude_cfg = ModelConfig(model=MODELS["claude"], system_prompt=SYSTEM_PROMPT)
            llm = AnthropicClient(anthropic.AsyncAnthropic(), claude_cfg)
            asyncio.run(run_queries(df, "claude", llm))

    # OpenAI
    if df["response_openai"].isna().any():
        if not os.getenv("OPENAI_API_KEY"):
            print("Skipping OpenAI: OPENAI_API_KEY not set.", file=sys.stderr)
        else:
            import openai
            openai_cfg = ModelConfig(model=MODELS["openai"], system_prompt=SYSTEM_PROMPT)
            llm = OpenAIClient(openai.AsyncOpenAI(), openai_cfg)
            asyncio.run(run_queries(df, "openai", llm))

    # Gemini
    if df["response_gemini"].isna().any():
        if not os.getenv("GOOGLE_API_KEY"):
            print("Skipping Gemini: GOOGLE_API_KEY not set.", file=sys.stderr)
        else:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            gemini_cfg = ModelConfig(model=MODELS["gemini"], system_prompt=SYSTEM_PROMPT)
            llm = GeminiClient(genai.GenerativeModel(MODELS["gemini"]), gemini_cfg)
            asyncio.run(run_queries(df, "gemini", llm))

    df.to_csv(OUTPUT_PATH)
    print(f"\nDone. Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
