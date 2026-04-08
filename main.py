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
import logging
from pathlib import Path

from db.database import init_db
from models import Model
from prompts import SystemPrompt
from response_collection import ResponseCollectionConfig, get_responses_for_models

#########################################################
# CONFIG - Change these variables to change the experiment settings
SYSTEM_PROMPT = SystemPrompt.HONEST_ASSISTANT
MODELS = [
    Model.GEMINI,
    Model.GPT_5_4_MINI,
]
MAX_RETRIES = 3
MAX_WORKERS_PER_MODEL = 5
LOGGING_LEVEL = logging.INFO

# Or None to pull all. By default will ONLY generate responses for prompts that haven't been processed already
MAX_RESPONSES = None

# Set to True to only make API calls and not append response records to database - used for testing AI connections
DRY_RUN = False

# Set to True to only get responses for 'YTA' prompts to get non-control group sycophancy rates.
YTA_ONLY = True
#########################################################

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).parent
DATASETS_DIR = REPO_ROOT / "datasets"
LOGS_DIR = REPO_ROOT / "logs"
LOG_FILE = LOGS_DIR / "app.log"

COLLECTION_CONFIG = ResponseCollectionConfig(
    system_prompt=SYSTEM_PROMPT,
    datasets_dir=DATASETS_DIR,
    max_retries=MAX_RETRIES,
    max_rows=MAX_RESPONSES,
    max_workers_per_model=MAX_WORKERS_PER_MODEL,
    yta_only=YTA_ONLY,
    dry_run=DRY_RUN,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=LOGGING_LEVEL,
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
)


async def main() -> None:
    """Run the configured model collection job."""

    await get_responses_for_models(MODELS, COLLECTION_CONFIG)


if __name__ == "__main__":
    init_db()
    asyncio.run(main())
