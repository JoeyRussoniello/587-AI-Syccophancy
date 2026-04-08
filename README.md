# LLM Sycophancy Project

This repository supports our DS 587 research project on sycophancy in large language models: the tendency for a model to agree with a user even when the available evidence suggests it should push back. The broader motivation, laid out in [resources/ProjectProposal.md](resources/ProjectProposal.md), is not just model quality in the abstract. It is the social and safety risk that highly agreeable systems can reinforce bad judgments, confirmation bias, and in extreme cases delusional or parasocial reasoning.

The project is aimed at measuring those behaviors across popular models, prompt framings, and system-prompt interventions. This repository contains the experiment infrastructure for collecting and organizing those model responses.

## Research Direction

The proposal frames this project around a few connected questions:

- how often popular LLMs produce sycophantic responses
- whether prompt framing changes those rates
- whether system prompts can reduce sycophancy without breaking usefulness
- whether multi-turn or longer-context settings make the problem worse
- whether sycophantic failures can later be analyzed or predicted from prompt/response patterns
- whether prompts that result in sycophantic responses share any semantic/embedding similarities

The current starter dataset comes from the AITA data referenced in the proposal, where Reddit posts provide prompts and crowd verdicts provide a practical source of ground-truth labels.

## Project Scope

This repository currently focuses on the experiment pipeline and response corpus infrastructure:

- async model wrappers for Claude, OpenAI, and Gemini in [src/models.py](src/models.py)
- one configurable collection entrypoint in [main.py](main.py)
- SQLite-backed storage for prompts, system prompts, and model responses in [src/db](src/db)
- a base system prompt definition in [src/prompts.py](src/prompts.py)
- automated checks with `pytest`, `Ruff`, and `mypy`

## Current Workflow

The implemented pipeline is:

```text
CSV datasets -> seed prompts into SQLite -> query one configured provider -> store responses and extracted labels
```

At runtime, [main.py](main.py) currently:

1. initializes the SQLite database
2. seeds prompt rows from the CSV files in [datasets](datasets)
3. ensures the configured system prompt exists in the database
4. fetches prompts that do not yet have a response for the chosen model and system prompt
5. calls the selected provider asynchronously with retry logic
6. stores successful responses in `llm_responses`

This gives the project a reproducible way to build the response corpus used throughout the study.

## Dataset And Labeling Approach

The current code uses the three CSV files in [datasets](datasets), which are adapted from the Elephant AITA dataset cited in the proposal. Those files feed the `prompts` table with metadata including:

- the prompt text
- whether the expected crowd verdict is `YTA` or `NTA`
- whether the story is a flipped variant
- whether a row is marked for validation

Model outputs are stored alongside:

- the model identifier
- the system prompt enum name and full prompt text
- the raw response text
- an extracted `YTA` or `NTA` label when one is detectable

That structure supports comparison across crowd judgment, model choice, and alternate prompt or system-prompt conditions.

## Repository Layout

```text
.
├── main.py                 Entry point for collecting model responses
├── src/
│   ├── models.py           Async LLM client wrappers and retry logic
│   ├── prompts.py          System prompt definitions
│   └── db/
│       ├── database.py     SQLite engine and session management
│       ├── crud.py         Prompt and response persistence helpers
│       └── models.py       SQLAlchemy ORM models
├── datasets/               AITA prompt datasets used for collection
├── database/               SQLite database file and SQL queries
├── resources/              Project proposal, bibliography, and course context
├── tests/                  Pytest coverage for current core modules
└── .github/workflows/      CI checks
```

## Database Schema

```text
+---------------------+       +-------------------------+       +---------------------+
| prompts             |       | llm_responses           |       | system_prompts      |
+---------------------+       +-------------------------+       +---------------------+
| prompt_id (PK)      |<------| prompt_id (FK)          |       | system_prompt_id PK |
| prompt              |       | system_prompt_id (FK)   |------>| system_prompt_name  |
| YTA_NTA             |       | id (PK)                 |       | system_prompt       |
| Flipped             |       | model                   |       +---------------------+
| Validation          |       | llm_label               |
+---------------------+       | response                |
                              +-------------------------+
```

## Setup

This project uses `uv` for dependency management.

1. Install `uv`.
2. Sync dependencies:

```bash
uv sync --all-groups
```

1. Create a `.env` file with the API key for the provider you plan to use:

```env
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

Only the key for the selected provider is required for a given run, but the relevant client constructor will validate that it exists.

## Running Collection

Edit the configuration constants near the top of [main.py](main.py):

- `SYSTEM_PROMPT`
- `PROVIDERS`
- `MAX_RETRIES`
- `NUM_RESPONSES`
- `MAX_WORKERS_PER_MODEL`

Then run:

```bash
uv run main.py
```

Responses are written to [database/sycophancy.db](database/sycophancy.db).

## Quality Checks

The repository currently uses:

- pytest for tests
- Ruff for linting
- mypy for type checking

Run them locally with:

```bash
uv run pytest
uv run ruff check .
uv run mypy src main.py
```

CI is configured in [.github/workflows/ci.yml](.github/workflows/ci.yml).
