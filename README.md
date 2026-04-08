# LLM Sycophancy & Resistance Research

This repository contains code and data for research tracking large language model (LLM) sycophancy and resistance. The project collects prompts, model responses, metadata, and human annotations to measure and analyze how models conform or resist across contexts and system prompts.

## Purpose

- Quantify and analyze sycophantic vs. resistant behaviors in LLM outputs
- Provide reproducible datasets and experiments for evaluating mitigation strategies
- Support research into evaluation metrics, annotation procedures, and model behavior across prompts and system instructions

## Repository layout

```bash
- src/          -- experiment code, data ingestion, evaluation scripts
- datasets/     -- raw reddit datasets from Elephant (LINK LATER) (CSV/JSON)
- database/     -- SQLite database files
- resources/    -- prompts, templates, and annotation guides
- tests/        -- unit and integration tests (not in place yet)
- main.py       -- entrypoint for running experiments
```

## Architecture
The system is an ETL + annotation pipeline that: collects model responses (via API or local runs), stores results in a relational database, runs automated analyses, and exports data for manual annotation. Analyses and visualizations are produced from aggregated tables.

**High-level flow:**
Prompt source (CSV) -> Model run (API/local) -> Store Response -> Automated metrics -> Human annotation -> Analysis

Components:
- Ingest: runs experiments and stores raw responses and metadata
- Storage: relational DB for structured storage and provenance
- Automated Metrics: Semantic similarity analysis using SBert and other semantic sentence similarity metrics
- Annotation: Graphing, quantiative analysis
- Analysis: scripts/notebooks computing metrics and producing plots

## Relational database schema (ASCII)

Tables and key columns (PK = primary key, FK = foreign key):

```bash
+---------------------+       +-------------------------+       +---------------------+
|   prompts           |       |   llm_responses         |       |   system_prompts    |
+---------------------+       +-------------------------+       +---------------------+
| prompt_id (PK)      |<------| prompt_id (FK)          |       | system_prompt_id(PK)|
| prompt              |       | system_prompt_id (FK)   |------>| system_prompt       |
| YTA_NTA             |       | id (PK)                 |       +---------------------+
| Flipped             |       | model                   |
| Validation          |       | llm_label               |
+---------------------+       | response                |
                              +-------------------------+
```

## How to run
- Configure constants `main.py` for the desired LLM (Claude, OpenAI, or Gemini), number of responses, token limits, and retry strategies
- Use uv to run scripts and tests (project preference):
  - `uv run main.py`
  - `uv run pytest` (tests not implemented yet)
- See `src/` for experiment entrypoints and `src/db` for database schemas and CRUD operations
