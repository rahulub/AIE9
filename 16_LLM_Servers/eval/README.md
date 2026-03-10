# RAG Evaluation with RAGAS

## Overview

This folder contains scripts to generate a RAGAS test set and evaluate both the **Fireworks AI** (open-source) and **OpenAI** RAG pipelines.

## Prerequisites

- `OPENAI_API_KEY` — required for test set generation and RAGAS evaluation metrics
- `FIREWORKS_API_KEY` — required for the Fireworks pipeline evaluation
- PDF documents in `data/` (e.g. `cat-health-guide.pdf`)

## Workflow

### 1. Generate RAGAS Test Set

```bash
uv run python eval/generate_testset.py
```

- Loads PDFs from `data/`
- Uses OpenAI (gpt-4o-mini) to synthesize questions with reference answers
- Output: `eval/data/ragas_testset.json`

Optional: set `RAGAS_TESTSET_SIZE` (default 10):

```bash
RAGAS_TESTSET_SIZE=15 uv run python eval/generate_testset.py
```

**If you hit async connection errors** (e.g. `APIConnectionError`, `Event loop is closed` on Python 3.13):

```bash
uv run python eval/generate_testset.py --sync
```

Uses synchronous API calls instead of RAGAS's async pipeline; produces a simpler but compatible test set.

### 2. Run Evaluation

```bash
uv run python eval/evaluate.py
```

- Runs both Fireworks and OpenAI pipelines over the test set

**OpenAI only** (skip Fireworks until deployment is set up):

```bash
uv run python eval/evaluate.py --provider openai
```
- Computes RAGAS metrics: context precision, context recall, faithfulness, answer relevancy, answer correctness
- Output: `eval/data/eval_fireworks.json`, `eval/data/eval_openai.json`

### 3. Provider Configuration

- Default RAG provider: set `RAG_PROVIDER=fireworks` or `RAG_PROVIDER=openai` in `.env`
- The agent tool uses the default provider; the evaluation script runs both pipelines
