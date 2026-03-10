#!/usr/bin/env python3
"""
Run RAGAS evaluation for Fireworks and/or OpenAI RAG pipelines.

Usage:
  1. Generate test set first: python eval/generate_testset.py [--sync]
  2. Run evaluation: python eval/evaluate.py
  3. OpenAI only: python eval/evaluate.py --provider openai

Output: eval/data/eval_fireworks.json, eval/data/eval_openai.json

Requires: OPENAI_API_KEY and ragas_testset.json. FIREWORKS_API_KEY only for --provider fireworks.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*ragas.metrics.*")

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVAL_DATA = SCRIPT_DIR / "data"

# Add project root so we can import app.rag
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local", override=True)


def _safe_float(v):
    """Convert score to float, or None if NaN."""
    try:
        f = float(v)
        return None if f != f else f  # NaN != NaN
    except (TypeError, ValueError):
        return None


def run_evaluation(provider: str) -> dict:
    """Run RAGAS evaluation for the given provider ("fireworks" or "openai")."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from app.rag import invoke_rag

    testset_path = EVAL_DATA / "ragas_testset.json"
    if not testset_path.exists():
        print(f"Error: {testset_path} not found. Run generate_testset.py first.")
        sys.exit(1)

    with open(testset_path) as f:
        test_records = json.load(f)

    # Collect eval samples by running the RAG pipeline
    eval_samples = []
    for rec in test_records:
        question = rec.get("user_input", "")
        if isinstance(question, list):
            question = question[-1] if question else ""
        if isinstance(question, dict) and "content" in question:
            question = question["content"]
        question = str(question)

        out = invoke_rag(question, provider=provider, return_context=True)
        eval_samples.append({
            "user_input": question,
            "retrieved_contexts": out.get("retrieved_contexts", []),
            "reference_contexts": rec.get("reference_contexts") or [],
            "response": out.get("response", ""),
            "reference": rec.get("reference", ""),
        })

    samples = [
        SingleTurnSample(
            user_input=s["user_input"],
            retrieved_contexts=s["retrieved_contexts"],
            reference_contexts=s["reference_contexts"],
            response=s["response"],
            reference=s["reference"],
        )
        for s in eval_samples
    ]
    dataset = EvaluationDataset(samples=samples)
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]

    # RAGAS needs embeddings for some metrics (e.g. answer_relevancy)
    from langchain_openai import OpenAIEmbeddings

    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    result = evaluate(dataset, metrics=metrics, embeddings=ragas_embeddings)

    scores = {k: _safe_float(v) for k, v in result._repr_dict.items()}
    return {
        "provider": provider,
        "scores": scores,
        "scores_per_row": result.to_pandas().to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation for RAG pipelines."
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "fireworks", "all"],
        default="all",
        help="Which provider to evaluate: openai, fireworks, or all (default: all)",
    )
    args = parser.parse_args()

    EVAL_DATA.mkdir(parents=True, exist_ok=True)

    if args.provider == "all":
        providers = ["fireworks", "openai"]
    else:
        providers = [args.provider]

    for provider in providers:
        key = "FIREWORKS_API_KEY" if provider == "fireworks" else "OPENAI_API_KEY"
        if not os.environ.get(key):
            print(f"Skipping {provider}: {key} not set.")
            continue

        print(f"\nEvaluating {provider} pipeline...")
        try:
            out = run_evaluation(provider)
            out_path = EVAL_DATA / f"eval_{provider}.json"
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved: {out_path}")
            print("Scores:", out["scores"])
        except Exception as e:
            print(f"Error evaluating {provider}: {e}")
            raise


if __name__ == "__main__":
    main()
