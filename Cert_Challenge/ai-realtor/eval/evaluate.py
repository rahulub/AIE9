#!/usr/bin/env python3
"""
Run RAGAS evaluation for AI Realtor. Use as standalone script to avoid
Python 3.14 + Jupyter asyncio timeout issues.

Usage: python evaluate.py
Output: eval/data/eval_results.json
"""

import json
import os
import sys
import warnings
from pathlib import Path

# Suppress deprecation warnings for ragas.metrics (use ragas.metrics.collections in v1.0)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*ragas.metrics.*")

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DATA = PROJECT_ROOT / "backend" / "data"
EVAL_DATA = SCRIPT_DIR / "data"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(SCRIPT_DIR / ".env.local", override=True)
load_dotenv(PROJECT_ROOT / "backend" / ".env")
load_dotenv(PROJECT_ROOT / "backend" / ".env.local", override=True)


def main():
    import numpy as np
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    )

    EVAL_DATA.mkdir(parents=True, exist_ok=True)
    testset_path = EVAL_DATA / "ragas_testset.json"
    if not testset_path.exists():
        print(f"Error: {testset_path} not found. Run generate_testset.py first.")
        sys.exit(1)

    with open(testset_path) as f:
        test_records = json.load(f)

    # Load documents and build retriever
    docs = []
    if BACKEND_DATA.exists():
        for p in BACKEND_DATA.glob("*.pdf"):
            docs.extend(PyPDFLoader(str(p)).load())
    if not docs:
        ref_file = EVAL_DATA / "home_inspection_reference.txt"
        if ref_file.exists():
            docs = TextLoader(str(ref_file)).load()
    if not docs:
        print("Error: No documents for RAG.")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    chunk_texts = [c.page_content for c in chunks]
    chunk_vectors = np.array(embeddings.embed_documents(chunk_texts))

    def retrieve(query: str, k: int = 4):
        qv = np.array(embeddings.embed_query(query)).reshape(1, -1)
        sims = np.dot(chunk_vectors, qv.T).flatten() / (
            np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(qv)
        )
        idx = np.argsort(sims)[::-1][:k]
        return [chunks[i] for i in idx]

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    RAG_PROMPT = """Answer the question based only on the following context. Be concise.

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rag_chain = (
        {"context": lambda q: format_docs(retrieve(q)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Collect eval samples
    eval_samples = []
    for rec in test_records:
        question = rec.get("user_input", "")
        if isinstance(question, list):
            question = question[-1] if question else ""
        if isinstance(question, dict) and "content" in question:
            question = question["content"]
        question = str(question)

        retrieved_docs = retrieve(question)
        retrieved_contexts = [d.page_content for d in retrieved_docs]
        response = rag_chain.invoke(question)

        eval_samples.append({
            "user_input": question,
            "retrieved_contexts": retrieved_contexts,
            "reference_contexts": rec.get("reference_contexts") or [],
            "response": response,
            "reference": rec.get("reference", ""),
        })

    # Build dataset and evaluate
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

    # Use LangchainEmbeddingsWrapper to satisfy RAGAS metrics that call embed_query
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    result = evaluate(dataset, metrics=metrics, embeddings=ragas_embeddings)

    # Save results
    out = {
        "scores": {k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                  for k, v in result._repr_dict.items()},
        "scores_per_row": result.to_pandas().to_dict(orient="records"),
    }
    out_path = EVAL_DATA / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")
    print("Scores:", result._repr_dict)

    # Optionally print per-sample breakdown (like the notebook's DataFrame display)
    if "--per-sample" in sys.argv:
        import pandas as pd
        df = pd.DataFrame(out.get("scores_per_row", []))
        if not df.empty:
            # Show key columns for readability
            cols = [c for c in ["user_input", "response", "reference"] if c in df.columns]
            if cols:
                print("\nPer-sample (excerpt):")
                print(df[cols].to_string(max_colwidth=60))


if __name__ == "__main__":
    main()
