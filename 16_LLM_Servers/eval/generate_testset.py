#!/usr/bin/env python3
"""
Generate a RAGAS test set for the LLM Servers RAG application (cat health guide).

Loads PDF documents from the project's data directory (RAG_DATA_DIR or "data").
Outputs to eval/data/:
  - ragas_testset.json  (RAGAS-compatible format for evaluation)

Uses OpenAI (gpt-4o-mini) for test set generation. Requires OPENAI_API_KEY.

Usage:
  python eval/generate_testset.py
  python eval/generate_testset.py --sync   # Fallback: sync-only, avoids async connection issues
  RAGAS_TESTSET_SIZE=15 python eval/generate_testset.py
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DATA = SCRIPT_DIR / "data"

load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env.local", override=True)

# Fallback: load from .env if OPENAI_API_KEY missing (e.g. load_dotenv blocked by sandbox)
if not os.environ.get("OPENAI_API_KEY"):
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        values = dotenv_values(env_path)
        if values.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = values["OPENAI_API_KEY"]

def load_documents():
    """Load PDF documents from the project data directory."""
    from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

    data_dir = os.environ.get("RAG_DATA_DIR", str(DATA_DIR))
    if not Path(data_dir).exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Add PDFs (e.g. cat-health-guide.pdf) to the data folder."
        )
    try:
        directory_loader = DirectoryLoader(
            data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader
        )
        documents = directory_loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load documents from {data_dir}: {e}") from e

    if not documents:
        raise ValueError(
            f"No PDF documents found in {data_dir}. "
            "Add at least one PDF file to generate a test set."
        )
    return documents


def save_json(testset, out_path: Path) -> None:
    """Save test set in RAGAS-compatible JSON format."""
    if isinstance(testset, list):
        records = testset
    else:
        df = testset.to_pandas()
        records = []
        for _, row in df.iterrows():
            rec = {
                "user_input": str(row.get("user_input", "")),
                "reference_contexts": (
                    row.get("reference_contexts", [])
                    if isinstance(row.get("reference_contexts"), list)
                    else [str(row.get("reference_contexts", ""))]
                ),
                "reference": str(row.get("reference", "")),
            }
            if "synthesizer_name" in row:
                rec["synthesizer_name"] = str(row["synthesizer_name"])
            records.append(rec)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _chunk_documents(documents):
    """Chunk documents to match the RAG pipeline (avoids HeadlineSplitter issues)."""
    import tiktoken
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    def _tiktoken_len(text: str) -> int:
        tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
        return len(tokens)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, length_function=_tiktoken_len
    )
    return splitter.split_documents(documents)


def generate_testset_sync(testset_size: int = 10) -> list[dict]:
    """Generate test set using synchronous LLM calls (avoids async connection issues).

    Uses sync invoke() instead of RAGAS async pipeline. Produces RAGAS-compatible
    JSON records: user_input, reference_contexts, reference.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY required. Set it in the project .env file."
        )

    docs = load_documents()
    chunks = _chunk_documents(docs)
    if not chunks:
        raise ValueError("No chunks produced from documents.")
    print(f"Chunked into {len(chunks)} chunks.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=120)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are creating evaluation questions for a cat health RAG system. "
         "Generate exactly one question and one reference answer based on the given context. "
         "Respond ONLY in this format, with no other text:\n"
         "QUESTION: <the question>\nREFERENCE: <the concise reference answer>"),
        ("human", "Context:\n\n{context}"),
    ])

    # Sample chunks; ensure we don't request more than we have
    n = min(testset_size, len(chunks))
    sampled = random.Random(42).sample(chunks, n)

    records = []
    for i, chunk in enumerate(sampled):
        ctx = chunk.page_content[:2000]  # Limit context length
        try:
            chain = prompt | llm
            resp = chain.invoke({"context": ctx})
            text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            print(f"  Chunk {i+1}/{n}: API error: {e}")
            continue

        # Parse QUESTION: ... REFERENCE: ...
        m_q = re.search(r"QUESTION:\s*(.+?)(?=REFERENCE:|$)", text, re.DOTALL | re.I)
        m_a = re.search(r"REFERENCE:\s*(.+?)$", text, re.DOTALL | re.I)
        question = m_q.group(1).strip() if m_q else text.split("\n")[0][:200]
        reference = m_a.group(1).strip() if m_a else ""

        if question:
            records.append({
                "user_input": question,
                "reference_contexts": [ctx],
                "reference": reference or "See context.",
            })
            print(f"  Generated {len(records)}/{n}")

    return records


def generate_ragas_testset(testset_size: int = 10):
    """Generate RAGAS test set with single-hop, multihop direct, multihop abstract.

    Uses pre-chunked documents (generate_with_chunks) to avoid the HeadlineSplitter
    pipeline that requires 'headlines' and can fail on some PDF content.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers.multi_hop import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
    )
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY required for test set generation. "
            "Set it in the project .env file."
        )

    docs = load_documents()
    chunks = _chunk_documents(docs)
    if not chunks:
        raise ValueError("No chunks produced from documents. Check document content.")
    print(f"Chunked into {len(chunks)} chunks (matches RAG pipeline).")

    # Explicit timeout avoids hangs; max_workers=1 reduces async connection pool issues
    llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=120)
    )
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(request_timeout=120))
    gen = TestsetGenerator(llm=llm, embedding_model=emb)

    query_dist = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 1 / 3),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 1 / 3),
        (MultiHopAbstractQuerySynthesizer(llm=llm), 1 / 3),
    ]
    # max_workers=1 reduces async connection pool issues (event loop closed) on Python 3.13
    run_config = RunConfig(max_workers=1, timeout=300)
    testset = gen.generate_with_chunks(
        chunks,
        testset_size=testset_size,
        query_distribution=query_dist,
        run_config=run_config,
    )
    return testset


def main():
    parser = argparse.ArgumentParser(
        description="Generate RAGAS test set for RAG evaluation."
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use sync-only generation (avoids async connection issues on Python 3.13)",
    )
    args = parser.parse_args()
    testset_size = int(os.environ.get("RAGAS_TESTSET_SIZE", "10"))

    EVAL_DATA.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DATA / "ragas_testset.json"

    if args.sync:
        print("Loading documents...")
        print("Generating test set (sync mode, avoids async issues)...")
        testset = generate_testset_sync(testset_size=testset_size)
        if not testset:
            raise RuntimeError("No test samples generated. Check documents and API.")
    else:
        print("Loading documents...")
        print(
            "Generating RAGAS test set (single-hop + multihop direct + multihop abstract)..."
        )
        testset = generate_ragas_testset(testset_size=testset_size)

    save_json(testset, json_path)
    print(f"Saved: {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
