#!/usr/bin/env python3
"""
Generate a RAGAS test set for the AI Realtor home inspection chatbot.

Loads documents from:
  1. backend/data/*.pdf (red flag guidelines) if present
  2. Otherwise: eval/data/home_inspection_reference.txt

Outputs to eval/data/:
  - ragas_testset.json  (RAGAS-compatible format)
  - ragas_testset.pdf   (human-readable report)

Includes: single-hop, multihop direct (specific), multihop abstract queries.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DATA = PROJECT_ROOT / "backend" / "data"
EVAL_DATA = SCRIPT_DIR / "data"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(SCRIPT_DIR / ".env.local", override=True)
load_dotenv(PROJECT_ROOT / "backend" / ".env")
load_dotenv(PROJECT_ROOT / "backend" / ".env.local", override=True)


def load_documents():
    """Load docs from PDFs in backend/data or fallback to eval/data reference text."""
    from langchain_community.document_loaders import TextLoader, PyPDFLoader

    docs = []
    if BACKEND_DATA.exists():
        for p in BACKEND_DATA.glob("*.pdf"):
            docs.extend(PyPDFLoader(str(p)).load())
    if not docs:
        ref_file = EVAL_DATA / "home_inspection_reference.txt"
        if ref_file.exists():
            docs = TextLoader(str(ref_file)).load()
        else:
            raise FileNotFoundError(
                f"No documents found. Add PDFs to {BACKEND_DATA} or ensure {ref_file} exists."
            )
    if not docs:
        raise ValueError("No documents loaded. Cannot generate test set.")
    return docs


def save_json(testset, out_path: Path) -> None:
    """Save test set in RAGAS-compatible JSON format."""
    df = testset.to_pandas()
    records = []
    for _, row in df.iterrows():
        rec = {
            "user_input": str(row.get("user_input", "")),
            "reference_contexts": row.get("reference_contexts", [])
            if isinstance(row.get("reference_contexts"), list)
            else [str(row.get("reference_contexts", ""))],
            "reference": str(row.get("reference", "")),
        }
        if "synthesizer_name" in row:
            rec["synthesizer_name"] = str(row["synthesizer_name"])
        records.append(rec)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _safe_text(s: str) -> str:
    """Replace Unicode chars that Helvetica cannot render."""
    return (
        s.replace("\u2014", "-")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def save_pdf(testset, out_path: Path) -> None:
    """Export test set as a human-readable PDF report."""
    from fpdf import FPDF

    df = testset.to_pandas()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _safe_text("AI Realtor - RAGAS Test Set"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Total samples: {len(df)}", ln=True)
    pdf.ln(8)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, f"Sample {i}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        q = _safe_text(str(row.get("user_input", ""))[:500])
        pdf.multi_cell(0, 5, f"Question: {q}")
        pdf.cell(0, 4, "", ln=True)
        ref = _safe_text(str(row.get("reference", ""))[:800])
        pdf.multi_cell(0, 5, f"Reference Answer: {ref}")
        pdf.ln(4)
        if pdf.get_y() > 260:
            pdf.add_page()

    pdf.output(str(out_path))


def generate_ragas_testset(testset_size: int = 10):
    """Generate RAGAS test set with single-hop, multihop direct, multihop abstract."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.testset.synthesizers.multi_hop import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY required. Set in eval/.env or backend/.env.local")

    docs = load_documents()
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    gen = TestsetGenerator(llm=llm, embedding_model=emb)

    query_dist = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 1 / 3),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 1 / 3),
        (MultiHopAbstractQuerySynthesizer(llm=llm), 1 / 3),
    ]
    testset = gen.generate_with_langchain_docs(
        docs, testset_size=testset_size, query_distribution=query_dist
    )
    return testset


def main():
    testset_size = int(os.environ.get("RAGAS_TESTSET_SIZE", "10"))
    print("Loading documents...")
    print("Generating RAGAS test set (single-hop + multihop direct + multihop abstract)...")
    testset = generate_ragas_testset(testset_size=testset_size)

    EVAL_DATA.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DATA / "ragas_testset.json"
    pdf_path = EVAL_DATA / "ragas_testset.pdf"

    save_json(testset, json_path)
    save_pdf(testset, pdf_path)

    print(f"Saved JSON: {json_path}")
    print(f"Saved PDF:  {pdf_path}")
    print("Done.")


if __name__ == "__main__":
    main()
