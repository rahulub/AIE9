"""
Ingestion routes:

  POST /api/ingest        → user uploads an inspection report PDF
                            → stored in USER_COLLECTION (user_reports)

  POST /api/ingest/local  → re-trigger reference doc ingestion via API

  auto_ingest_data_dir()  → called on startup, ingests backend/data/ PDFs
                            → stored in REFERENCE_COLLECTION (reference_guidelines)
"""

import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pypdf import PdfReader
import io

from rag.chunker import chunk_by_pages
from rag.embedder import embed_batch
from rag.store import (
    ensure_collection,
    upsert_chunks,
    source_exists,
    REFERENCE_COLLECTION,
    USER_COLLECTION,
)

router = APIRouter()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def extract_pages_from_pdf(file_bytes: bytes) -> list[str]:
    """Returns a list of strings — one per page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    return [page.extract_text() or "" for page in reader.pages]


async def ingest_pdf_bytes(
    file_bytes: bytes,
    filename: str,
    collection: str,
    skip_if_exists: bool = True,
) -> dict:
    """
    Core ingestion pipeline for either collection:
    1. Skip if already ingested (deduplication)
    2. Extract text page-by-page (preserving page numbers)
    3. Chunk with overlap
    4. Embed in one batch API call
    5. Upsert into the specified Qdrant collection
    """
    if skip_if_exists and await source_exists(filename, collection):
        return {"filename": filename, "pages": 0, "chunks_ingested": 0, "skipped": True}

    pages = extract_pages_from_pdf(file_bytes)
    if not any(p.strip() for p in pages):
        raise ValueError(f"Could not extract text from '{filename}'.")

    chunks = chunk_by_pages(pages, chunk_size=500, overlap=50)
    if not chunks:
        raise ValueError(f"No chunks produced from '{filename}'.")

    texts = [c["text"] for c in chunks]
    vectors = await embed_batch(texts)

    await ensure_collection(collection)
    count = await upsert_chunks(chunks, vectors, source=filename, collection=collection)

    return {"filename": filename, "pages": len(pages), "chunks_ingested": count, "skipped": False}


# ---------------------------------------------------------------------------
# Route: user uploads their inspection report
# ---------------------------------------------------------------------------

@router.post("/ingest")
async def ingest_upload(file: UploadFile = File(...)):
    """
    Accepts a user's inspection report PDF.
    Stored in USER_COLLECTION so the agent can search it with page citations.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()

    try:
        result = await ingest_pdf_bytes(file_bytes, file.filename, USER_COLLECTION)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result["skipped"]:
        return {
            **result,
            "message": f"'{file.filename}' was already indexed — skipping re-ingestion.",
        }

    return {
        **result,
        "message": (
            f"Indexed '{file.filename}': {result['pages']} pages, "
            f"{result['chunks_ingested']} chunks ready for analysis."
        ),
    }


# ---------------------------------------------------------------------------
# Route: manually re-trigger reference doc ingestion
# ---------------------------------------------------------------------------

@router.post("/ingest/local")
async def ingest_local():
    """Re-ingests all PDFs from backend/data/ into the reference guidelines collection."""
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        return {"message": f"No PDF files found in {DATA_DIR}", "results": []}

    results = []
    for pdf_path in pdf_files:
        try:
            result = await ingest_pdf_bytes(
                pdf_path.read_bytes(), pdf_path.name, REFERENCE_COLLECTION
            )
            results.append({**result, "status": "ok"})
        except Exception as e:
            results.append({"filename": pdf_path.name, "status": "error", "error": str(e)})

    total = sum(r.get("chunks_ingested", 0) for r in results)
    return {
        "message": f"Processed {len(pdf_files)} reference file(s), {total} chunks ingested.",
        "results": results,
    }


# ---------------------------------------------------------------------------
# Startup utility — ingests backend/data/ into reference collection
# ---------------------------------------------------------------------------

async def auto_ingest_data_dir() -> None:
    """
    Called from main.py on server startup.
    Ingests PDFs from backend/data/ into REFERENCE_COLLECTION.
    These teach the agent what inspection red flags look like.
    """
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"No reference PDFs found in {DATA_DIR} — skipping auto-ingest.")
        return

    logger.info(f"Loading {len(pdf_files)} reference PDF(s) into '{REFERENCE_COLLECTION}' ...")
    for pdf_path in pdf_files:
        try:
            result = await ingest_pdf_bytes(
                pdf_path.read_bytes(), pdf_path.name, REFERENCE_COLLECTION
            )
            if result["skipped"]:
                logger.info(f"  ↷ {pdf_path.name} already indexed — skipped.")
            else:
                logger.info(
                    f"  ✓ {pdf_path.name} → {result['pages']} pages, "
                    f"{result['chunks_ingested']} chunks"
                )
        except Exception as e:
            logger.error(f"  ✗ {pdf_path.name} failed: {e}")
