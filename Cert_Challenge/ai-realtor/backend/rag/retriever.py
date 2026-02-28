"""
Two retrieval functions — one per collection:

  retrieve_from_reference()  → searches red flag guidelines (backend/data/ PDFs)
  retrieve_from_report()     → searches the user's uploaded inspection report

Keeping them separate means the agent can:
1. Learn WHAT to look for from reference guidelines
2. Find WHERE those issues appear in the user's report (with page numbers)
"""

from rag.embedder import embed_text
from rag.store import search, REFERENCE_COLLECTION, USER_COLLECTION


def _format_results(results, label: str) -> str:
    if not results:
        return f"No relevant content found in {label}."

    chunks = []
    for i, hit in enumerate(results, 1):
        source = hit.payload.get("source", "unknown")
        page = hit.payload.get("page_number", "?")
        text = hit.payload.get("text", "")
        chunks.append(f"[{i}] {source} — Page {page}\n{text}")

    return "\n\n---\n\n".join(chunks)


async def retrieve_from_reference(query: str, top_k: int = 5) -> str:
    """
    Searches the reference guidelines collection (PDFs from backend/data/).
    Use this to understand what red flags and defects to look for.
    """
    query_vector = await embed_text(query)
    results = await search(query_vector, REFERENCE_COLLECTION, top_k)
    return _format_results(results, "reference guidelines")


async def retrieve_from_report(query: str, top_k: int = 5) -> str:
    """
    Searches the user's uploaded inspection report collection.
    Results include page numbers from the user's report for direct citation.
    """
    query_vector = await embed_text(query)
    results = await search(query_vector, USER_COLLECTION, top_k)
    return _format_results(results, "the uploaded inspection report")
