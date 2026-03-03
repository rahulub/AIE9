"""
Advanced retrieval with Cohere Rerank — improves relevance over vector search alone.

Uses the same underlying vector store and embedder as retriever.py, but reranks
candidates with Cohere's rerank model before returning results.

  retrieve_from_reference_cohere()  → reference guidelines (reranked)
  retrieve_from_report_cohere()     → user's inspection report (reranked)

Requires: COHERE_API_KEY
"""

import os
from rag.embedder import embed_text
from rag.store import search, REFERENCE_COLLECTION, USER_COLLECTION


def _format_results(results: list, label: str) -> str:
    """Format search results for agent consumption."""
    if not results:
        return f"No relevant content found in {label}."

    chunks = []
    for i, hit in enumerate(results, 1):
        source = hit.payload.get("source", "unknown")
        page = hit.payload.get("page_number", "?")
        text = hit.payload.get("text", "")
        chunks.append(f"[{i}] {source} — Page {page}\n{text}")

    return "\n\n---\n\n".join(chunks)


def _rerank_with_cohere(query: str, documents: list[str], top_n: int) -> list[int]:
    """
    Rerank documents using Cohere API. Returns indices of top_n documents in relevance order.
    """
    import cohere

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY required for Cohere rerank.")

    client = cohere.Client(api_key=api_key)
    model = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")

    response = client.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
    )

    return [r.index for r in response.results]


async def _retrieve_with_rerank(
    query: str, collection: str, label: str, top_k: int = 5, candidate_factor: int = 4
) -> str:
    """
    Retrieve via vector search, then rerank with Cohere.

    Fetches candidate_factor * top_k candidates from the vector store, reranks
    with Cohere, and returns the top_k most relevant.
    """
    query_vector = await embed_text(query)
    candidates = await search(
        query_vector, collection, top_k=max(top_k * candidate_factor, 20)
    )

    if not candidates:
        return f"No relevant content found in {label}."

    doc_texts = [c.payload.get("text", "") for c in candidates]
    if not any(doc_texts):
        return f"No relevant content found in {label}."

    indices = _rerank_with_cohere(query, doc_texts, top_n=top_k)
    reranked = [candidates[i] for i in indices]
    return _format_results(reranked, label)


async def retrieve_from_reference_cohere(query: str, top_k: int = 5) -> str:
    """
    Searches the reference guidelines collection with Cohere reranking.
    Use when you need higher-precision results than basic vector search.
    """
    return await _retrieve_with_rerank(
        query, REFERENCE_COLLECTION, "reference guidelines", top_k
    )


async def retrieve_from_report_cohere(query: str, top_k: int = 5) -> str:
    """
    Searches the user's inspection report with Cohere reranking.
    Use when you need higher-precision results and page-level citations.
    """
    return await _retrieve_with_rerank(
        query, USER_COLLECTION, "the uploaded inspection report", top_k
    )
