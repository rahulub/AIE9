"""
Qdrant vector store — supports two separate collections:

  REFERENCE_COLLECTION  → PDFs from backend/data/ (red flag guidelines)
  USER_COLLECTION       → PDFs uploaded by the user (inspection reports)

Keeping them separate ensures page number citations always refer to
the user's report, not the reference material.
"""

import os
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScoredPoint,
    Filter,
    FieldCondition,
    MatchValue,
)
from rag.embedder import VECTOR_SIZE

REFERENCE_COLLECTION = os.getenv("QDRANT_REFERENCE_COLLECTION", "reference_guidelines")
USER_COLLECTION = os.getenv("QDRANT_USER_COLLECTION", "user_reports")


def get_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


async def ensure_collection(collection: str) -> None:
    """Creates the given Qdrant collection if it doesn't already exist."""
    client = get_client()
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    if collection not in names:
        await client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


async def source_exists(source: str, collection: str) -> bool:
    """Returns True if this source filename is already in the given collection."""
    client = get_client()
    try:
        result = await client.count(
            collection_name=collection,
            count_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
            exact=True,
        )
        return result.count > 0
    except Exception:
        return False


async def upsert_chunks(
    chunks: list[dict],
    vectors: list[list[float]],
    source: str,
    collection: str,
) -> int:
    """Stores text chunks with page numbers into the specified collection."""
    client = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk["text"],
                "source": source,
                "page_number": chunk.get("page_number", 0),
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    await client.upsert(collection_name=collection, points=points)
    return len(points)


async def search(
    query_vector: list[float],
    collection: str,
    top_k: int = 5,
) -> list[ScoredPoint]:
    """Returns the top-k most similar points from the specified collection."""
    client = get_client()
    result = await client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
    )
    return result.points
