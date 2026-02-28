"""
Converts text into vector embeddings using OpenAI's embedding model.

The embedding model maps text to a high-dimensional vector space where
semantically similar texts are close together — this is what makes
semantic search possible.
"""

import os
from openai import AsyncOpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536  # dimensions produced by text-embedding-3-small


async def embed_text(text: str) -> list[float]:
    """Returns the embedding vector for a single string."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embeds multiple texts in a single API call — more efficient than looping."""
    if not texts:
        return []
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # The API returns embeddings in the same order as the input
    return [item.embedding for item in response.data]
