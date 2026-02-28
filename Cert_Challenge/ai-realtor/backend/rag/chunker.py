"""
Text chunking utilities.

chunk_by_pages() — page-aware chunking that preserves the page number
for each chunk. This is critical for inspection reports so the agent
can cite exact page numbers when reporting red flags.
"""


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Plain character-based chunking (no page tracking)."""
    if not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_by_pages(
    pages: list[str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Splits each page's text into overlapping chunks while keeping track of
    which page each chunk came from.

    Returns a list of dicts:
        { "text": str, "page_number": int }

    The page_number (1-indexed) is stored in Qdrant so the agent can cite it.
    """
    result = []
    for page_idx, page_text in enumerate(pages):
        page_number = page_idx + 1
        if not page_text.strip():
            continue

        start = 0
        while start < len(page_text):
            chunk = page_text[start : start + chunk_size].strip()
            if chunk:
                result.append({"text": chunk, "page_number": page_number})
            start += chunk_size - overlap

    return result
