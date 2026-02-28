"""
Tavily web search — for additional context not found in the reference guidelines.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from tavily import AsyncTavilyClient

_base_dir = Path(__file__).resolve().parent
load_dotenv(_base_dir / ".env")
load_dotenv(_base_dir / ".env.local", override=True)


async def search_web(query: str, max_results: int = 5) -> str:
    """
    Runs a web search via Tavily and returns formatted results for the LLM.
    Returns empty string if TAVILY_API_KEY is not set or search fails.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search is not configured (TAVILY_API_KEY missing)."

    try:
        client = AsyncTavilyClient(api_key=api_key)
        response = await client.search(
            query=query,
            search_depth="basic",
            max_results=min(max_results, 10),
            include_answer=False,
        )

        if not response.get("results"):
            return "No relevant web results found."

        parts = []
        for i, r in enumerate(response["results"], 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "").strip()
            if content:
                parts.append(f"[{i}] {title}\n{content}\nSource: {url}")

        return "\n\n---\n\n".join(parts) if parts else "No content extracted from results."
    except Exception as e:
        return f"Web search failed: {str(e)}"
