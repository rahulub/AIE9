"""
LangChain tools for the inspection agent.
"""

import os
from langchain_core.tools import tool
from rag.retriever import retrieve_from_reference, retrieve_from_report
from web_search import search_web


@tool
async def search_red_flag_guidelines(query: str, top_k: int = 5) -> str:
    """Searches the reference knowledge base (expert inspection guidelines) to understand
    what defects and red flags to look for in a given category.
    Use this FIRST to learn what to look for before searching the user's report."""
    return await retrieve_from_reference(query, top_k)


@tool
async def search_inspection_report(query: str, top_k: int = 5) -> str:
    """Searches the user's uploaded inspection report for property/structure issues.
    Use ONLY for: foundation, roof, plumbing, electrical, HVAC, structure, water damage, mold, etc.
    Does NOT contain: schools, neighborhood, walkability, safety/crime, amenities — use web_search for those."""
    return await retrieve_from_report(query, top_k)


@tool
async def search_red_flag_guidelines_advanced(query: str, top_k: int = 3) -> str:
    """Searches the reference knowledge base with Cohere reranking for higher-precision results.
    Use when standard search_red_flag_guidelines returns too many irrelevant results."""
    from rag.retriever_cohere import retrieve_from_reference_cohere
    return await retrieve_from_reference_cohere(query, top_k)


@tool
async def search_inspection_report_advanced(query: str, top_k: int = 3) -> str:
    """Searches the user's inspection report with Cohere reranking for higher-precision results.
    Use when standard search_inspection_report returns too many irrelevant results."""
    from rag.retriever_cohere import retrieve_from_report_cohere
    return await retrieve_from_report_cohere(query, top_k)


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Searches the web for information NOT in the inspection report.
    For schools: search for property address + 'schools' and 'school ratings' to get elementary, middle,
    high school names and ratings (e.g. GreatSchools, Niche). Include address in query.
    Also use for: neighborhood, walkability, safety, amenities, repair costs, building codes."""
    return await search_web(query, max_results)


def get_tools():
    """Returns tools. Includes Cohere-reranked versions when COHERE_API_KEY is set."""
    tools = [web_search, search_red_flag_guidelines, search_inspection_report]
    if os.getenv("COHERE_API_KEY"):
        tools.extend([search_red_flag_guidelines_advanced, search_inspection_report_advanced])
    return tools
