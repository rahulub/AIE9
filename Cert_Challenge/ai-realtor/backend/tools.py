"""
LangChain tools for the inspection agent.
"""

from langchain_core.tools import tool
from rag.retriever import retrieve_from_reference, retrieve_from_report
from web_search import search_web


@tool
async def search_red_flag_guidelines(query: str, top_k: int = 4) -> str:
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
async def web_search(query: str, max_results: int = 5) -> str:
    """Searches the web for information NOT in the inspection report.
    For schools: search for property address + 'schools' and 'school ratings' to get elementary, middle,
    high school names and ratings (e.g. GreatSchools, Niche). Include address in query.
    Also use for: neighborhood, walkability, safety, amenities, repair costs, building codes."""
    return await search_web(query, max_results)


def get_tools():
    """Returns tools. web_search first so agent considers it for neighborhood queries."""
    return [web_search, search_red_flag_guidelines, search_inspection_report]
