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
    """Searches the user's uploaded inspection report for specific issues.
    Results include the exact page numbers from the user's report.
    Use this to find where issues appear in the report."""
    return await retrieve_from_report(query, top_k)


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Searches the web for additional information when the reference guidelines and
    inspection report do not contain enough detail.
    Use for: repair cost estimates, remediation steps, building codes, material lifespan."""
    return await search_web(query, max_results)


def get_tools():
    """Returns the list of tools for the LangChain agent."""
    return [search_red_flag_guidelines, search_inspection_report, web_search]
