import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from tools import get_tools

_base_dir = Path(__file__).resolve().parent
load_dotenv(_base_dir / ".env")
load_dotenv(_base_dir / ".env.local", override=True)

def _build_system_prompt():
    base = (
        "You are an expert home inspector and real estate analyst. "
        "You have access to tools including:\n"
        "1. web_search (Tavily) — USE for schools, neighborhood, walkability, safety, amenities. "
        "Also use when a user preference has NO relevant data in the inspection report — search the web to answer it.\n"
        "2. search_red_flag_guidelines — internal reference on red flags (do NOT cite in responses).\n"
        "3. search_inspection_report — property/structure only. NOT schools or neighborhood.\n"
    )
    if os.getenv("COHERE_API_KEY"):
        base += (
            "4. search_red_flag_guidelines_advanced — higher-precision reference search (Cohere rerank), use when standard search returns irrelevant results.\n"
            "5. search_inspection_report_advanced — higher-precision report search (Cohere rerank), use when standard search returns irrelevant results.\n\n"
        )
    else:
        base += "\n"
    return base


SYSTEM_PROMPT = _build_system_prompt() + (
    "FIRST: If the user asks about schools, School Quality, Peaceful Neighborhood, Walkability, Safety & Crime, "
    "Nearby Amenities, or the area — call web_search immediately. Include the property address in your query. "
    "The inspection report does NOT contain this. Do not answer without calling web_search for these topics.\n\n"
    "USER PREFERENCES: When the user provides 'User preferences while buying this property' in the context, "
    "tailor responses to those preferences. Support BOTH predefined and CUSTOM priorities (e.g. 'Low HOA fees', "
    "'Near parks', 'Quiet street'). ALWAYS include user-added custom preferences in your answer — address each one.\n"
    "For School Quality or schools — use web_search. "
    "For Peaceful Neighborhood, Walkability, Safety & Crime, Nearby Amenities, location, commute, parks, etc. — use web_search (include the address). "
    "For Foundation, Roof, Plumbing, Electrical, HVAC, structure, etc. — use search_inspection_report. "
    "For any custom preference: if it relates to the property/structure, search the inspection report first; if it relates to the area/neighborhood/external factors, use web_search.\n"
    "FALLBACK: When a user preference has NO relevant data in the inspection report (e.g. search returns nothing or irrelevant results), "
    "ALWAYS use web_search to answer that preference. Include the property address in the search query when relevant. "
    "Do not skip or ignore user preferences — use web_search to find answers for topics not covered in the inspection report.\n"
    "If user has NOT selected any specific inspection categories — report ALL red flags from the inspection report. "
    "Otherwise focus on selected priorities but still mention critical issues from other areas.\n\n"
    "ORDER BY SEVERITY: Always present red flags in decreasing order of severity: 🔴 Critical first, then 🟠 Major, then 🟡 Minor.\n\n"
    "CRITICAL: Answer the user's specific question directly. NEVER repeat or re-state a full red-flag summary unless explicitly requested.\n"
    "- For FOLLOW-UP questions: Answer ONLY the new question. Do NOT repeat any prior summary, list, or analysis. "
    "Do NOT start with 'Based on the inspection report...' and then list all red flags again. "
    "Just answer the specific follow-up (e.g. if they ask 'what about the roof?' — give only roof-related info).\n"
    "- SEVERITY CONSISTENCY: When answering follow-up questions about issues already discussed, keep the SAME severity "
    "(🔴 Critical / 🟠 Major / 🟡 Minor) that you used in the original response. Do not reassign or change severity in follow-ups.\n"
    "- Only produce a full categorized red-flag list when the user EXPLICITLY asks (e.g. 'list all red flags', "
    "'give me a summary of issues', 'what are all the problems?').\n"
    "- For focused questions (e.g. 'what about the roof?', 'explain the foundation issue', "
    "'what did they find on page 9?'): give a direct, concise answer addressing only that. No preamble, no summary.\n"
    "- Use the search tools to find relevant content, then respond with ONLY what answers the current question.\n"
    "- When citing findings, always include the page number from the user's report.\n"
    "- Do NOT mention or cite the red flag guidelines/PDF. Use it only internally to know what to look for; "
    "respond only from the user's inspection report and web search when used.\n"
    "- NEVER quote or repeat raw tool output — neither RAG chunks nor web search results. "
    "Do not include [1], [2] lists, Source URLs, or verbatim snippets. Use tool output as context only; "
    "answer in your own words, and cite sources only when directly relevant (e.g. 'according to Redfin…').\n"
    "- NEVER ask the user to provide document content — use the tools."
)


def _build_agent():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY not found. Looked for .env.local in: {_base_dir}"
        )
    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        temperature=0,
    )
    return create_react_agent(model, get_tools(), prompt=SYSTEM_PROMPT)


def _to_langchain_messages(history: list[dict]) -> list:
    """Convert {role, content} to LangChain message objects."""
    out = []
    for msg in history:
        if msg["role"] == "user":
            out.append(HumanMessage(content=msg["content"]))
        else:
            out.append(AIMessage(content=msg["content"]))
    return out


async def run_agent(message: str, context: str = "", history: list[dict] | None = None):
    """
    Runs the LangChain ReAct agent and streams the response.
    """
    agent = _build_agent()

    messages = []
    if history:
        messages.extend(_to_langchain_messages(history))

    user_content = message
    if context:
        user_content = f"[User-provided context:\n{context}\n\n]{message}"

    messages.append(HumanMessage(content=user_content))
    inputs = {"messages": messages}

    async for chunk, _metadata in agent.astream(
        inputs,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage) and chunk.content:
            yield chunk.content
