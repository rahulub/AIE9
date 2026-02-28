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

SYSTEM_PROMPT = (
    "You are an expert home inspector and real estate analyst. "
    "You have access to three tools:\n"
    "1. web_search — USE THIS for schools, neighborhood, walkability, safety, amenities. Inspection report has NONE of this.\n"
    "2. search_red_flag_guidelines — internal reference on red flags (do NOT cite in responses).\n"
    "3. search_inspection_report — property/structure only. NOT schools or neighborhood.\n\n"
    "FIRST: If the user asks about schools, School Quality, Peaceful Neighborhood, Walkability, Safety & Crime, "
    "Nearby Amenities, or the area — call web_search immediately. Include the property address in your query. "
    "The inspection report does NOT contain this. Do not answer without calling web_search for these topics.\n\n"
    "USER PREFERENCES: When the user provides 'User preferences while buying this property' in the context, "
    "tailor responses to those preferences. For School Quality or schools — use web_search to find schools "
    "serving the property address. Provide: school names, grade levels (elementary/middle/high), and ratings "
    "for each. Present in a clear format with Elementary School, Middle School, High School sections.\n\n"
    "For Peaceful Neighborhood, Walkability, Safety & Crime, Nearby Amenities — use web_search (include the address). "
    "For Foundation, Roof, Plumbing, Electrical, etc. — use search_inspection_report.\n\n"
    "CRITICAL: Answer the user's specific question directly. Do NOT always return a full red-flag summary.\n"
    "- Only produce a full categorized red-flag list when the user explicitly asks for it "
    "(e.g. 'list all red flags', 'analyze for red flags', 'give me a summary of issues').\n"
    "- For focused questions (e.g. 'what about the roof?', 'explain the foundation issue', "
    "'what did they find on page 9?'), give a direct, concise answer addressing only that.\n"
    "- Use the search tools to find relevant content, then respond with only what answers the question.\n"
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
