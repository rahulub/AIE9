"""An agent graph with a post-response vibe (friendliness) check loop.

After the agent responds, a secondary node evaluates whether the response
has a friendly, approachable tone. If it passes, end; otherwise, loop back
with feedback so the agent can try again.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from app.state import MessagesState
from app.models import get_chat_model
from app.tools import get_tool_belt


class VibeResult(BaseModel):
    is_friendly: bool = Field(
        description="Whether the response has a friendly, warm, and approachable tone"
    )


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_vibe(state: MessagesState):
    """Decide whether to execute tools or run the vibe checker."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "vibe_check"


_vibe_prompt = ChatPromptTemplate.from_template(
    "Given a user's query and an AI assistant's response, determine if the response "
    "has a friendly, warm, and approachable tone. A good vibe means the response "
    "sounds helpful, personable, and welcoming—not cold, robotic, or dismissive.\n\n"
    "User Query:\n{initial_query}\n\n"
    "Assistant Response:\n{final_response}\n\n"
    "Is the assistant's response friendly and approachable?"
)


def vibe_check_node(state: MessagesState) -> dict:
    """Evaluate whether the latest response has a friendly, approachable vibe."""
    # Guard against infinite loops
    if len(state["messages"]) > 12:
        return {"messages": [AIMessage(content="VIBE:END")]}

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    structured_model = get_chat_model(
        model_name="gpt-4.1-mini"
    ).with_structured_output(VibeResult)
    result = (_vibe_prompt | structured_model).invoke(
        {
            "initial_query": initial_query.content,
            "final_response": final_response.content,
        }
    )

    if result.is_friendly:
        return {"messages": [AIMessage(content="VIBE:Y")]}
    return {
        "messages": [
            AIMessage(
                content="VIBE:N - The response was not friendly enough. "
                "Please reply again with a warmer, more approachable tone."
            )
        ]
    }


def vibe_decision(state: MessagesState):
    """Terminate on 'VIBE:Y' or 'VIBE:END'; loop back to agent otherwise."""
    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if "VIBE:END" in text or "VIBE:Y" in text:
        return END
    return "agent"


def build_graph():
    """Build an agent graph with a vibe-check evaluation loop."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("vibe_check", vibe_check_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_vibe,
        {"action": "action", "vibe_check": "vibe_check"},
    )
    graph.add_conditional_edges(
        "vibe_check",
        vibe_decision,
        {"agent": "agent", END: END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()
