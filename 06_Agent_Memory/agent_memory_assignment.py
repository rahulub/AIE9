# Core imports
import os
import getpass
from uuid import uuid4
from typing import Annotated, TypedDict

import nest_asyncio
nest_asyncio.apply()  # Required for async operations in Jupyter

# Set API Keys
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")

# Optional: LangSmith for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE9 - Agent Memory - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key (press Enter to skip): ") or ""

if not os.environ["LANGCHAIN_API_KEY"]:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("LangSmith tracing disabled")
else:
    print(f"LangSmith tracing enabled. Project: {os.environ['LANGCHAIN_PROJECT']}")

# Initialize LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Test the connection
response = llm.invoke("Say 'Memory systems ready!' in exactly those words.")
print(response.content)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Define the state schema for our graph
# The `add_messages` annotation tells LangGraph how to update the messages list
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Define our wellness chatbot node
def wellness_chatbot(state: State):
    """Process the conversation and generate a wellness-focused response."""
    system_prompt = SystemMessage(content="""You are a friendly Personal Wellness Assistant. 
Help users with exercise, nutrition, sleep, and stress management questions.
Be supportive and remember details the user shares about themselves.""")
    
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build the graph
builder = StateGraph(State)
builder.add_node("chatbot", wellness_chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Compile with a checkpointer for short-term memory
checkpointer = MemorySaver()
wellness_graph = builder.compile(checkpointer=checkpointer)

print("Wellness chatbot compiled with short-term memory (checkpointing)")

# Test short-term memory within a thread
config = {"configurable": {"thread_id": "wellness_thread_1"}}

# First message - introduce ourselves
response = wellness_graph.invoke(
    {"messages": [HumanMessage(content="Hi! My name is Sarah and I want to improve my sleep.")]},
    config
)
print("User: Hi! My name is Sarah and I want to improve my sleep.")
print(f"Assistant: {response['messages'][-1].content}")
print()

# Second message - test if it remembers (same thread)
response = wellness_graph.invoke(
    {"messages": [HumanMessage(content="What's my name and what am I trying to improve?")]},
    config  # Same config = same thread_id
)
print("User: What's my name and what am I trying to improve?")
print(f"Assistant: {response['messages'][-1].content}")

# New thread - it won't remember Sarah!
different_config = {"configurable": {"thread_id": "wellness_thread_2"}}

response = wellness_graph.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    different_config  # Different thread_id = no memory of Sarah
)
print("User (NEW thread): What's my name?")
print(f"Assistant: {response['messages'][-1].content}")
print()
print("Notice: The agent doesn't know our name because this is a new thread!")

# Inspect the state of thread 1
state = wellness_graph.get_state(config)
print(f"Thread 1 has {len(state.values['messages'])} messages:")
for msg in state.values['messages']:
    role = "User" if isinstance(msg, HumanMessage) else "Assistant"
    content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
    print(f"  {role}: {content}")

from langgraph.store.memory import InMemoryStore

# Create a store for long-term memory
store = InMemoryStore()

# Namespaces organize memories - typically by user_id and category
user_id = "user_sarah"
profile_namespace = (user_id, "profile")
preferences_namespace = (user_id, "preferences")

# Store Sarah's wellness profile
store.put(profile_namespace, "name", {"value": "Sarah"})
store.put(profile_namespace, "goals", {"primary": "improve sleep", "secondary": "reduce stress"})
store.put(profile_namespace, "conditions", {"allergies": ["peanuts"], "injuries": ["bad knee"]})

# Store Sarah's preferences
store.put(preferences_namespace, "communication", {"style": "friendly", "detail_level": "moderate"})
store.put(preferences_namespace, "schedule", {"preferred_workout_time": "morning", "available_days": ["Mon", "Wed", "Fri"]})

print("Stored Sarah's profile and preferences in long-term memory")

# Retrieve specific memories
name = store.get(profile_namespace, "name")
print(f"Name: {name.value}")

goals = store.get(profile_namespace, "goals")
print(f"Goals: {goals.value}")

# List all memories in a namespace
print("\nAll profile items:")
for item in store.search(profile_namespace):
    print(f"  {item.key}: {item.value}")

from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

# Define state with user_id for personalization
class PersonalizedState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


def personalized_wellness_chatbot(state: PersonalizedState, config: RunnableConfig, *, store: BaseStore):
    """A wellness chatbot that uses long-term memory for personalization."""
    user_id = state["user_id"]
    profile_namespace = (user_id, "profile")
    preferences_namespace = (user_id, "preferences")
    
    # Retrieve user profile from long-term memory
    profile_items = list(store.search(profile_namespace))
    pref_items = list(store.search(preferences_namespace))
    
    # Build context from profile
    profile_text = "\n".join([f"- {p.key}: {p.value}" for p in profile_items])
    pref_text = "\n".join([f"- {p.key}: {p.value}" for p in pref_items])
    
    system_msg = f"""You are a Personal Wellness Assistant. You know the following about this user:

PROFILE:
{profile_text if profile_text else 'No profile stored.'}

PREFERENCES:
{pref_text if pref_text else 'No preferences stored.'}

Use this information to personalize your responses. Be supportive and helpful."""
    
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build the personalized graph
builder2 = StateGraph(PersonalizedState)
builder2.add_node("chatbot", personalized_wellness_chatbot)
builder2.add_edge(START, "chatbot")
builder2.add_edge("chatbot", END)

# Compile with BOTH checkpointer (short-term) AND store (long-term)
personalized_graph = builder2.compile(
    checkpointer=MemorySaver(),
    store=store
)

print("Personalized graph compiled with both short-term and long-term memory")

# Test the personalized chatbot - it knows Sarah's profile!
config = {"configurable": {"thread_id": "personalized_thread_1"}}

response = personalized_graph.invoke(
    {
        "messages": [HumanMessage(content="What exercises would you recommend for me?")],
        "user_id": "user_sarah"
    },
    config
)

print("User: What exercises would you recommend for me?")
print(f"Assistant: {response['messages'][-1].content}")
print()
print("Notice: The agent knows about Sarah's bad knee without her mentioning it!")

# Even in a NEW thread, it still knows Sarah's profile
# because long-term memory is cross-thread!

new_config = {"configurable": {"thread_id": "personalized_thread_2"}}

response = personalized_graph.invoke(
    {
        "messages": [HumanMessage(content="Can you suggest a snack for me?")],
        "user_id": "user_sarah"
    },
    new_config
)

print("User (NEW thread): Can you suggest a snack for me?")
print(f"Assistant: {response['messages'][-1].content}")
print()
print("Notice: Even in a new thread, the agent knows Sarah has a peanut allergy!")

from langchain_core.messages import trim_messages

# Create a trimmer that keeps only recent messages
trimmer = trim_messages(
    max_tokens=500,  # Keep messages up to this token count
    strategy="last",  # Keep the most recent messages
    token_counter=llm,  # Use the LLM to count tokens
    include_system=True,  # Always keep system messages
    allow_partial=False,  # Don't cut messages in half
)

# Example: Create a long conversation
long_conversation = [
    SystemMessage(content="You are a wellness assistant."),
    HumanMessage(content="I want to improve my health."),
    AIMessage(content="Great goal! Let's start with exercise. What's your current activity level?"),
    HumanMessage(content="I walk about 30 minutes a day."),
    AIMessage(content="That's a good foundation. For cardiovascular health, aim for 150 minutes of moderate activity per week."),
    HumanMessage(content="What about nutrition?"),
    AIMessage(content="Focus on whole foods: vegetables, lean proteins, whole grains. Limit processed foods and added sugars."),
    HumanMessage(content="And sleep?"),
    AIMessage(content="Aim for 7-9 hours. Maintain a consistent sleep schedule and create a relaxing bedtime routine."),
    HumanMessage(content="What's the most important change I should make first?"),
]

# Trim to fit context window
trimmed = trimmer.invoke(long_conversation)
print(f"Original: {len(long_conversation)} messages")
print(f"Trimmed: {len(trimmed)} messages")
print("\nTrimmed conversation:")
for msg in trimmed:
    role = type(msg).__name__.replace("Message", "")
    content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
    print(f"  {role}: {content}")

# Summarization approach for longer conversations

def summarize_conversation(messages: list, max_messages: int = 6) -> list:
    """Summarize older messages to manage context length."""
    if len(messages) <= max_messages:
        return messages
    
    # Keep the system message and last few messages
    system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
    content_messages = messages[1:] if system_msg else messages
    
    if len(content_messages) <= max_messages:
        return messages
    
    old_messages = content_messages[:-max_messages+1]
    recent_messages = content_messages[-max_messages+1:]
    
    # Summarize old messages
    summary_prompt = f"""Summarize this conversation in 2-3 sentences, 
capturing key wellness topics discussed and any important user information:

{chr(10).join([f'{type(m).__name__}: {m.content[:200]}' for m in old_messages])}"""
    
    summary = llm.invoke(summary_prompt)
    
    # Return: system + summary + recent messages
    result = []
    if system_msg:
        result.append(system_msg)
    result.append(SystemMessage(content=f"[Previous conversation summary: {summary.content}]"))
    result.extend(recent_messages)
    
    return result


# Test summarization
summarized = summarize_conversation(long_conversation, max_messages=4)
print(f"Summarized: {len(summarized)} messages")
print("\nSummarized conversation:")
for msg in summarized:
    role = type(msg).__name__.replace("Message", "")
    content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
    print(f"  {role}: {content}")

### YOUR CODE HERE ###

# Step 1: Define a wellness profile schema
# Example attributes: name, age, goals, conditions, allergies, fitness_level, preferred_activities
class WellnessProfile(TypedDict, total=False):
    name: str
    age: int
    goals: dict
    conditions: dict
    fitness_level: str

# Step 2: Create helper functions to store and retrieve profiles
def store_wellness_profile(store, user_id: str, profile: dict):
    """Store a user's wellness profile."""
    namespace = (user_id, "profile")   # or "wellness_profile"
    for key, value in profile.items():
        store.put(namespace, key, value)


def get_wellness_profile(store, user_id: str) -> dict:
    """Retrieve a user's wellness profile."""
    namespace = (user_id, "profile")
    result = {}
    for item in store.search(namespace):
        result[item.key] = item.value
    return result


# Step 3: Create two different user profiles
profile_aron = {
    "name": "Aron",
    "age": 28,
    "goals": {"primary": "improve sleep", "secondary": "reduce stress"},
    "conditions": {"allergies": ["nuts"], "injuries": ["back"]},
    "fitness_level": "beginner",
}

profile_bean = {
    "name": "Bean",
    "age": 35,
    "goals": {"primary": "lose weight", "secondary": "reduce stress"},
    "conditions": {"allergies": ["cinnamon, nutmeg"], "injuries": ["knee"]},
    "fitness_level": "intermediate",
}

store_wellness_profile(store, "user_aron", profile_aron)
store_wellness_profile(store, "user_bean", profile_bean)

# Step 4: Build a personalized agent that uses profiles
def personalized_wellness_agent(state: PersonalizedState, config: RunnableConfig, *, store: BaseStore):
    """A wellness chatbot that uses long-term memory for personalization."""
    user_id = state["user_id"]
    # Get wellness profile from store using your helper
    profile = get_wellness_profile(store, user_id)
    profile_text = "\n".join([f"- {k}: {v}" for k, v in profile.items()]) if profile else "No profile stored."
    system_msg = f"""You are a Personal Wellness Assistant. You know the following about this user:

WELLNESS PROFILE:
{profile_text}

Use this information to personalize your responses. Be supportive and helpful."""
    
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build the personalized graph
builder3 = StateGraph(PersonalizedState)
builder3.add_node("agent", personalized_wellness_agent)
builder3.add_edge(START, "agent")
builder3.add_edge("agent", END)

# Compile with BOTH checkpointer (short-term) AND store (long-term)
personalized_graph = builder3.compile(
    checkpointer=MemorySaver(),
    store=store
)


# Step 5: Test with different users - they should get different advice
print("="*70)
config = {"configurable": {"thread_id": "personalized_thread_1"}}
response = personalized_graph.invoke(
    {
        "messages": [HumanMessage(content="What exercises would you recommend for me?")],
        "user_id": "user_aron"
    },
    config
)

print("User: What exercises would you recommend for me?")
print(f"Assistant: {response['messages'][-1].content}")
print("="*70)

config = {"configurable": {"thread_id": "personalized_thread_2"}}
response = personalized_graph.invoke(
    {
        "messages": [HumanMessage(content="What exercises would you recommend for me?")],
        "user_id": "user_bean"
    },
    config
)

print("User: What exercises would you recommend for me?")
print(f"Assistant: {response['messages'][-1].content}")
print("="*70)



config = {"configurable": {"thread_id": "personalized_thread_2"}}
response = personalized_graph.invoke(
    {
        "messages": [HumanMessage(content="What sweets would you recommend for me?")],
        "user_id": "user_bean"
    },
    config
)

print("User: What sweets would you recommend for me?")
print(f"Assistant: {response['messages'][-1].content}")

from langchain_openai import OpenAIEmbeddings

# Create embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a store with semantic search enabled
semantic_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,  # Dimension of text-embedding-3-small
    }
)

print("Semantic memory store created with embedding support")

# Store various wellness facts as semantic memories
namespace = ("wellness", "facts")

wellness_facts = [
    ("fact_1", {"text": "Drinking water can help relieve headaches caused by dehydration"}),
    ("fact_2", {"text": "Regular exercise improves sleep quality and helps you fall asleep faster"}),
    ("fact_3", {"text": "Deep breathing exercises can reduce stress and anxiety within minutes"}),
    ("fact_4", {"text": "Eating protein at breakfast helps maintain steady energy levels throughout the day"}),
    ("fact_5", {"text": "Blue light from screens can disrupt your circadian rhythm and sleep"}),
    ("fact_6", {"text": "Walking for 30 minutes daily can improve cardiovascular health"}),
    ("fact_7", {"text": "Magnesium-rich foods like nuts and leafy greens can help with muscle cramps"}),
    ("fact_8", {"text": "A consistent sleep schedule, even on weekends, improves overall sleep quality"}),
]

for key, value in wellness_facts:
    semantic_store.put(namespace, key, value)

print(f"Stored {len(wellness_facts)} wellness facts in semantic memory")

# Search semantically - notice we don't need exact matches!

queries = [
    "My head hurts, what should I do?",
    "How can I get better rest at night?",
    "I'm feeling stressed and anxious",
    "What should I eat in the morning?",
]

for query in queries:
    print(f"\nQuery: {query}")
    results = semantic_store.search(namespace, query=query, limit=2)
    for r in results:
        print(f"   {r.value['text']} (score: {r.score:.3f})")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load and chunk the wellness document
loader = TextLoader("data/HealthWellnessGuide.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print(f"Loaded and split into {len(chunks)} chunks")
print(f"\nSample chunk:\n{chunks[0].page_content[:200]}...")

# Store chunks in semantic memory
knowledge_namespace = ("wellness", "knowledge")

for i, chunk in enumerate(chunks):
    semantic_store.put(
        knowledge_namespace,
        f"chunk_{i}",
        {"text": chunk.page_content, "source": "HealthWellnessGuide.txt"}
    )

print(f"Stored {len(chunks)} chunks in semantic knowledge base")

# Build a semantic search wellness chatbot

class SemanticState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


def semantic_wellness_chatbot(state: SemanticState, config: RunnableConfig, *, store: BaseStore):
    """A wellness chatbot that retrieves relevant facts using semantic search."""
    user_message = state["messages"][-1].content
    
    # Search for relevant knowledge
    knowledge_results = store.search(
        ("wellness", "knowledge"),
        query=user_message,
        limit=3
    )
    
    # Build context from retrieved knowledge
    if knowledge_results:
        knowledge_text = "\n\n".join([f"- {r.value['text']}" for r in knowledge_results])
        system_msg = f"""You are a Personal Wellness Assistant with access to a wellness knowledge base.

Relevant information from your knowledge base:
{knowledge_text}

Use this information to answer the user's question. If the information doesn't directly answer their question, use your general knowledge but mention what you found."""
    else:
        system_msg = "You are a Personal Wellness Assistant. Answer wellness questions helpfully."
    
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build and compile
builder3 = StateGraph(SemanticState)
builder3.add_node("chatbot", semantic_wellness_chatbot)
builder3.add_edge(START, "chatbot")
builder3.add_edge("chatbot", END)

semantic_graph = builder3.compile(
    checkpointer=MemorySaver(),
    store=semantic_store
)

print("Semantic wellness chatbot ready")

# Test semantic retrieval
config = {"configurable": {"thread_id": "semantic_thread_1"}}

questions = [
    "What exercises can help with lower back pain?",
    "How can I improve my sleep quality?",
    "What should I eat for better gut health?",
]

for q in questions:
    response = semantic_graph.invoke(
        {"messages": [HumanMessage(content=q)], "user_id": "test_user"},
        config
    )
    print(f"\nUser: {q}")
    print(f"Assistant: {response['messages'][-1].content[:500]}...")

# Create episodic memory with past successful interactions
episodic_namespace = ("agent", "episodes")

successful_episodes = [
    {
        "situation": "User asked about starting exercise with joint pain",
        "input": "I want to start exercising but my knees hurt. What should I do?",
        "output": """For knee-friendly exercise, I'd recommend starting with low-impact activities:

1. **Swimming or water aerobics** - Water supports your joints while providing great cardio
2. **Cycling** - Either stationary or regular bikes are gentle on knees
3. **Elliptical machine** - Provides cardio without impact

Start slowly with 15-20 minutes and gradually increase. If you experience pain, stop and consult a healthcare provider. Would you like specific exercises for any of these?""",
        "feedback": "User appreciated the specific alternatives and safety advice"
    },
    {
        "situation": "User struggling with sleep and stress",
        "input": "I can't fall asleep because I'm always thinking about work. Help!",
        "output": """Racing thoughts at bedtime is common, especially with work stress. Here's a proven approach:

**Before bed:**
- Write down tomorrow's to-do list (gets thoughts out of your head)
- Set a "worry time" earlier in the evening - not at bedtime

**At bedtime:**
- Try the 4-7-8 breathing technique: Inhale 4 counts, hold 7, exhale 8
- Use the "cognitive shuffle" - think of random, unrelated objects (apple, guitar, mountain...)

Which of these would you like to try first?""",
        "feedback": "User found the structured approach and specific techniques helpful"
    },
    {
        "situation": "User asking about healthy eating on a budget",
        "input": "How can I eat healthy when I don't have much money?",
        "output": """Eating healthy on a budget is absolutely possible! Here are practical strategies:

**Budget-friendly nutritious foods:**
- Eggs, beans, lentils (cheap protein)
- Frozen vegetables (same nutrition as fresh, longer lasting)
- Oats, rice, whole grain bread
- Canned fish (tuna, salmon)

**Money-saving tips:**
- Buy in bulk when possible
- Plan meals around sales
- Cook in batches and freeze portions

What's your typical weekly food budget? I can help you create a specific meal plan.""",
        "feedback": "User valued the practical, actionable advice without judgment"
    },
]

for i, episode in enumerate(successful_episodes):
    semantic_store.put(
        episodic_namespace,
        f"episode_{i}",
        {
            "text": episode["situation"],  # Used for semantic search
            **episode
        }
    )

print(f"Stored {len(successful_episodes)} episodic memories (past successful interactions)")

class EpisodicState(TypedDict):
    messages: Annotated[list, add_messages]


def episodic_wellness_chatbot(state: EpisodicState, config: RunnableConfig, *, store: BaseStore):
    """A chatbot that learns from past successful interactions."""
    user_question = state["messages"][-1].content
    
    # Search for similar past experiences
    similar_episodes = store.search(
        ("agent", "episodes"),
        query=user_question,
        limit=1
    )
    
    # Build few-shot examples from past episodes
    if similar_episodes:
        episode = similar_episodes[0].value
        few_shot_example = f"""Here's an example of a similar wellness question I handled well:

User asked: {episode['input']}

My response was:
{episode['output']}

The user feedback was: {episode['feedback']}

Use this as inspiration for the style, structure, and tone of your response, but tailor it to the current question."""
        
        system_msg = f"""You are a Personal Wellness Assistant. Learn from your past successes:

{few_shot_example}"""
    else:
        system_msg = "You are a Personal Wellness Assistant. Be helpful, specific, and supportive."
    
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build the episodic memory graph
builder4 = StateGraph(EpisodicState)
builder4.add_node("chatbot", episodic_wellness_chatbot)
builder4.add_edge(START, "chatbot")
builder4.add_edge("chatbot", END)

episodic_graph = builder4.compile(
    checkpointer=MemorySaver(),
    store=semantic_store
)

print("Episodic memory chatbot ready")

# Test episodic memory - similar question to stored episode
config = {"configurable": {"thread_id": "episodic_thread_1"}}

response = episodic_graph.invoke(
    {"messages": [HumanMessage(content="I want to exercise more but I have a bad hip. What can I do?")]},
    config
)

print("User: I want to exercise more but I have a bad hip. What can I do?")
print(f"\nAssistant: {response['messages'][-1].content}")
print("\nNotice: The response structure mirrors the successful knee pain episode!")

# Initialize procedural memory with base instructions
procedural_namespace = ("agent", "instructions")

initial_instructions = """You are a Personal Wellness Assistant.

Guidelines:
- Be supportive and non-judgmental
- Provide evidence-based wellness information
- Ask clarifying questions when needed
- Encourage healthy habits without being preachy"""

semantic_store.put(
    procedural_namespace,
    "wellness_assistant",
    {"instructions": initial_instructions, "version": 1}
)

print("Initialized procedural memory with base instructions")
print(f"\nCurrent Instructions (v1):\n{initial_instructions}")

class ProceduralState(TypedDict):
    messages: Annotated[list, add_messages]
    feedback: str  # Optional feedback from user


def get_instructions(store: BaseStore) -> tuple[str, int]:
    """Retrieve current instructions from procedural memory."""
    item = store.get(("agent", "instructions"), "wellness_assistant")
    if item is None:
        return "You are a helpful wellness assistant.", 0
    return item.value["instructions"], item.value["version"]


def procedural_assistant_node(state: ProceduralState, config: RunnableConfig, *, store: BaseStore):
    """Respond using current procedural instructions."""
    instructions, version = get_instructions(store)
    
    messages = [SystemMessage(content=instructions)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def reflection_node(state: ProceduralState, config: RunnableConfig, *, store: BaseStore):
    """Reflect on feedback and update instructions if needed."""
    feedback = state.get("feedback", "")
    
    if not feedback:
        return {}  # No feedback, no update needed
    
    # Get current instructions
    current_instructions, version = get_instructions(store)
    
    # Ask the LLM to reflect and improve instructions
    reflection_prompt = f"""You are improving a wellness assistant's instructions based on user feedback.

Current Instructions:
{current_instructions}

User Feedback:
{feedback}

Based on this feedback, provide improved instructions. Keep the same general format but incorporate the feedback.
Only output the new instructions, nothing else."""
    
    response = llm.invoke([HumanMessage(content=reflection_prompt)])
    new_instructions = response.content
    
    # Update procedural memory with new instructions
    store.put(
        ("agent", "instructions"),
        "wellness_assistant",
        {"instructions": new_instructions, "version": version + 1}
    )
    
    print(f"\nInstructions updated to version {version + 1}")
    return {}


def should_reflect(state: ProceduralState) -> str:
    """Decide whether to reflect on feedback."""
    if state.get("feedback"):
        return "reflect"
    return "end"


# Build the procedural memory graph
builder5 = StateGraph(ProceduralState)
builder5.add_node("assistant", procedural_assistant_node)
builder5.add_node("reflect", reflection_node)

builder5.add_edge(START, "assistant")
builder5.add_conditional_edges("assistant", should_reflect, {"reflect": "reflect", "end": END})
builder5.add_edge("reflect", END)

procedural_graph = builder5.compile(
    checkpointer=MemorySaver(),
    store=semantic_store
)

print("Procedural memory graph ready (with self-improvement capability)")

# Test with initial instructions
config = {"configurable": {"thread_id": "procedural_thread_1"}}

response = procedural_graph.invoke(
    {
        "messages": [HumanMessage(content="How can I reduce stress?")],
        "feedback": ""  # No feedback yet
    },
    config
)

print("User: How can I reduce stress?")
print(f"\nAssistant (v1 instructions):\n{response['messages'][-1].content}")

# Now provide feedback - the agent will update its own instructions!
response = procedural_graph.invoke(
    {
        "messages": [HumanMessage(content="How can I reduce stress?")],
        "feedback": "Your responses are too long. Please be more concise and give me 3 actionable tips maximum."
    },
    {"configurable": {"thread_id": "procedural_thread_2"}}
)

# Check the updated instructions
new_instructions, version = get_instructions(semantic_store)
print(f"Updated Instructions (v{version}):\n")
print(new_instructions)

# Test with updated instructions - should be more concise now!
response = procedural_graph.invoke(
    {
        "messages": [HumanMessage(content="How can I sleep better?")],
        "feedback": ""  # No feedback this time
    },
    {"configurable": {"thread_id": "procedural_thread_3"}}
)

print(f"User: How can I sleep better?")
print(f"\nAssistant (v{version} instructions - after feedback):")
print(response['messages'][-1].content)
print("\nNotice: The response should now be more concise based on the feedback!")

class UnifiedState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    feedback: str


def unified_wellness_assistant(state: UnifiedState, config: RunnableConfig, *, store: BaseStore):
    """An assistant that uses all five memory types."""
    user_id = state["user_id"]
    user_message = state["messages"][-1].content
    
    # 1. PROCEDURAL: Get current instructions
    instructions_item = store.get(("agent", "instructions"), "wellness_assistant")
    base_instructions = instructions_item.value["instructions"] if instructions_item else "You are a helpful wellness assistant."
    
    # 2. LONG-TERM: Get user profile
    profile_items = list(store.search((user_id, "profile")))
    pref_items = list(store.search((user_id, "preferences")))
    profile_text = "\n".join([f"- {p.key}: {p.value}" for p in profile_items]) if profile_items else "No profile stored."
    
    # 3. SEMANTIC: Search for relevant knowledge
    relevant_knowledge = store.search(("wellness", "knowledge"), query=user_message, limit=2)
    knowledge_text = "\n".join([f"- {r.value['text'][:200]}..." for r in relevant_knowledge]) if relevant_knowledge else "No specific knowledge found."
    
    # 4. EPISODIC: Find similar past interactions
    similar_episodes = store.search(("agent", "episodes"), query=user_message, limit=1)
    if similar_episodes:
        ep = similar_episodes[0].value
        episode_text = f"Similar past interaction:\nUser: {ep.get('input', 'N/A')}\nResponse style: {ep.get('feedback', 'N/A')}"
    else:
        episode_text = "No similar past interactions found."
    
    # Build comprehensive system message
    system_message = f"""{base_instructions}

=== USER PROFILE ===
{profile_text}

=== RELEVANT WELLNESS KNOWLEDGE ===
{knowledge_text}

=== LEARNING FROM EXPERIENCE ===
{episode_text}

Use all of this context to provide the best possible personalized response."""
    
    # 5. SHORT-TERM: Full conversation history is automatically managed by the checkpointer
    # Use summarization for long conversations
    trimmed_messages = summarize_conversation(state["messages"], max_messages=6)
    
    messages = [SystemMessage(content=system_message)] + trimmed_messages
    response = llm.invoke(messages)
    return {"messages": [response]}


def unified_feedback_node(state: UnifiedState, config: RunnableConfig, *, store: BaseStore):
    """Update procedural memory based on feedback."""
    feedback = state.get("feedback", "")
    if not feedback:
        return {}
    
    item = store.get(("agent", "instructions"), "wellness_assistant")
    if item is None:
        return {}
    
    current = item.value
    reflection_prompt = f"""Update these instructions based on feedback:

Current: {current['instructions']}
Feedback: {feedback}

Output only the updated instructions."""
    
    response = llm.invoke([HumanMessage(content=reflection_prompt)])
    store.put(
        ("agent", "instructions"),
        "wellness_assistant",
        {"instructions": response.content, "version": current["version"] + 1}
    )
    print(f"Procedural memory updated to v{current['version'] + 1}")
    return {}


def unified_route(state: UnifiedState) -> str:
    return "feedback" if state.get("feedback") else "end"


# Build the unified graph
unified_builder = StateGraph(UnifiedState)
unified_builder.add_node("assistant", unified_wellness_assistant)
unified_builder.add_node("feedback", unified_feedback_node)

unified_builder.add_edge(START, "assistant")
unified_builder.add_conditional_edges("assistant", unified_route, {"feedback": "feedback", "end": END})
unified_builder.add_edge("feedback", END)

# Compile with both checkpointer (short-term) and store (all other memory types)
unified_graph = unified_builder.compile(
    checkpointer=MemorySaver(),
    store=semantic_store
)

print("Unified wellness assistant ready with all 5 memory types!")

# Test the unified assistant
config = {"configurable": {"thread_id": "unified_thread_1"}}

# First interaction - should use semantic + long-term + episodic memory
response = unified_graph.invoke(
    {
        "messages": [HumanMessage(content="What exercises would you recommend for my back?")],
        "user_id": "user_sarah",  # Sarah has a bad knee in her profile!
        "feedback": ""
    },
    config
)

print("User: What exercises would you recommend for my back?")
print(f"\nAssistant: {response['messages'][-1].content}")
print("\n" + "="*60)
print("Memory types used:")
print("  Long-term: Knows Sarah has a bad knee")
print("  Semantic: Retrieved back exercise info from knowledge base")
print("  Episodic: May use similar joint pain episode as reference")
print("  Procedural: Following current instructions")
print("  Short-term: Will remember this in follow-up questions")

# Follow-up question (tests short-term memory)
response = unified_graph.invoke(
    {
        "messages": [HumanMessage(content="Can you show me how to do the first one?")],
        "user_id": "user_sarah",
        "feedback": ""
    },
    config  # Same thread
)

print("User: Can you show me how to do the first one?")
print(f"\nAssistant: {response['messages'][-1].content}")
print("\nNotice: The agent remembers the context from the previous message!")

### YOUR CODE HERE ###


# Step 1: Define wellness metrics schema and storage functions
def log_wellness_metric(store, user_id: str, date: str, metric_type: str, value: float, notes: str = ""):
    """Log a wellness metric for a user. metric_type: 'mood' | 'energy' | 'sleep_quality'."""
    namespace = (user_id, "wellness_metrics")
    key = f"{date}_{metric_type}"  # one entry per (date, metric_type) so we don't overwrite
    store.put(namespace, key, {"metric_type": metric_type, "value": value, "notes": notes, "date": date})


def get_wellness_history(store, user_id: str, metric_type: str = None, days: int = 7) -> list:
    """Get wellness history for a user. Use store.search to get all entries, then filter."""
    from datetime import datetime, timedelta
    namespace = (user_id, "wellness_metrics")
    items = list(store.search(namespace))
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d") if days else None
    out = []
    for item in items:
        v = item.value
        if metric_type and v.get("metric_type") != metric_type:
            continue
        if cutoff and v.get("date", "") < cutoff:
            continue
        out.append({"key": item.key, **v})
    return sorted(out, key=lambda x: (x.get("date", ""), x.get("metric_type", "")))


# Step 2: Create sample wellness data for a user (simulate a week)
# Example: log mood, energy, sleep_quality for several dates (1-5 scale)
from langgraph.store.memory import InMemoryStore
from src.wellness_memory.stores import create_memory_store
store = create_memory_store(with_embeddings=False)
USER = "mike"
for date, mood, energy, sleep in [("2025-01-26", 4, 3, 5), ("2025-01-27", 3, 2, 4), ("2025-01-28", 5, 4, 3)]:
    log_wellness_metric(store, USER, date, "mood", mood)
    log_wellness_metric(store, USER, date, "energy", energy, "tired, less sleep")
    log_wellness_metric(store, USER, date, "sleep_quality", sleep, "not enough sleep")
    get_wellness_history(store, USER, days=7)  # or metric_type="mood" for one metric


# Step 3: Build a wellness dashboard agent that:
#   - Retrieves user's wellness history
#   - Searches for relevant advice based on patterns (semantic memory)
#   - Uses episodic memory for what worked before
#   - Generates a personalized summary

def get_relevant_advice(store, user_id: str, days: int = 7, limit: int = 3):
    """Use semantic memory to find advice relevant to the user's wellness patterns.
    Builds a query from recent metrics (e.g. low energy/sleep) and searches (wellness, knowledge).
    Store must have embeddings and (wellness, knowledge) namespace populated (e.g. Task 7 chunks)."""
    history = get_wellness_history(store, user_id, days=days)
    # Build a natural-language query from patterns: which metrics are low or mentioned
    parts = []
    for h in history:
        mt, val = h.get("metric_type"), h.get("value")
        if isinstance(val, (int, float)) and val < 3 and mt:
            parts.append(f"improve {mt} and ")
    query = (" ".join(parts) + "wellness advice").strip() if parts else "general wellness and sleep mood energy advice"
    results = list(store.search(("wellness", "knowledge"), query=query, limit=limit))
    return [{"text": r.value.get("text", ""), "score": getattr(r, "score", None)} for r in results]

# Optional: seed semantic knowledge if not using Task 7 (HealthWellnessGuide).
# Store must be created with create_memory_store(with_embeddings=True) for semantic search.
sem_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,  # Dimension of text-embedding-3-small
    }
)

namespace = ("wellness_metrics", "facts")

wellness_metrics_facts = [
    ("mood_1", {"text": "Physical activity, social connection, and enough sleep support mood."}),
    ("energy_1", {"text": "Regular exercise and balanced meals help maintain energy levels throughout the day."}),
    ("sleep_1", {"text": "Consistent sleep schedule and a dark, cool room improve sleep quality."}),
]

for key, value in wellness_metrics_facts:
    sem_store.put(namespace, key, value)

print(f"Stored {len(wellness_metrics_facts)} wellness facts in semantic memory")

#from src.wellness_memory.memory_types import SemanticMemory
#sem = SemanticMemory(store, ("wellness", "knowledge"))
sem.store_fact("sleep_1", "Consistent sleep schedule and a dark, cool room improve sleep quality.")
sem.store_fact("energy_1", "Regular exercise and balanced meals help maintain energy levels throughout the day.")
sem.store_fact("mood_1", "Physical activity, social connection, and enough sleep support mood.")
get_relevant_advice(store, "mike", days=7, limit=2)


# Step 4: Test the dashboard
# Example: "Give me a summary of my wellness this week"
# Example: "I've been feeling tired lately. What might help?"


