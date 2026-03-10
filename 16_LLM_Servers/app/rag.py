"""Retrieval-Augmented Generation (RAG) utilities and tool.

This module builds an in-memory RAG pipeline that:
- Loads PDF documents from `RAG_DATA_DIR` (default: "data").
- Splits documents into chunks using a token-aware splitter.
- Embeds chunks and stores vectors in an in-memory Qdrant store.
- Exposes a LangChain Tool `retrieve_information` that retrieves relevant
  context and generates a response constrained to that context.

Supports two providers:
- "fireworks": Fireworks AI (open-source) embedding + chat models
- "openai": OpenAI embedding + gpt-4.1-mini chat model
"""

from __future__ import annotations

import os
from typing import Annotated, Literal, TypedDict

import tiktoken
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph


RAGProvider = Literal["fireworks", "openai"]
_RAG_CACHE: dict[tuple[RAGProvider, str], object] = {}


def _tiktoken_len(text: str) -> int:
    """Return token length using tiktoken; used for chunk length measurement."""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


class _RAGState(TypedDict):
    """State schema for the simple two-step RAG graph: retrieve then generate."""

    question: str
    context: list[Document]
    response: str


def _get_embedding_model(provider: RAGProvider) -> OpenAIEmbeddings:
    """Return embedding model for the given provider."""
    if provider == "fireworks":
        return OpenAIEmbeddings(
            model=os.environ.get(
                "FIREWORKS_EMBEDDING_MODEL", "fireworks/qwen3-embedding-8b"
            ),
            openai_api_key=os.environ["FIREWORKS_API_KEY"],
            openai_api_base=os.environ.get(
                "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
            ),
            check_embedding_ctx_length=False,
            dimensions=4096,
        )
    # openai
    return OpenAIEmbeddings(
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


def _get_chat_model(provider: RAGProvider) -> ChatOpenAI:
    """Return chat model for the given provider."""
    if provider == "fireworks":
        return ChatOpenAI(
            model=os.environ.get(
                "FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-20b"
            ),
            openai_api_key=os.environ["FIREWORKS_API_KEY"],
            openai_api_base=os.environ.get(
                "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
            ),
        )
    # openai - gpt-4.1-mini per Activity 1 requirements
    return ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


def _build_rag_graph(provider: RAGProvider, data_dir: str):
    """Construct and compile a minimal RAG graph for the given provider.

    Steps:
    1) Load PDFs from `data_dir` recursively (best-effort).
    2) Split documents into token-aware chunks.
    3) Create embeddings and an in-memory Qdrant vector store retriever.
    4) Define a chat prompt and generation model.
    5) Wire a two-node graph: retrieve -> generate.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load PDFs from data directory (recursive)
    try:
        directory_loader = DirectoryLoader(
            data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader
        )
        documents = directory_loader.load()
    except Exception:
        documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, length_function=_tiktoken_len
    )
    chunks = text_splitter.split_documents(documents) if documents else []

    embedding_model = _get_embedding_model(provider)
    qdrant_vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        location=":memory:",
        collection_name=f"rag_collection_{provider}",
    )
    retriever = qdrant_vectorstore.as_retriever()

    human_template = (
        "\n#CONTEXT:\n{context}\n\nQUERY:\n{query}\n\n"
        "Use the provide context to answer the provided user query. "
        "Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with \"I don't know\""
    )
    chat_prompt = ChatPromptTemplate.from_messages([("human", human_template)])
    generator_llm = _get_chat_model(provider)

    def retrieve(state: _RAGState) -> _RAGState:
        retrieved_docs = retriever.invoke(state["question"]) if retriever else []
        return {"context": retrieved_docs}  # type: ignore

    def generate(state: _RAGState) -> _RAGState:
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response_text = generator_chain.invoke(
            {"query": state["question"], "context": state.get("context", [])}
        )
        return {"response": response_text}  # type: ignore

    graph_builder = StateGraph(_RAGState)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


def get_rag_graph(provider: RAGProvider | None = None):
    """Return a cached compiled RAG graph for the given provider."""
    provider = provider or _default_provider()
    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    cache_key = (provider, data_dir)
    if cache_key not in _RAG_CACHE:
        _RAG_CACHE[cache_key] = _build_rag_graph(provider, data_dir)
    return _RAG_CACHE[cache_key]


def _default_provider() -> RAGProvider:
    """Return the default RAG provider from environment."""
    p = os.environ.get("RAG_PROVIDER", "fireworks").lower()
    return "openai" if p == "openai" else "fireworks"


def invoke_rag(
    question: str,
    provider: RAGProvider | None = None,
    *,
    return_context: bool = False,
) -> str | dict:
    """Invoke the RAG pipeline for a question.

    Args:
        question: User query.
        provider: "fireworks" or "openai". Defaults to RAG_PROVIDER env.
        return_context: If True, return dict with response and retrieved context strings.

    Returns:
        Response string, or dict with "response" and "retrieved_contexts" if return_context.
    """
    graph = get_rag_graph(provider)
    result = graph.invoke({"question": question})
    if not return_context:
        return result.get("response", result) if isinstance(result, dict) else result
    # Return context for RAGAS evaluation
    ctx = result.get("context", [])
    ctx_strs = [d.page_content if hasattr(d, "page_content") else str(d) for d in ctx]
    return {
        "response": result.get("response", ""),
        "retrieved_contexts": ctx_strs,
    }


@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"],
):
    """Use Retrieval Augmented Generation to retrieve information about feline health, including life stage care, nutrition, vaccinations, parasite control, behavior, diagnostics, and veterinary guidelines for cats."""
    return invoke_rag(query, provider=_default_provider())
