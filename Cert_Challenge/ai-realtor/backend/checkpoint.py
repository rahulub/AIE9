"""
Redis-backed checkpointer for conversation persistence.

Stores chat history per thread_id so users can resume conversations
across requests. Keys: chat:{thread_id} → JSON array of {role, content}.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import redis.asyncio as redis

_base_dir = Path(__file__).resolve().parent
load_dotenv(_base_dir / ".env")
load_dotenv(_base_dir / ".env.local", override=True)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TTL_DAYS = int(os.getenv("REDIS_CHAT_TTL_DAYS", "7"))

_KEY_PREFIX = "chat:"


async def get_history(thread_id: str) -> list[dict]:
    """
    Load previous messages for this thread.
    Returns a list of {role, content} dicts.
    """
    if not thread_id:
        return []
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        data = await client.get(f"{_KEY_PREFIX}{thread_id}")
        await client.aclose()
        if not data:
            return []
        return json.loads(data)
    except redis.RedisError:
        return []


async def save_messages(thread_id: str, messages: list[dict]) -> None:
    """
    Append new messages to the thread and persist to Redis.
    messages: list of {role, content}
    """
    if not thread_id:
        return
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        key = f"{_KEY_PREFIX}{thread_id}"
        existing = await client.get(key)
        history = json.loads(existing) if existing else []
        history.extend(messages)
        await client.set(key, json.dumps(history), ex=TTL_DAYS * 86400)
        await client.aclose()
    except redis.RedisError:
        pass


async def clear_thread(thread_id: str) -> None:
    """Delete all checkpointed messages for this thread."""
    if not thread_id:
        return
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        await client.delete(f"{_KEY_PREFIX}{thread_id}")
        await client.aclose()
    except redis.RedisError:
        pass
