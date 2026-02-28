import { useState, useCallback } from "react";

/**
 * useStreamingChat — manages chat with Redis-backed checkpointing.
 *
 * threadId: persisted conversation id; sent with each request so the backend
 *           can load prior messages. Cleared on "Clear Chat" so next message
 *           starts a fresh thread.
 */
export default function useStreamingChat() {
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (message, context = "") => {
    setIsStreaming(true);
    setError(null);

    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      const body = { message, context };
      if (threadId) body.thread_id = threadId;

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const newThreadId = res.headers.get("X-Thread-Id");
      if (newThreadId && !threadId) setThreadId(newThreadId);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: updated[updated.length - 1].content + chunk,
          };
          return updated;
        });
      }
    } catch (err) {
      setError(err.message);
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `Error: ${err.message}`,
        };
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  }, [threadId]);

  const clearChat = useCallback(async () => {
    if (threadId) {
      try {
        await fetch(`/api/chat/thread/${threadId}`, { method: "DELETE" });
      } catch (_) {}
    }
    setThreadId(null);
    setMessages([]);
    setError(null);
  }, [threadId]);

  return { messages, isStreaming, error, sendMessage, clearChat };
}
