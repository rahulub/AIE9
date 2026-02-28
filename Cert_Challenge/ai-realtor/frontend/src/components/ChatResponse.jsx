import React, { useEffect, useRef } from "react";

const styles = {
  container: {
    marginTop: 24,
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  bubble: (role) => ({
    maxWidth: "80%",
    padding: "10px 14px",
    borderRadius: 12,
    lineHeight: 1.6,
    whiteSpace: "pre-wrap",
    fontFamily: "inherit",
    fontSize: 15,
    alignSelf: role === "user" ? "flex-end" : "flex-start",
    background: role === "user" ? "#0070f3" : "#f1f1f1",
    color: role === "user" ? "#fff" : "#111",
  }),
  label: (role) => ({
    fontSize: 11,
    fontWeight: 600,
    color: "#888",
    marginBottom: 2,
    textAlign: role === "user" ? "right" : "left",
  }),
  streamingDot: {
    display: "inline-block",
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#888",
    marginLeft: 6,
    animation: "blink 1s infinite",
  },
};

export default function ChatResponse({ messages, isStreaming }) {
  const bottomRef = useRef(null);

  // Auto-scroll to the latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (!messages || messages.length === 0) return null;

  return (
    <>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
      <div style={styles.container}>
        {messages.map((msg, i) => {
          const isLastAssistant =
            msg.role === "assistant" && i === messages.length - 1;
          return (
            <div key={i}>
              <div style={styles.label(msg.role)}>
                {msg.role === "user" ? "You" : "Assistant"}
              </div>
              <div style={styles.bubble(msg.role)}>
                {msg.content}
                {isLastAssistant && isStreaming && (
                  <span style={styles.streamingDot} />
                )}
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </>
  );
}
