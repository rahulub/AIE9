import React, { useState } from "react";
import ChatInput from "./components/ChatInput";
import ChatResponse from "./components/ChatResponse";
import InspectionUpload from "./components/InspectionUpload";
import useStreamingChat from "./hooks/useStreamingChat";

export default function App() {
  const [message, setMessage] = useState("");

  const { messages, isStreaming, error, sendMessage, clearChat } = useStreamingChat();

  const handleSubmit = () => {
    if (!message.trim() || isStreaming) return;
    sendMessage(message);
    setMessage("");
  };

  // Called by InspectionUpload after a PDF is successfully ingested.
  // Automatically fires the red-flags analysis prompt.
  const handleAnalyze = (filename) => {
    const prompt =
      `The inspection report "${filename}" has been indexed into the knowledge base. ` +
      "Search for and identify ALL red flags across these areas: " +
      "structural issues, roof, foundation, electrical, plumbing, HVAC, water damage, mold, safety hazards. " +
      "For each red flag found, provide: " +
      "1) The issue description, " +
      "2) Severity (🔴 Critical / 🟠 Major / 🟡 Minor), " +
      "3) Page number from the report. " +
      "Search thoroughly — make multiple retrieve_context calls covering different problem areas.";
    sendMessage(prompt);
  };

  return (
    <div style={{ maxWidth: 820, margin: "40px auto", padding: "0 20px" }}>

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24 }}>🏠 Home Inspection Assistant</h1>
          <p style={{ margin: "4px 0 0", color: "#666", fontSize: 14 }}>
            Upload an inspection report to identify red flags with page references
          </p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            disabled={isStreaming}
            style={{
              padding: "6px 14px",
              background: "transparent",
              border: "1px solid #ccc",
              borderRadius: 6,
              cursor: "pointer",
              color: "#555",
              fontSize: 13,
            }}
          >
            Clear Chat
          </button>
        )}
      </div>

      {/* PDF Upload */}
      <InspectionUpload onAnalyze={handleAnalyze} disabled={isStreaming} />

      {/* Chat history */}
      {error && <p style={{ color: "red", margin: "8px 0" }}>Error: {error}</p>}
      <ChatResponse messages={messages} isStreaming={isStreaming} />

      {/* Message input — always visible for follow-up questions */}
      <div style={{ marginTop: 16 }}>
        <ChatInput
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onSubmit={handleSubmit}
          disabled={isStreaming}
        />
        <p style={{ fontSize: 12, color: "#aaa", margin: "4px 0 0" }}>
          Ask follow-up questions about the report after the analysis completes.
        </p>
      </div>

    </div>
  );
}
