import React from "react";

/**
 * ChatInput — renders the message input field and submit button.
 *
 * Props:
 *   value      - controlled input value
 *   onChange   - handler for input changes
 *   onSubmit   - handler called when user clicks Send or presses Enter
 *   disabled   - disables input while streaming
 */
export default function ChatInput({ value, onChange, onSubmit, disabled }) {
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
      <input
        type="text"
        value={value}
        onChange={onChange}
        onKeyDown={handleKeyDown}
        placeholder="Type your message..."
        disabled={disabled}
        style={{ flex: 1, padding: "8px 12px", fontSize: 16 }}
      />
      <button onClick={onSubmit} disabled={disabled}>
        {disabled ? "Streaming..." : "Send"}
      </button>
    </div>
  );
}
