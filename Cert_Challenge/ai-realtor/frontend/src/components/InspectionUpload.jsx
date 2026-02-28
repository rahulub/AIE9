import React, { useState } from "react";

const STATUS = { IDLE: "idle", UPLOADING: "uploading", READY: "ready", ERROR: "error" };

/**
 * InspectionUpload
 *
 * Props:
 *   onAnalyze(filename) — called after successful ingest;
 *                         parent uses this to trigger the red-flag chat message
 */
export default function InspectionUpload({ onAnalyze, disabled }) {
  const [status, setStatus] = useState(STATUS.IDLE);
  const [filename, setFilename] = useState(null);
  const [detail, setDetail] = useState("");

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = "";

    setStatus(STATUS.UPLOADING);
    setFilename(file.name);
    setDetail("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/ingest", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed.");
      setStatus(STATUS.READY);
      setDetail(`${data.pages} pages · ${data.chunks_ingested} chunks indexed`);
    } catch (err) {
      setStatus(STATUS.ERROR);
      setDetail(err.message);
    }
  };

  const statusColor = {
    [STATUS.IDLE]: "#555",
    [STATUS.UPLOADING]: "#888",
    [STATUS.READY]: "#16a34a",
    [STATUS.ERROR]: "#dc2626",
  }[status];

  return (
    <div style={{
      border: "1px solid #e5e7eb",
      borderRadius: 10,
      padding: 16,
      marginBottom: 16,
      background: "#fafafa",
    }}>
      <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 15 }}>
        📄 Upload Inspection Report
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <label style={{
          padding: "7px 14px",
          background: "#0070f3",
          color: "#fff",
          borderRadius: 6,
          cursor: disabled || status === STATUS.UPLOADING ? "not-allowed" : "pointer",
          fontSize: 14,
          opacity: disabled || status === STATUS.UPLOADING ? 0.6 : 1,
        }}>
          {status === STATUS.UPLOADING ? "Uploading…" : "Choose PDF"}
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            disabled={disabled || status === STATUS.UPLOADING}
            style={{ display: "none" }}
          />
        </label>

        {filename && (
          <span style={{ fontSize: 13, color: statusColor }}>
            {filename}
            {detail && <span style={{ color: "#888" }}> — {detail}</span>}
          </span>
        )}
      </div>

      {status === STATUS.READY && (
        <button
          onClick={() => onAnalyze(filename)}
          disabled={disabled}
          style={{
            marginTop: 12,
            padding: "8px 18px",
            background: "#16a34a",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: disabled ? "not-allowed" : "pointer",
            fontWeight: 600,
            fontSize: 14,
          }}
        >
          🔍 Analyze for Red Flags
        </button>
      )}
    </div>
  );
}
