import React, { useState } from "react";

/**
 * PdfUpload — lets the user upload a PDF to the /api/ingest endpoint.
 * On success, shows how many chunks were stored in Qdrant.
 */
export default function PdfUpload() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setStatus(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        setStatus({ type: "error", message: data.detail || "Upload failed." });
      } else {
        setStatus({ type: "success", message: data.message });
      }
    } catch (err) {
      setStatus({ type: "error", message: err.message });
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  };

  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ fontWeight: "bold" }}>Upload PDF to knowledge base: </label>
      <input
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        disabled={loading}
        style={{ marginLeft: 8 }}
      />
      {loading && <span style={{ marginLeft: 8, color: "#888" }}>Ingesting...</span>}
      {status && (
        <p style={{ color: status.type === "error" ? "red" : "green", margin: "4px 0" }}>
          {status.message}
        </p>
      )}
    </div>
  );
}
