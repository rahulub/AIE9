#!/usr/bin/env python3
"""Generate a PDF report from ragas_testset.json."""
import json
from pathlib import Path

from fpdf import FPDF

SCRIPT_DIR = Path(__file__).resolve().parent
JSON_PATH = SCRIPT_DIR / "ragas_testset.json"
PDF_PATH = SCRIPT_DIR / "ragas_testset.pdf"


def _safe_text(s: str) -> str:
    """Replace Unicode chars that Helvetica cannot render."""
    return (
        s.replace("\u2014", "-")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        records = json.load(f)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _safe_text("AI Realtor - RAGAS Test Set"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Total samples: {len(records)}", ln=True)
    pdf.ln(8)

    for i, rec in enumerate(records, 1):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, f"Sample {i}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        q = _safe_text(str(rec.get("user_input", ""))[:500])
        pdf.multi_cell(0, 5, f"Question: {q}")
        pdf.cell(0, 4, "", ln=True)
        ref = _safe_text(str(rec.get("reference", ""))[:800])
        pdf.multi_cell(0, 5, f"Reference Answer: {ref}")
        pdf.ln(4)
        if pdf.get_y() > 260:
            pdf.add_page()

    pdf.output(str(PDF_PATH))
    print(f"Saved PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
