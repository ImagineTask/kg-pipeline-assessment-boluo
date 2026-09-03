"""Stage 1.1 cross-check - diff Document AI against `pdftotext -layout`.

Near-free, and it catches OCR regressions that no downstream check would notice.
A page where the two extractors disagree by more than 2% on character count is
flagged for review.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from src.common import ROOT, SETTINGS, load_jsonl, path, report, write_jsonl

THRESHOLD = 0.02


def _norm(text: str) -> str:
    """Compare content, not whitespace: -layout pads with spaces, DocAI does not."""
    return re.sub(r"\s+", "", text)


def pdftotext_pages(pdf: Path) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.txt"
        subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), str(out)],
            check=True,
            capture_output=True,
        )
        raw = out.read_text(encoding="utf-8", errors="replace")
    # pdftotext separates pages with a form feed
    pages = raw.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return [{"page": i + 1, "text": t} for i, t in enumerate(pages)]


def main() -> None:
    pdf = ROOT / SETTINGS["paths"]["raw_pdf"]
    ref = pdftotext_pages(pdf)
    write_jsonl(path("pdftotext_pages"), ref)

    docai = {p["page"]: p for p in load_jsonl(path("raw_pages"))}
    ref_by_page = {p["page"]: p for p in ref}

    rows, flagged = [], []
    for page in sorted(set(docai) | set(ref_by_page)):
        a = _norm(docai.get(page, {}).get("text", ""))
        b = _norm(ref_by_page.get(page, {}).get("text", ""))
        denom = max(len(a), len(b), 1)
        delta = abs(len(a) - len(b)) / denom
        row = {"page": page, "docai_chars": len(a), "pdftotext_chars": len(b), "delta": round(delta, 4)}
        rows.append(row)
        if delta > THRESHOLD:
            flagged.append(row)

    within = len(rows) - len(flagged)
    summary = {
        "pages_compared": len(rows),
        "pages_within_2pct": within,
        "pct_within_2pct": round(100 * within / max(len(rows), 1), 2),
        "pages_flagged": len(flagged),
        "flagged": flagged[:50],
        "acceptance_95pct": (within / max(len(rows), 1)) >= 0.95,
    }
    report("pdftotext_diff.json", summary)
    print(
        f"{within}/{len(rows)} pages within 2% "
        f"({summary['pct_within_2pct']}%) - acceptance {'PASS' if summary['acceptance_95pct'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
