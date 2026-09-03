"""Stage 1.2 - page-boundary reconstruction.

The page is a printing artefact, not a unit of meaning. This stage destroys page
boundaries and emits one continuous, ordered block stream, keeping the page only
as citation metadata. Everything downstream reads the stream, never pages.

Order of operations:
  1. strip repeated boilerplate                       (src/ingest/boilerplate.py)
  2. merge hanging-indent rows into single blocks
  3. detect and merge tables, including across pages
  4. attach footnotes to the block that references them
  5. repair splits by joining blocks broken at a page break

Step 2 is not in the spec but is load-bearing for this PDF. Most schedules set
the clause number in its own column ("2.1.3" at x=359, its text at x=514), and
Joint Schedule 1 sets the defined term in a left column against its definition on
the right. Without a row merge the clause numbers arrive as free-floating blocks
and every clause in every schedule loses its number.
"""
from __future__ import annotations

import re
from collections import defaultdict

from src.common import SETTINGS, load_jsonl, path, report, write_jsonl
from src.ingest.boilerplate import detect as detect_boilerplate, shape
from src.textutils import (
    LETTERED_LIMB,
    ends_hyphenated,
    ends_open,
    is_doc_heading,
    is_heading_like,
    is_truncated,
    starts_new_unit,
    unbalanced_parens,
)

FOOTNOTE_MARKER = re.compile(r"^\s*(\d{1,2}|\*{1,3})\s+(?=[A-Z(\"'])")
ROW_TOLERANCE = 16.0   # points; blocks within this vertical distance share a row
X_TOLERANCE = 30.0     # points; column positions match within this

# a left-column cell that is a numbering label rather than prose
LABEL_NUMBER = re.compile(r"^\(?\s*(?:\d+(?:\.\d+){0,3}\.?|[a-z]{1,2}|[ivxlcdm]{1,6})\s*[\.\)]?$", re.I)
BULLET = re.compile(r"^[\u2022\u25aa\u25cb\u25e6\u00b7\u039f\-o]$")
TERM_LIKE = re.compile(r'^[“"\'‘]?[A-Za-z]')


# --------------------------------------------------------------------------- #
# 1. flatten pages into blocks
# --------------------------------------------------------------------------- #
def flatten(pages: list[dict], plan: dict) -> list[dict]:
    """Drop boilerplate blocks, and scrub boilerplate *lines* out of the blocks
    that survive.

    Document AI sometimes merges a running header into the same block as body
    text ("[insert licence terms] RM6116 Call-Off Schedule 25 ... Crown Copyright
    2018"). Dropping such a block would delete real contract text, so confirmed
    boilerplate lines are removed individually instead.
    """
    drop, keep, boiler_lines = plan["drop"], plan["keep"], plan["lines"]
    blocks: list[dict] = []
    for page in pages:
        for i, block in enumerate(page["blocks"]):
            if (page["page"], i) in drop:
                continue
            raw = block["text"]
            if (page["page"], i) not in keep:
                lines = [l for l in raw.splitlines() if shape(l) not in boiler_lines or not l.strip()]
                raw = "\n".join(lines)
            text = re.sub(r"[ \t]+", " ", raw).strip()
            if not text:
                continue
            blocks.append(
                {
                    "text": text,
                    "page": page["page"],
                    "page_height": page["page_height"],
                    "page_width": page["page_width"],
                    "y_top": block.get("y_top"),
                    "y_bottom": block["bbox"][3] if block.get("bbox") else block.get("y_top"),
                    "x_left": block.get("x_left"),
                    "x_right": block.get("x_right"),
                }
            )
    return blocks


# --------------------------------------------------------------------------- #
# 2. row merging
# --------------------------------------------------------------------------- #
def row_bands(page_blocks: list[dict]) -> list[list[dict]]:
    bands: list[list[dict]] = []
    for b in sorted(page_blocks, key=lambda b: ((b["y_top"] or 0), (b["x_left"] or 0))):
        if bands and abs((b["y_top"] or 0) - (bands[-1][0]["y_top"] or 0)) <= ROW_TOLERANCE:
            bands[-1].append(b)
        else:
            bands.append([b])
    for band in bands:
        band.sort(key=lambda b: b["x_left"] or 0)
    return bands


def classify_row(band: list[dict]) -> str:
    """single | label | term | table_row"""
    if len(band) == 1:
        return "single"
    if len(band) > 2:
        return "table_row"
    left, right = band
    width = left["page_width"]
    if (right["x_left"] or 0) <= (left["x_right"] or 0):   # overlapping columns
        return "table_row"
    lt = re.sub(r"\s+", " ", left["text"]).strip()
    if BULLET.match(lt):
        return "bullet"
    if LABEL_NUMBER.match(lt):
        return "label"
    # Joint Schedule 1 lays the defined term in a narrow left column against the
    # definition body on the right.
    narrow_left = (left["x_left"] or 0) <= width * 0.32
    far_right = (right["x_left"] or 0) >= width * 0.28
    letters = sum(1 for ch in lt if ch.isalpha())
    if (
        narrow_left
        and far_right
        and len(lt) <= 90
        and not lt.endswith(".")
        and letters >= 2
        and TERM_LIKE.match(lt)
    ):
        return "term"
    return "table_row"


def merge_row(band: list[dict], kind: str) -> dict:
    cells = [re.sub(r"\s+", " ", b["text"]).strip() for b in band]
    merged = dict(band[0])
    merged["y_top"] = min(b["y_top"] or 0 for b in band)
    merged["y_bottom"] = max(b["y_bottom"] or 0 for b in band)
    merged["x_left"] = min(b["x_left"] or 0 for b in band)
    merged["x_right"] = max(b["x_right"] or 0 for b in band)
    if kind == "bullet":
        merged["text"] = f"\u2022 {cells[1]}"
    elif kind == "label":
        label = cells[0].strip(" .)")
        merged["text"] = f"{label} {cells[1]}"
        merged["label"], merged["label_type"], merged["body"] = label, "number", cells[1]
    elif kind == "term":
        term = cells[0].strip(' "“”\'’')
        merged["text"] = f'"{term}" {cells[1]}'
        merged["label"], merged["label_type"], merged["body"] = term, "term", cells[1]
    else:
        merged["text"] = " ".join(cells)
    return merged


# --------------------------------------------------------------------------- #
# 3. tables
# --------------------------------------------------------------------------- #
def _columns(band: list[dict]) -> list[float]:
    return sorted(b["x_left"] or 0 for b in band)


def _similar(a: list[float], b: list[float]) -> bool:
    if not a or not b or abs(len(a) - len(b)) > 1:
        return False
    hits = sum(1 for x in a if any(abs(x - y) <= X_TOLERANCE for y in b))
    return hits >= max(2, int(0.7 * min(len(a), len(b))))


def to_markdown(rows: list[list[str]]) -> str:
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return ""
    padded = [r + [""] * (width - len(r)) for r in rows]
    clean = [[re.sub(r"\s+", " ", c).replace("|", r"\|").strip() for c in r] for r in padded]
    head, *body = clean
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def rejoin_wrapped_terms(blocks: list[dict]) -> list[dict]:
    """Re-join a defined term whose *label* wraps across two printed rows.

    "Framework Contract Period" is set as "Framework Contract" on one row and
    "Period" on the next, each paired with half of its definition. Treated as two
    terms, both halves are wrong and the definition is truncated mid-sentence.
    """
    out: list[dict] = []
    for b in blocks:
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.get("label_type") == "term"
            and b.get("label_type") == "term"
            and prev["page"] == b["page"]
            and ends_open(prev.get("body", ""))
        ):
            prev["label"] = f"{prev['label']} {b['label']}".strip()
            prev["body"] = f"{prev.get('body', '')} {b.get('body', '')}".strip()
            prev["text"] = f'"{prev["label"]}" {prev["body"]}'
            prev["y_bottom"] = max(prev.get("y_bottom") or 0, b.get("y_bottom") or 0)
            continue
        out.append(b)
    return out


# --------------------------------------------------------------------------- #
# 4. footnotes
# --------------------------------------------------------------------------- #
def attach_footnotes(blocks: list[dict]) -> tuple[list[dict], int]:
    """Footnotes sit at the bottom of a page but belong to a sentence in the body.

    Left free-floating they merge into whichever clause happens to follow,
    silently corrupting it.
    """
    by_page: dict[int, list[int]] = defaultdict(list)
    for idx, b in enumerate(blocks):
        by_page[b["page"]].append(idx)

    drop: set[int] = set()
    attached = 0
    for idxs in by_page.values():
        if len(idxs) < 2:
            continue
        h = blocks[idxs[0]]["page_height"]
        for idx in idxs[-3:]:
            b = blocks[idx]
            if b["y_top"] is None or b["y_top"] < h * 0.80 or b.get("table_id"):
                continue
            m = FOOTNOTE_MARKER.match(b["text"])
            if not m or len(b["text"]) > 600:
                continue
            marker, body = m.group(1), b["text"][m.end():].strip()
            host = None
            for cand in idxs:
                if cand == idx or cand in drop:
                    continue
                if re.search(rf"[a-z\)]{re.escape(marker)}\b", blocks[cand]["text"]):
                    host = cand
            if host is None:
                continue
            blocks[host]["text"] += f" [fn: {body}]"
            drop.add(idx)
            attached += 1
    return [b for i, b in enumerate(blocks) if i not in drop], attached


# --------------------------------------------------------------------------- #
# 5. join repair
# --------------------------------------------------------------------------- #
def should_join(prev: dict, nxt: dict) -> str | None:
    a, b = prev["text"], nxt["text"]
    if prev.get("block_kind") == "table" or nxt.get("block_kind") == "table":
        return None
    if nxt.get("label"):          # a labelled row is always a new unit
        return None
    # A heading that wraps across two printed lines leaves its parenthesis open:
    # "Framework Schedule 6 (Order Form Template and" / "Call-Off Schedules)".
    if is_doc_heading(a) and unbalanced_parens(a):
        return "wrapped_heading"
    # Truncation beats structure. Clause 10.6.1 continues onto a line beginning
    # "20.2 or a Contract expires..." - a cross-reference number, not a clause
    # number - and refusing to join there silently drops the rest of the clause.
    # A genuine new clause never follows a sentence that stops mid-phrase.
    if is_truncated(a) and not is_heading_like(a):
        return "truncated"
    if nxt.get("label"):          # a labelled row is otherwise a new unit
        return None
    if starts_new_unit(b):
        return None
    if is_heading_like(a) and not ends_hyphenated(a):
        return None

    if ends_hyphenated(a) and b[:1].islower():
        return "dehyphenate"

    limb = LETTERED_LIMB.match(b)
    if limb and not a.rstrip().endswith("."):
        return "lettered_limb"

    if ends_open(a):
        # The literal rule "join only when the next line starts lowercase" is
        # wrong for this document. Clause 3.2.9 breaks as
        #   "...the Buyer needs to make use of the" / "Goods."
        # and "Goods" is a capitalised *defined term*, not a new sentence. An
        # unterminated line is the reliable signal; capitalisation is not.
        return "unterminated"

    if a.rstrip().endswith((":", ";")) and b[:1].islower():
        return "list_continuation"

    return None


def join_blocks(blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    out: list[dict] = []
    joins: list[dict] = []
    for b in blocks:
        if not out:
            out.append(dict(b, page_end=b["page"], spans_pages=False))
            continue
        prev = out[-1]
        reason = should_join(prev, b)
        if reason is None:
            out.append(dict(b, page_end=b["page"], spans_pages=False))
            continue
        if reason == "dehyphenate":
            prev["text"] = prev["text"].rstrip()[:-1] + b["text"].lstrip()
        else:
            # A term row absorbed into a truncated sentence contributes its body,
            # not its quoted term: Document AI split "Framework Contract Period"
            # across two rows, and the second row's label is half a term, not a
            # new one.
            tail = b["body"] if reason == "truncated" and b.get("label_type") == "term" else b["text"]
            prev["text"] = prev["text"].rstrip() + " " + tail.lstrip()
        if b["page"] != prev["page_end"]:
            joins.append(
                {
                    "reason": reason,
                    "from_page": prev["page_end"],
                    "to_page": b["page"],
                    "text": prev["text"][-120:],
                }
            )
        prev["page_end"] = max(prev["page_end"], b["page"])
        prev["spans_pages"] = prev["page_end"] != prev["page"]
    return out, joins


# --------------------------------------------------------------------------- #
def build_stream(pages: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    plan, _, page_headers = detect_boilerplate(pages)
    blocks = flatten(pages, plan)

    by_page: dict[int, list[dict]] = defaultdict(list)
    for b in blocks:
        by_page[b["page"]].append(b)

    # --- rows -> blocks, collecting table candidates --------------------------
    merged_blocks: list[dict] = []
    tables: list[dict] = []
    row_kinds: dict[str, int] = defaultdict(int)
    table_seq = 0

    for page in sorted(by_page):
        run: list[list[dict]] = []

        def flush_run() -> None:
            nonlocal run, table_seq
            if len(run) >= 2:
                table_seq += 1
                first = run[0][0]
                tables.append(
                    {
                        "table_id": table_seq,
                        "page": page,
                        "rows": [[re.sub(r"\s+", " ", c["text"]).strip() for c in band] for band in run],
                        "columns": _columns(run[0]),
                        "y_top": min(c["y_top"] or 0 for c in run[0]),
                        "y_bottom": max(c["y_bottom"] or 0 for band in run for c in band),
                        "page_height": first["page_height"],
                        "anchor": first,
                    }
                )
            else:
                for band in run:   # an isolated multi-column row is not a table
                    merged_blocks.append(merge_row(band, "table_row"))
            run = []

        for band in row_bands(by_page[page]):
            kind = classify_row(band)
            row_kinds[kind] += 1
            if kind == "table_row":
                if run and not _similar(_columns(run[-1]), _columns(band)):
                    flush_run()
                run.append(band)
                continue
            flush_run()
            merged_blocks.append(merge_row(band, kind) if len(band) > 1 else band[0])
        flush_run()

    # --- merge tables continued on the next page ------------------------------
    logical: list[dict] = []
    for t in sorted(tables, key=lambda t: (t["page"], t["y_top"])):
        prev = logical[-1] if logical else None
        continued = (
            prev is not None
            and t["page"] == prev["page_end"] + 1
            and _similar(prev["columns"], t["columns"])
            and t["y_top"] < t["page_height"] * 0.30
        )
        if continued:
            rows = t["rows"]
            if rows and prev["rows"] and rows[0] == prev["rows"][0]:
                rows = rows[1:]          # repeated header row
            prev["rows"].extend(rows)
            prev["page_end"] = t["page"]
        else:
            logical.append(
                {
                    "table_id": t["table_id"],
                    "page_start": t["page"],
                    "page_end": t["page"],
                    "columns": t["columns"],
                    "rows": list(t["rows"]),
                    "anchor": t["anchor"],
                }
            )

    for t in logical:
        a = t["anchor"]
        merged_blocks.append(
            {
                "text": to_markdown(t["rows"]),
                "page": t["page_start"],
                "page_end_hint": t["page_end"],
                "page_height": a["page_height"],
                "page_width": a["page_width"],
                "y_top": a["y_top"],
                "y_bottom": a["y_bottom"],
                "x_left": a["x_left"],
                "x_right": a["x_right"],
                "block_kind": "table",
                "table_id": t["table_id"],
            }
        )

    merged_blocks.sort(key=lambda b: (b["page"], b["y_top"] or 0, b["x_left"] or 0))
    merged_blocks = rejoin_wrapped_terms(merged_blocks)
    merged_blocks, n_footnotes = attach_footnotes(merged_blocks)
    stream, joins = join_blocks(merged_blocks)

    stats = {
        "row_kinds": dict(row_kinds),
        "footnotes_attached": n_footnotes,
        "page_tables": len(tables),
        "logical_tables": len(logical),
        "tables_spanning_pages": sum(1 for t in logical if t["page_end"] > t["page_start"]),
        "boilerplate_blocks_stripped": len(plan["drop"]),
        "page_headers": page_headers,
    }
    return stream, joins, logical, stats


def main() -> None:
    pages = load_jsonl(path("raw_pages"))
    raw_chars = sum(len(re.sub(r"\s+", "", b["text"])) for p in pages for b in p["blocks"])

    stream, joins, tables, stats = build_stream(pages)
    page_headers = stats.pop("page_headers")
    report("page_headers.json", page_headers)

    rows = []
    for i, b in enumerate(stream, start=1):
        page_end = max(b.get("page_end", b["page"]), b.get("page_end_hint", b["page"]))
        rows.append(
            {
                "block_id": i,
                "text": b["text"],
                "page": b["page"],
                "page_end": page_end,
                "spans_pages": page_end > b["page"],
                "y_top": b.get("y_top"),
                "x_left": b.get("x_left"),
                "label": b.get("label"),
                "label_type": b.get("label_type"),
                "block_type": "table" if b.get("block_kind") == "table" else "text",
            }
        )
    n = write_jsonl(path("document_stream"), rows)
    write_jsonl(
        path("tables"),
        [
            {
                "table_id": t["table_id"],
                "page_start": t["page_start"],
                "page_end": t["page_end"],
                "n_rows": len(t["rows"]),
                "spans_pages": t["page_end"] > t["page_start"],
                "markdown": to_markdown(t["rows"]),
            }
            for t in tables
        ],
    )

    kept_chars = sum(len(re.sub(r"\s+", "", r["text"])) for r in rows)
    retention = kept_chars / max(raw_chars, 1)
    summary = {
        **stats,
        "raw_blocks": sum(len(p["blocks"]) for p in pages),
        "stream_blocks": n,
        "blocks_spanning_pages": sum(1 for r in rows if r["spans_pages"]),
        "labelled_blocks": sum(1 for r in rows if r["label"]),
        "term_rows": sum(1 for r in rows if r["label_type"] == "term"),
        "cross_page_joins": len(joins),
        "join_reasons": {r: sum(1 for j in joins if j["reason"] == r) for r in {j["reason"] for j in joins}},
        "raw_chars": raw_chars,
        "stream_chars": kept_chars,
        "retention_vs_raw": round(retention, 4),
        "sample_cross_page_joins": joins[:20],
    }
    report("stitch_report.json", summary)
    print(
        f"stream: {n} blocks | {len(joins)} cross-page joins | "
        f"{len(tables)} tables ({summary['tables_spanning_pages']} spanning) | "
        f"retention {retention:.2%}"
    )


if __name__ == "__main__":
    main()
