"""Stage 1.3 - structure-aware hierarchical chunking.

No fixed-size or recursive-character splitting: the document carries its own
numbering hierarchy, and that hierarchy is the chunk boundary. Two levels of
segmentation - documents (schedules) then clauses - produce leaf chunks that are
complete provisions, with parent links for small-to-big retrieval.

Operates on document_stream.jsonl. Pages are never containers here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.common import ROOT, SETTINGS, load_jsonl, path, report, write_jsonl
from src.textutils import (
    CLAUSE_NUMBER,
    DOC_HEADING,
    SUB_HEADING,
    clause_number,
    is_all_caps_heading,
    is_truncated,
    unbalanced_parens,
)

CFG = SETTINGS["chunk"]

# A defined term in Joint Schedule 1: `"Term" means ...` / `Term has the meaning ...`
# Document AI sometimes merges the term column into the definition column, so a
# defined term arrives as one block: 'Framework Contract" the framework
# agreement established between...'. Without this the term is lost and its text
# is appended to whichever definition came before it.
INLINE_DEFINITION = re.compile(
    r'^\s*[\u201c"\']?([A-Z][A-Za-z0-9 ,\-/&\'\u2019\(\)]{2,80}?)[\u201d"\']\s+(?=[a-z(])'
)

DEFINITION_START = re.compile(
    r'^\s*[“"\'‘]?([A-Z][A-Za-z0-9 ,\-/&\'’\(\)\.]{2,90}?)[”"\'’]?\s+'
    r"(means\b|shall mean\b|has the meaning\b|shall have the meaning\b|includes\b)",
)


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", s)


BOILER_LINE = re.compile(r"^\s*(Crown Copyright|Call[- ]?Off Ref|Framework Ref|Project Version|Model Version)", re.I)


def clean_title(text: str) -> str:
    """Recover a schedule title from a heading block.

    Two complications. Document AI merges the copyright line into the same block
    as the running header, so trailing boilerplate lines are dropped. And titles
    wrap across two printed lines - "Call-Off Schedule 24 (Supplier Furnished" /
    "Terms)" - so lines keep being absorbed while the parenthesis is still open.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    title = ""
    for line in lines:
        if BOILER_LINE.match(line):
            break
        title = f"{title} {line}".strip() if title else line
        if not unbalanced_parens(title):
            break
    title = re.sub(r"\s*(Crown Copyright|Call[- ]?Off Ref|Framework Ref).*$", "", title, flags=re.I)
    return re.sub(r"\s+", " ", title).strip()


RM_PREFIX = re.compile(r"^\s*RM\d+\s+(?:Network Services\s+\d+\s+)?", re.I)


def doc_id_for(heading: str) -> str | None:
    t = RM_PREFIX.sub("", re.sub(r"\s+", " ", heading)).strip()
    m = re.match(r"^(Core Terms)\b", t, re.I)
    if m:
        return "core_terms"
    m = re.match(r"^(Framework Award Form)\b", t, re.I)
    if m:
        return "framework_award_form"
    m = re.match(r"^(Framework|Joint|Call[- ]?Off)\s+Schedule\s+(\d+)\b", t, re.I)
    if m:
        family = m.group(1).lower().replace("-", "_").replace(" ", "_")
        family = "call_off" if family.startswith("call") else family
        return f"{family}_schedule_{m.group(2)}"
    return None


@dataclass
class Clause:
    clause_id: str
    doc_id: str
    doc_type: str
    number: str | None
    parent_id: str | None
    depth: int
    heading: str | None
    text: str
    hierarchy_path: str
    page_start: int
    page_end: int
    chunk_type: str = "clause"
    section: str | None = None
    is_split: bool = False
    blocks: list[int] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "clause_id": self.clause_id,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "number": self.number,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "heading": self.heading,
            "text": self.text.strip(),
            "hierarchy_path": self.hierarchy_path,
            "section": self.section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "spans_pages": self.page_end > self.page_start,
            "chunk_type": self.chunk_type,
            "is_split": self.is_split,
            "char_count": len(self.text.strip()),
        }


def doc_type_for(doc_id: str) -> str:
    if doc_id.startswith("framework_schedule"):
        return "framework_schedule"
    if doc_id.startswith("joint_schedule"):
        return "joint_schedule"
    if doc_id.startswith("call_off_schedule"):
        return "call_off_schedule"
    return doc_id


# --------------------------------------------------------------------------- #
# document segmentation
# --------------------------------------------------------------------------- #
def page_document_map(page_headers: dict[int, list[str]], n_pages: int) -> dict[int, str | None]:
    """Assign each page to a document using its running header.

    This is the reliable boundary signal in this PDF. A heading regex over body
    text is not: the Framework Award Form lists all 25 Call-Off Schedules by name
    on pages 25-26, and segmenting on those names opens twenty empty documents
    and mis-assigns every schedule that follows. The running header, by contrast,
    names the schedule each page actually belongs to.
    """
    out: dict[int, str | None] = {}
    for page in range(1, n_pages + 1):
        did = None
        for candidate in page_headers.get(page, []) or page_headers.get(str(page), []):
            did = doc_id_for(clean_title(candidate))
            if did:
                break
        out[page] = did
    return out


def segment_documents(blocks: list[dict], page_doc: dict[int, str | None]) -> list[dict]:
    """Split the stream into documents.

    Header-derived assignment first. Short schedules carry no running header at
    all (Core Terms, and fourteen others), so those runs fall back to a heading
    match on a standalone body block - safe here precisely because the run is
    bounded by pages the headers already claimed.
    """
    docs: list[dict] = []
    current: dict | None = None
    seen: set[str] = set()

    def open_doc(did: str, title: str, page: int) -> None:
        nonlocal current
        seen.add(did)
        current = {"doc_id": did, "title": title, "page_start": page, "blocks": []}
        docs.append(current)

    for b in blocks:
        text = b["text"].strip()
        title = clean_title(text)
        header_doc = page_doc.get(b["page"])

        if header_doc:
            if current is None or current["doc_id"] != header_doc:
                if header_doc in seen:
                    # a header reappearing after the document closed: keep the
                    # existing document rather than opening a duplicate
                    current = next(d for d in docs if d["doc_id"] == header_doc)
                else:
                    open_doc(header_doc, title if doc_id_for(title) == header_doc else header_doc, b["page"])
            if doc_id_for(title) == header_doc and len(title) <= 140 and not current["blocks"]:
                current["title"] = title      # prefer the document's own title block
                continue
        else:
            did = doc_id_for(title) if DOC_HEADING.match(title) else None
            standalone = len(title) <= 140 and not title.endswith((".", ";")) and not b.get("label")
            if did and did not in seen and standalone:
                open_doc(did, title, b["page"])
                continue

        if current is None:
            open_doc("front_matter", "Front Matter", b["page"])
        current["blocks"].append(b)

    for d in docs:
        d["page_end"] = max((b["page_end"] for b in d["blocks"]), default=d["page_start"])
    return docs


# --------------------------------------------------------------------------- #
# clause segmentation
# --------------------------------------------------------------------------- #
def _plausible_number(line: str) -> bool:
    """A clause number, not a year or a quantity. "2018 was the year..." matches
    the numbering regex; a clause numbered 2018 does not exist in this document."""
    m = CLAUSE_NUMBER.match(line)
    if not m:
        return False
    parts = m.group(1).split(".")
    return len(parts) > 1 and all(int(x) <= 200 for x in parts)


def explode_embedded_numbers(blocks: list[dict]) -> list[dict]:
    """Split a block that carries a second clause number on a later line.

    Document AI occasionally emits two consecutive provisions as one block. Left
    alone, the second clause is swallowed by the first and never gets an id.
    """
    out: list[dict] = []
    for b in blocks:
        lines = b["text"].splitlines()
        if b.get("block_type") == "table" or len(lines) < 2:
            out.append(b)
            continue
        cuts = [
            i for i, line in enumerate(lines)
            if i and _plausible_number(line) and not is_truncated("\n".join(lines[:i]))
        ]
        if not cuts:
            out.append(b)
            continue
        for start, end in zip([0] + cuts, cuts + [len(lines)]):
            part = "\n".join(lines[start:end]).strip()
            if part:
                keep_label = start == 0
                out.append(dict(
                    b, text=part,
                    label=b.get("label") if keep_label else None,
                    label_type=b.get("label_type") if keep_label else None,
                ))
    return out



def _parent_of(number: str) -> str | None:
    parts = number.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


def segment_clauses(doc: dict) -> list[Clause]:
    doc_id = doc["doc_id"]
    dtype = doc_type_for(doc_id)
    clauses: list[Clause] = []
    by_number: dict[tuple[str | None, str], Clause] = {}
    section: str | None = None
    section_title: str | None = None
    part: str | None = None
    part_title: str | None = None
    current: Clause | None = None
    used_ids: set[str] = set()
    collisions: list[str] = []

    def new_id(number: str) -> tuple[str, str | None]:
        """Clause ids must be unique. Numbering restarts inside each Part and
        Annex, so the section qualifies the id; where even that collides (a
        schedule that repeats a Part heading) a suffix disambiguates, and the
        count is reported so the collision is visible rather than silent."""
        sec = section
        base = f"{doc_id}.{number}"
        if sec is None and base not in used_ids:
            return base, None
        candidate = f"{doc_id}.{sec}.{number}" if sec else base
        if candidate not in used_ids:
            return candidate, sec
        n = 2
        while f"{candidate}__{n}" in used_ids:
            n += 1
        collisions.append(candidate)
        return f"{candidate}__{n}", sec

    def path_for(number: str | None, heading: str | None) -> str:
        parts = [doc["title"]]
        if part_title:
            parts.append(part_title)
        if section_title and section_title != part_title:
            parts.append(section_title)
        if number:
            # walk up the numeric ancestry, using their headings where present
            chain = number.split(".")
            for i in range(1, len(chain) + 1):
                anc_num = ".".join(chain[:i])
                anc = by_number.get((section, anc_num))
                label = f"{anc_num} {anc.heading}" if anc and anc.heading else anc_num
                parts.append(label)
        elif heading:
            parts.append(heading)
        return " > ".join(parts)

    for b in explode_embedded_numbers(doc["blocks"]):
        text = b["text"].strip()
        if not text:
            continue

        if b.get("block_type") == "table":
            tid = f"{doc_id}.table.{b['block_id']}"
            clauses.append(
                Clause(
                    clause_id=tid, doc_id=doc_id, doc_type=dtype, number=None,
                    parent_id=current.clause_id if current else doc_id,
                    depth=(current.depth + 1) if current else 1,
                    heading=None, text=text,
                    hierarchy_path=path_for(current.number if current else None, "table"),
                    page_start=b["page"], page_end=b["page_end"], chunk_type="table",
                    section=section, blocks=[b["block_id"]],
                )
            )
            used_ids.add(tid)
            current = None
            continue

        # Part / Annex / Appendix headings scope the numbering that follows:
        # paragraph numbering restarts inside each Part, so the section becomes
        # part of the clause id.
        sub = SUB_HEADING.match(text)
        if sub and len(text) <= 140:
            label = re.sub(r"\s+", " ", text).strip()
            if re.match(r"^Part\s", label, re.I):
                part, part_title = slug(sub.group(1)), label
                section, section_title = part, label
            else:
                section, section_title = slug(sub.group(1)), label
                if part:
                    section = f"{part}.{section}"
            current = None
            continue

        # A term-labelled row is a defined term set against its definition in a
        # two-column layout. One defined term = one chunk, keyed by the term,
        # not by a clause number. Local definitions in other schedules use the
        # same layout and override Joint Schedule 1 within their own document.
        inline_term = None
        if b.get("label_type") != "term":
            m = INLINE_DEFINITION.match(text)
            if m:
                inline_term = m.group(1).strip(' "\u201c\u201d')
        if b.get("label_type") == "term" or inline_term:
            term = inline_term or b["label"]
            key = slug(term) or f"t{b['block_id']}"
            cid = f"{doc_id}.def.{key}"
            if cid in used_ids:
                cid = f"{cid}_{b['block_id']}"
            clause = Clause(
                clause_id=cid, doc_id=doc_id, doc_type=dtype, number=None,
                parent_id=doc_id, depth=1, heading=term, text=text,
                hierarchy_path=f'{doc["title"]} > "{term}"',
                page_start=b["page"], page_end=b["page_end"],
                chunk_type="definition", section=section, blocks=[b["block_id"]],
            )
            clauses.append(clause)
            used_ids.add(cid)
            current = clause
            continue

        num = clause_number(text)
        if num:
            body = text[CLAUSE_NUMBER.match(text).end():].strip()
            heading = None
            # "3. What needs to be delivered", "3.1 All deliverables" - a numbered
            # heading, not a provision. Requiring no terminal punctuation is what
            # keeps short real provisions ("3.3.1 Late Delivery ... Contract.")
            # out of this branch.
            if (
                body
                and len(body) < 90
                and len(body.split()) <= 12
                and not body.rstrip().endswith((".", ":", ";", "?", "!"))
                and body[:1].isupper()
            ):
                heading, body = body, ""
            cid, sec = new_id(num)
            parent_num = _parent_of(num)
            parent_id = None
            if parent_num and (sec, parent_num) in by_number:
                parent_id = by_number[(sec, parent_num)].clause_id
            else:
                parent_id = doc_id
            clause = Clause(
                clause_id=cid, doc_id=doc_id, doc_type=dtype, number=num,
                parent_id=parent_id, depth=num.count(".") + 1, heading=heading,
                text=body, hierarchy_path="", page_start=b["page"], page_end=b["page_end"],
                section=sec, blocks=[b["block_id"]],
            )
            by_number[(sec, num)] = clause
            clause.hierarchy_path = path_for(num, heading)
            clauses.append(clause)
            used_ids.add(cid)
            current = clause
            continue

        if current is not None:
            current.text = (current.text + " " + text).strip()
            current.page_end = max(current.page_end, b["page_end"])
            current.blocks.append(b["block_id"])
            continue

        # unnumbered narrative before the first clause of a document/section
        heading = text if is_all_caps_heading(text) else None
        pid = f"{doc_id}.{section}.pre.{b['block_id']}" if section else f"{doc_id}.pre.{b['block_id']}"
        clauses.append(
            Clause(
                clause_id=pid, doc_id=doc_id, doc_type=dtype, number=None,
                parent_id=doc_id, depth=1, heading=heading, text=text,
                hierarchy_path=path_for(None, heading or section_title),
                page_start=b["page"], page_end=b["page_end"], chunk_type="preamble",
                section=section, blocks=[b["block_id"]],
            )
        )
        used_ids.add(pid)

    doc["id_collisions"] = collisions
    return clauses


# --------------------------------------------------------------------------- #
def split_long(clauses: list[Clause]) -> list[dict]:
    """Split over-long leaves on sentence boundaries, flagged so retrieval can
    always pull the siblings back together."""
    limit = CFG["max_leaf_chars"]
    rows: list[dict] = []
    for c in clauses:
        row = c.as_row()
        if row["chunk_type"] == "table" or row["char_count"] <= limit:
            rows.append(row)
            continue
        sentences = re.split(r"(?<=[.;:])\s+", row["text"])
        parts, buf = [], ""
        for s in sentences:
            if buf and len(buf) + len(s) + 1 > limit:
                parts.append(buf.strip())
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            parts.append(buf.strip())
        if len(parts) == 1:
            rows.append(row)
            continue
        first_id = f"{row['clause_id']}#p1"
        for i, part in enumerate(parts, start=1):
            sub = dict(row)
            sub["clause_id"] = f"{row['clause_id']}#p{i}"
            # later parts hang off the first, which is the surviving node; the
            # unsplit id no longer exists, so parenting onto it would orphan them
            sub["parent_id"] = first_id if i > 1 else row["parent_id"]
            sub["text"] = part
            sub["char_count"] = len(part)
            sub["is_split"] = True
            rows.append(sub)
    return rows


def main() -> None:
    import json

    blocks = load_jsonl(path("document_stream"))
    headers_path = ROOT / SETTINGS["paths"]["reports"] / "page_headers.json"
    page_headers = {int(k): v for k, v in json.loads(headers_path.read_text()).items()}
    n_pages = max(b["page_end"] for b in blocks)
    docs = segment_documents(blocks, page_document_map(page_headers, n_pages))

    all_clauses: list[Clause] = []
    doc_rows: list[dict] = []
    for doc in docs:
        clauses = segment_clauses(doc)
        all_clauses.extend(clauses)
        doc_rows.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"][:120],
                "doc_type": doc_type_for(doc["doc_id"]),
                "page_start": doc["page_start"],
                "page_end": doc["page_end"],
                "n_blocks": len(doc["blocks"]),
                "n_chunks": len(clauses),
                "id_collisions": len(doc.get("id_collisions", [])),
            }
        )

    rows = split_long(all_clauses)
    n = write_jsonl(path("clauses"), rows)
    report("documents.json", doc_rows)
    report(
        "chunk_report.json",
        {
            "documents": len(docs),
            "chunks": n,
            "by_type": {
                t: sum(1 for r in rows if r["chunk_type"] == t)
                for t in {r["chunk_type"] for r in rows}
            },
            "spanning_pages": sum(1 for r in rows if r["spans_pages"]),
            "split_leaves": sum(1 for r in rows if r["is_split"]),
            "mean_chars": round(sum(r["char_count"] for r in rows) / max(n, 1), 1),
        },
    )
    print(f"{len(docs)} documents, {n} chunks")


if __name__ == "__main__":
    main()
