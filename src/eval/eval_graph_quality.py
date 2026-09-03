"""Stage 4b - quality of the graph itself: coverage, recall and precision by type.

This is a different question from `run_eval.py`. That measures whether retrieval
finds what the graph knows. This measures whether the graph knows enough, which
nothing else here tests - and the golden set structurally *cannot* test, because
its cross-reference questions are generated from edges the resolver already
produced. A resolver that misses a whole class of references scores perfectly on
a set built from its own output.

Ground truth therefore comes from outside the graph:

  1. Coverage and reference-detection recall are measured against the parallel
     `pdftotext -layout` extraction, which shares no code with the Document AI
     path the graph was built from.
  2. Precision and recall by node/edge type are measured against a silver
     standard: a model reads raw page text it has never seen in chunked form and
     enumerates what is on the page, and the graph is scored against that.

Nothing here can prove the graph correct. It can catch a class of thing that is
systematically missing, which is the failure the rest of the harness is blind to.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types
from pydantic import BaseModel
from rapidfuzz import fuzz

from src.common import ROOT, SETTINGS, load_jsonl, path, report, write_jsonl
from src.extract.crossrefs import REFERENCE
from src.retrieval import queries as q
from src.vertex import VertexClient

SHINGLE = 5
SAMPLE_PAGES = 24
FUZZY = 85


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def shingles(text: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i: i + n]) for i in range(max(len(w) - n + 1, 0))}


def norm_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def fuzzy_in(needle: str, haystack: list[str], threshold: int = FUZZY) -> bool:
    n = norm_phrase(needle)
    return any(fuzz.ratio(n, norm_phrase(h)) >= threshold for h in haystack)


MARKER = re.compile(
    r"^(Crown Copyright|Framework Ref|Project Version|Model Version|Call[- ]?Off Ref|"
    r"RM\d+ Network Services)\b", re.I)
PAGE_NUMBER = re.compile(r"^-?\s*\d{1,3}\s*-?$")


def boilerplate_lines() -> set[str]:
    """Line shapes the stitcher removed, so the reference side is not penalised
    for the header and footer the graph was supposed to strip."""
    f = ROOT / SETTINGS["paths"]["reports"] / "boilerplate_report.json"
    if not f.exists():
        return set()
    return {p["shape"] for p in json.loads(f.read_text())["patterns"]}


def page_headers() -> dict[int, list[str]]:
    f = ROOT / SETTINGS["paths"]["reports"] / "page_headers.json"
    if not f.exists():
        return {}
    return {int(k): v for k, v in json.loads(f.read_text()).items()}


def strip_boilerplate(page_text: str, shapes: set[str], header_lines: list[str] | None = None) -> str:
    """Remove header and footer lines from the *reference* extraction.

    Shape matching alone is not enough on this side. `pdftotext -layout` pads with
    spaces and joins the footer's three fields onto one physical line, so its
    shapes do not match the ones Document AI produced - which left the running
    schedule title in place on every page and made the graph look as though it had
    dropped 138 references to "Framework Schedule 1" that were never references at
    all, just the header. Marker tokens and the page's own known header close that.

    This excludes text from the denominator, never adds any: boilerplate is by
    design absent from the graph, so counting it as missing measures nothing.
    """
    headers = [norm_phrase(h.splitlines()[0]) for h in (header_lines or []) if h.strip()]
    out = []
    for line in page_text.splitlines():
        flat = re.sub(r"\s+", " ", line).strip()
        if not flat:
            continue
        if MARKER.match(flat) or PAGE_NUMBER.match(flat):
            continue
        if re.sub(r"\d", "#", flat) in shapes:
            continue
        # the running title of this page's own schedule, however it is spaced
        if headers and any(fuzz.partial_ratio(norm_phrase(flat), h) >= 95
                           and len(flat) <= len(h) + 25 for h in headers):
            continue
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 1. coverage, against the independent extraction
# --------------------------------------------------------------------------- #
def coverage(clauses: list[dict], pdf_pages: dict[int, str]) -> dict:
    by_page: dict[int, list[dict]] = defaultdict(list)
    for c in clauses:
        for p in range(c["page_start"], c["page_end"] + 1):
            by_page[p].append(c)

    shapes, headers = boilerplate_lines(), page_headers()
    per_page, word_page, raw_per_page, thin = [], [], [], []
    for page, text in sorted(pdf_pages.items()):
        graph_text = " ".join(c["text"] + " " + (c["heading"] or "") for c in by_page.get(page, []))
        clean = strip_boilerplate(text, shapes, headers.get(page))
        # Two questions, deliberately kept apart. Word coverage asks whether the
        # text reached the graph at all. Shingle coverage asks whether it reached
        # it in the same order - which two-column definitions and tables break
        # even when every word is present.
        for source, sink, n in ((clean, word_page, 1), (clean, per_page, SHINGLE),
                                (text, raw_per_page, SHINGLE)):
            pdf_sh = shingles(source, n)
            if not pdf_sh:
                continue
            sink.append(len(pdf_sh & shingles(graph_text, n)) / len(pdf_sh))
        if word_page and word_page[-1] < 0.90:
            thin.append({"page": page, "word_coverage": round(word_page[-1], 3),
                         "shingle_coverage": round(per_page[-1], 3) if per_page else None,
                         "clauses_on_page": len(by_page.get(page, [])),
                         "has_table": any(c["chunk_type"] == "table" for c in by_page.get(page, []))})

    semantic = q.query("""
        MATCH (c:Clause) WHERE c.text <> ''
        RETURN count(c) AS total,
               count { (c)<-[:STATED_IN]-() } AS _ignore""")
    covered = q.query("""
        MATCH (c:Clause) WHERE c.text <> '' AND (c)<-[:STATED_IN]-()
        RETURN count(c) AS n""")[0]["n"]
    total_text_clauses = q.query(
        "MATCH (c:Clause) WHERE c.text <> '' RETURN count(c) AS n")[0]["n"]

    pages_with_clauses = len([p for p in pdf_pages if by_page.get(p)])
    return {
        "method": "overlap against `pdftotext -layout`, per page, boilerplate excluded from the denominator",
        "word_coverage_mean": round(sum(word_page) / max(len(word_page), 1), 4),
        "shingle_coverage_mean": round(sum(per_page) / max(len(per_page), 1), 4),
        "text_coverage_including_boilerplate": round(
            sum(raw_per_page) / max(len(raw_per_page), 1), 4),
        "pages_compared": len(per_page),
        "pages_below_90pct_words": len(thin),
        "thin_pages": sorted(thin, key=lambda r: r["word_coverage"])[:25],
        "thin_pages_with_a_table": sum(1 for t in thin if t["has_table"]),
        "page_coverage": round(pages_with_clauses / max(len(pdf_pages), 1), 4),
        "pages_with_no_clause": [p for p in sorted(pdf_pages) if not by_page.get(p)],
        "documents": q.query("MATCH (d:Document) RETURN count(d) AS n")[0]["n"],
        "clauses_with_a_semantic_node": round(covered / max(total_text_clauses, 1), 4),
        "clauses_without_a_semantic_node": total_text_clauses - covered,
    }


# --------------------------------------------------------------------------- #
# 2. cross-reference detection recall, against the independent extraction
# --------------------------------------------------------------------------- #
def reference_detection_recall(edges: list[dict], unresolved: list[dict],
                               pdf_pages: dict[int, str]) -> dict:
    """Scan the *other* extraction for references and ask whether we saw each one.

    This is the check the golden set cannot do. A reference class the resolver has
    no regex for is invisible to every other measurement in this repo, because
    everything else is built from what the resolver produced.
    """
    seen: dict[int, list[str]] = defaultdict(list)
    for row in edges + unresolved:
        seen[row.get("page") or 0].append(norm_phrase(row["reference_text"]))

    found, missed = 0, []
    total = 0
    shapes, headers = boilerplate_lines(), page_headers()
    for page, text in sorted(pdf_pages.items()):
        clean = strip_boilerplate(text, shapes, headers.get(page))
        # the page a clause is cited on may be either side of a page break
        nearby = seen[page] + seen[page - 1] + seen[page + 1]
        # scan line by line: `-layout` puts a heading and the clause number under
        # it on consecutive lines, and a scanner allowed to span the break invents
        # references like "clauses\n3.2.1" that are not in the text at all
        for line in clean.splitlines():
            for m in REFERENCE.finditer(line):
                phrase = norm_phrase(m.group(0))
                if len(phrase) < 4:
                    continue
                total += 1
                if fuzzy_in(phrase, nearby, threshold=90):
                    found += 1
                elif len(missed) < 400:
                    missed.append({"page": page, "reference_text": m.group(0)[:60]})

    by_class: dict[str, int] = defaultdict(int)
    for m in missed:
        head = re.match(r"\s*(this\s+\w+|[A-Za-z-]+)", m["reference_text"])
        by_class[(head.group(1) if head else "?").lower()] += 1

    return {
        "method": "reference regex run over `pdftotext` text, matched against what the resolver saw",
        "mentions_in_reference_extraction": total,
        "matched_in_graph": found,
        "detection_recall": round(found / max(total, 1), 4),
        "missed": len(missed),
        "missed_by_leading_token": dict(sorted(by_class.items(), key=lambda kv: -kv[1])[:15]),
        "missed_sample": missed[:30],
    }


# --------------------------------------------------------------------------- #
# 3. silver standard: precision and recall by type on sampled pages
# --------------------------------------------------------------------------- #
class PageContents(BaseModel):
    internal_references: list[str]
    defined_terms_used: list[str]
    obligations: list[str]


ENUMERATE = """Below is the raw text of one page of the RM6116 Network Services 3 framework
agreement (a UK public-sector procurement framework). Enumerate what is on THIS page.

- internal_references: every pointer to another part of this same agreement, quoted
  as printed - "Clause 10.4.1", "Paragraph 3", "Joint Schedule 1", "Part B",
  "Annex 1", "this Clause". Do NOT include references to legislation (Acts,
  Regulations, Directives).
- defined_terms_used: capitalised defined terms used on this page (e.g. "Working
  Day", "Deliverables", "Call-Off Contract"). Not headings, not ordinary
  capitalised words.
- obligations: one short phrase per normative provision stated on this page - a
  duty, a prohibition or a permission - each beginning with the party it binds or
  entitles: "Supplier must provide...", "Buyer must pay...", "Either Party may
  request...".

Be exhaustive but do not invent. If a category is empty, return an empty list.

Page {page}:
{text}"""


def enumerate_page(client: VertexClient, page: int, text: str) -> dict:
    try:
        response = client.generate_content(
            model=SETTINGS["llm"]["model"],
            contents=ENUMERATE.format(page=page, text=text[:12000]),
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json",
                response_schema=PageContents, max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(thinking_budget=128)),
        )
        return {"page": page, **PageContents.model_validate_json(response.text).model_dump()}
    except Exception as exc:  # noqa: BLE001
        return {"page": page, "error": f"{type(exc).__name__}: {exc}"[:200]}


class Alignment(BaseModel):
    matched: list[str]
    missing_from_graph: list[str]
    extra_in_graph: list[str]


class PageAlignment(BaseModel):
    internal_references: Alignment
    defined_terms_used: Alignment
    obligations: Alignment


ALIGN = """Two lists describe the same page of a UK framework agreement. One was read
off the page. The other is what a knowledge graph holds for that page. Align them.

For each category decide, item by item, whether the two lists refer to the SAME thing.
Judge by what an item refers to, not by wording:
- "Clause 34 (Resolving Disputes)" and "Clause 34" are the same reference.
- "Supplier must provide the Deliverables" and "Supplier / must / provides deliverables
  in accordance with the Specification" are the same obligation.
- "Service Offers" and "Service Offer" are the same defined term.

Then classify every item:
- matched: appears in both lists.
- missing_from_graph: on the page but absent from the graph. This is the graph
  failing to capture something.
- extra_in_graph: in the graph but not on the page list. Note the page list is NOT
  exhaustive - a real item the reader did not bother to list belongs here too, so
  this is disagreement, not necessarily error.

Exclude from ALL categories, listing them nowhere: references to legislation, to
Lots or sub-Lots (out of scope by design), to documents outside this agreement
(e.g. "the Order Form"), and bare anaphora ("that clause", "the said paragraph").

Page {page}

READ OFF THE PAGE
internal_references: {e_refs}
defined_terms_used: {e_terms}
obligations: {e_obs}

HELD IN THE GRAPH
internal_references: {a_refs}
defined_terms_used: {a_terms}
obligations: {a_obs}"""


def align_with_judge(client: VertexClient, page: int, expected: dict, actual: dict) -> dict | None:
    """Let a judge decide what matches what.

    Mechanical matching proved to be the dominant source of error in this
    measurement, not the graph: switching from whole-string fuzzy comparison to a
    canonical (kind, number) key moved reference recall from 0.313 to 0.809
    without a single edge changing. Anything left to a string comparison here is
    measuring the comparison.
    """
    try:
        response = client.generate_content(
            model=SETTINGS["llm"]["model"],
            contents=ALIGN.format(
                page=page,
                e_refs=json.dumps(expected["internal_references"][:40]),
                e_terms=json.dumps(expected["defined_terms_used"][:40]),
                e_obs=json.dumps(expected["obligations"][:30]),
                a_refs=json.dumps(actual["internal_references"][:40]),
                a_terms=json.dumps(actual["defined_terms_used"][:40]),
                a_obs=json.dumps(actual["obligations"][:30]),
            ),
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json",
                response_schema=PageAlignment, max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_budget=128)),
        )
        return PageAlignment.model_validate_json(response.text).model_dump()
    except Exception:  # noqa: BLE001
        return None


def score_alignment(a: dict) -> dict:
    matched, missing, extra = len(a["matched"]), len(a["missing_from_graph"]), len(a["extra_in_graph"])
    return {
        "expected": matched + missing, "actual": matched + extra,
        "recall": matched / (matched + missing) if matched + missing else None,
        "precision": matched / (matched + extra) if matched + extra else None,
    }


def graph_contents(page: int) -> dict:
    refs = q.query("""
        MATCH (c:Clause)-[r:CROSS_REFERENCES|REFERENCES_DOCUMENT]->()
        WHERE c.page_start <= $p AND c.page_end >= $p
        RETURN collect(DISTINCT r.reference_text) AS refs""", p=page)[0]["refs"]
    terms = q.query("""
        MATCH (c:Clause)-[:USES_TERM]->(t:Definition)
        WHERE c.page_start <= $p AND c.page_end >= $p
        RETURN collect(DISTINCT t.term) AS terms""", p=page)[0]["terms"]
    # A reader listing "obligations" on a page does not split them the way the
    # ontology does: "Either Party can request a Variation" is a duty to that
    # reader and a `right` to the extraction schema, so it lands on a Remedy node.
    # Comparing against Obligation alone measures the label, not the coverage.
    obligations = q.query("""
        MATCH (n)-[:STATED_IN]->(c:Clause)
        WHERE c.page_start <= $p AND c.page_end >= $p
          AND any(l IN labels(n) WHERE l IN ['Obligation','Remedy','Provision'])
          AND (n.modality IS NOT NULL OR n.actor IS NOT NULL)
        RETURN collect(DISTINCT coalesce(n.actor,'?') + ' ' +
                       coalesce(n.modality,'') + ' ' + n.summary) AS obs""", p=page)[0]["obs"]
    return {"internal_references": refs, "defined_terms_used": terms, "obligations": obligations}


NUMBERS = re.compile(r"\d+(?:\.\d+)*")


def canonical_reference(text: str) -> str | None:
    """Reduce a quoted reference to what it points *at*.

    Comparing quoted strings does not work: a reader writes "Clause 34 (Resolving
    Disputes)" and the graph stores "Clause 34", and a fuzzy string test scores
    that a miss on length alone. What matters is whether both name the same
    target, so both sides are reduced to (kind, numbers).

    Returns None for anything that is not a structured internal pointer - "the
    Order Form", "that clause", "Lot 3a". Those are excluded from the comparison
    rather than counted against the graph: Lots are out of scope for v1 by
    decision, and anaphora has no target to resolve to.
    """
    flat = norm_phrase(text)
    m = re.match(
        r"^(?:the\s+)?(this\s+)?(clause|paragraph|schedule|annex|part|appendix|"
        r"framework schedule|joint schedule|call[- ]?off schedule)s?\b",
        flat,
    )
    if not m:
        return None
    kind = re.sub(r"\s+", " ", m.group(2))
    kind = {"framework schedule": "schedule", "joint schedule": "schedule",
            "call-off schedule": "schedule", "call off schedule": "schedule"}.get(kind, kind)
    nums = NUMBERS.findall(flat)
    if m.group(1) and not nums:
        return f"this {kind}"
    if not nums:
        # "Annex B", "Part C" - the label is a letter
        label = re.match(rf"^(?:the\s+)?{re.escape(kind)}s?\s+([a-z]\d*)\b", flat)
        return f"{kind} {label.group(1)}" if label else None
    return f"{kind} {nums[0]}"


def canonical_set(items: list[str]) -> tuple[set[str], list[str]]:
    keys, unstructured = set(), []
    for i in items:
        k = canonical_reference(i)
        keys.add(k) if k else unstructured.append(i)
    return keys, unstructured


def token_overlap(a: str, b: str) -> float:
    """Content-word overlap, which survives the wording differences that a whole-
    string ratio does not - an obligation phrased two ways shares its nouns."""
    stop = {"the", "a", "an", "of", "to", "in", "for", "and", "or", "must", "shall",
            "will", "any", "all", "with", "on", "by", "at", "is", "are", "be", "that"}
    wa = {w for w in words(a) if w not in stop and len(w) > 2}
    wb = {w for w in words(b) if w not in stop and len(w) > 2}
    return len(wa & wb) / max(min(len(wa), len(wb)), 1) if wa and wb else 0.0


def score_overlap(expected: list[str], actual: list[str], threshold: float = 0.5) -> dict:
    hit_e = sum(1 for e in expected if any(token_overlap(e, a) >= threshold for a in actual))
    hit_a = sum(1 for a in actual if any(token_overlap(a, e) >= threshold for e in expected))
    return {
        "expected": len(expected), "actual": len(actual),
        "recall": hit_e / len(expected) if expected else None,
        "precision": hit_a / len(actual) if actual else None,
    }


def score_keys(expected: set[str], actual: set[str]) -> dict:
    return {
        "expected": len(expected), "actual": len(actual),
        "recall": len(expected & actual) / len(expected) if expected else None,
        "precision": len(expected & actual) / len(actual) if actual else None,
    }


def score(expected: list[str], actual: list[str], threshold: int = FUZZY) -> dict:
    hit_e = sum(1 for e in expected if fuzzy_in(e, actual, threshold))
    hit_a = sum(1 for a in actual if fuzzy_in(a, expected, threshold))
    return {
        "expected": len(expected), "actual": len(actual),
        "recall": hit_e / len(expected) if expected else None,
        "precision": hit_a / len(actual) if actual else None,
    }


def silver_standard(client: VertexClient, pdf_pages: dict[int, str], pages: list[int],
                    matcher: str = "llm") -> dict:
    shapes, headers = boilerplate_lines(), page_headers()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(enumerate_page, client, p,
                        strip_boilerplate(pdf_pages[p], shapes, headers.get(p))): p
            for p in pages
        }
        expected = [f.result() for f in as_completed(futures)]

    rows, per_type = [], defaultdict(list)
    for exp in sorted(expected, key=lambda r: r["page"]):
        if exp.get("error"):
            rows.append(exp)
            continue
        got = graph_contents(exp["page"])
        # an obligation is matched loosely: the same party bound to the same act,
        # not the same wording
        page_row = {"page": exp["page"], "matcher": matcher}

        if matcher == "llm":
            aligned = align_with_judge(client, exp["page"], exp, got)
            if aligned is not None:
                for key in ("internal_references", "defined_terms_used", "obligations"):
                    page_row[key] = score_alignment(aligned[key])
                    per_type[key].append(page_row[key])
                page_row["missed_references"] = aligned["internal_references"]["missing_from_graph"][:8]
                page_row["missed_terms"] = aligned["defined_terms_used"]["missing_from_graph"][:8]
                page_row["missed_obligations"] = aligned["obligations"]["missing_from_graph"][:6]
                rows.append(page_row)
                continue
            page_row["matcher"] = "mechanical (judge failed)"

        exp_keys, exp_unstructured = canonical_set(exp["internal_references"])
        got_keys, _ = canonical_set(got["internal_references"])
        page_row["internal_references"] = score_keys(exp_keys, got_keys)
        page_row["unstructured_mentions_excluded"] = exp_unstructured[:8]

        page_row["defined_terms_used"] = score(exp["defined_terms_used"],
                                               got["defined_terms_used"], 90)
        page_row["obligations"] = score_overlap(exp["obligations"], got["obligations"])
        for key in ("internal_references", "defined_terms_used", "obligations"):
            per_type[key].append(page_row[key])

        page_row["missed_references"] = sorted(exp_keys - got_keys)[:8]
        page_row["missed_terms"] = [
            t for t in exp["defined_terms_used"]
            if not fuzzy_in(t, got["defined_terms_used"], 90)
        ][:8]
        rows.append(page_row)

    def agg(key: str, metric: str) -> float | None:
        vals = [s[metric] for s in per_type[key] if s[metric] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    write_jsonl(ROOT / SETTINGS["paths"]["reports"] / "graph_quality_pages.jsonl", rows)
    return {
        "method": "a model enumerates each sampled page from raw text; a second pass aligns "
                  "that enumeration with the graph's contents for the page",
        "matcher": matcher,
        "pages_sampled": len(pages),
        "pages_failed": sum(1 for r in rows if r.get("error")),
        "by_type": {
            key: {"recall": agg(key, "recall"), "precision": agg(key, "precision"),
                  "pages": len(per_type[key])}
            for key in ("internal_references", "defined_terms_used", "obligations")
        },
        "matching": {
            "internal_references": "reduced on both sides to (kind, number), so "
                                   "'Clause 34 (Resolving Disputes)' and 'Clause 34' agree; "
                                   "unstructured mentions (Lots, 'the Order Form', anaphora) "
                                   "are excluded rather than counted against the graph",
            "defined_terms_used": "90% fuzzy on the term",
            "obligations": "content-word overlap >= 0.5",
        },
        "caveat": (
            "The silver standard is model-generated and carries its own error rate, and "
            "the model is not exhaustive. Precision against a non-exhaustive annotator is "
            "not precision: a term the graph linked and the model did not list is not "
            "thereby wrong. Read recall as the useful signal and precision as agreement."
        ),
    }


# --------------------------------------------------------------------------- #
# 4. edge precision - is a resolved cross-reference pointing at the right clause
# --------------------------------------------------------------------------- #
class EdgeVerdict(BaseModel):
    correct: bool
    reason: str


EDGE_PROMPT = """A cross-reference in a UK framework agreement was resolved to a target clause.
Judge whether the target is the provision the reference points at.

Reference text found in the source: "{reference_text}"
Source clause {source}: {source_text}
Resolved target {target}: {target_text}

Scope rule applied: {scope_rule}
All targets emitted for this same reference phrase: {all_targets_for_this_reference}

Judge only whether THIS target is a correct destination. If the reference names a
range or a list, the other targets are emitted as separate edges and are listed
above - do not mark this edge wrong for being one of several.
"Clause N" always means Core Terms; "Paragraph N" means paragraph N of the schedule
the reference sits in. A reference to a Part or Annex resolves to that section's
first substantive clause, since Parts have no node of their own."""


def edge_precision(client: VertexClient, sample_size: int = 100) -> dict:
    """A wrong edge is worse than a missing one - it actively misroutes the agent -
    so edges are graded one at a time and the sample is written out for review."""
    edges = q.query("""
        MATCH (a:Clause)-[r:CROSS_REFERENCES]->(b:Clause)
        CALL (a, r) {
            MATCH (a)-[s:CROSS_REFERENCES {reference_text: r.reference_text}]->(t:Clause)
            RETURN collect(t.clause_id) AS siblings
        }
        RETURN a.clause_id AS source, left(a.text, 700) AS source_text,
               b.clause_id AS target, left(b.text, 400) AS target_text,
               r.reference_text AS reference_text, r.scope_rule AS scope_rule,
               siblings AS all_targets_for_this_reference
        ORDER BY rand() LIMIT $n""", n=sample_size)

    def grade(e: dict) -> dict:
        try:
            response = client.generate_content(
                model=SETTINGS["llm"]["model"], contents=EDGE_PROMPT.format(**e),
                config=types.GenerateContentConfig(
                    temperature=0, response_mime_type="application/json",
                    response_schema=EdgeVerdict, max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=128)),
            )
            v = EdgeVerdict.model_validate_json(response.text)
            return {**e, "correct": v.correct, "reason": v.reason}
        except Exception as exc:  # noqa: BLE001
            return {**e, "correct": None, "reason": f"judge failed: {exc}"[:150]}

    with ThreadPoolExecutor(max_workers=10) as pool:
        graded = list(pool.map(grade, edges))
    decided = [g for g in graded if isinstance(g["correct"], bool)]
    write_jsonl(ROOT / SETTINGS["paths"]["reports"] / "edge_precision_sample.jsonl", graded)
    return {
        "sample_size": len(graded),
        "graded": len(decided),
        "precision": round(sum(g["correct"] for g in decided) / max(len(decided), 1), 4),
        "incorrect": [{k: g[k] for k in ("source", "target", "reference_text",
                                          "scope_rule", "reason")}
                      for g in decided if not g["correct"]][:20],
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=SAMPLE_PAGES,
                    help="pages to sample for the silver standard")
    ap.add_argument("--skip-llm", action="store_true",
                    help="deterministic coverage and detection recall only")
    ap.add_argument("--matcher", choices=("llm", "mechanical"), default="llm",
                    help="how the page enumeration is aligned with the graph")
    args = ap.parse_args()

    import random

    clauses = load_jsonl(path("clauses"))
    edges = load_jsonl(path("edges"))
    unresolved = load_jsonl(path("unresolved"))
    pdf_pages = {p["page"]: p["text"] for p in load_jsonl(path("pdftotext_pages"))}

    result = {
        "coverage": coverage(clauses, pdf_pages),
        "reference_detection": reference_detection_recall(edges, unresolved, pdf_pages),
    }
    if not args.skip_llm:
        rng = random.Random(6116)
        candidates = [p for p, t in pdf_pages.items() if len(words(t)) > 120]
        sample = sorted(rng.sample(candidates, min(args.pages, len(candidates))))
        try:
            client = VertexClient()
            result["silver_standard"] = silver_standard(client, pdf_pages, sample, args.matcher)
            result["edge_precision"] = edge_precision(client)
        except Exception as exc:  # noqa: BLE001
            # The deterministic half needs no cloud access and has already run;
            # losing credentials should not throw away a completed measurement.
            result["silver_standard"] = {
                "skipped": True,
                "reason": f"{type(exc).__name__}: {exc}"[:300],
                "hint": "run `gcloud auth login`, then `make graph-eval` again",
            }
            print(f"model-graded checks skipped: {type(exc).__name__} - "
                  f"run `gcloud auth login` and retry\n")

    report("graph_quality.json", result)

    c, d = result["coverage"], result["reference_detection"]
    print(f"coverage   words {c['word_coverage_mean']:.1%} | 5-grams {c['shingle_coverage_mean']:.1%} "
          f"| pages {c['page_coverage']:.1%} | clauses with a semantic node "
          f"{c['clauses_with_a_semantic_node']:.1%}")
    print(f"           {c['pages_below_90pct_words']} pages below 90% word coverage "
          f"({c['thin_pages_with_a_table']} of them hold a table)")
    print(f"detection  {d['matched_in_graph']}/{d['mentions_in_reference_extraction']} references "
          f"seen ({d['detection_recall']:.1%}); missed by token: "
          f"{list(d['missed_by_leading_token'].items())[:5]}")
    if "edge_precision" in result:
        print(f"edges      precision {result['edge_precision']['precision']:.3f} "
              f"on {result['edge_precision']['graded']} sampled cross-reference edges")
    if "by_type" in result.get("silver_standard", {}):
        for key, v in result["silver_standard"]["by_type"].items():
            r = f"{v['recall']:.3f}" if v["recall"] is not None else "  -  "
            pr = f"{v['precision']:.3f}" if v["precision"] is not None else "  -  "
            print(f"silver     {key:22s} recall {r}  precision {pr}")


if __name__ == "__main__":
    main()
