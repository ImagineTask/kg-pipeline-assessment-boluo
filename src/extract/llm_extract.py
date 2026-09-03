"""Stage 1.6 - LLM extraction against the slim ontology.

The model sees one clause at a time, with its hierarchy path and parent text for
context, and returns one flat record. It is never asked for cross-references or
defined terms: both are already resolved deterministically, and asking a model
to redo deterministic work only introduces an error rate.

Output shape is enforced by a Pydantic model passed as the response schema, not
by parsing free-form JSON. Failures retry twice, then quarantine.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.common import SETTINGS, load_jsonl, path, report, write_jsonl
from src.vertex import VertexClient

CFG = SETTINGS["llm"]
BATCH_SIZE = 6


class ProvisionType(str, Enum):
    obligation = "obligation"
    right = "right"
    definition = "definition"
    liability = "liability"
    payment = "payment"
    procedure = "procedure"
    statement = "statement"


class Actor(str, Enum):
    ccs = "CCS"
    buyer = "Buyer"
    supplier = "Supplier"
    subcontractor = "Subcontractor"
    guarantor = "Guarantor"
    auditor = "Auditor"
    other = "Other"


class Modality(str, Enum):
    must = "must"
    must_not = "must_not"
    may = "may"


class Provision(BaseModel):
    """One normative statement. A clause may contain several."""

    provision_type: ProvisionType
    actor: Actor | None = None
    counterparty: Actor | None = None
    modality: Modality | None = None
    summary: str = Field(description="one normalised sentence, max 25 words")
    trigger: str | None = None
    deadline: str | None = None
    amount: str | None = None
    uncapped: bool | None = None
    confidence: float


class ClauseRecord(BaseModel):
    """Every distinct provision in one clause.

    v1 emitted a single flat record per clause, which put a hard ceiling on the
    semantic layer: a clause imposing three duties became one node with one
    summary, and measured provision recall against a page reading stalled at 0.65.
    The extraction unit is still the clause - it is what the model is shown, and
    what a citation points at - but the clause now yields one record per duty,
    right or prohibition it actually contains.
    """

    clause_id: str
    provisions: list[Provision] = Field(
        description="one entry per distinct provision; usually 1-3, rarely more")


class Batch(BaseModel):
    records: list[ClauseRecord]


SYSTEM = """You are extracting structured data from a UK public-sector framework contract.

Return one JSON object per clause given, echoing its clause_id exactly, containing
a list of the distinct provisions in that clause.

A provision is one normative statement: a duty, a prohibition, a right, a payment
term, a liability position, a procedural step, or a statement of fact. Split a
clause where it genuinely says several things - a clause reading "The Supplier must
provide X, must notify the Buyer within 5 Working Days, and may charge for Y"
contains three provisions. A clause with a lead-in and lettered limbs that are
facets of one duty is ONE provision; do not split a single obligation into its
conditions. Most clauses contain one or two. Do not invent provisions to pad the
list, and do not merge genuinely separate duties to shorten it.

Every clause carrying substantive text yields at least one provision. Where a
clause states no duty, right or prohibition - a recital, a definition, a piece of
narrative - emit a single provision of type `statement` or `definition` saying
what it establishes. Only a clause that is purely a heading, a form placeholder or
an empty field yields none.

For each provision:

- provision_type: exactly one of [obligation|right|definition|liability|payment|procedure|statement].
  Pick the dominant function of the clause.
- actor: who must act or who holds the right. counterparty: who benefits or is acted against.
  Each is one of [CCS|Buyer|Supplier|Subcontractor|Guarantor|Auditor|Other] or null.
- modality: one of [must|must_not|may] or null.
- summary: a restatement in one sentence, at most 25 words. Not a quotation of the clause.
- trigger: the condition or defined event that activates this, verbatim if capitalised, else null.
- deadline: verbatim, e.g. "30 days", "3 Working Days". Do not normalise or convert.
- amount: verbatim money or percentage, e.g. "£1,000,000", "1%". Do not normalise.
- uncapped: true only where the clause expressly excludes a liability cap, else null.
- confidence: 0-1.

Rules:
- Use null rather than guessing. A null field is correct; an invented one is not.
- Never invent enum values outside the lists above.
- Copy amounts, deadlines and defined terms verbatim from the clause text.
- Do not list cross-references or defined terms anywhere; they are extracted separately.
- Each provision must stand on its own: its summary should be readable without the others."""


def prompt_for(items: list[dict], parents: dict[str, dict]) -> str:
    parts = []
    for c in items:
        parent = parents.get(c["parent_id"] or "")
        parent_text = (parent["text"][:400] if parent else "") or "(none)"
        parts.append(
            f"---\nclause_id: {c['clause_id']}\n"
            f"Context path: {c['hierarchy_path']}\n"
            f"Parent clause: {parent_text}\n"
            f"Clause {c['number'] or c['heading'] or ''}: {c['text']}"
        )
    return "\n".join(parts)


class Extractor:
    def __init__(self, model: str | None = None):
        self.client = VertexClient()
        self.model = model or CFG["model"]
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=CFG["temperature"],
            response_mime_type="application/json",
            response_schema=Batch,
            # Pro cannot disable thinking; the floor keeps latency and cost down
            # on what is a mechanical labelling task.
            thinking_config=types.ThinkingConfig(thinking_budget=128),
        )
        self.usage = {"prompt": 0, "output": 0, "calls": 0}
        self.lock = threading.Lock()

    @retry(
        stop=stop_after_attempt(CFG["max_retries"] + 1),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((ValidationError, ValueError, json.JSONDecodeError)),
        reraise=True,
    )
    def _call(self, items: list[dict], parents: dict[str, dict]) -> list[Record]:
        response = self.client.generate_content(
            model=self.model,
            contents=prompt_for(items, parents),
            config=self.config,
        )
        with self.lock:
            u = response.usage_metadata
            self.usage["calls"] += 1
            self.usage["prompt"] += getattr(u, "prompt_token_count", 0) or 0
            self.usage["output"] += (getattr(u, "candidates_token_count", 0) or 0) + (
                getattr(u, "thoughts_token_count", 0) or 0
            )
        batch = Batch.model_validate_json(response.text)
        wanted = {c["clause_id"] for c in items}
        got = {r.clause_id for r in batch.records}
        if got != wanted:
            raise ValueError(f"clause_id mismatch: missing {wanted - got}, extra {got - wanted}")
        return batch.records

    def run(self, items: list[dict], parents: dict[str, dict]) -> tuple[list[dict], list[dict]]:
        try:
            return flatten(self._call(items, parents)), []
        except Exception as exc:  # noqa: BLE001 - the reason is recorded, not swallowed
            if len(items) == 1:
                return [], [{"clause_id": items[0]["clause_id"], "error": f"{type(exc).__name__}: {exc}"[:400]}]
            ok, bad = [], []
            for one in items:                       # fall back to one at a time
                o, b = self.run([one], parents)
                ok += o
                bad += b
            return ok, bad


def flatten(records: list[ClauseRecord]) -> list[dict]:
    """One row per provision. `provision_id` is the node key in the graph;
    `clause_id` remains the citation and the link back to the source text."""
    rows: list[dict] = []
    for record in records:
        for i, provision in enumerate(record.provisions, start=1):
            rows.append({
                "provision_id": f"{record.clause_id}::{i}",
                "clause_id": record.clause_id,
                "provision_index": i,
                "provisions_in_clause": len(record.provisions),
                **provision.model_dump(mode="json"),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="extract only the first N clauses")
    ap.add_argument("--doc", action="append", help="restrict to these doc_ids")
    ap.add_argument("--restart", action="store_true", help="ignore existing records and start over")
    args = ap.parse_args()

    clauses = load_jsonl(path("clauses"))
    parents = {c["clause_id"]: c for c in clauses}

    # tables need a separate extraction path and are out of scope for v1
    todo = [c for c in clauses if c["chunk_type"] != "table" and c["text"].strip()]
    if args.doc:
        todo = [c for c in todo if c["doc_id"] in set(args.doc)]

    # Keep the existing rows as a *list*. Keying them by clause_id silently
    # collapses a clause's several provisions into one, which is exactly what a
    # resumed run must not do now that a clause yields more than one record.
    existing: list[dict] = []
    if not args.restart and path("records").exists():
        existing = load_jsonl(path("records"))
    already = {r["clause_id"] for r in existing}
    todo = [c for c in todo if c["clause_id"] not in already]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(existing)} records over {len(already)} clauses already present; "
          f"extracting {len(todo)} clauses")
    extractor = Extractor()
    batches = [todo[i: i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

    records = list(existing)
    quarantine: list[dict] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=CFG["concurrency"]) as pool:
        futures = {pool.submit(extractor.run, b, parents): b for b in batches}
        for fut in as_completed(futures):
            ok, bad = fut.result()
            records.extend(ok)
            quarantine.extend(bad)
            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                write_jsonl(path("records"), records)
                print(f"  {completed}/{len(batches)} batches | {len(records)} records "
                      f"| {len(quarantine)} quarantined", flush=True)

    write_jsonl(path("records"), records)
    write_jsonl(path("quarantine"), quarantine)

    attempted = len(todo) + len(already)
    report(
        "llm_extract_report.json",
        {
            "model": extractor.model,
            "clauses_attempted": attempted,
            "records": len(records),
            "quarantined": len(quarantine),
            "schema_valid_rate": round(len(records) / max(attempted, 1), 4),
            "acceptance_ge_98pct": len(records) / max(attempted, 1) >= 0.98,
            "batch_size": BATCH_SIZE,
            "token_usage": extractor.usage,
            "quarantine_sample": quarantine[:20],
        },
    )
    print(f"{len(records)} records, {len(quarantine)} quarantined, usage {extractor.usage}")


if __name__ == "__main__":
    main()
