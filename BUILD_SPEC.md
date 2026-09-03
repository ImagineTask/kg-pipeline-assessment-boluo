# RM6116 GraphRAG — Build Specification

Target document: `RM6116 Network Services 3 — Framework Agreement` (475 pages, A4, digital-native PDF with a clean text layer, Crown Copyright 2018, v3.0.11).

This spec is written for a coding agent. Each stage has **inputs, outputs, acceptance criteria, and gotchas**. Do not advance to the next stage until the acceptance criteria pass.

---

## 0. Architecture

```
PDF ──▶ Document AI (batch OCR/Layout)
          │
          ▼
     raw_pages.jsonl
          │
          ▼
     PAGE STITCHING ──▶ document_stream.jsonl   ◀── page boundaries destroyed here, on purpose
          │
          ▼
     structural chunker ──▶ clauses.jsonl
          │
   deterministic validation ─┤ (gate 1)
          │
          ▼
     definitions + CROSS-REFERENCE RESOLUTION ──▶ edges.jsonl
          │
          ▼
     LLM extraction (ontology) ──▶ records.jsonl
          │
   deterministic QA ─────────┤ (gate 2)
          │
          ▼
     graph materialiser ──▶ Neo4j (nodes + edges + vector index)
          │
          ▼
     MCP server (graph tools) ──▶ LangGraph agent ──▶ LangSmith eval
```

**Two design rules govern everything below.**

1. **Structure is deterministic, semantics are probabilistic.** Anything derivable by a parser must not go through the LLM.
2. **The page is a printing artefact, not a unit of meaning.** Destroy page boundaries early and completely. Every downstream stage operates on a continuous text stream, with page numbers retained only as metadata for citation.

---

## 1. Repository layout

```
rm6116-graphrag/
├── pyproject.toml
├── .env.example
├── config/
│   ├── ontology_extraction.json     # slim LLM-facing schema (v1)
│   ├── ontology_graph.json          # target graph model (v2)
│   └── settings.yaml
├── data/
│   ├── raw/                         # source PDF
│   ├── interim/                     # docai output, stitched stream, chunks
│   └── processed/                   # validated records, edges
├── src/
│   ├── ingest/
│   │   ├── docai_batch.py
│   │   ├── boilerplate.py           # header/footer detection
│   │   └── stitch.py                # page-boundary reconstruction
│   ├── chunk/
│   │   ├── structural.py
│   │   └── validate.py
│   ├── extract/
│   │   ├── definitions.py           # deterministic
│   │   ├── crossrefs.py             # deterministic — scope-aware resolver
│   │   ├── llm_extract.py
│   │   └── qa.py
│   ├── graph/
│   │   ├── ddl.cypher
│   │   ├── loader.py
│   │   └── embeddings.py
│   ├── retrieval/
│   │   ├── queries.py               # parameterised Cypher
│   │   └── mcp_server.py
│   ├── agent/
│   │   ├── graph.py                 # LangGraph state machine
│   │   └── prompts.py
│   └── eval/
│       ├── golden_set.jsonl
│       └── run_eval.py
└── tests/
```

**Dependencies:** `google-cloud-documentai`, `google-cloud-storage`, `neo4j`, `langchain-neo4j`, `langgraph`, `langsmith`, `pydantic>=2`, `mcp`, `tenacity`, `rapidfuzz`.

**Environment variables** (`.env.example`):
```
GCP_PROJECT_ID=
GCP_LOCATION=eu
DOCAI_PROCESSOR_ID=
GCS_INPUT_URI=
GCS_OUTPUT_URI=
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
ANTHROPIC_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=rm6116-graphrag
```

---

## 2. Stage 1 — Data processing

### 1.1 Document AI extraction

**Use batch (asynchronous) processing, not online.** Online processing has a low per-request page limit; this document is 475 pages. Batch requires GCS input and output buckets.

- Processor: **Enterprise Document OCR**. Do not use a specialised parser (Invoice/Contract) — those are trained for field extraction and will not help here.
- Enable **layout/paragraph detection** so you get block-level bounding boxes and reading order.
- Region: choose `eu` for a UK Crown Copyright document if data residency matters.
- Verify current page-per-batch limits against Google's quota docs before running. If the document exceeds the limit, split the PDF and merge results **preserving a global page offset** — an off-by-N page offset silently corrupts every citation downstream.

**Output normalisation** → `data/interim/raw_pages.jsonl`, one record per page:
```json
{
  "page": 1,
  "text": "...",
  "blocks": [{"text": "...", "bbox": [x0,y0,x1,y1], "type": "paragraph", "y_top": 52.4}]
}
```

Retain `bbox` / `y_top` on every block. **Page stitching depends on vertical position** — without it you cannot reliably tell a header from a first line of body text.

**Cross-check:** run `pdftotext -layout` as a parallel extraction and diff. If Document AI and pdftotext disagree by more than 2% on character count for a page, flag it. Near-free, catches OCR regressions.

**Acceptance:** 475 page records; every block has a bbox; character-count diff vs `pdftotext` under 2% on 95%+ of pages.

---

### 1.2 Page-boundary reconstruction (content crossing pages)

This document breaks clauses across pages constantly. Chunking per page, or chunking after naive page concatenation, produces broken clauses and orphaned fragments. Reconstruct the stream **before** chunking.

#### Step 1 — Strip repeated boilerplate

Every page carries:
- header line 1: `Crown Copyright 2018.` + `Version: 3.0.11`
- header line 2: the running schedule name (e.g. `Core Terms`, `Joint Schedule 1 (Definitions)`)
- footer: a bare page number

**Detection algorithm** (do not hard-code the strings — the running header changes per schedule):

```
for each text block:
    key = (rounded_y_top, normalised_text_shape)
    # normalised_text_shape: digits -> '#', so page numbers collapse together
count occurrences of each key across all pages
mark as boilerplate any key appearing on > 30% of pages
    AND positioned in the top 8% or bottom 8% of the page height
```

Emit a `boilerplate_report.json` listing every stripped pattern with its page count. **Review it manually once.** A false positive here deletes real contract text.

> Do not strip by regex on `Crown Copyright` alone. The running schedule header is the more dangerous one, because its text is identical to legitimate heading text that appears once at the true start of each schedule. Disambiguate by position and repetition, and keep the **first** occurrence when it appears at a schedule boundary.

#### Step 2 — Concatenate into a continuous stream

Produce `data/interim/document_stream.jsonl` — an ordered list of blocks with page provenance, not a list of pages:

```json
{"block_id": 1042, "text": "3.2.10 The Supplier must indemnify the Buyer against...", "page": 4, "y_top": 118.2}
```

Every block keeps its `page`. Nothing downstream reads pages as containers again.

#### Step 3 — Repair cross-page splits

Apply these joins when concatenating the last block of page *N* with the first block of page *N+1*:

| Condition at end of page N | Action |
|---|---|
| Line ends with a hyphen and next line starts lowercase | **De-hyphenate and join** — 72 candidate line-final hyphens exist in this document |
| Line ends without terminal punctuation, next starts lowercase or with `(` | **Join with a single space** — sentence continues |
| Line ends mid-list, next line starts `(b)`, `(c)`… | **Join** — lettered limbs must stay in one clause |
| Next page starts with a clause-number pattern (`^\d+\.\d+`) | **Do not join** — genuine new clause |
| Next page starts with a schedule heading | **Do not join** — hard document boundary |
| Table rows continue with the same column geometry | **Join as a continued table** (see below) |

**Reference case in this document:** clause `3.2.9` ends on page 3 with `"...the Buyer needs to make use of the"` and continues on page 4 with `"Goods."` after the stripped header. If your pipeline produces a chunk ending in `"make use of the"`, page stitching is broken. **Use this as a unit test.**

#### Step 4 — Tables spanning pages

Framework Schedule 3 (Framework Prices), Annex 1 (Rates and Prices) and the MI Reporting Template contain tables running over several pages.

- Detect a continued table when the following page's first blocks share the previous table's column x-positions **and** no header row is repeated.
- Merge continued tables into a single logical table object before chunking.
- If a table repeats its header row on each page, drop the repeated header and keep one.
- Serialise each merged table to markdown as a **single chunk** with `chunk_type: "table"`. Never split a table across chunks; a half-price-table chunk is worse than no chunk.

#### Step 5 — Footnotes and page-anchored artefacts

Footnotes sit at the bottom of the page but belong to a sentence in the body. Attach them to the block containing their marker, appending as ` [fn: ...]` rather than leaving them as free-floating blocks that would otherwise merge into whatever clause happens to follow.

**Acceptance for 1.2:**
- Zero chunks in the final output end without terminal punctuation, unless the chunk ends with a colon introducing a list.
- Clause `core_terms.3.2.9` reconstructs to full text ending in `"...make use of the Goods."`
- `boilerplate_report.json` reviewed and signed off.
- Reconstructed stream character count ≥ 97% of raw (boilerplate removal accounts for the difference; anything below 97% means real text was stripped).

---

### 1.3 Chunking

**Do not use fixed-size or naive recursive character splitting.** For legal and regulatory documents the established approach is **structure-aware hierarchical chunking**: split on the document's own numbering hierarchy, then apply a parent–child ("small-to-big") retrieval pattern.

Chunking operates on `document_stream.jsonl`, never on pages.

**Two-level segmentation:**

1. **Document segmentation.** Split on schedule headings. The document contains, in order: Core Terms; Framework Schedules 1–9; Joint Schedules 1–12; Call-Off Schedules 1–25; plus Annexes and Parts nested inside them. Anchored regex:
   ```
   ^(Core Terms|(Framework|Joint|Call-Off) Schedule \d+ \(.+?\)|Annex [A-Z0-9]+|Part [A-Z]|Appendix \d+)
   ```
   Handle **headings that wrap across two lines** — e.g. `Framework Schedule 6 (Order Form Template and` / `Call-Off Schedules)`. Join a candidate heading with the next line when the parenthesis is unbalanced. Note this interacts with page stitching: a heading can wrap across a page break.

2. **Clause segmentation.** Within each document:
   - `^\d+\.\s` — top-level clause (`3. What needs to be delivered`)
   - `^\d+\.\d+\s` — sub-clause (`3.1 All deliverables`)
   - `^\d+\.\d+\.\d+\s` — leaf clause (`3.1.1 The Supplier must...`)
   - `^\([a-z]\)\s` — lettered limb; attaches to the preceding clause, never split out

**Chunk record schema** (`data/interim/clauses.jsonl`):
```json
{
  "clause_id": "core_terms.3.1.1",
  "doc_id": "core_terms",
  "doc_type": "core_terms",
  "number": "3.1.1",
  "parent_id": "core_terms.3.1",
  "depth": 3,
  "heading": null,
  "text": "The Supplier must provide Deliverables: (a) that comply with...",
  "hierarchy_path": "Core Terms > 3. What needs to be delivered > 3.1 All deliverables > 3.1.1",
  "page_start": 3,
  "page_end": 4,
  "spans_pages": true,
  "chunk_type": "clause",
  "char_count": 412
}
```

`page_start` / `page_end` / `spans_pages` replace a single `page`. Citations quote the range.

**Rules:**
- Lettered limbs stay inside their parent clause text.
- Leaf clause = retrieval unit. If a leaf exceeds ~1,500 characters, split on sentence boundaries into `clause_id#p1`, `#p2`, set `is_split: true`, and **always retrieve siblings together**.
- **Definitions chunk differently.** In Joint Schedule 1, one defined term = one chunk keyed by the term, not a clause number. Definitions also run across pages — the same stitching rules apply.
- Tables are single chunks with `chunk_type: "table"`.

**Acceptance:** every chunk has a resolvable `parent_id` or is a document root; `hierarchy_path` reconstructs to a valid chain; no empty chunks; reconstructed text ≥99% of the stitched stream.

---

### 1.4 Deterministic validation (gate 1)

Run before any LLM call. Fail the build on any hard check.

**Hard checks:**
- **Coverage:** concatenating chunks in order reproduces ≥99% of the stitched stream. Report missing spans.
- **Numbering continuity:** within each document, clause numbers ascend with no gaps. A jump from `10.2` to `10.4` almost always means `10.3` was lost **at a page break** — this check is your primary page-stitching alarm. Log every gap with its page number.
- **Sentence integrity:** no chunk ends without terminal punctuation (`.`, `:`, `;`) — the direct test for a missed cross-page join.
- **No orphans:** every `parent_id` resolves.
- **No duplicates:** `clause_id` unique.
- **Boundary sanity:** no chunk starts mid-sentence.
- **Boilerplate residue:** zero chunks contain `Crown Copyright` or `Version: 3.0.11`.

**Soft checks (warn):**
- Chunks under 40 characters — mis-split heading.
- Chunks over 2,000 characters — missed boundary.
- More than one clause-number pattern at line start within a chunk — missed split.
- `spans_pages: true` on more than ~25% of chunks — plausible for this document, but verify the stitcher is not over-joining.

**Known-good spot checks:**

| Clause | Why |
|---|---|
| `core_terms.2.5` | lettered limbs (a)–(d) |
| `core_terms.3.2.9` | **sentence split across a page break** |
| `core_terms.3.2.10` | two-digit sub-number immediately after a page break |
| `joint_schedule_1` | definition-style chunking, not numeric |
| `framework_schedule_6` | two-line wrapped heading |
| `framework_schedule_3` Annex 1 | multi-page table |
| `call_off_schedule_2` | nested Parts A–E with Annexes D1–D4 |

**Acceptance:** all hard checks pass; numbering gaps reviewed and either fixed or explicitly whitelisted.

---

### 1.5 Definitions and cross-reference resolution

No model needed. Run before the LLM pass — the LLM pass depends on definitions being loaded.

#### 1.5.1 Definitions

Parse Joint Schedule 1 into `Definition` records: `{term, definition_text, scope, defined_in}`. Expect several hundred terms.

**Local overrides:** individual schedules define terms locally that override Joint Schedule 1 within their own document. Detect `"means"` / `"shall mean"` constructions inside a schedule and record them with `scope: "document_local"`. Local wins within its document.

**Defined-term linking:** build a gazetteer, match every occurrence in every clause. Exact capitalised form, word boundaries, **gazetteer sorted longest-first** so `Key Subcontractor` wins over `Subcontractor` and `Material Default` over `Default`. Emit `USES_TERM` edges with an occurrence count.

#### 1.5.2 Cross-reference resolution

This document is dense with internal pointers. Resolving them is what makes multi-hop retrieval work, and it is entirely deterministic. Observed frequencies from the source text are given below — use them as regression targets.

**Reference classes to handle:**

| Class | Example from the document | Frequency | Resolution |
|---|---|---|---|
| Clause reference | `Clause 10.4.1`, `Clause 34` | high | Resolve within **Core Terms** |
| Paragraph reference | `Paragraph 3`, `Paragraph 5.1`, `paragraph 10` | very high | Resolve within the **containing schedule** |
| Schedule reference | `Joint Schedule 1`, `Call-Off Schedule 2` (65×), `Framework Schedule 7` | very high | Resolve to `doc_id` |
| Schedule + paragraph | `Framework Schedule 7, Paragraph 2.1` | medium | Resolve to a clause in the named schedule |
| Annex / Part / Appendix | `Part B` (55×), `Annex 1` (50×), `Annex D3`, `Appendix 2` | high | Resolve within the containing schedule |
| Plural / list | `Clauses 10.6.1 and 10.6.2` | medium | Expand to multiple edges |
| Range | `Paragraphs 4.3 to 4.6`, `Clauses 27 to 32` | medium | **Expand the full range** into individual edges |
| Self-reference | `this Clause`, `this Paragraph`, `this Schedule` | ~166× | Resolve to the containing clause / document |
| Relative | `above`, `below` | ~121× | Usually attached to a numbered reference — resolve the number, ignore the direction word. Log bare `above`/`below` as unresolvable. |

**The critical resolution rule — scope matters more than the number.**

`Clause` and `Paragraph` are not interchangeable in this document:
- **`Clause N`** refers to **Core Terms**, regardless of which schedule the reference is written in.
- **`Paragraph N`** refers to a paragraph of **the schedule the reference sits in**, unless a schedule is named explicitly.

So `Paragraph 3` appearing inside Joint Schedule 7 resolves to `joint_schedule_7.3`, and the identical string inside Call-Off Schedule 14 resolves to `call_off_schedule_14.3`. **A resolver that ignores the containing document will produce mostly wrong edges** — and they will look plausible, which is worse.

**Resolution algorithm:**

```
for each clause C:
    for each regex match M in C.text:
        1. classify M  (clause | paragraph | schedule | annex | part | self | range | list)
        2. determine scope:
             explicit schedule named in M            -> that doc_id
             M is "Clause N"                          -> core_terms
             M is "Paragraph N"                       -> C.doc_id
             M is "this Clause/Paragraph"             -> C.clause_id
             M is "Annex/Part/Appendix X"             -> C.doc_id
        3. expand ranges and lists into individual targets
        4. resolve each target to an existing clause_id or doc_id
        5. emit edge or log to unresolved.jsonl with the reason
```

**Edge record** (`data/processed/edges.jsonl`):
```json
{
  "type": "CROSS_REFERENCES",
  "source": "joint_schedule_7.4.2",
  "target": "core_terms.10.4.1",
  "reference_text": "Clause 10.4.1",
  "ref_class": "clause",
  "scope_rule": "clause_to_core_terms",
  "resolved": true
}
```

**Never drop an unresolved reference silently.** Write it to `unresolved.jsonl` with the source clause and the matched string. Review the file — a cluster of failures usually reveals a missing regex class or a chunking bug, not a genuinely dangling reference.

**Bidirectional traversal:** store the edge once, directed. Query both ways in Cypher (`-[:CROSS_REFERENCES]-`). "What refers *to* this clause?" is as important as "what does this clause refer to?", because it reveals which obligations depend on the clause you are reading.

**Reference-chain depth:** cap traversal at depth 3 at query time. Cross-references in this document form cycles (Core Terms points at schedules that point back at Core Terms). **Cycle detection is mandatory** in the traversal query, or `trace_references` will not terminate.

**Acceptance for 1.5:**
- ≥95% of detected references resolve to an existing node.
- Zero `Paragraph N` references resolved to Core Terms (the scope-rule regression test).
- `unresolved.jsonl` reviewed; remaining failures categorised.
- Spot check: `Clause 10.4.1` (11 occurrences) resolves to the same Core Terms target every time, from whichever schedule it is cited in.
- Cycle detection verified on a known cycle.

---

### 1.6 LLM extraction

**Input:** one clause, plus its `hierarchy_path` and parent clause text.
**Schema:** `config/ontology_extraction.json` (flat, one record per clause).
**Output:** `data/processed/records.jsonl`.

**Rules:**
- Enforce output shape with a **Pydantic model + structured/tool-call output**. Do not parse free-form JSON.
- Batch by document for cost; one record per clause in the response.
- Temperature 0. Retry on schema-validation failure via `tenacity`, max 2, then quarantine.
- Inline the enum lists in the prompt. Do **not** paste the full ontology JSON — it bloats context and invites invented structure.
- Do **not** ask for cross-references or defined terms. Already done deterministically. Instruct the model to leave `refers_to` empty.
- Do **not** ask for normalisation of money or dates. Verbatim only; normalise in code.
- Skip `chunk_type: "table"` chunks in this pass — tables need a separate extraction path, out of scope for v1.

**Prompt skeleton:**
```
You are extracting structured data from a UK public-sector framework contract.

Context path: {hierarchy_path}
Parent clause: {parent_text}
Clause {number}: {text}

Return one JSON object matching the schema. Rules:
- provision_type: exactly one of [obligation|right|definition|liability|payment|procedure|statement]
- actor/counterparty: one of [CCS|Buyer|Supplier|Subcontractor|Guarantor|Auditor|Other|null]
- Copy amounts, deadlines and defined terms verbatim.
- Use null rather than guessing. Leave refers_to empty.
```

**Acceptance:** ≥98% of clauses produce a schema-valid record; quarantine file reviewed.

---

### 1.7 Extraction QA (gate 2)

- **Enum conformance:** zero values outside the allowed lists.
- **Verbatim check:** every non-null `amount` and `deadline` appears literally in the source clause text. Reject on failure. This is the single most effective hallucination detector.
- **Actor plausibility:** `provision_type == "obligation"` implies `actor` is not null.
- **Modality consistency:** `modality == "must"` should co-occur with `must`/`shall`/`will`. Flag mismatches.
- **Distribution sanity:** `Supplier` obligations should dominate. If `Buyer` obligations exceed `Supplier` obligations, there is a role-swap bug.
- **Human sample:** review a random 50 records, record the error rate, treat it as your baseline.

**Acceptance:** verbatim check 100%; enum conformance 100%; sampled error rate documented.

---

## 3. Stage 2 — Neo4j graph

### 2.1 Schema DDL

Create constraints and indexes **before** loading. `src/graph/ddl.cypher`:

```cypher
CREATE CONSTRAINT clause_id IF NOT EXISTS
  FOR (c:Clause) REQUIRE c.clause_id IS UNIQUE;
CREATE CONSTRAINT doc_id IF NOT EXISTS
  FOR (d:Document) REQUIRE d.doc_id IS UNIQUE;
CREATE CONSTRAINT definition_term IF NOT EXISTS
  FOR (t:Definition) REQUIRE t.term IS UNIQUE;
CREATE CONSTRAINT actor_name IF NOT EXISTS
  FOR (a:Actor) REQUIRE a.name IS UNIQUE;

CREATE FULLTEXT INDEX clause_fulltext IF NOT EXISTS
  FOR (c:Clause) ON EACH [c.text, c.heading, c.hierarchy_path];
CREATE INDEX clause_doc IF NOT EXISTS
  FOR (c:Clause) ON (c.doc_id);
```

### 2.2 Loading

- Order: `Document` → `Clause` (+ `PART_OF`) → `Definition` → `USES_TERM` → `CROSS_REFERENCES` → semantic nodes from `records.jsonl`.
- `UNWIND $batch` with batches of 1,000. Never string-concatenate Cypher.
- `MERGE` on the constrained key, `SET` properties — reloads stay idempotent.
- Apply the `graph_materialisation` mapping rules from the ontology file. Normalise money and durations **here** (`£1m` → `1000000` GBP, `30 days` → `P30D`).
- Every semantic node gets `-[:STATED_IN]->(:Clause)`. Assert zero orphans.
- Load `CROSS_REFERENCES` with `reference_text`, `ref_class` and `scope_rule` as edge properties — you will need them to debug bad traversals.

### 2.3 Vector index

Embed **`Clause.text`** — the clause is the retrieval unit.

```cypher
CREATE VECTOR INDEX clause_embedding IF NOT EXISTS
FOR (c:Clause) ON (c.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};
```

- Set `vector.dimensions` to match your model. Changing it later means dropping and rebuilding.
- Embed `hierarchy_path + "\n" + text`, not `text` alone. The path carries the schedule name, which disambiguates boilerplate recurring across schedules.
- Second index on `Definition.definition_text` — definition lookups are a distinct pattern.
- Store `embedding_model` and `embedding_version` as node properties to detect a stale index.

**Acceptance:** node and edge counts match source files; zero orphaned semantic nodes; `CROSS_REFERENCES` count matches `edges.jsonl`; a known query (`"liability cap"`) returns relevant clauses in the top 5.

---

## 4. Stage 3 — MCP tools + LangGraph agent

### 3.1 MCP server

Expose **narrow, parameterised tools** — not a raw Cypher endpoint. A general "run this Cypher" tool will be abused by the agent, returns unbounded results, and is a security hole.

| Tool | Signature | Returns |
|---|---|---|
| `search_clauses` | `query, doc_filter?, top_k=5` | clause text + hierarchy_path + clause_id |
| `expand_context` | `clause_id` | parents, siblings, defined terms, outbound refs |
| `lookup_definition` | `term, scope_doc?` | definition text + defining document; local override if one exists |
| `get_obligations` | `actor, doc_filter?` | obligations with modality, deadline, source |
| `get_termination_rights` | `actor?, trigger?` | who / why / notice / against what |
| `get_liability_position` | `instrument?` | cap **plus** carve-outs, never the cap alone |
| `trace_references` | `clause_id, depth=2, direction="out"` | reference chain; `direction="in"` for inbound |

**Every tool returns `clause_id` and `hierarchy_path` with each result.** Answers without citations are unusable for contract work.

`trace_references` must implement **cycle detection** and hard-cap depth at 3.

Cap result sizes (default 10 rows, 8,000 characters) so a broad query cannot blow the agent's context.

### 3.2 LangGraph agent

**State:**
```python
class AgentState(TypedDict):
    question: str
    plan: list[str]
    retrieved: list[dict]      # accumulating evidence, each with clause_id
    visited: set[str]          # cycle guard for reference traversal
    hops: int
    answer: str
    citations: list[str]
```

**Nodes:**
1. `classify` — route: definition lookup / single-clause / multi-hop / comparison / aggregation. Route determines available tools.
2. `retrieve` — vector search plus fulltext in parallel, merged with reciprocal rank fusion.
3. `expand` — for each promising hit, call `expand_context`, `lookup_definition` on unfamiliar capitalised terms, and `trace_references` when the clause text points elsewhere. Add every visited `clause_id` to `visited`.
4. `reflect` — conditional edge. Is the evidence sufficient? If not and `hops < 3`, loop back with a refined query. **Cap hops at 3.**
5. `synthesise` — answer with inline clause citations.
6. `verify` — every citation in the answer must exist in `retrieved`. Strip or flag any that does not.

**Synthesis rules:**
- Cite `hierarchy_path` (and page range) for every substantive claim.
- If evidence is insufficient, say so — do not fill gaps from general contract knowledge.
- For liability questions, state cap **and** carve-outs together.
- **When a clause cross-references another, follow the reference before answering.** A clause that says "subject to Clause 34" is not fully answered without Clause 34.
- Flag when a Call-Off Schedule overrides Core Terms rather than reporting both as equally valid.

---

## 5. Stage 4 — Evaluation with LangSmith

### 4.1 Golden set

**60–100 question/answer pairs** with ground-truth `clause_id` lists, hand-labelled. Distribution:

| Type | Share | Example |
|---|---|---|
| Definition lookup | 15% | "What is an Occasion of Tax Non-Compliance?" |
| Single-clause fact | 20% | "What is the minimum warranty period on Deliverables?" |
| **Cross-page clause** | 10% | Questions whose answer sits in a clause spanning a page break — direct test of Stage 1.2 |
| **Cross-reference chain** | 15% | "What conditions apply to the Clause 34 rights referenced in Joint Schedule 7?" |
| Multi-hop | 15% | "What happens to Call-Off Contracts if the Framework Contract is terminated?" |
| Aggregation | 15% | "List all Supplier obligations with a deadline under 5 Working Days." |
| Negative / unanswerable | 10% | "What is the penalty for late delivery of hardware?" (not in document — correct answer is to say so) |

The negative set is not optional. It is the only way to measure hallucination.

### 4.2 Metrics

**Retrieval:**
- `recall@k` — target **≥0.90 @ k=10**.
- `precision@k` — target ≥0.60.
- `MRR`.

**Stage 1 quality (measured once against a labelled sample):**
- **Chunk integrity rate** — chunks that are complete, unbroken provisions. Target ≥0.98.
- **Cross-reference resolution rate** — target ≥0.95, and precision on a hand-checked sample of 100 edges ≥0.95. A wrong edge is worse than a missing one: it actively misroutes the agent.
- Node and edge precision/recall by type.

**Answer:**
- **Citation accuracy** — every citation resolves and supports the claim. Target 1.0; below that is a correctness bug.
- **Faithfulness** — no unsupported claims (LLM-as-judge).
- **Abstention rate on the negative set** — target ≥0.90.

### 4.3 LangSmith wiring

- Trace every run; tag by question type so you can see which category fails.
- Upload the golden set as a LangSmith dataset; run evaluators against it.
- Custom evaluators for `recall@k`, `citation_accuracy`, `abstention`, `crossref_precision`.
- **Establish a baseline first:** run plain vector RAG (no graph) over the same chunks against the same golden set. If GraphRAG does not beat it on the multi-hop and cross-reference categories, the graph is not earning its complexity — fix the graph rather than adding agent nodes.

---

## 6. Build order and gates

| Step | Gate before proceeding |
|---|---|
| 1 | Document AI output diffs cleanly against `pdftotext`; bboxes present |
| 2 | Boilerplate report reviewed; `core_terms.3.2.9` reconstructs across its page break |
| 3 | Chunk coverage ≥99%; zero chunks ending without terminal punctuation |
| 4 | All hard validation checks pass; numbering gaps reviewed |
| 5 | Definitions loaded; ≥95% of cross-references resolve; scope-rule regression test passes |
| 6 | LLM records: 100% verbatim check, error rate documented |
| 7 | Graph loads with zero orphans; edge counts match |
| 8 | Vector search sensible on 10 manual probes; `trace_references` terminates on a cyclic case |
| 9 | MCP tools correct when called directly, without the agent |
| 10 | Agent beats the flat-RAG baseline on multi-hop and cross-reference questions |

**Build stages 1–2 fully before starting stage 3.** A retrieval bug looks exactly like an agent bug from the outside, and debugging both at once wastes days.

---

## 7. Known pitfalls specific to this document

- **Page breaks mid-clause are the norm, not the exception.** Any per-page processing produces broken text. Stitch first; treat pages as citation metadata only.
- **The running header is identical to legitimate heading text.** Strip by position + repetition frequency, never by string match alone, and keep the first occurrence at a true schedule boundary.
- **`Clause` vs `Paragraph` is a scope distinction, not a synonym.** `Clause N` → Core Terms; `Paragraph N` → the containing schedule. Getting this wrong produces confidently wrong edges throughout the graph.
- **Cross-reference cycles exist.** Core Terms points at schedules that point back. Always carry a visited set.
- **Repeated boilerplate clauses.** Standard provisions recur near-verbatim across schedules. Vector search alone returns the wrong copy — hence embedding `hierarchy_path` with the text, plus `doc_filter`.
- **Order of precedence.** Core Terms, Schedules and the Order Form rank against each other. Quoting Core Terms where a Call-Off Schedule overrides it is a wrong answer. Capture the precedence clause early.
- **Optional schedules.** Buyers may exclude optional Call-Off Schedules or add Special Terms. Flag `is_optional` so answers can be qualified.
- **Two definition scopes.** Joint Schedule 1 is global; schedules define terms locally that override it within their own document.
- **Jurisdiction variants.** Scottish Law, Northern Ireland Law and MOD Terms schedules modify standard terms. Out of scope for v1 — tag those documents so the agent can warn a variant exists.
- **Lots.** Prices and specification vary by Lot 1–4. Out of scope for v1; do not let the agent state Lot-specific prices as general facts.
