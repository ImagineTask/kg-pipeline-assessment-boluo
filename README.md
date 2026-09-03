# RM6116 GraphRAG

A GraphRAG pipeline over **RM6116 Network Services 3 — Framework Agreement**
(475 pages, 48 constituent documents, Crown Copyright 2018, v3.0.11), built to
`BUILD_SPEC.md`.

**Current state at a glance.** Both gates pass. All 48 documents are recovered
from the document itself and match the published page map exactly. 97.1% of the
text reaches the graph, verified against an independent extraction. Cross-
references resolve at 96.8% with zero scope-rule violations. Every citation the
agent produces resolves to a real clause it actually saw. Answers are judged
faithful 0.963 of the time and multi-hop questions are answered 0.917 of the time.

It is **not production ready** — see *Known limits*. The absence of any
human-validated error rate is the most serious gap; aggregation returning half a
complete answer is the most visible one.

```
PDF ─▶ Document AI batch OCR ─▶ raw_pages.jsonl
                                     │
                     boilerplate strip + row merge + page stitching
                                     ▼
                            document_stream.jsonl          ← page boundaries destroyed here
                                     │
                        structure-aware hierarchical chunker
                                     ▼
                              clauses.jsonl ──── gate 1 (deterministic validation)
                                     │
              definitions + scope-aware cross-reference resolution
                                     ▼
                    definitions.jsonl · term_edges.jsonl · edges.jsonl
                                     │
       LLM extraction (Gemini 2.5 Pro) — one record per *provision*
                                     ▼
                              records.jsonl ──── gate 2 (extraction QA)
                                     │
                     graph materialiser ─▶ Neo4j + vector indexes
                                     │
                MCP tools ─▶ LangGraph agent ─▶ golden-set evaluation
```

Two rules govern the design, and most of the interesting decisions follow from them:

1. **Structure is deterministic, semantics are probabilistic.** Document
   boundaries, clause numbering, definitions and cross-references are all parsed.
   The model is only asked what a provision *means*.
2. **The page is a printing artefact, not a unit of meaning.** Page boundaries are
   destroyed before chunking. Pages survive only as citation metadata.

---

## Running it

```bash
make setup            # uv venv + dependencies (Python 3.12)
make neo4j-up         # Neo4j 5 in Docker with APOC on :7687
cp .env.example .env  # fill in GCP project, DocAI processor, GCS buckets, Neo4j password

make all              # stages 1 and 2, end to end, with both gates
make test             # 20 regression tests

make ask Q="What is the cap on the Supplier's liability?"
make mcp              # MCP server on stdio (see .mcp.json)

make golden           # build both golden sets
make eval             # 80 questions: recall@k, citations, judged answers
make eval-agg         # 12 complete-answer questions: set precision and recall
make graph-eval       # measure the graph itself: coverage, recall, precision
```

Every stage writes a report to `data/reports/`. Both gates exit non-zero on a
hard-check failure, so `make all` stops rather than propagating bad data.

**Credentials.** Everything runs on Google Cloud: Document AI for OCR, Gemini 2.5
Pro and `gemini-embedding-001` on Vertex AI. `src/gcp_auth.py` uses standard
Application Default Credentials where they work and otherwise mints short-lived
tokens from the `gcloud` CLI — no long-lived service-account key on disk.

---

## Stage 1 — data processing

### 1.1 Document AI

Batch (asynchronous) processing against an **Enterprise Document OCR** processor
in `eu`. The PDF is sharded into 100-page pieces before upload and each shard
carries a **global page offset**, so the merge cannot silently shift page numbers —
an off-by-N offset would corrupt every citation downstream, invisibly. The merged
output asserts contiguous pages 1..N.

`pdftotext -layout` runs as a parallel extraction and the two are diffed per page.
It is nearly free and it earns its place twice: once as a regression check, and
again in §1.5 where it repairs OCR errors in defined terms.

| Check | Result |
|---|---|
| Pages | **475 / 475** |
| Blocks | 10,442, **0 without a bounding box** |
| Pages within 2% of `pdftotext` | **468 / 475 (98.5%)**, target ≥95% |

### 1.2 Page-boundary reconstruction

**Boilerplate stripping.** Detection is by position and repetition, never by
string match: the running schedule header is textually identical to the real
heading at the start of that schedule, so a regex on `Joint Schedule 1
(Definitions)` deletes contract text.

Two departures from the spec's algorithm were needed, both forced by the data:

- *Per-cluster density, not a global page fraction.* The spec's ">30% of pages"
  never fires here, because the running header **changes per schedule**: Joint
  Schedule 1's header covers 28 of 475 pages (5.9%). Repetition is instead
  measured as density over the contiguous page run a pattern occupies. Clustering
  is also required in the other direction — a bare page number `3` appears once
  per schedule, so its *global* span is the whole document and its global density
  is near zero.
- *A second, line-level pass.* Document AI merges the two header lines into one
  block on 23 pages of Joint Schedule 1 and leaves them separate on 5. A
  block-shape key misses those 5. Confirmed boilerplate *lines* are therefore also
  scrubbed out of blocks that survive — which additionally cleans up headers that
  Document AI merged into the middle of body text.

Result: **162 patterns, 2,045 blocks stripped**. Every page sheds something except
pages 461–469 (Call-Off Schedule 23, HMRC Terms), which genuinely carry no header
or footer. `data/reports/boilerplate_report.json` lists every pattern with its page
span and density, for the one-time manual review the spec asks for.

**Row merging — a stage the spec does not have, and the pipeline cannot work
without.** Most schedules set the clause number in its own column:

```
y=375  x=359  "2.1.3"
y=376  x=514  "any proceeding, claim or demand by HMRC or other statutory authority…"
```

and Joint Schedule 1 sets the defined term in a left column against its definition
on the right. Without merging rows before anything else, every clause number in
every schedule arrives as a free-floating block and no clause outside Core Terms
gets an id. Rows are classified as `label` (a numbering label), `term` (a defined
term), `bullet`, or `table_row`, and merged accordingly — which is also what keeps
the two-column definitions schedule from being swallowed by the table detector.

**Join repair.** The spec's literal rule — join only when the next line starts
lowercase — is wrong for this document, and its own worked example proves it.
Clause 3.2.9 breaks as `"...the Buyer needs to make use of the"` / `"Goods."`, and
`Goods` is a capitalised **defined term**, not a new sentence. An unterminated line
is the reliable signal; capitalisation is not.

A second rule, **truncation beats structure**: clause 10.6.1 continues onto a line
beginning `20.2 or a Contract expires…` — a cross-reference number that looks
exactly like a clause number. A genuine new clause never follows a sentence that
stops mid-phrase, so truncation wins. Both cases are unit tests.

| | |
|---|---|
| Stream blocks | 4,601 (from 10,442 raw) |
| Cross-page joins | 131 |
| Tables detected / spanning pages | 29 / 4 |
| Character retention vs raw | **93.6%** |

> **On the 97% retention target.** 6.4% of characters are boilerplate: 2,045
> header/footer blocks over 475 pages is ~116 characters per page, which matches
> the measured header (`Joint Schedule 1 (Definitions) Crown Copyright 2018`, 50
> chars) plus footer (`Framework Ref: RM6116 Project Version: v1.0 Model Version:
> v3.10`, 64 chars) plus page number almost exactly. The spec's 97% floor assumed
> less furniture than this document carries. The number is reported rather than
> tuned away, and §1.4's coverage check (99.2%) is the one that actually guards
> against lost text.

### 1.3 Chunking

Structure-aware hierarchical chunking on the document's own numbering, never
fixed-size splitting.

**Document segmentation is derived from the running headers, not from a heading
regex over body text.** This matters more than it sounds. The Framework Award Form
lists all 25 Call-Off Schedules by name on pages 25–26; segmenting on those names
opens twenty empty documents and mis-assigns every schedule that follows (in an
early run, `call_off_schedule_6` swallowed pages 302–426 and eight other
schedules). The running header instead names the schedule each page actually
belongs to. Short schedules with no header at all — Core Terms among them — fall
back to a standalone heading match, which is safe precisely because those runs are
bounded by pages the headers already claimed.

The derived page map reproduces the published one in `DOCUMENT_NOTES.md` exactly,
including the two-line wrapped title `Framework Schedule 6 (Order Form Template and
Call-Off Schedules)` and `RM6116 Call-Off Schedule 24 (Supplier Furnished Terms)`,
whose framework-reference prefix and wrapped parenthesis both had to be handled.

**48 / 48 documents recovered, 3,502 chunks.** Clause numbering restarts inside
each Part and Annex, so the section qualifies the clause id
(`call_off_schedule_2.part_c.1.5.2`).

Definitions chunk differently: one defined term is one chunk keyed by the term.
Local definitions in other schedules use the same layout and are captured the same
way, with `scope: document_local`.

### 1.4 Gate 1 — deterministic validation

```
coverage 99.24% | chunks 3502 | docs 48
  [PASS] coverage_ge_99pct
  [PASS] no_duplicate_clause_ids
  [PASS] no_orphan_parents
  [PASS] no_empty_chunks
  [PASS] no_boilerplate_residue
  [PASS] sentence_integrity
  [PASS] no_chunk_starts_mid_sentence
```

**Sentence integrity is retargeted, deliberately.** "No chunk ends without terminal
punctuation" fails 457 times here, almost all of them legitimate: Framework
Schedule 1 is full of bulleted service lists that end bare, and list limbs end
`"...; and"`. The check therefore looks for chunks that stop **mid-sentence** — a
trailing comma, hyphen, or function word — which is the shape a missed cross-page
join actually produces, and is exactly what clause 3.2.9 looked like before the
stitcher was fixed.

That leaves 9 chunks out of 3,502 (0.26%). Each was compared against
`pdftotext -layout` on its page. **None has lost source text**; all are trailing
fragments picked up from an adjacent column during row merging, or side-column text
in a form template. They are recorded with their reasons in
`config/accepted_truncations.json`, and the gate fails on anything not in that
file — including a stale entry.

Two things found along the way that are worth stating: page 230 of the source PDF
is itself malformed (`2.1.3 monitor the number, type an` / `2.1.4 d value of…` —
`pdftotext` shows the same break), and the 51 numbering gaps include the Framework
Award Form's, which the document explains on page 25: *"Where numbers are missing
we are not using these schedules."*

### 1.5 Definitions and cross-references

**514 definitions** — 237 global (Joint Schedule 1), 277 document-local, with **16
terms locally overriding a global definition**. **9,446 `USES_TERM` edges** over
2,779 clauses, matched longest-first with span masking so `Key Subcontractor` wins
over `Subcontractor`.

**OCR repair of defined terms.** Document AI reads *"Occasion of Tax
**Non**-Compliance"* as *"Occasion of Tax **Jon**-Compliance"* — a single-character
substitution, invisible to a character-count diff, that silently breaks the
gazetteer entry for one of the document's more important terms. The parallel
`pdftotext` extraction already exists, so each term is checked against it. Two
guards keep this from doing damage: a repair is only accepted if the corrected
phrase **is actually used elsewhere in the document** (which rejects the fragments
that `-layout` produces by straddling the term and definition columns), and the
term must still look like a defined term. **17 terms repaired**, all genuine —
they share a pattern of dropped leading capitals (`I`→`h`, `N`→`l`,
`T`→`emplate`).

**Cross-reference resolution.** The rule that matters more than any regex:

> `Clause N` always means **Core Terms**, wherever it is written.
> `Paragraph N` means paragraph N of **the schedule the reference sits in**.

| | |
|---|---|
| References detected | 1,680 |
| Resolved | **1,544 (97.0%)** of resolvable, target ≥95% |
| Unresolvable by design | 88 (76 bare `above`/`below`, 12 statutory) |
| **`Paragraph N` leaking into Core Terms** | **0** — the scope-rule regression test |
| `Clause 10.4.1` (17 occurrences, 5 documents) | resolves to `core_terms.10.4.1` **every time** |

Ranges expand in full (`Paragraphs 4.3 to 4.6` → four edges), lists split, and
`Annex`/`Part`/`Appendix` are matched **case-sensitively** — lower-case "part of"
appears constantly in prose and matching it manufactured 208 edges to nothing.
Statutory references (`Part 7 of the Finance Act`) are classified as external
rather than resolved inside the schedule. Nothing is dropped silently: 136
unresolved references sit in `data/processed/unresolved.jsonl` with a reason each.

### 1.6 / 1.7 LLM extraction and gate 2

Gemini 2.5 Pro on Vertex AI, temperature 0, output shape enforced by a Pydantic
response schema rather than JSON parsing. Batches of 6 clauses with clause ids
echoed back and checked; a mismatch falls back to one clause at a time. Resumable —
a re-run only extracts what is missing. Tables are skipped, per the spec.

**3,563 records over 2,840 clauses (1.25 provisions per clause). Zero quarantined.**
(834k input / 381k output tokens, 515 calls.) See *What the ontology change bought*
under stage 4b for why a clause now yields more than one record.

```
records 3563 over 2840 clauses (1.25 per clause) | verbatim 98.03% (9/458 nulled and logged)
  [PASS] enum_conformance_100pct
  [PASS] verbatim_100pct_after_rejection
  [PASS] obligations_have_an_actor_ge_97pct
  [PASS] supplier_obligations_dominate
  obligations by actor: {'Supplier': 1082, 'Buyer': 132, 'Other': 99, 'Guarantor': 24, 'CCS': 10, 'Subcontractor': 1}
```

The **verbatim check** is the load-bearing one — amounts and deadlines are copied
character-for-character by instruction, so anything not in the source was invented.
It is checked against *exactly what the model was shown* (clause text, heading,
hierarchy path, and the parent clause supplied as context); checking the clause
text alone reported the £10,000,000 cap printed in the Framework Award Form's
heading as a hallucination. Nine genuine failures remain and, as the spec requires,
they are **rejected**: the field is nulled and the rejection logged to
`data/processed/rejected_values.jsonl`. Supplier obligations outnumber Buyer
obligations 1,082 to 132, so there is no role-swap bug.

A 50-record random sample is written to `data/reports/qa_human_sample.json` with
empty `reviewer_verdict` fields, for the human baseline the spec asks for. **That
review has not been done** — it needs a person, and the error rate is therefore not
yet established.

---

## Stage 2 — the graph

| Nodes | | Relationships | |
|---|---|---|---|
| Clause | 3,482 | USES_TERM | 10,083 |
| Provision | 1,601 | STATED_IN | 3,563 |
| Obligation | 1,378 | PART_OF | 3,483 |
| Definition | 514 | IN_DOCUMENT | 3,482 |
| Remedy | 306 | BOUND_BY | 1,348 |
| Liability_Cap | 195 | OWED_TO | 1,112 |
| Defined_Event | 142 | CROSS_REFERENCES | 1,097 |
| Financial_Term | 83 | DEFINED_IN | 514 |
| Document | 48 | REFERENCES_DOCUMENT | 444 |
| Actor | 7 | HAS_REMEDY / TRIGGERED_BY / PAYS / EXCLUDED_FROM_CAP | 306 / 170 / 76 / 6 |

Constraints and indexes are created before loading; every write is `UNWIND $batch`
in batches of 1,000, `MERGE` on the constrained key, so reloads are idempotent.
Money and durations are normalised **here** (`£1m` → `1000000 GBP`, `30 days` →
`P30D`), with Working Days kept flagged — conflating five Working Days with five
calendar days makes every deadline comparison wrong.

All five load acceptance checks pass: clause count matches, cross-reference count
matches `edges.jsonl`, semantic node count matches `records.jsonl`, **zero orphaned
semantic nodes**, zero clauses outside a document.

> **Reloads rebuild rather than merge.** `MERGE` is idempotent for what is present
> and silent about what has been removed, so re-chunking would leave orphaned
> Clause nodes behind — still searchable, still citable. Clauses and definitions
> are therefore pruned against the source files, and the semantic layer is dropped
> and rebuilt outright: a semantic node is keyed by provision id but *labelled* by
> provision type, so a provision reclassified between runs (`obligation` →
> `right`) would otherwise survive under its old label with an id that still looked
> current. `semantic_nodes_match_records` is an acceptance check.

> **Deviation:** the spec's `Definition.term IS UNIQUE` constraint cannot hold,
> because 16 terms are defined both globally and locally. The key is
> `defined_in:term`, and `lookup_definition` resolves the override at query time.

**Vector index.** `gemini-embedding-001` at 1,536 dimensions, cosine. What is
embedded is `hierarchy_path + "\n" + text`, not text alone: standard provisions
recur near-verbatim across schedules, and the path carries the schedule name that
separates them. A second index covers `Definition.definition_text`.
`embedding_model` and `embedding_version` are stored per node so a stale index is
detectable. **3,129 clauses + 514 definitions embedded, none missing.**

---

## Stage 3 — MCP tools and the agent

Eight narrow, parameterised MCP tools: `search_clauses`, `expand_context`,
`lookup_definition`, `get_obligations`, `get_termination_rights`,
`get_liability_position`, `trace_references`, `list_documents`. There is
deliberately **no raw-Cypher tool** — it returns unbounded results, hands the
model a route to anything in the database, and cannot be evaluated. Every row
carries `clause_id` and `hierarchy_path`; every result set is capped at 10 rows
and 8,000 characters.

`get_liability_position` never returns a cap without its carve-outs, because a cap
quoted alone is a wrong answer.

`trace_references` enforces acyclic paths and a hard depth cap of 3. This is not
theoretical: `core_terms.15.1` and `15.2`, `15.3`, `15.4` point at each other, and
the traversal is tested against that cycle at depths 1–3 and at a requested depth
of 99.

**Both agents reach the graph only through those tools**, over stdio to the server
as a subprocess ([`src/agent/mcp_client.py`](src/agent/mcp_client.py)). Neither
imports a query module or opens a Neo4j session. That was not true of the first
version: the pipeline agent imported `src.retrieval.queries` directly, which made
the `MCP server → agent` edge in the architecture diagram a claim the code did not
support and left the server exercised only by external clients. It is now load
bearing — if a tool is broken, both agents are broken.

### The agent

[`graph.py`](src/agent/graph.py) is the spec's §3.2 state machine:
`classify → retrieve → expand → reflect → synthesise → verify`, looping back to
`retrieve` at most 3 hops, carrying a `visited` set as the cycle guard. `retrieve`
fuses several `search_clauses` rankings with reciprocal rank fusion; `expand` pulls
parents, follows outbound references, and looks up capitalised terms it has not
seen. `verify` checks every citation against the retrieved evidence and rewrites
anything it cannot find as `[unverified: …]` — an invented citation is a
correctness bug, not a formatting one.

It is a **pipeline, not a tool-choosing agent**, and worth calling that plainly.
Three LLM calls decide anything: `classify` picks the route and search queries,
`reflect` judges sufficiency and drives the only loop, `synthesise` writes the
answer. Which specialised tool gets called is decided in Python — by the route,
and in two places by keyword (`if "terminat" in question.lower()`). That keyword
fires on six golden-set questions including *"What is the definition of the
Termination Assistance Period?"*, a pure definition lookup that gets a
termination-rights query bolted on for nothing. It is the weakest part of the
design.

What the fixed shape buys is bounded cost and evaluability: every question takes
the same path, so a change in the numbers is a change in the retrieval or the
graph, not in how many tools the model felt like calling.

**Three retrieval behaviours are non-obvious and were each arrived at by
measurement**, so they are worth stating rather than leaving to be rediscovered:

- **`doc_filter` is a boost, not a gate.** Applied as a hard filter it is worse
  than useless: the classifier guesses a schedule from the question's topic, and a
  wrong guess makes the answer unreachable — in one case retrieving entirely from
  Joint Schedule 5 for a question answered in Framework Schedule 1. The unfiltered
  ranking always runs; a filtered one is fused into it.
- **The user's own question is always searched, first and at double weight.** The
  classifier's paraphrases retrieve *worse* than the question as asked (0.587
  against 0.736 recall@10, measured directly). Rewriting a question is a way to
  lose the words that made it findable.
- **Rank fusion weights vector search above full-text, 2:1.** Unweighted fusion of
  the two scores *below* vector search alone (0.736 against 0.764). Full-text still
  earns its place — it is what catches an exact defined term — at half the weight.

## Stage 4 — evaluation

Three questions, measured separately, because they fail for different reasons.

```
make golden      # build both golden sets
make eval        # can the agent answer? - 80 questions, recall@k and judged answers
make eval-agg    # can it return a complete set? - 12 questions, set precision/recall
make graph-eval  # does the graph itself hold enough? - stage 4b, below
```

### The golden set

**80 questions** in the spec's distribution: definition 12, single-clause 16,
cross-page 8, cross-reference chain 12, multi-hop 12, aggregation 12, negative 8.

**How ground truth is established, plainly.** Questions are *generated from*
clauses selected structurally — cross-page questions from clauses where
`spans_pages` is true, cross-reference questions from a resolved
`CROSS_REFERENCES` edge crossing documents, multi-hop from a two-edge chain — so
the ground-truth `clause_id` is correct **by construction** rather than by later
judgement, and every id is checked to exist in the graph. The eight negatives are
hand-written.

This is not the same thing as a hand-labelled set, and it has **two blind spots
that flatter every number computed on it**. Both are worth knowing before reading
the table below.

**It inherits the graph's own view of what refers to what.** Cross-reference
questions are generated from edges the resolver produced, so a reference class the
resolver never found is invisible here. Stage 4b exists to cover that.

**Its questions are near-paraphrases of the clause they came from**, which hands
retrieval a text-similarity shortcut that real users do not provide. Measured
directly: plain full-text search *alone*, with no vector index, no graph and no
agent, puts the answer in the top 5 for **26 of 40** questions. On the aggregation
set — whose questions describe a predicate rather than restating a clause — the
same search recovers **6.3%**.

So recall@10 of 0.738 should be read as an upper bound on real-world performance,
not an estimate of it. Someone asking "what must we do if the supplier goes into
administration" is not phrasing a paraphrase of the clause they need.

**The two sets are not comparable, and the difference is mostly construction:**

| | questions | ground truth per question | scoring |
|---|---|---|---|
| Main set | 80 | **1.67 clauses** — half have exactly one | recall@10: ten slots to find one |
| Aggregation set | 12 | **12.33 clauses** | set recall: must find all of them |

Finding 1 of 1 scores 1.000; finding 6 of 12 scores 0.500. The aggregation set's
lower numbers are mostly this and the absent text shortcut — not the system being
worse at aggregation than at everything else.

### Scoring

Retrieval and citation checks are arithmetic. Only the answer judgements need a
model, and they are marked as estimates wherever they appear.

| | | Target |
|---|---|---|
| recall@10 | 0.738 | ≥0.90 |
| recall@20 | 0.799 | — |
| precision@1 | **0.625** | ≥0.60 |
| MRR | 0.704 | — |
| Citations resolve to a real clause | **1.000** | 1.0 |
| Citations present in the evidence | **1.000** | 1.0 |
| Citations support their claim *(judged)* | 0.975 | 1.0 |
| Answers the question *(judged)* | 0.838 | — |
| Faithfulness *(judged)* | 0.963 | — |
| Abstention on negatives | 0.875 | ≥0.90 |
| False abstention on answerable | 0.028 | — |

80 questions, zero errors.

| By category | n | recall@10 | Answers the question |
|---|---|---|---|
| definition | 12 | 1.000 | 1.000 |
| single_clause | 16 | 1.000 | 1.000 |
| **cross_reference_chain** | 12 | 0.750 | **1.000** |
| **multi_hop** | 12 | 0.708 | **0.917** |
| cross_page | 8 | 0.875 | 0.625 |
| negative | 8 | — | 0.875 |
| aggregation | 12 | 0.056 | 0.333 |

**The two citation checks are structural and they pass exactly.** Every citation
the agent produced names a real clause and one it had actually seen — computed in
code, not judged. That is the property that makes an answer checkable, and it is
worth more than any of the judged numbers below it.

**Evidence is ranked, not ordered by arrival.** Every item carries a score:
search hits keep their fused RRF score, and an expanded neighbour inherits its
seed's score decayed by how it was reached — a referenced clause 0.85, a traced
one 0.70, a parent 0.55, a definition 0.50. A clause reached more than one way
keeps its best score. Ordering by arrival instead pushes the sixth search hit past
rank 10, because everything the first five seeds pull in sits directly behind them.

**Where it falls short.** recall@10 of 0.738 misses the 0.90 target, and
abstention is one negative question short of 0.90. recall@20 is 0.799, so the
gap is not a ranking artefact — it is evidence that is not retrieved at all.

**The aggregation numbers in the table above are not meaningful.** Those questions
are generated from clauses, so "all Supplier obligations with a deadline under 5
Working Days" gets three arbitrary clauses as ground truth when thirty satisfy it.
The second golden set below exists to measure that properly.


## Stage 4b — testing the graph itself

`make graph-eval` → [`src/eval/eval_graph_quality.py`](src/eval/eval_graph_quality.py)

This answers a different question from the golden set, and one nothing else here
could answer. `run_eval.py` measures whether retrieval finds **what the graph
knows**. It cannot measure whether the graph **knows enough**, because its
cross-reference questions are generated from edges the resolver already produced:
a resolver missing an entire class of reference would score perfectly on a set
built from its own output.

So ground truth has to come from outside the graph, and there are two sources:

1. **The parallel `pdftotext -layout` extraction**, which shares no code with the
   Document AI path the graph was built from. Used for coverage and for
   reference-detection recall.
2. **A silver standard**: a model reads raw page text it has never seen in chunked
   form and enumerates what is on the page; the graph is then scored against that,
   giving precision *and* recall per node/edge type.

### Coverage — against the independent extraction

| | |
|---|---|
| **Word coverage** (is the text in the graph at all) | **97.1%** |
| 5-gram coverage (is it there in the same order) | 81.5% |
| Pages with at least one clause | 98.5% |
| Clauses carrying a semantic node | 99.1% |
| Pages below 90% word coverage | 16 of 475 |

The two coverage numbers are deliberately kept apart because they answer different
questions. Word coverage says the text reached the graph; the 5-gram figure is
lower because two-column definitions and tables are *ordered* differently by
`-layout` than by the stitcher, not because anything is missing. Reporting only
the 5-gram number would understate the pipeline; reporting only the word number
would hide that adjacency is not preserved in tables.

The seven pages with no clause at all are genuinely near-empty — a signature block,
an empty form field, a residual footer variant. Nothing of substance is lost.

### Reference-detection recall — the check the golden set cannot do

The reference scanner is run over the *other* extraction and every mention is
matched against what the resolver saw:

| | |
|---|---|
| Reference mentions in the reference extraction | 1,430 |
| Seen by the resolver | 1,364 |
| **Detection recall** | **95.4%** |

The 66 residual misses cluster on `Annex` (25) and `Part` (11) and mostly sit in
tables and contents lists. That is an **upper bound** on real misses: the matcher
allows only ±1 page between a mention and the clause carrying it, so a reference
inside a clause spanning several pages can fall outside the window and be counted
as missed when it was found.


### Precision and recall by type — the silver standard

Twenty-four pages are sampled. A model enumerates the internal references, defined
terms and normative provisions on each from raw page text; a second pass aligns
that enumeration with what the graph holds for the page.

| | Recall | Precision |
|---|---|---|
| Internal references | 0.708 | 0.838 |
| Defined terms used | 0.801 | 0.618 |
| Normative provisions (duty / prohibition / permission) | **0.745** | **0.772** |

### Edge precision — is a resolved reference pointing at the right clause

**~0.93** on 100-edge random samples, graded one edge at a time
(`data/reports/edge_precision_sample.jsonl`), against the spec's ≥0.95 target.

Successive samples of 100 have returned 0.91, 0.94, 0.950 and 0.930, so the
underlying value is around 0.93 with a sampling spread of a few points. It has not
cleared the target; the one sample that hit 0.950 was the top of that range. A wrong edge misroutes the agent, so the residue still
matters: it concentrates on references to a Part or Annex *as a whole*, which have
no node of their own and resolve to the section's first substantive clause.


---
## Known limits

- **Tables are detected and merged but their content is not extracted.** 29 tables,
  4 spanning pages, serialised to markdown as single chunks and skipped by the
  extraction pass per the spec. Framework Schedule 3's prices and Call-Off
  Schedule 14's service levels are retrievable as text but not queryable as data —
  and for a procurement framework those are among the most-asked questions. The
  81.5% 5-gram coverage against 97.1% word coverage is this: every word present,
  adjacency lost.

- **No human validation baseline exists.** A 50-record sample sits in
  `data/reports/qa_human_sample.json` with empty `reviewer_verdict` fields. Until
  someone fills it in, the extraction has no human-checked error rate — which for
  a system answering contractual questions is the most serious gap here.
- **Unknown OCR residue.** 17 corrupted defined terms were found and repaired, but
  only because an independent extraction happened to surface them. There is no
  estimate of how many remain.
- **Retrieval still misses the target.** recall@10 is 0.738 against ≥0.90. Score-
  based ranking closed most of the gap that was an ordering artefact (recall@20 is
  0.799, so the two are now close); what remains is evidence the search does not
  surface at all.
- **Cross-reference edge precision is ~0.93, short of the 0.95 target.** The
  residue is concentrated on references to a Part or Annex as a whole, which have
  no node of their own and resolve to the section's first substantive clause.
- **Jurisdiction variants and Lots are out of scope but live in the document.**
  MOD Terms, Scottish Law and Northern Ireland Law modify the standard terms;
  prices vary by Lot. Documents are tagged so the agent can warn, and the synthesis
  prompt instructs it to — but nothing enforces it.
- **Order of precedence** between Core Terms, schedules and the Order Form is
  captured as text and instructed in the prompt, not modelled as a relationship,
  so precedence is not enforced structurally.
- **Nine chunks carry a trailing fragment** from an adjacent column (listed with
  reasons in `config/accepted_truncations.json`), and the source PDF is itself
  malformed at page 230.
---

