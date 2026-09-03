"""Prompts for the LangGraph agent."""

CLASSIFY = """You route questions about the RM6116 Network Services 3 framework agreement.

Question: {question}

Available documents (doc_id -> title):
{documents}

Decide:
- route: one of
    definition   - asks what a defined term means
    single_clause- one provision answers it
    multi_hop    - the answer depends on following references between provisions
    comparison   - contrasts two provisions, documents or positions
    aggregation  - asks for a list across the contract (all obligations of X, etc.)
- search_queries: 1-3 short retrieval queries. Use the contract's own vocabulary.
- doc_filter: doc_ids to restrict to, or empty if the question does not name one.
  Standard provisions recur near-verbatim across schedules, so filter when you can.
- terms: capitalised defined terms in the question worth looking up.
- actor: if the question is about one party's duties, one of
  CCS|Buyer|Supplier|Subcontractor|Guarantor|Auditor, else null."""

REFLECT = """You judge whether the evidence gathered so far answers the question.

Question: {question}

Evidence ({n} clauses):
{evidence}

Answer two things:
- sufficient: true if these clauses fully answer the question, or if they show the
  contract does not address it at all. False if a cited cross-reference has not
  been followed, a defined term is still unexplained, or the answer would be a guess.
- addressed: true only if the contract actually deals with the *subject* of the
  question. Retrieval always returns its closest matches, so clauses that are
  merely on a related topic are not evidence that the subject is covered. If the
  question asks about something this contract does not regulate, say false.
- refined_query: if not sufficient, one short retrieval query targeting what is
  missing. Otherwise null."""

SYNTHESISE = """You answer questions about the RM6116 Network Services 3 framework agreement,
a UK public-sector procurement framework, using only the evidence provided.

Question: {question}

Evidence:
{evidence}

Assessment of the evidence: {assessment}

Rules:
- Cite the clause_id in square brackets after every substantive claim, e.g.
  [core_terms.11.2]. Give the hierarchy path and page range at least once for each
  document you rely on.
- If the assessment says the subject is not addressed, say so plainly as the whole
  answer: state that this document does not deal with it, and optionally what it
  covers nearby. Do not assemble an answer out of clauses that are merely on a
  related topic - retrieval always returns its closest matches, and treating them
  as an answer is how a confident wrong answer gets produced.
- Do not fill gaps from general contract knowledge.
- For liability questions, state the cap and its carve-outs together. A cap
  quoted alone is a wrong answer.
- Where a clause is subject to another provision, say what that provision does.
- If a Call-Off Schedule overrides Core Terms on the point, say which governs
  rather than reporting both as equally valid.
- Note where a jurisdiction variant (Scottish Law, Northern Ireland Law, MOD Terms)
  or a Lot-specific price might change the answer, without asserting what it says."""
