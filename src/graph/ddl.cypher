// Constraints and indexes. Create these BEFORE loading: MERGE without a
// supporting constraint scans, and the load degrades to quadratic.

CREATE CONSTRAINT clause_id IF NOT EXISTS
  FOR (c:Clause) REQUIRE c.clause_id IS UNIQUE;
CREATE CONSTRAINT doc_id IF NOT EXISTS
  FOR (d:Document) REQUIRE d.doc_id IS UNIQUE;
CREATE CONSTRAINT definition_term IF NOT EXISTS
  FOR (t:Definition) REQUIRE t.key IS UNIQUE;
CREATE CONSTRAINT actor_name IF NOT EXISTS
  FOR (a:Actor) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT obligation_id IF NOT EXISTS
  FOR (o:Obligation) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT remedy_id IF NOT EXISTS
  FOR (r:Remedy) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT liability_id IF NOT EXISTS
  FOR (l:Liability_Cap) REQUIRE l.id IS UNIQUE;
CREATE CONSTRAINT financial_id IF NOT EXISTS
  FOR (f:Financial_Term) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT event_name IF NOT EXISTS
  FOR (e:Defined_Event) REQUIRE e.name IS UNIQUE;

CREATE FULLTEXT INDEX clause_fulltext IF NOT EXISTS
  FOR (c:Clause) ON EACH [c.text, c.heading, c.hierarchy_path];
CREATE FULLTEXT INDEX definition_fulltext IF NOT EXISTS
  FOR (d:Definition) ON EACH [d.term, d.definition_text];
CREATE INDEX clause_doc IF NOT EXISTS
  FOR (c:Clause) ON (c.doc_id);
CREATE INDEX clause_type IF NOT EXISTS
  FOR (c:Clause) ON (c.chunk_type);
CREATE INDEX definition_term_lookup IF NOT EXISTS
  FOR (d:Definition) ON (d.term_lower);
CREATE INDEX obligation_actor IF NOT EXISTS
  FOR (o:Obligation) ON (o.actor);
