### Steps to load data into Neo4j

#### STEP 0 — Start with a clean database
// Run in Neo4j Browser:
MATCH (n)
DETACH DELETE n;
Verify:   MATCH (n) RETURN count(n);
Expected: 0

#### STEP 1 — Load NODES (only once)
// Speakers
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.type = 'Speaker'
CREATE (:Speaker {
  id: row.id,
  name: row.label,
  source_row: toInteger(row.source_row)
});

// Speeches
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.type = 'Speech'
CREATE (:Speech {
  id: row.id,
  title: row.label,
  source_row: toInteger(row.source_row)
});

// Topics
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.type = 'Topic'
CREATE (:Topic {
  id: row.id,
  label: row.label,
  source_row: toInteger(row.source_row)
});

// Dates
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.type = 'Date'
CREATE (:Date {
  id: row.id,
  date: date(row.label)
});

#### STEP 2 — Load BASE RELATIONSHIPS (only source-of-truth)
// Speaker → Speech (DELIVERED)
LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
WITH row WHERE row.type = 'DELIVERED'
MATCH (sp:Speaker {id: row.source})
MATCH (s:Speech  {id: row.target})
CREATE (sp)-[:DELIVERED {confidence: toFloat(row.confidence)}]->(s);

// Speaker → Topic (PROFILE_COVERS)
LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
WITH row WHERE row.type = 'PROFILE_COVERS'
MATCH (sp:Speaker {id: row.source})
MATCH (t:Topic   {id: row.target})
CREATE (sp)-[:PROFILE_COVERS {
  confidence: toFloat(row.confidence),
  edge_key: row.key
}]->(t);

// Speech → Date (HELD_ON)
LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
WITH row WHERE row.type = 'HELD_ON'
MATCH (s:Speech {id: row.source})
MATCH (d:Date   {id: row.target})
CREATE (s)-[:HELD_ON]->(d);

#### STEP 3 — DERIVE Speech → Topic (COVERS) inside Neo4j
// Derive Speech → Topic (COVERS) from Speaker → PROFILE_COVERS
MATCH (sp:Speaker)-[r:PROFILE_COVERS]->(t:Topic)
WITH r, t,
     split(r.edge_key, '-') AS parts
WITH t, r, parts
WHERE size(parts) >= 2
WITH
  'SPH' + substring(parts[0], 3) AS speech_id,
  t,
  r
MATCH (s:Speech {id: speech_id})
MERGE (s)-[:COVERS {
  derived: true,
  confidence: r.confidence,
  source_edge: r.edge_key
}]->(t);


### Create indexes and run sanity checks

#### STEP 4 — Indexes (required for queries)
// CREATE INDEX IF NOT EXISTS FOR (s:Speaker) ON (s.id);
// CREATE INDEX IF NOT EXISTS FOR (s:Speech)  ON (s.id);
// CREATE INDEX IF NOT EXISTS FOR (t:Topic)   ON (t.id);
// CREATE INDEX IF NOT EXISTS FOR (d:Date)    ON (d.id);

// CREATE INDEX topic_text_idx
CREATE FULLTEXT INDEX topic_text_idx
IF NOT EXISTS
FOR (t:Topic)
ON EACH [t.label];

// run topic_text_idx
CALL db.index.fulltext.queryNodes(
  'topic_text_idx',
  'monetary policy'
)
YIELD node, score
RETURN node.id, score
LIMIT 5;

// SHOW INDEXES
SHOW INDEXES
YIELD name, type, entityType, labelsOrTypes, properties
WHERE type = 'FULLTEXT'
RETURN name, labelsOrTypes, properties;


#### STEP 5 — Sanity checks (must pass)
// Structure
MATCH (a)-[r]->(b)
RETURN labels(a), type(r), labels(b), count(*)
ORDER BY count(*) DESC;

#### Expected patterns:
Speaker — DELIVERED→ Speech
Speaker — PROFILE_COVERS→ Topic
Speech — COVERS→ Topic
Speech — HELD_ON→ Date

#### Ground-truth query (THIS is what API will use)
MATCH (sp:Speaker)-[:DELIVERED]->(s:Speech)
MATCH (s)-[:COVERS]->(t:Topic)
MATCH (s)-[:HELD_ON]->(d:Date)
WHERE t.label CONTAINS 'monetary'
RETURN s.title, sp.name, d.date, t.label
LIMIT 5;

-- Note:
-- If this returns rows → you are done.
-- Find full and final version of query in logic.