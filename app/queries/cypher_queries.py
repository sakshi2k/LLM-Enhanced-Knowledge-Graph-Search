CYPHER_QUERY_FETCH_GROUNDED_SPEECHES = """
// Phase 1: find speeches that match ANY queried topic
MATCH (s:Speech)-[c_match:COVERS]->(t_match:Topic)
WHERE t_match.id IN $topic_ids

// Phase 2: expand full speech context
MATCH (sp:Speaker)-[:DELIVERED]->(s)
MATCH (s)-[:HELD_ON]->(d:Date)
MATCH (s)-[c_all:COVERS]->(t_all:Topic)

WHERE ($speaker IS NULL OR toLower(sp.name) CONTAINS toLower($speaker))
  AND ($from_date IS NULL OR d.date >= date($from_date))
  AND ($to_date IS NULL OR d.date <= date($to_date))

WITH
  s,
  sp,
  max(d.date) AS date,
  max(c_match.confidence) AS confidence,
  collect(DISTINCT {
    id: t_all.id,
    label: t_all.label
  }) AS topics_with_id,
  collect(DISTINCT c_all.source_edge) AS provenance

// ---- ORDER TOPICS PROPERLY (Neo4j-safe) ----
UNWIND topics_with_id AS t
WITH
  s,
  sp,
  date,
  confidence,
  provenance,
  t
ORDER BY toInteger(substring(t.id, 3)) ASC
WITH
  s,
  sp,
  date,
  confidence,
  provenance,
  collect(t.label) AS ordered_topics

RETURN
  s.id AS speech_id,
  s.title AS title,
  sp.name AS speaker,
  date AS date,
  ordered_topics AS topics,
  confidence AS confidence,
  provenance AS provenance
ORDER BY confidence DESC
LIMIT $limit;

"""

CYPHER_QUERY_FETCH_GROUNDED_SPEECHES_old = """
    MATCH (s:Speech)-[c:COVERS]->(t:Topic)
    MATCH (sp:Speaker)-[:DELIVERED]->(s)
    MATCH (s)-[:HELD_ON]->(d:Date)
    WHERE t.id IN $topic_ids
      AND ($speaker IS NULL OR toLower(sp.name) CONTAINS toLower($speaker))
      AND ($from_date IS NULL OR d.date >= date($from_date))
      AND ($to_date IS NULL OR d.date <= date($to_date))
    WITH
      s,
      sp,
      max(d.date) AS date,
      max(c.confidence) AS confidence,
      collect(DISTINCT t.label) AS topics,
      collect(DISTINCT c.source_edge) AS provenance
    RETURN
      s.id AS speech_id,
      s.title AS title,
      sp.name AS speaker,
      date AS date,
      topics AS topics,
      confidence AS confidence,
      provenance AS provenance
    ORDER BY confidence DESC
    LIMIT $limit;
    """