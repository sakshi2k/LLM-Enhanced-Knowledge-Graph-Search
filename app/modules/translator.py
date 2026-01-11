from config import EDGE_MAP, MAX_RESULTS

def translate(normalized: dict) -> dict:
    cypher = f"""
    MATCH (sp:Speaker)-[:{EDGE_MAP['DELIVERED_SPEECH']}]->(s:Speech)
    MATCH (s)-[r:{EDGE_MAP['HAS_TOPIC']}]->(t:Topic)
    MATCH (s)-[:{EDGE_MAP['HELD_ON']}]->(d:Date)
    WHERE t.id IN $topics
    RETURN
        s.id AS speech_id,
        s.source_row AS source_row,
        s.title AS title,
        d.date AS date,
        sp.name AS speaker,
        s.content AS content,
        r.confidence AS confidence
    ORDER BY confidence DESC
    LIMIT $limit
    """

    return {
        "cypher": cypher,
        "params": {
            "topics": normalized["topics"],
            "limit": normalized.get("limit", MAX_RESULTS)
        }
    }
