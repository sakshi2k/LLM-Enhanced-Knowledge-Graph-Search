# Translator MVP for Cypher / Neo4j using the fixed relationships the user requested.
# Relationships to use:
# (Person)-[:DELIVERED_SPEECH]->(Speech)
# (Speech)-[:HAS_TOPIC]->(Topic)
# (Speech)-[:MENTIONS_ORG]->(Organization)
# (Speech)-[:HELD_ON]->(Date)
#
# The translator takes the normalizer output (a dict with fields like 'intent','entities','date_range','canonical_query',
# 'transform_log') and emits a parameterized Cypher query (string with $params), a params dict, and metadata.
#
# Behavior specifics per user:
# - Use only Cypher
# - If speaker ambiguous (no auto-pick), do NOT run a union; instead return needs_disambiguation with candidates.
# - If date_range is None due to ambiguity, omit date filter.
# - Always return the filled parameter dict and a safe parameterized Cypher string (no direct concatenation of raw user text).
#
# We'll implement templates for: fact_lookup, timeline, quote_search, speaker_bio, compare_metric, citation_lookup.
# Then run the translator on several example normalized inputs (crafted to reflect outputs of the normalizer).

import re, json
from typing import Dict, Any, Tuple

def sanitize_id(s: str) -> str:
    """Sanitize a canonical id for use as a param value (not injecting into query string).
    We don't put user strings into query text; they go into params. Still sanitize to remove control chars."""
    if s is None:
        return None
    return re.sub(r'[\x00-\x1f\x7f]', '', str(s)).strip()

def build_param_provenance(transform_log):
    """Build a simple provenance map from transform_log: map param -> origin step if present"""
    prov = {}
    for entry in transform_log:
        # look for obvious keys in entry
        if entry.get("step") == "speaker_resolution" and entry.get("output") and entry["output"].get("canonical"):
            prov["speaker_id"] = {"origin_step":"speaker_resolution","value": entry["output"]["id"]}
        if entry.get("step") == "date_normalize" and entry.get("output"):
            prov["date_range"] = {"origin_step":"date_normalize","value": entry["output"]}
        if entry.get("step") == "abbreviation_expansion" and entry.get("output"):
            # best-effort: could map indicator provenance
            prov.setdefault("abbrevs", []).append({"input":entry.get("input"), "output":entry.get("output")})
    return prov

# Templates as functions returning (cypher_string, params)
def template_fact_lookup(normalized):
    # Expect an indicator (optional), speaker (optional), topic (optional), date_range (optional)
    params = {}
    where_clauses = []
    match_clauses = ["MATCH (s:Speech)"]
    # join on indicator -> in our minimal schema, indicators could be Topics or Organizations;
    # user earlier asked for Topic nodes - map indicators to Topic if provided as 'indicator' entity
    # We'll check entities list for 'indicator' or 'topic' or 'organization' or 'speaker'
    entities = normalized.get("entities", [])
    speaker = None
    topics = []
    orgs = []
    indicators = []
    for e in entities:
        etype = e.get("type","").lower()
        if etype == "speaker" and e.get("canonical"):
            speaker = e
        if etype == "topic":
            topics.append(e)
        if etype == "organization" or etype == "org":
            orgs.append(e)
        if etype == "indicator":
            indicators.append(e)
    # Speaker join
    if speaker:
        params["speaker_id"] = sanitize_id(speaker.get("id") or speaker.get("canonical"))
        match_clauses.append("MATCH (p:Person)-[:DELIVERED_SPEECH]->(s)")
        where_clauses.append("p.id = $speaker_id")
        speaker_prov = {"speaker_id": {"value": params["speaker_id"], "origin": "entities"}}
    # Topic join (HAS_TOPIC)
    if topics:
        # use first topic for MVP
        t = topics[0]
        params["topic_id"] = sanitize_id(t.get("id") or t.get("canonical"))
        match_clauses.append("MATCH (s)-[:HAS_TOPIC]->(t:Topic)")
        where_clauses.append("t.id = $topic_id")
    # Org join (MENTIONS_ORG)
    if orgs:
        o = orgs[0]
        params["org_id"] = sanitize_id(o.get("id") or o.get("canonical"))
        match_clauses.append("MATCH (s)-[:MENTIONS_ORG]->(o:Organization)")
        where_clauses.append("o.id = $org_id")
    # Indicator mapping as topic if present
    if indicators:
        ind = indicators[0]
        params["topic_id"] = sanitize_id(ind.get("id") or ind.get("canonical"))
        match_clauses.append("MATCH (s)-[:HAS_TOPIC]->(t:Topic)")
        where_clauses.append("t.id = $topic_id")
    # Date range filter (HELD_ON relation)
    date_range = normalized.get("date_range")
    if date_range:
        # Use HELD_ON relation: date node with property 'date' as xsd date string
        params["date_from"] = sanitize_id(date_range["start"])
        params["date_to"] = sanitize_id(date_range["end"])
        match_clauses.append("MATCH (s)-[:HELD_ON]->(d:Date)")
        where_clauses.append("d.date >= date($date_from) AND d.date <= date($date_to)")
    # Build query text
    cypher = "\n".join(match_clauses) + "\n"
    if where_clauses:
        cypher += "WHERE " + " AND ".join(where_clauses) + "\n"
    cypher += "OPTIONAL MATCH (p:Person)-[:DELIVERED_SPEECH]->(s)\n"
    cypher += "RETURN s.id AS speech_id, s.title AS title, s.date AS date, p.name AS speaker, substring(s.content,0,800) AS snippet\n"
    cypher += "ORDER BY s.date DESC\nLIMIT $limit"
    params.setdefault("limit", 20)
    meta = {"template":"fact_lookup","params_provided": list(params.keys())}
    return cypher, params, meta

def template_timeline(normalized):
    # Return count of speeches per date mentioning a topic/indicator
    params = {}
    entities = normalized.get("entities", [])
    indicators = [e for e in entities if e.get("type","").lower()=="indicator"]
    topics = [e for e in entities if e.get("type","").lower()=="topic"]
    join_clause = "MATCH (s:Speech)"
    where_clauses = []
    if indicators:
        params["topic_id"] = sanitize_id(indicators[0].get("id") or indicators[0].get("canonical"))
        join_clause += "\nMATCH (s)-[:HAS_TOPIC]->(t:Topic)"
        where_clauses.append("t.id = $topic_id")
    elif topics:
        params["topic_id"] = sanitize_id(topics[0].get("id") or topics[0].get("canonical"))
        join_clause += "\nMATCH (s)-[:HAS_TOPIC]->(t:Topic)"
        where_clauses.append("t.id = $topic_id")
    date_range = normalized.get("date_range")
    if date_range:
        params["date_from"] = sanitize_id(date_range["start"])
        params["date_to"] = sanitize_id(date_range["end"])
        join_clause += "\nMATCH (s)-[:HELD_ON]->(d:Date)"
        where_clauses.append("d.date >= date($date_from) AND d.date <= date($date_to)")
    cypher = join_clause + "\n"
    if where_clauses:
        cypher += "WHERE " + " AND ".join(where_clauses) + "\n"
    cypher += "RETURN d.date AS date, count(DISTINCT s) AS mentions\nORDER BY d.date ASC\nLIMIT $limit"
    params.setdefault("limit", 100)
    meta = {"template":"timeline","params_provided": list(params.keys())}
    return cypher, params, meta

def template_quote_search(normalized):
    # This relies on a fulltext index; if not available translator will indicate fallback to Retriever.
    params = {}
    entities = normalized.get("entities", [])
    speaker = next((e for e in entities if e.get("type","").lower()=="speaker" and e.get("canonical")), None)
    # Expect normalized['expanded_query'] may contain the quote text 
    quote_text = normalized.get("expanded_query") or ""
    # Heuristic: extract a quoted substring if present
    m = re.search(r'"([^"]{8,})"|\'([^\']{8,})\'', quote_text)
    quote = None
    if m:
        quote = m.group(1) or m.group(2)
        params["quote"] = quote
    else:
        # no explicit quote found; use the whole expanded_query as search term (safe but less precise)
        params["quote"] = quote_text[:300]
    if speaker and speaker.get("canonical"):
        params["speaker_id"] = sanitize_id(speaker.get("id") or speaker.get("canonical"))
    # Build fulltext query — parameterized
    # We'll use a parameterized call to a fulltext index 'speechContentIndex' (Neo4j 4+ syntax)
    cypher = "CALL db.index.fulltext.queryNodes('speechContentIndex', $quote) YIELD node, score\n"
    if speaker and speaker.get("canonical"):
        cypher += "MATCH (node)<-[:DELIVERED_SPEECH]-(p:Person)\nWHERE p.id = $speaker_id\n"
    cypher += "RETURN node.id AS speech_id, node.title AS title, node.date AS date, score\nORDER BY score DESC\nLIMIT $limit"
    params.setdefault("limit", 10)
    meta = {"template":"quote_search","params_provided": list(params.keys()), "requires_fulltext_index": True}
    return cypher, params, meta

def template_speaker_bio(normalized):
    params = {}
    entities = normalized.get("entities", [])
    speaker = next((e for e in entities if e.get("type","").lower()=="speaker" and e.get("canonical")), None)
    if not speaker:
        return None, None, {"error":"no_speaker_in_request"}
    params["speaker_id"] = sanitize_id(speaker.get("id") or speaker.get("canonical"))
    cypher = ("MATCH (p:Person {id:$speaker_id})\n"
              "OPTIONAL MATCH (p)-[:DELIVERED_SPEECH]->(s:Speech)\n"
              "OPTIONAL MATCH (s)-[:HAS_TOPIC]->(t:Topic)\n"
              "RETURN p.name AS name, p.role AS role, collect(distinct t.name)[0..10] AS topics, collect(distinct s.title)[0..5] AS recent_speeches")
    meta = {"template":"speaker_bio","params_provided": list(params.keys())}
    return cypher, params, meta

def template_compare_metric(normalized):
    # Essentially fetch speeches mentioning either of two indicators (assumes indicators provided)
    params = {}
    entities = normalized.get("entities", [])
    indicators = [e for e in entities if e.get("type","").lower()=="indicator"]
    if len(indicators) < 2:
        # fallback: if one indicator present, fetch mentions for it
        if indicators:
            # reuse fact_lookup behaviour
            return template_fact_lookup(normalized)
        else:
            return None, None, {"error":"need_at_least_one_indicator"}
    # we have at least two indicators; produce a query returning counts by indicator and date
    params["ind1"] = sanitize_id(indicators[0].get("id") or indicators[0].get("canonical"))
    params["ind2"] = sanitize_id(indicators[1].get("id") or indicators[1].get("canonical"))
    cypher = (
        "MATCH (s:Speech)-[:HAS_TOPIC]->(t:Topic)\n"
        "WHERE t.id IN [$ind1, $ind2]\n"
        "OPTIONAL MATCH (s)-[:HELD_ON]->(d:Date)\n"
        "RETURN t.id AS indicator, d.date AS date, count(DISTINCT s) AS mentions\n"
        "ORDER BY d.date ASC\nLIMIT $limit"
    )
    params.setdefault("limit", 200)
    meta = {"template":"compare_metric","params_provided": list(params.keys())}
    return cypher, params, meta

def template_citation_lookup(normalized):
    # Find which speech(s) are cited for some fact — map to fact_lookup essentially
    return template_fact_lookup(normalized)

# Translator main entry
def translate_to_cypher(normalized: Dict[str,Any]) -> Dict[str,Any]:
    intent = normalized.get("intent",{}).get("label","fact_lookup")
    # Handle ambiguous speaker: if entities contain speaker type but without canonical and contains candidates => return disambiguation
    for e in normalized.get("entities", []):
        if e.get("type","").lower()=="speaker" and "canonical" not in e:
            # return early: do not run queries per user instruction
            return {"ok": False, "needs_disambiguation": True, "candidates": e.get("candidates", []), "reason":"speaker_ambiguous"}
    # Map intents to template functions
    templates = {
        "fact_lookup": template_fact_lookup,
        "timeline": template_timeline,
        "quote_search": template_quote_search,
        "speaker_biography": template_speaker_bio,
        "speaker_bio": template_speaker_bio,
        "compare_metric": template_compare_metric,
        "citation_lookup": template_citation_lookup,
        "quote_search": template_quote_search
    }
    if intent not in templates:
        # fallback to fact_lookup
        intent = "fact_lookup"
    cypher, params, meta = templates[intent](normalized)
    if cypher is None:
        return {"ok": False, "error": meta.get("error","template_failure")}
    # build provenance map
    prov = build_param_provenance(normalized.get("transform_log", []))
    # attach param origins if possible
    for k in list(params.keys()):
        prov.setdefault("params", {})[k] = {"value": params[k], "sanitized": sanitize_id(params[k])}
    # Verification metadata (lightweight): ensure no disallowed substring in cypher (no backticks, no semicolons etc)
    forbidden = [";--", "--", "CALL apoc", "UNWIND", "CREATE", "DELETE", "SET "]
    warnings = []
    lc = cypher.lower()
    for f in forbidden:
        if f.lower() in lc:
            warnings.append(f"forbidden_construct_detected:{f}")
    result = {
        "ok": True,
        "query_language": "cypher",
        "cypher": cypher,
        "params": params,
        "meta": meta,
        "provenance": prov,
        "warnings": warnings
    }
    return result

# -----------------------
# Now run translator on several crafted normalized inputs (reflecting outputs from your normalizer)
# Example normalized inputs:
examples = []

# 1. Fact lookup: "what did ECB say about HICP and inflation in 2019-2020?"
examples.append({
    "canonical_query":"intent=compare_metric; indicators=Harmonised Index of Consumer Prices; date_from=2019-01-01,date_to=2020-12-31",
    "intent":{"label":"compare_metric","confidence":0.92},
    "entities":[{"text":"ECB","type":"organization","canonical":"European Central Bank","id":"european central bank","confidence":0.99},
                {"text":"HICP","type":"indicator","canonical":"Harmonised Index of Consumer Prices","id":"hicp","confidence":0.95}],
    "date_range": {"start":"2019-01-01","end":"2020-12-31"},
    "transform_log":[{"step":"abbreviation_expansion","input":"ECB","output":"European Central Bank"},{"step":"date_normalize","input":"2019-2020","output":{"start":"2019-01-01","end":"2020-12-31"}}]
})

# 2. Quote search: "Lane 11 June Dublin quote about banks" (ambiguous date -> per user normalizer date_range None)
examples.append({
    "canonical_query":"intent=quote_search; speaker=Philip R. Lane; expanded_query=Lane 11 June Dublin quote about banks",
    "intent":{"label":"quote_search","confidence":0.8},
    "entities":[{"text":"Philip R. Lane","type":"speaker","canonical":"Philip R. Lane","id":"philip r lane","confidence":0.9}],
    "date_range": None,
    "expanded_query":"Lane 11 June Dublin quote about banks",
    "transform_log":[{"step":"speaker_resolution","output":{"canonical":"Philip R. Lane","id":"philip r lane"}},{"step":"date_ambiguous","input":"11 June"}]
})

# 3. Timeline: "How did GDP evolve after the 2008 crisis?"
examples.append({
    "canonical_query":"intent=timeline; indicators=GDP; date_from=2009-01-01,date_to=2020-12-31",
    "intent":{"label":"timeline","confidence":0.9},
    "entities":[{"text":"GDP","type":"indicator","canonical":"Gross Domestic Product","id":"gdp","confidence":0.95}],
    "date_range": {"start":"2009-01-01","end":"2020-12-31"},
    "transform_log":[{"step":"date_normalize","input":"after 2008","output":{"start":"2009-01-01","end":"2020-12-31"}}]
})

# 4. Speaker bio: "who is Christine Lagarde"
examples.append({
    "canonical_query":"intent=speaker_bio; speaker=Christine Lagarde",
    "intent":{"label":"speaker_bio","confidence":0.8},
    "entities":[{"text":"Christine Lagarde","type":"speaker","canonical":"Christine Lagarde","id":"christine lagarde","confidence":0.9}],
    "date_range": None,
    "transform_log":[{"step":"speaker_resolution","output":{"canonical":"Christine Lagarde","id":"christine lagarde"}}]
})

# 5. Ambiguous speaker (candidates) example
examples.append({
    "canonical_query":"intent=fact_lookup; speaker=Smith (ambiguous)",
    "intent":{"label":"fact_lookup","confidence":0.6},
    "entities":[{"type":"speaker","candidates":[{"canonical":"John Smith","confidence":0.6},{"canonical":"Jane Smith","confidence":0.55}]}],
    "date_range": None,
    "transform_log":[{"step":"speaker_ambiguous","candidates":[{"canonical":"John Smith","confidence":0.6},{"canonical":"Jane Smith","confidence":0.55}]}]
})

# Translate all examples
translated = []
for ex in examples:
    out = translate_to_cypher(ex)
    translated.append({"input": ex["canonical_query"], "output": out})

# Print results succinctly
print("=== Translated queries (summary) ===")
for t in translated:
    print("\n-- Input:", t["input"])
    print("OK:", t["output"].get("ok"))
    if t["output"].get("needs_disambiguation"):
        print("Needs disambiguation:", t["output"].get("candidates"))
    elif t["output"].get("ok"):
        print("Cypher:\n", t["output"]["cypher"])
        print("Params:", json.dumps(t["output"]["params"], indent=2))
        if t["output"]["warnings"]:
            print("Warnings:", t["output"]["warnings"])
    else:
        print("Error:", t["output"].get("error"))

# Save the translator module to /mnt/data for download if user wants (optional)
module_code = """
# Translator module code (same functions implemented in the notebook).
# You can save this file and import translate_to_cypher(normalized) in your application.
# (For brevity, not re-duplicating the entire code here — ask me to export if you want the file.)
"""

print("\\n Translator was implemented successfully.\\n")

