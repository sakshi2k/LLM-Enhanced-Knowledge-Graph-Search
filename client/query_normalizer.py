# Retry implementation without ace_tools to avoid environment issues.
import re, json, unicodedata
from difflib import get_close_matches
from dateutil import parser as dateparser
from sample_input_jsons_for_dev.speakers_list import speakers


def norm_name(n):
    return re.sub(r'[^a-z0-9 ]','', n.lower())

speaker_map = {}
variant_to_canonical = {}


for s in speakers:
    canonical = s.strip()
    speaker_map[canonical] = {'canonical': canonical, 'norm': norm_name(canonical)}
    parts = canonical.split()
    variants = set()
    variants.add(canonical)
    if len(parts) > 1:
        variants.add(parts[-1])
        variants.add(' '.join(parts[:-1]))
        initials = ' '.join([p[0]+'.' for p in parts[:-1]]) + ' ' + parts[-1] if len(parts) > 1 else canonical
        variants.add(initials)
    for v in list(variants):
        variant_to_canonical[norm_name(v)] = canonical
    variant_to_canonical[norm_name(canonical)] = canonical

# Abbrev and indicators
abbrev_dict = {
    "ECB": "European Central Bank",
    "EU": "European Union",
    "US": "United States",
    "GDP": "Gross Domestic Product",
    "HICP": "Harmonised Index of Consumer Prices",
    "SSM": "Single Supervisory Mechanism",
    "IMF": "International Monetary Fund",
    "EUR": "Euro",
    "CPI": "Consumer Price Index",
    "OECD": "Organisation for Economic Co-operation and Development",
    "BCE": "European Central Bank",
    "EZB": "European Central Bank",
    "ER": "Exchange Rate",
    "APP": "Asset Purchase Programme",
    "NBER": "National Bureau of Economic Research",
    "SEPA": "Single Euro Payments Area",
    "ERM": "Exchange Rate Mechanism",
    "ESCB": "European System of Central Banks",
    "ESRB": "European Systemic Risk Board",
}

indicator_glossary = {
    "gdp": "Gross Domestic Product",
    "hicp": "Harmonised Index of Consumer Prices",
    "cpi": "Consumer Price Index",
    "unemployment": "Unemployment Rate",
    "inflation": "Inflation",
    "interest rate": "Interest Rate",
    "repo rate": "Repo Rate",
    "policy rate": "Policy Rate",
}

def unicode_clean(s: str) -> str:
    s = '' if s is None else s
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\u00A0',' ')
    s = s.replace('\u2013','-').replace('\u2014','-')
    s = s.strip()
    return s

def tokenize(s: str):
    return re.findall(r"\b[\w'./%€$-]+\b", s)

def expand_abbrevs(tokens):
    out = []
    log = []
    for t in tokens:
        if t.upper() in abbrev_dict:
            out.append(abbrev_dict[t.upper()])
            log.append({"step":"abbreviation_expansion","input":t,"output":abbrev_dict[t.upper()],"confidence":0.95})
        else:
            out.append(t)
    return out, log

def detect_intent(query_lower):
    intents = [
        ("compare_metric", ["compare", " vs ", " vs.", "versus", "difference", "change in", "evolution of", "evolve"]),
        ("timeline", ["trend", "evolution", "how did", "how has", "evolve", "after", "before", "since", "recent", "recently"]),
        ("fact_lookup", ["what is", "what did", "find", "tell me", "show me", "list", "which speech"]),
        ("definition", ["define", "definition", "what is"]),
        ("quote_search", ["quote", "said", "statement", "speech where", "where he said", "where she said", "quote where"]),
        ("speaker_biography", ["who is", "biography", "about"]),
        ("citation_lookup", ["cite", "reference", "source", "which speech"]),
    ]
    scores = {}
    for label, keywords in intents:
        for kw in keywords:
            if kw in query_lower:
                scores[label] = scores.get(label,0) + 1
    if scores:
        label = max(scores, key=scores.get)
        confidence = min(0.6 + 0.1*scores[label], 0.99)
        detected_intent = {"label": label, "confidence": confidence}
    detected_intent = {"label":"fact_lookup","confidence":0.5}

    print('detected_intent set to : ', detected_intent)

    return detected_intent

def parse_date_expression(q):
    logs = []
    m = re.search(r'(\b[12][0-9]{3})\s*[-–to]+\s*(\b[12][0-9]{3})', q)
    if m:
        y1 = int(m.group(1)); y2 = int(m.group(2))
        start = f"{y1}-01-01"; end = f"{y2}-12-31"
        logs.append({"step":"date_normalize","input":m.group(0),"output":{"start":start,"end":end},"confidence":0.95})
        return ({"start":start,"end":end}, logs, False)
    m2 = re.search(r'\b(19|20)\d{2}\b', q)
    if m2:
        y = int(m2.group(0))
        start = f"{y}-01-01"; end = f"{y}-12-31"
        logs.append({"step":"date_normalize","input":m2.group(0),"output":{"start":start,"end":end},"confidence":0.9})
        return ({"start":start,"end":end}, logs, False)
    m3 = re.search(r'Q([1-4])\s*(?:of\s*)?([12][0-9]{3})', q, re.I)
    if m3:
        qn = int(m3.group(1)); y = int(m3.group(2))
        quarter_months = {1:("01-01","03-31"),2:("04-01","06-30"),3:("07-01","09-30"),4:("10-01","12-31")}
        start = f"{y}-{quarter_months[qn][0]}"; end = f"{y}-{quarter_months[qn][1]}"
        logs.append({"step":"date_normalize","input":m3.group(0),"output":{"start":start,"end":end},"confidence":0.92})
        return ({"start":start,"end":end}, logs, False)
    try:
        m4 = re.search(r'\b(\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?)\b', q)
        if m4:
            dstr = m4.group(1)
            dt = None
            try:
                dt = dateparser.parse(dstr, dayfirst=True)
            except Exception:
                dt = None
            if dt and dt.year and dt.year > 1900:
                start = dt.strftime("%Y-%m-%d"); end = start
                logs.append({"step":"date_normalize","input":dstr,"output":{"start":start,"end":end},"confidence":0.9})
                return ({"start":start,"end":end}, logs, False)
            else:
                logs.append({"step":"date_ambiguous","input":dstr,"note":"day-month found without year; treating as ambiguous per user preference","confidence":0.6})
                return (None, logs, True)
    except Exception:
        pass
    return (None, [], False)

def find_speaker_candidates(query):
    qnorm = norm_name(query)
    candidates = []
    for vnorm, canon in variant_to_canonical.items():
        if vnorm in qnorm or qnorm in vnorm:
            candidates.append((canon, 0.9))
    names = list(speaker_map.keys())
    matches = get_close_matches(query, names, n=5, cutoff=0.7)
    for m in matches:
        candidates.append((m, 0.85))
    seen = {}
    for c,score in candidates:
        if c not in seen or seen[c] < score:
            seen[c]=score
    sorted_cands = sorted([(c,seen[c]) for c in seen], key=lambda x:-x[1])
    return sorted_cands

def map_indicators(query_tokens):
    found = []
    for t in query_tokens:
        key = t.lower()
        if key in indicator_glossary:
            found.append({"text":t,"canonical":indicator_glossary[key],"id":key,"confidence":0.95})
    return found

def normalize(query: str):
    transform_log = []
    raw = query
    q = unicode_clean(query)
    transform_log.append({"step":"unicode_normalize","input":raw,"output":q})

    tokens = tokenize(q)
    expanded_tokens, logs = expand_abbrevs(tokens)
    for l in logs:
        transform_log.append(l)

    query_expanded = " ".join(expanded_tokens)

    # detect intent step
    intent = detect_intent(query_expanded.lower())
    transform_log.append({"step":"intent_detection","output":intent})

    # detect dates step
    date_range, date_logs, ambiguous_date = parse_date_expression(query_expanded)
    for l in date_logs:
        transform_log.append(l)
    if ambiguous_date:
        transform_log.append({"step":"date_resolution","note":"Ambiguous date found; per user preference ignoring date filter and returning matches by speaker/title instead.","confidence":0.3})
        date_range = None

    # detect candidate speakers
    speaker_candidates = find_speaker_candidates(q)
    speaker_entity = None
    if speaker_candidates:
        top, score = speaker_candidates[0]
        if score >= 0.85:
            speaker_entity = {"text": top, "canonical": top, "id": norm_name(top), "confidence": score}
            transform_log.append({"step":"speaker_resolution","input":query,"output":speaker_entity,"confidence":score})
        else:
            candlist = [{"canonical":c,"confidence":s} for c,s in speaker_candidates]
            transform_log.append({"step":"speaker_ambiguous","candidates":candlist})
            speaker_entity = {"candidates": candlist}
    indicators = map_indicators(expanded_tokens)
    if indicators:
        transform_log.append({"step":"indicator_extraction","output":indicators})
    cq_parts = []
    if intent and intent.get("label"):
        cq_parts.append(f"intent={intent['label']}")
    if speaker_entity and "canonical" in speaker_entity:
        cq_parts.append(f"speaker={speaker_entity['canonical']}")
    if indicators:
        cq_parts.append("indicators=" + ",".join([ind['canonical'] for ind in indicators]))
    if date_range:
        cq_parts.append(f"date_from={date_range['start']},date_to={date_range['end']}")
    canonical_query = "; ".join(cq_parts) if cq_parts else query_expanded
    conf_components = [intent.get("confidence",0.5)]
    if speaker_entity and "confidence" in speaker_entity:
        conf_components.append(speaker_entity["confidence"])
    if indicators:
        conf_components.append(sum([i['confidence'] for i in indicators])/len(indicators))
    overall_confidence = sum(conf_components)/len(conf_components) if conf_components else 0.5
    entities = []
    if speaker_entity:
        if "canonical" in speaker_entity:
            entities.append({"text": speaker_entity["canonical"],"type":"speaker","canonical":speaker_entity["canonical"],"id":speaker_entity["id"],"confidence":speaker_entity["confidence"]})
        else:
            entities.append({"type":"speaker","candidates":speaker_entity.get("candidates",[])})
    for ind in indicators:
        entities.append({"text": ind["text"],"type":"indicator","canonical":ind["canonical"],"id":ind["id"],"confidence":ind["confidence"]})
    result = {
        "raw_query": raw,
        "canonical_query": canonical_query,
        "expanded_query": query_expanded,
        "intent": intent,
        "entities": entities,
        "date_range": date_range,
        "transform_log": transform_log,
        "confidence": round(overall_confidence, 3)
    }
    return result


def test_harness():
    example_queries = [
        "what did ECB say about HICP and inflation in 2019-2020?",
        "Lane 11 June Dublin quote about banks",
        "How did GDP evolve after the 2008 crisis?",
        "what did lane say in his 'Monetary policy' speech?",
        "Find speeches by Draghi on 2012",
        "quote where 'we will act if needed' by ECB president",
        "compare CPI vs HICP 2015 to 2018",
        "who is Christine Lagarde",
        "Q1 2020 inflation comments by ECB",
        "Philip R. Lane speech on unemployment"
    ]

    results = [normalize(q) for q in example_queries]

    # Print concise summary
    summary_rows = []
    for r in results:
        summary_rows.append({
            "raw_query": r["raw_query"],
            "canonical_query": r["canonical_query"],
            "intent": r["intent"]["label"],
            "entities": ";".join([e.get("canonical", str(e.get("candidates",[]))) for e in r["entities"]]) if r["entities"] else "",
            "date_range": (r["date_range"]["start"] + " to " + r["date_range"]["end"]) if r["date_range"] else "None",
            "confidence": r["confidence"]
        })
    summary_df = pd.DataFrame(summary_rows)
    print("=== Normalizer test summary ===")
    print(summary_df.to_string(index=False))

    # Print full JSON for first 4 for inspection
    for i, r in enumerate(results[:4]):
        print(f"\n--- Full result {i+1} ---")
        print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_harness()