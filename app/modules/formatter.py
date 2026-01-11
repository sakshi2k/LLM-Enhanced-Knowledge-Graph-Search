# formatter.py
from urllib.parse import urlencode

BASE_URL = "https://www.ecb.europa.eu/press/pubbydate/html/index.en.html"

def build_ecb_url(topic: str, speaker: str, date: str) -> str:
    year = date[:4] if date else ""

    params = {
        "search_term": topic,
        "year": year,
        "boardmember": speaker
    }

    return f"{BASE_URL}?{urlencode(params)}"


def format_results(query, rows):
    if not rows:
        return []

    results = []

    for r in rows:
        # topic = r["topics"][0] if isinstance(r.get("topics"), list) and r["topics"] else r.get("topic")

        url = build_ecb_url(
            topic=r["title"],
            speaker=r["speaker"],
            date=str(r["date"])
        )

        results.append({
            "speech_id": r["speech_id"],
            "title": r["title"],
            "speaker": r["speaker"],
            "date": str(r["date"]),
            "topics": r.get("topics"),
            "confidence": r["confidence"],
            "provenance": r["provenance"],
            "url": url
        })

    return results
