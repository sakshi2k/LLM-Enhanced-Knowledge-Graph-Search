def normalize(query: str) -> dict:
    tokens = [t.lower() for t in query.split() if len(t) > 2]

    return {
        "topics": tokens,
        "limit": 5
    }
