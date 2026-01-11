# api.py
from fastapi import FastAPI, Query
from neo4j import GraphDatabase
from pipeline import SearchPipeline

app = FastAPI()

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

pipeline = SearchPipeline(driver)

@app.get("/search")
def search(
    q: str = Query(..., min_length=2),
    speaker: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int | None = 5
):
    return pipeline.run(
        query=q,
        speaker=speaker,
        from_date=from_date,
        to_date=to_date,
        limit=limit
    )

