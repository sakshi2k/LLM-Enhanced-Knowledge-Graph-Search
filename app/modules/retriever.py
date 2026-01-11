# retriever.py
from typing import List
from neo4j import Driver

class TopicFullTextRetriever:
    def __init__(self, driver: Driver, limit: int = 20):
        self.driver = driver
        self.limit = limit

    def retrieve_topic_ids(self, query: str) -> List[str]:
        cypher = """
        CALL db.index.fulltext.queryNodes(
          'topic_text_idx',
          $q
        )
        YIELD node, score
        RETURN node.id AS topic_id, score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            rows = session.run(
                cypher,
                q=query,
                limit=self.limit
            ).data()

        return [r["topic_id"] for r in rows]

