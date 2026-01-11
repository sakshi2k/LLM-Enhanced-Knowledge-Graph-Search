# kg_executor.py
from queries.cypher_queries import CYPHER_QUERY_FETCH_GROUNDED_SPEECHES_old, CYPHER_QUERY_FETCH_GROUNDED_SPEECHES


class KGExecutor:
    def __init__(self, driver):
        self.driver = driver

    def fetch_grounded_speeches(
        self,
        topic_ids,
        speaker=None,
        from_date=None,
        to_date=None,
        limit=5
    ):
        
        cypher = CYPHER_QUERY_FETCH_GROUNDED_SPEECHES

        with self.driver.session() as session:
            return session.run(
                cypher,
                topic_ids=topic_ids,
                speaker=speaker,
                from_date=from_date,
                to_date=to_date,
                limit=limit
            ).data()
