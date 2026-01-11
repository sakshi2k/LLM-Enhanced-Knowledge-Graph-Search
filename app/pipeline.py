# pipeline.py

from retriever import TopicFullTextRetriever
from kg_executor import KGExecutor
from formatter import format_results
from verifier import verify_results

class SearchPipeline:
    def __init__(self, driver):
        self.retriever = TopicFullTextRetriever(driver)
        self.executor = KGExecutor(driver)

    def run(self, query, speaker=None, from_date=None, to_date=None, limit=None):
        topic_ids = self.retriever.retrieve_topic_ids(query)

        if not topic_ids:
            return {
                "query": query,
                "results": [],
                "verified": False
            }

        rows = self.executor.fetch_grounded_speeches(
            topic_ids,
            speaker=speaker,
            from_date=from_date,
            to_date=to_date,
            limit=limit
        )

        formatted = format_results(query, rows)

        return {
            "query": query,
            "filters": {
                "speaker": speaker,
                "from_date": from_date,
                "to_date": to_date
            },
            "results": formatted,
            "verified": verify_results(formatted)
        }

