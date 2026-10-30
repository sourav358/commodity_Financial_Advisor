
import unittest
from unittest.mock import Mock

from commodity_finance_assistance.rag.query import RAGQueryService


class TestRAGQueryService(unittest.TestCase):
    def setUp(self):
        self.embedding_service = Mock()
        self.embedding_service.model = "test-model"
        self.embedding_service.embed_query.return_value = [1.0, 0.0, 0.0]

        self.retriever = Mock()
        self.retriever.search.return_value = [
            {
                "chunk_id": "doc-1:0",
                "title": "Gold Guide",
                "source": "gold.txt",
                "text": "Gold futures contracts have expiry dates.",
                "score": 0.95,
            }
        ]

        self.service = RAGQueryService(
            embedding_service=self.embedding_service,
            retriever=self.retriever,
        )

    def test_query_embeds_question_and_returns_results(self):
        results = self.service.query("What is a gold futures contract?")

        self.embedding_service.embed_query.assert_called_once_with(
            "What is a gold futures contract?"
        )
        self.retriever.search.assert_called_once_with(
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
            min_score=None,
        )
        self.assertEqual(results[0]["chunk_id"], "doc-1:0")

    def test_query_strips_question_whitespace(self):
        self.service.query("  gold futures  ")

        self.embedding_service.embed_query.assert_called_once_with(
            "gold futures"
        )

    def test_query_passes_search_options(self):
        self.service.query(
            "gold futures",
            top_k=3,
            min_score=0.6,
        )

        self.retriever.search.assert_called_once_with(
            query_vector=[1.0, 0.0, 0.0],
            top_k=3,
            min_score=0.6,
        )

    def test_query_rejects_empty_question(self):
        for question in ("", "   "):
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    self.service.query(question)

        self.embedding_service.embed_query.assert_not_called()
        self.retriever.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
