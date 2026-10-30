
import unittest
from unittest.mock import Mock

from commodity_finance_assistance.rag.retriever import RAGRetriever


class TestRAGRetriever(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.store.get_embeddings.return_value = [
            {
                "chunk_id": "chunk-a",
                "document_id": "doc-a",
                "text": "Gold futures contracts have expiry dates.",
                "title": "Gold Guide",
                "source": "gold.txt",
                "model": "test-model",
                "dimensions": 3,
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "chunk_id": "chunk-b",
                "document_id": "doc-b",
                "text": "Silver is an industrial precious metal.",
                "title": "Silver Guide",
                "source": "silver.txt",
                "model": "test-model",
                "dimensions": 3,
                "embedding": [0.0, 1.0, 0.0],
            },
            {
                "chunk_id": "chunk-c",
                "document_id": "doc-c",
                "text": "Crude oil is an energy commodity.",
                "title": "Oil Guide",
                "source": "oil.txt",
                "model": "test-model",
                "dimensions": 3,
                "embedding": [0.8, 0.6, 0.0],
            },
        ]
        self.retriever = RAGRetriever(
            store=self.store,
            model="test-model",
        )

    def test_cosine_similarity_identical_vectors(self):
        score = self.retriever.cosine_similarity(
            [1.0, 0.0],
            [1.0, 0.0],
        )
        self.assertAlmostEqual(score, 1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        score = self.retriever.cosine_similarity(
            [1.0, 0.0],
            [0.0, 1.0],
        )
        self.assertAlmostEqual(score, 0.0)

    def test_search_ranks_most_similar_chunks_first(self):
        results = self.retriever.search([1.0, 0.0, 0.0])

        self.assertEqual(
            [result["chunk_id"] for result in results],
            ["chunk-a", "chunk-c", "chunk-b"],
        )
        self.assertAlmostEqual(results[0]["score"], 1.0)
        self.assertAlmostEqual(results[1]["score"], 0.8)
        self.assertEqual(results[0]["source"], "gold.txt")
        self.assertEqual(results[0]["title"], "Gold Guide")
        self.assertIn("expiry dates", results[0]["text"])

    def test_search_respects_top_k(self):
        results = self.retriever.search(
            [1.0, 0.0, 0.0],
            top_k=2,
        )
        self.assertEqual(len(results), 2)

    def test_search_applies_minimum_score(self):
        results = self.retriever.search(
            [1.0, 0.0, 0.0],
            min_score=0.5,
        )
        self.assertEqual(
            [result["chunk_id"] for result in results],
            ["chunk-a", "chunk-c"],
        )

    def test_search_passes_configured_model_to_store(self):
        self.retriever.search([1.0, 0.0, 0.0])

        self.store.get_embeddings.assert_called_once_with(
            model="test-model"
        )

    def test_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match stored"):
            self.retriever.search([1.0, 0.0])

    def test_rejects_zero_query_vector(self):
        with self.assertRaisesRegex(ValueError, "zero vector"):
            self.retriever.search([0.0, 0.0, 0.0])

    def test_rejects_non_finite_query_vector(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.retriever.search([1.0, float("nan"), 0.0])

    def test_rejects_invalid_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.retriever.search([1.0, 0.0, 0.0], top_k=0)

    def test_empty_store_returns_no_results(self):
        self.store.get_embeddings.return_value = []

        results = self.retriever.search([1.0, 0.0, 0.0])

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
