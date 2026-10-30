import unittest
from unittest.mock import Mock

from commodity_finance_assistance.rag.evaluation import (
    RAGEvaluator,
    RetrievalTestCase,
    evaluate_single_case,
)


class TestRAGEvaluation(unittest.TestCase):

    def setUp(self):
        self.case = RetrievalTestCase(
            question="What affects commodity prices?",
            relevant_sources=frozenset({
                "commodity_market_basics.md",
            }),
        )

    def test_perfect_retrieval(self):
        results = [
            {"source": "commodity_market_basics.md"},
        ]

        result = evaluate_single_case(results, self.case, k=1)

        self.assertEqual(result.precision_at_k, 1.0)
        self.assertEqual(result.precision_at_retrieved, 1.0)
        self.assertEqual(result.recall_at_k, 1.0)
        self.assertEqual(result.reciprocal_rank, 1.0)
        self.assertEqual(result.hit_rate_at_k, 1.0)

    def test_first_relevant_result_rank(self):
        results = [
            {"source": "irrelevant.md"},
            {"source": "commodity_market_basics.md"},
        ]

        result = evaluate_single_case(results, self.case, k=2)

        self.assertEqual(result.precision_at_k, 0.5)
        self.assertEqual(result.precision_at_retrieved, 0.5)
        self.assertEqual(result.recall_at_k, 1.0)
        self.assertEqual(result.reciprocal_rank, 0.5)
        self.assertEqual(result.hit_rate_at_k, 1.0)

    def test_no_relevant_results(self):
        results = [
            {"source": "irrelevant.md"},
            {"source": "another_document.md"},
        ]

        result = evaluate_single_case(results, self.case, k=2)

        self.assertEqual(result.precision_at_k, 0.0)
        self.assertEqual(result.precision_at_retrieved, 0.0)
        self.assertEqual(result.recall_at_k, 0.0)
        self.assertEqual(result.reciprocal_rank, 0.0)
        self.assertEqual(result.hit_rate_at_k, 0.0)

    def test_precision_at_retrieved_with_partial_results(self):
        results = [
            {"source": "commodity_market_basics.md"},
            {"source": "irrelevant.md"},
        ]

        # Request 5 results, but only 2 are returned.
        result = evaluate_single_case(results, self.case, k=5)

        self.assertEqual(result.precision_at_k, 0.2)
        self.assertEqual(result.precision_at_retrieved, 0.5)

    def test_precision_at_retrieved_with_no_results(self):
        result = evaluate_single_case(
            results=[],
            test_case=self.case,
            k=5,
        )

        self.assertEqual(result.precision_at_k, 0.0)
        self.assertEqual(result.precision_at_retrieved, 0.0)
        self.assertEqual(result.recall_at_k, 0.0)
        self.assertEqual(result.hit_rate_at_k, 0.0)

    def test_source_path_normalization(self):
        results = [
            {
                "source": (
                    "data/knowledge/commodity_market_basics.md"
                )
            },
        ]

        result = evaluate_single_case(results, self.case, k=1)

        self.assertEqual(result.hit_rate_at_k, 1.0)
        self.assertEqual(result.precision_at_retrieved, 1.0)

    def test_duplicate_sources_do_not_inflate_recall(self):
        results = [
            {"source": "commodity_market_basics.md"},
            {"source": "commodity_market_basics.md"},
        ]

        result = evaluate_single_case(results, self.case, k=2)

        self.assertEqual(result.precision_at_k, 1.0)
        self.assertEqual(result.precision_at_retrieved, 1.0)
        self.assertEqual(result.recall_at_k, 1.0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            evaluate_single_case([], self.case, k=0)

    def test_evaluator_aggregates_results(self):
        query_service = Mock()

        query_service.query.side_effect = [
            [{"source": "commodity_market_basics.md"}],
            [{"source": "irrelevant.md"}],
        ]

        evaluator = RAGEvaluator(
            query_service=query_service,
            k=1,
        )

        cases = [
            self.case,
            RetrievalTestCase(
                question="What is a spot market?",
                relevant_sources=frozenset({
                    "commodity_market_basics.md",
                }),
            ),
        ]

        report = evaluator.evaluate(cases)

        self.assertEqual(report.total_questions, 2)
        self.assertEqual(report.precision_at_k, 0.5)
        self.assertEqual(report.precision_at_retrieved, 0.5)
        self.assertEqual(report.recall_at_k, 0.5)
        self.assertEqual(report.mrr, 0.5)
        self.assertEqual(report.hit_rate_at_k, 0.5)

    def test_empty_dataset_is_rejected(self):
        evaluator = RAGEvaluator(query_service=Mock())

        with self.assertRaises(ValueError):
            evaluator.evaluate([])

    def test_invalid_evaluator_k_is_rejected(self):
        with self.assertRaises(ValueError):
            RAGEvaluator(
                query_service=Mock(),
                k=0,
            )

    def test_invalid_min_score_is_rejected(self):
        with self.assertRaises(ValueError):
            RAGEvaluator(
                query_service=Mock(),
                min_score=1.5,
            )


if __name__ == "__main__":
    unittest.main()