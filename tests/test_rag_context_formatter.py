
import unittest

from commodity_finance_assistance.rag.context_formatter import (
    RAGContextFormatter,
)


class TestRAGContextFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = RAGContextFormatter()

    def test_formats_source_title_and_score(self):
        context = self.formatter.format_context(
            [{
                "title": "Gold Futures Guide",
                "source": "gold.txt",
                "text": "Gold futures have expiry dates.",
                "score": 0.92,
            }]
        )
        self.assertIn("Gold Futures Guide", context)
        self.assertIn("gold.txt", context)
        self.assertIn("0.9200", context)
        self.assertIn("Gold futures have expiry dates.", context)
        self.assertIn("untrusted data", context)

    def test_empty_results_return_clear_message(self):
        self.assertIn("No relevant", self.formatter.format_context([]))

    def test_chunk_count_is_limited(self):
        formatter = RAGContextFormatter(max_chunks=1)
        results = [
            {"title": "First", "source": "a.txt", "text": "Alpha"},
            {"title": "Second", "source": "b.txt", "text": "Beta"},
        ]
        context = formatter.format_context(results)
        self.assertIn("Alpha", context)
        self.assertNotIn("Beta", context)

    def test_chunk_text_is_limited(self):
        formatter = RAGContextFormatter(max_chars_per_chunk=5)
        context = formatter.format_context(
            [{"text": "ABCDEFGHIJKLMN"}]
        )
        self.assertIn("ABCDE", context)
        self.assertNotIn("ABCDEFGHIJKLMN", context)

    def test_total_context_length_is_bounded(self):
        formatter = RAGContextFormatter(
            max_chunks=5,
            max_chars_per_chunk=100,
            max_total_chars=180,
        )
        results = [
            {
                "title": f"Document {index}",
                "source": f"doc{index}.txt",
                "text": "X" * 100,
                "score": 0.8,
            }
            for index in range(5)
        ]
        context = formatter.format_context(results)
        self.assertLessEqual(len(context), 180)

    def test_blank_chunks_are_skipped(self):
        context = self.formatter.format_context([{"text": "   "}])
        self.assertEqual(
            context,
            "No usable reference text was retrieved.",
        )

    def test_constructor_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            RAGContextFormatter(max_chunks=0)
        with self.assertRaises(ValueError):
            RAGContextFormatter(max_chars_per_chunk=0)
        with self.assertRaises(ValueError):
            RAGContextFormatter(max_total_chars=0)


if __name__ == "__main__":
    unittest.main()
