
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from commodity_finance_assistance.rag.embeddings import EmbeddingService


def make_response(vectors):
    """Build a minimal mock response using (index, vector) pairs."""
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in vectors
        ]
    )


class TestEmbeddingService(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.service = EmbeddingService(
            client=self.client,
            batch_size=2,
        )

    def vector(self, value):
        return [float(value)] * self.service.dimensions

    def test_embeds_texts_in_batches_and_preserves_order(self):
        first = self.vector(1)
        second = self.vector(2)
        third = self.vector(3)

        # Deliberately return the first batch in reverse order.
        self.client.embeddings.create.side_effect = [
            make_response([(1, second), (0, first)]),
            make_response([(0, third)]),
        ]

        result = self.service.embed_texts(["gold", "silver", "crude oil"])

        self.assertEqual(result, [first, second, third])
        self.assertEqual(self.client.embeddings.create.call_count, 2)

        first_call = self.client.embeddings.create.call_args_list[0]
        self.assertEqual(first_call.kwargs["input"], ["gold", "silver"])
        self.assertEqual(first_call.kwargs["model"], self.service.model)
        self.assertEqual(
            first_call.kwargs["dimensions"],
            self.service.dimensions,
        )

        second_call = self.client.embeddings.create.call_args_list[1]
        self.assertEqual(second_call.kwargs["input"], ["crude oil"])

    def test_empty_input_returns_empty_list_without_api_call(self):
        self.assertEqual(self.service.embed_texts([]), [])
        self.client.embeddings.create.assert_not_called()

    def test_rejects_a_single_string_argument(self):
        with self.assertRaises(TypeError):
            self.service.embed_texts("gold")

        self.client.embeddings.create.assert_not_called()

    def test_rejects_blank_text(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.service.embed_texts(["gold", "   "])

        self.client.embeddings.create.assert_not_called()

    def test_rejects_wrong_vector_dimensions(self):
        self.client.embeddings.create.return_value = make_response(
            [(0, [0.1, 0.2])]
        )

        with self.assertRaisesRegex(ValueError, "Expected"):
            self.service.embed_texts(["gold"])

    def test_rejects_duplicate_response_indexes(self):
        self.client.embeddings.create.return_value = make_response(
            [(0, self.vector(1)), (0, self.vector(2))]
        )

        with self.assertRaisesRegex(ValueError, "duplicate indexes"):
            self.service.embed_texts(["gold", "silver"])

    def test_rejects_non_finite_values(self):
        vector = self.vector(1)
        vector[0] = float("nan")
        self.client.embeddings.create.return_value = make_response(
            [(0, vector)]
        )

        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.service.embed_texts(["gold"])

    def test_embed_query_returns_one_vector(self):
        expected = self.vector(5)
        self.client.embeddings.create.return_value = make_response(
            [(0, expected)]
        )

        result = self.service.embed_query("What is a gold futures contract?")

        self.assertEqual(result, expected)
        self.client.embeddings.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
