
from __future__ import annotations

import math
from typing import Any

from commodity_finance_assistance.rag.store import RAGStore


class RAGRetriever:
    """Rank stored document chunks by cosine similarity to a query vector."""

    def __init__(
        self,
        store: RAGStore | None = None,
        model: str | None = None,
    ) -> None:
        self.store = store or RAGStore()
        self.model = model

    @staticmethod
    def _validate_vector(
        vector: list[float],
        name: str = "vector",
    ) -> list[float]:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError(f"{name} must be a non-empty vector")

        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains non-numeric values") from exc

        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} contains non-finite values")

        if math.sqrt(sum(value * value for value in values)) == 0:
            raise ValueError(f"{name} must not be a zero vector")

        return values

    @staticmethod
    def cosine_similarity(
        query_vector: list[float],
        document_vector: list[float],
    ) -> float:
        """Calculate cosine similarity between two finite, non-zero vectors."""
        query = RAGRetriever._validate_vector(query_vector, "query_vector")
        document = RAGRetriever._validate_vector(
            document_vector,
            "document_vector",
        )

        if len(query) != len(document):
            raise ValueError("Vectors must have the same dimensions")

        dot_product = sum(
            q_value * d_value
            for q_value, d_value in zip(query, document)
        )
        query_norm = math.sqrt(sum(value * value for value in query))
        document_norm = math.sqrt(sum(value * value for value in document))

        return dot_product / (query_norm * document_norm)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the most similar stored chunks.

        Each result contains chunk/document metadata, the embedding model,
        and a cosine similarity score. Scores are not confidence probabilities.
        """
        query = self._validate_vector(query_vector, "query_vector")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if min_score is not None:
            if not math.isfinite(min_score) or not -1.0 <= min_score <= 1.0:
                raise ValueError("min_score must be between -1 and 1")

        stored_embeddings = self.store.get_embeddings(model=self.model)
        results = []

        for item in stored_embeddings:
            vector = self._validate_vector(
                item["embedding"],
                f"stored embedding for {item['chunk_id']}",
            )

            expected_dimensions = item["dimensions"]
            if len(vector) != expected_dimensions:
                raise ValueError(
                    f"Stored embedding dimension metadata mismatch for "
                    f"{item['chunk_id']}"
                )

            if len(vector) != len(query):
                raise ValueError(
                    f"Query dimension ({len(query)}) does not match stored "
                    f"embedding dimension ({len(vector)}) for "
                    f"{item['chunk_id']}"
                )

            score = self.cosine_similarity(query, vector)

            if min_score is not None and score < min_score:
                continue

            result = dict(item)
            result["score"] = score
            results.append(result)

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
