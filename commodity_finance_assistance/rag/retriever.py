
from __future__ import annotations

import math
from typing import Any

from commodity_finance_assistance.rag.store import RAGStore


class RAGRetriever:
    """Rank stored document chunks by cosine similarity."""

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

        if any(isinstance(value, bool) for value in vector):
            raise ValueError(f"{name} must not contain Boolean values")

        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} contains invalid numeric values"
            ) from exc

        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} contains non-finite values")

        if not any(value != 0.0 for value in values):
            raise ValueError(f"{name} must not be a zero vector")

        return values

    @staticmethod
    def cosine_similarity(
        query_vector: list[float],
        document_vector: list[float],
    ) -> float:
        """Calculate numerically stable cosine similarity."""

        query = RAGRetriever._validate_vector(
            query_vector, "query_vector"
        )
        document = RAGRetriever._validate_vector(
            document_vector, "document_vector"
        )

        if len(query) != len(document):
            raise ValueError("Vectors must have the same dimensions")

        # Scale before multiplication to prevent overflow.
        query_scale = max(abs(value) for value in query)
        document_scale = max(abs(value) for value in document)

        normalized_query = [
            value / query_scale for value in query
        ]
        normalized_document = [
            value / document_scale for value in document
        ]

        dot_product = math.fsum(
            q * d
            for q, d in zip(normalized_query, normalized_document)
        )

        query_norm = math.sqrt(
            math.fsum(value * value for value in normalized_query)
        )
        document_norm = math.sqrt(
            math.fsum(value * value for value in normalized_document)
        )

        score = dot_product / (query_norm * document_norm)

        # Protect against tiny floating-point rounding deviations.
        return max(-1.0, min(1.0, score))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most similar stored chunks."""

        query = self._validate_vector(
            query_vector, "query_vector"
        )

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")

        if min_score is not None:
            if isinstance(min_score, bool) or not isinstance(
                min_score, (int, float)
            ):
                raise ValueError("min_score must be numeric")

            if (
                not math.isfinite(min_score)
                or not -1.0 <= min_score <= 1.0
            ):
                raise ValueError("min_score must be between -1 and 1")

        stored_embeddings = self.store.get_embeddings(
            model=self.model
        )

        results = []

        for item in stored_embeddings:
            if not isinstance(item, dict):
                raise ValueError(
                    "Stored embedding record must be a dictionary"
                )

            chunk_id = item.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Stored embedding has an invalid chunk ID")

            vector = self._validate_vector(
                item.get("embedding"),
                f"stored embedding for {chunk_id}",
            )

            expected_dimensions = item.get("dimensions")

            if (
                not isinstance(expected_dimensions, int)
                or isinstance(expected_dimensions, bool)
                or expected_dimensions <= 0
            ):
                raise ValueError(
                    f"Invalid dimension metadata for {chunk_id}"
                )

            if len(vector) != expected_dimensions:
                raise ValueError(
                    f"Stored embedding dimension metadata mismatch "
                    f"for {chunk_id}"
                )

            if len(vector) != len(query):
                raise ValueError(
                    f"Query dimension ({len(query)}) does not match "
                    f"stored embedding dimension ({len(vector)}) "
                    f"for {chunk_id}"
                )

            score = self.cosine_similarity(query, vector)

            if min_score is not None and score < min_score:
                continue

            result = dict(item)
            result["score"] = score
            results.append(result)

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]
