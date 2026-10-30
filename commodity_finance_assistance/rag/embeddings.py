
from __future__ import annotations

import math
from typing import Any, Sequence

from openai import OpenAI

from commodity_finance_assistance.config_loader import get_model_config


class EmbeddingService:
    """Generate validated embedding vectors using the configured provider."""

    def __init__(
        self,
        client: Any | None = None,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        config = get_model_config("embeddings")
        self.provider = config["provider"]
        self.model = config["model"]
        self.dimensions = int(config["dimensions"])
        self.batch_size = batch_size

        if self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero")

        if self.provider != "openai":
            raise ValueError(
                f"Unsupported embedding provider: {self.provider}"
            )

        # Create the real client only when one wasn't injected.
        # Tests can inject a mock client to avoid API calls.
        self.client = client if client is not None else OpenAI()

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Embed texts in batches and preserve their original order."""
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a string")

        if not texts:
            return []

        normalized_texts = list(texts)

        for index, text in enumerate(normalized_texts):
            if not isinstance(text, str):
                raise TypeError(f"Text at index {index} must be a string")
            if not text.strip():
                raise ValueError(f"Text at index {index} must not be empty")

        all_embeddings: list[list[float]] = []

        for start in range(0, len(normalized_texts), self.batch_size):
            batch = normalized_texts[start : start + self.batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
            )

            data = response.data
            if len(data) != len(batch):
                raise ValueError(
                    "Embedding response count does not match input count"
                )

            # API results have an index. Use it rather than assuming the
            # response is already in input order.
            ordered_batch: list[list[float] | None] = [None] * len(batch)

            for item in data:
                index = item.index
                vector = item.embedding

                if not isinstance(index, int) or not 0 <= index < len(batch):
                    raise ValueError("Embedding response contains an invalid index")

                if ordered_batch[index] is not None:
                    raise ValueError("Embedding response contains duplicate indexes")

                if len(vector) != self.dimensions:
                    raise ValueError(
                        f"Expected {self.dimensions} dimensions, "
                        f"received {len(vector)}"
                    )

                validated_vector = [float(value) for value in vector]

                if not all(math.isfinite(value) for value in validated_vector):
                    raise ValueError("Embedding vector contains non-finite values")

                ordered_batch[index] = validated_vector

            if any(vector is None for vector in ordered_batch):
                raise ValueError("Embedding response is missing an input index")

            all_embeddings.extend(
                vector for vector in ordered_batch if vector is not None
            )

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed one query using the same model as document chunks."""
        return self.embed_texts([text])[0]
