
from __future__ import annotations

from typing import Any

from commodity_finance_assistance.rag.embeddings import EmbeddingService
from commodity_finance_assistance.rag.retriever import RAGRetriever
from commodity_finance_assistance.rag.store import RAGStore


class RAGQueryService:
    """Embed a question and retrieve relevant stored knowledge chunks."""

    def __init__(
        self,
        embedding_service: Any | None = None,
        retriever: RAGRetriever | None = None,
        store: RAGStore | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()

        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = RAGRetriever(
                store=store,
                model=self.embedding_service.model,
            )

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked evidence for a natural-language question."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must not be empty")

        query_vector = self.embedding_service.embed_query(question.strip())

        return self.retriever.search(
            query_vector=query_vector,
            top_k=top_k,
            min_score=min_score,
        )
