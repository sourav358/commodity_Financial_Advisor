from __future__ import annotations

from pathlib import Path
from typing import Any

from commodity_finance_assistance.rag.embeddings import EmbeddingService
from commodity_finance_assistance.rag.document_loader import (
    SUPPORTED_EXTENSIONS,
)
from commodity_finance_assistance.rag.ingestion import (
    IngestionResult,
    RAGIngestionService,
)
from commodity_finance_assistance.rag.store import RAGStore


class RAGPipeline:
    """Orchestrate document ingestion, embedding, and atomic persistence."""

    def __init__(
        self,
        ingestion_service: RAGIngestionService,
        store: RAGStore,
        embedding_service: Any | None = None,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.store = store
        self._embedding_service = embedding_service

    @property
    def embedding_service(self) -> Any:
        """Initialize the embedding client only when ingestion is requested."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()

        return self._embedding_service

    def ingest_file(self, path: str | Path) -> IngestionResult:
        """Index one document as an atomic operation."""

        document, chunks = self.ingestion_service.prepare_file(path)

        texts = [chunk.text for chunk in chunks]

        vectors = self.embedding_service.embed_texts(texts)

        if len(vectors) != len(chunks):
            raise ValueError(
                "Embedding count does not match the number of chunks"
            )

        embeddings = {
            chunk.chunk_id: vector
            for chunk, vector in zip(chunks, vectors, strict=True)
        }

        saved_count = self.store.save_document_with_embeddings(
            document=document,
            chunks=chunks,
            embeddings=embeddings,
            model=self.embedding_service.model,
            dimensions=self.embedding_service.dimensions,
        )

        return IngestionResult(
            document_id=document.document_id,
            source=document.source,
            title=document.title,
            chunk_count=saved_count,
        )

    def ingest_directory(
        self,
        directory: str | Path,
    ) -> list[IngestionResult]:
        """Ingest supported files recursively, atomically per document."""

        root = Path(directory)

        if not root.is_dir():
            raise NotADirectoryError(
                f"Knowledge directory not found: {root}"
            )

        results = []

        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        for path in paths:
            results.append(self.ingest_file(path))

        return results