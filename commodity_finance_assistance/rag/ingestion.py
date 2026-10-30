
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from commodity_finance_assistance.rag.chunker import chunk_document
from commodity_finance_assistance.rag.document_loader import (
    SUPPORTED_EXTENSIONS,
    load_document,
)
from commodity_finance_assistance.rag.store import RAGStore


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    source: str
    title: str
    chunk_count: int


class RAGIngestionService:
    """Coordinate document loading, chunking, and SQLite persistence."""

    def __init__(
        self,
        store: RAGStore | None = None,
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller than chunk_size"
            )

        self.store = store or RAGStore()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_file(self, path: str | Path) -> IngestionResult:
        """Load, chunk, and persist one supported document."""
        document = load_document(path)

        chunks = chunk_document(
            document,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        if not chunks:
            raise ValueError(
                f"No chunks were generated for: {Path(path).name}"
            )

        saved_count = self.store.save_document(document, chunks)

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
        """Ingest all supported documents under a directory recursively."""
        root = Path(directory)

        if not root.is_dir():
            raise NotADirectoryError(
                f"Knowledge directory not found: {root}"
            )

        results = []

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            results.append(self.ingest_file(path))

        return results
