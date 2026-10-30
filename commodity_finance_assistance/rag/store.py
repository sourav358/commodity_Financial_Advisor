
from __future__ import annotations

import json
import math

from commodity_finance_assistance.memory.database import get_connection
from commodity_finance_assistance.rag.chunker import DocumentChunk
from commodity_finance_assistance.rag.document_loader import Document


class RAGStore:
    """Persist RAG documents, chunks, and embeddings in SQLite."""

    def initialize_schema(self) -> None:
        """Create RAG tables and indexes if they do not exist."""
        with get_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    document_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    FOREIGN KEY (document_id)
                        REFERENCES rag_documents(document_id)
                        ON DELETE CASCADE,
                    UNIQUE (document_id, chunk_index),
                    CHECK (chunk_index >= 0),
                    CHECK (start_char >= 0),
                    CHECK (end_char >= start_char)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_index
                ON rag_chunks(document_id, chunk_index)
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id)
                        REFERENCES rag_chunks(chunk_id)
                        ON DELETE CASCADE,
                    CHECK (dimensions > 0)
                )
                """
            )

    def save_document(
        self,
        document: Document,
        chunks: list[DocumentChunk],
    ) -> int:
        """
        Save document metadata and replace its existing chunks.

        Replacing chunks also removes embeddings belonging to the old chunks
        through the foreign-key cascade.
        """
        if not chunks:
            raise ValueError("At least one chunk is required")

        seen_chunk_ids: set[str] = set()
        seen_chunk_indexes: set[int] = set()

        for chunk in chunks:
            if chunk.document_id != document.document_id:
                raise ValueError(
                    "Every chunk must belong to the supplied document"
                )
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError("Duplicate chunk IDs are not allowed")
            if chunk.chunk_index in seen_chunk_indexes:
                raise ValueError("Duplicate chunk indexes are not allowed")

            seen_chunk_ids.add(chunk.chunk_id)
            seen_chunk_indexes.add(chunk.chunk_index)

        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO rag_documents (
                    document_id, source, title, file_type
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source = excluded.source,
                    title = excluded.title,
                    file_type = excluded.file_type,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    document.document_id,
                    document.source,
                    document.title,
                    document.file_type,
                ),
            )

            # Replacing chunks makes repeated ingestion idempotent.
            # Existing embeddings are removed through ON DELETE CASCADE.
            connection.execute(
                "DELETE FROM rag_chunks WHERE document_id = ?",
                (document.document_id,),
            )

            connection.executemany(
                """
                INSERT INTO rag_chunks (
                    chunk_id,
                    document_id,
                    text,
                    chunk_index,
                    start_char,
                    end_char
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.text,
                        chunk.chunk_index,
                        chunk.start_char,
                        chunk.end_char,
                    )
                    for chunk in chunks
                ],
            )

        return len(chunks)

    def get_document(self, document_id: str) -> dict | None:
        """Return document metadata, or None if the document is absent."""
        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT document_id, source, title, file_type,
                       created_at, updated_at
                FROM rag_documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        return dict(row) if row is not None else None

    def get_chunks(self, document_id: str) -> list[dict]:
        """Return a document's chunks in their original chunk order."""
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    c.chunk_id,
                    c.document_id,
                    d.source,
                    d.title,
                    c.text,
                    c.chunk_index,
                    c.start_char,
                    c.end_char
                FROM rag_chunks AS c
                JOIN rag_documents AS d
                    ON d.document_id = c.document_id
                WHERE c.document_id = ?
                ORDER BY c.chunk_index ASC
                """,
                (document_id,),
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks and embeddings."""
        with get_connection() as connection:
            cursor = connection.execute(
                "DELETE FROM rag_documents WHERE document_id = ?",
                (document_id,),
            )

        return cursor.rowcount > 0

    def save_embeddings(
        self,
        embeddings: dict[str, list[float]],
        model: str,
        dimensions: int,
    ) -> int:
        """Save or update chunk vectors and their model metadata."""
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Embedding model must not be empty")

        if not isinstance(dimensions, int) or isinstance(dimensions, bool):
            raise ValueError("Embedding dimensions must be an integer")

        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero")

        if not embeddings:
            return 0

        rows = []

        for chunk_id, vector in embeddings.items():
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Chunk ID must not be empty")

            if not isinstance(vector, (list, tuple)):
                raise ValueError(f"Embedding for {chunk_id} must be a vector")

            if len(vector) != dimensions:
                raise ValueError(
                    f"Expected {dimensions} dimensions for {chunk_id}, "
                    f"received {len(vector)}"
                )

            try:
                normalized = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Embedding for {chunk_id} contains non-numeric values"
                ) from exc

            if not all(math.isfinite(value) for value in normalized):
                raise ValueError(
                    f"Embedding vector for {chunk_id} contains non-finite values"
                )

            rows.append(
                (
                    chunk_id,
                    model,
                    dimensions,
                    json.dumps(normalized, allow_nan=False),
                )
            )

        with get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO rag_embeddings (
                    chunk_id, model, dimensions, vector_json
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    model = excluded.model,
                    dimensions = excluded.dimensions,
                    vector_json = excluded.vector_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                rows,
            )

        return len(rows)

    def get_embedding(self, chunk_id: str) -> dict | None:
        """Return one stored vector and its metadata."""
        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT chunk_id, model, dimensions, vector_json, created_at
                FROM rag_embeddings
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["embedding"] = json.loads(result.pop("vector_json"))
        return result

    def get_embeddings(self, model: str | None = None) -> list[dict]:
        """Return stored vectors, optionally filtered by model."""
        with get_connection() as connection:
            query = """
                SELECT
                    e.chunk_id,
                    e.model,
                    e.dimensions,
                    e.vector_json,
                    c.document_id,
                    c.text,
                    c.chunk_index,
                    d.title,
                    d.source
                FROM rag_embeddings AS e
                JOIN rag_chunks AS c ON c.chunk_id = e.chunk_id
                JOIN rag_documents AS d
                    ON d.document_id = c.document_id
            """
            parameters: tuple = ()

            if model is not None:
                query += " WHERE e.model = ?"
                parameters = (model,)

            query += " ORDER BY c.document_id, c.chunk_index"

            rows = connection.execute(query, parameters).fetchall()

        results = []
        for row in rows:
            result = dict(row)
            result["embedding"] = json.loads(result.pop("vector_json"))
            results.append(result)

        return results
