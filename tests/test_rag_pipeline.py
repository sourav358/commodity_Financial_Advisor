
from contextlib import contextmanager
import sqlite3

import pytest

from commodity_finance_assistance.rag.ingestion import RAGIngestionService
from commodity_finance_assistance.rag.pipeline import RAGPipeline
from commodity_finance_assistance.rag.store import RAGStore


class FakeEmbeddingService:
    model = "test-embedding-model"
    dimensions = 3

    def __init__(
        self,
        failure=None,
        wrong_count=False,
        value=0.1,
    ):
        self.failure = failure
        self.wrong_count = wrong_count
        self.value = value

    def embed_texts(self, texts):
        if self.failure:
            raise self.failure

        vectors = [
            [self.value, 0.2, 0.3]
            for _ in texts
        ]

        if self.wrong_count and vectors:
            return vectors[:-1]

        return vectors


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    db_path = tmp_path / "rag_test.db"

    @contextmanager
    def test_get_connection():
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # Patch the exact function imported by RAGStore.
    monkeypatch.setattr(
        "commodity_finance_assistance.rag.store.get_connection",
        test_get_connection,
    )

    store = RAGStore()
    store.initialize_schema()

    ingestion = RAGIngestionService(store=store)
    embeddings = FakeEmbeddingService()

    rag_pipeline = RAGPipeline(
        ingestion_service=ingestion,
        store=store,
        embedding_service=embeddings,
    )

    return rag_pipeline, store, test_get_connection


def create_document(
    tmp_path,
    content="Commodity market fundamentals " * 100,
):
    path = tmp_path / "commodity.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_successful_ingestion_saves_chunks_and_embeddings(
    pipeline,
    tmp_path,
):
    rag_pipeline, store, _ = pipeline
    path = create_document(tmp_path)

    result = rag_pipeline.ingest_file(path)

    chunks = store.get_chunks(result.document_id)
    embeddings = store.get_embeddings(
        model="test-embedding-model"
    )

    assert result.chunk_count == len(chunks)
    assert len(chunks) > 0
    assert len(embeddings) == len(chunks)


def test_embedding_failure_preserves_existing_index(
    pipeline,
    tmp_path,
):
    rag_pipeline, store, _ = pipeline
    path = create_document(tmp_path)

    first = rag_pipeline.ingest_file(path)

    old_chunks = store.get_chunks(first.document_id)
    old_embeddings = store.get_embeddings(
        model="test-embedding-model"
    )

    rag_pipeline._embedding_service = FakeEmbeddingService(
        failure=RuntimeError("Embedding API unavailable")
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding API unavailable",
    ):
        rag_pipeline.ingest_file(path)

    assert store.get_chunks(first.document_id) == old_chunks
    assert store.get_embeddings(
        model="test-embedding-model"
    ) == old_embeddings


def test_embedding_count_mismatch_preserves_existing_index(
    pipeline,
    tmp_path,
):
    rag_pipeline, store, _ = pipeline
    path = create_document(tmp_path)

    first = rag_pipeline.ingest_file(path)

    old_chunks = store.get_chunks(first.document_id)
    old_embeddings = store.get_embeddings(
        model="test-embedding-model"
    )

    rag_pipeline._embedding_service = FakeEmbeddingService(
        wrong_count=True
    )

    with pytest.raises(ValueError, match="Embedding count"):
        rag_pipeline.ingest_file(path)

    assert store.get_chunks(first.document_id) == old_chunks
    assert store.get_embeddings(
        model="test-embedding-model"
    ) == old_embeddings


def test_reingestion_replaces_embeddings_for_same_document(
    pipeline,
    tmp_path,
):
    rag_pipeline, store, _ = pipeline
    path = create_document(tmp_path)

    first = rag_pipeline.ingest_file(path)

    old_embeddings = store.get_embeddings(
        model="test-embedding-model"
    )

    rag_pipeline._embedding_service = FakeEmbeddingService(
        value=0.9
    )

    second = rag_pipeline.ingest_file(path)

    new_embeddings = store.get_embeddings(
        model="test-embedding-model"
    )

    assert second.document_id == first.document_id
    assert len(new_embeddings) == len(old_embeddings)
    assert new_embeddings[0]["embedding"] == [0.9, 0.2, 0.3]
    assert new_embeddings[0]["embedding"] != old_embeddings[0]["embedding"]


def test_database_failure_rolls_back_document_replacement(
    pipeline,
    tmp_path,
):
    rag_pipeline, store, get_connection = pipeline
    path = create_document(tmp_path)

    first = rag_pipeline.ingest_file(path)

    old_document = store.get_document(first.document_id)
    old_chunks = store.get_chunks(first.document_id)
    old_embeddings = store.get_embeddings(
        model="test-embedding-model"
    )

    # Fail during embedding insertion, after the pipeline has
    # started replacing the document's chunks.
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_embedding_insert
            BEFORE INSERT ON rag_embeddings
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'simulated embedding persistence failure'
                );
            END;
            """
        )

    rag_pipeline._embedding_service = FakeEmbeddingService(
        value=0.9
    )

    with pytest.raises(
        sqlite3.IntegrityError,
        match="simulated embedding persistence failure",
    ):
        rag_pipeline.ingest_file(path)

    # The previous document, chunks and vectors must survive.
    assert store.get_document(first.document_id) == old_document
    assert store.get_chunks(first.document_id) == old_chunks
    assert store.get_embeddings(
        model="test-embedding-model"
    ) == old_embeddings


def test_directory_ingestion_processes_multiple_documents(
    pipeline,
    tmp_path,
):
    rag_pipeline, store, _ = pipeline

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()

    (knowledge / "oil.txt").write_text(
        "Oil market information " * 100,
        encoding="utf-8",
    )

    (knowledge / "gas.md").write_text(
        "Natural gas market information " * 100,
        encoding="utf-8",
    )

    (knowledge / "ignored.csv").write_text(
        "Unsupported file",
        encoding="utf-8",
    )

    results = rag_pipeline.ingest_directory(knowledge)

    assert len(results) == 2

    for result in results:
        assert store.get_document(result.document_id) is not None

        chunks = store.get_chunks(result.document_id)
        assert len(chunks) > 0

        for chunk in chunks:
            assert store.get_embedding(chunk["chunk_id"]) is not None

def test_changed_file_content_keeps_same_id_and_removes_old_version(
    pipeline,
    tmp_path,
):
    rag_pipeline, store, _ = pipeline

    path = create_document(
        tmp_path,
        "Original commodity information " * 100,
    )

    first = rag_pipeline.ingest_file(path)

    old_chunks = store.get_chunks(first.document_id)
    assert old_chunks

    # Change the content of the same physical file.
    path.write_text(
        "Updated commodity market information " * 100,
        encoding="utf-8",
    )

    second = rag_pipeline.ingest_file(path)

    assert second.document_id == first.document_id

    new_chunks = store.get_chunks(second.document_id)

    assert new_chunks
    assert new_chunks != old_chunks

    # Verify that all new chunks have embeddings.
    for chunk in new_chunks:
        assert store.get_embedding(chunk["chunk_id"]) is not None

    assert store.get_document(first.document_id) is not None
def test_different_files_have_different_document_ids(tmp_path):
    from commodity_finance_assistance.rag.document_loader import load_document

    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"

    first_path.write_text("Same commodity content", encoding="utf-8")
    second_path.write_text("Same commodity content", encoding="utf-8")

    first = load_document(first_path)
    second = load_document(second_path)

    assert first.document_id != second.document_id