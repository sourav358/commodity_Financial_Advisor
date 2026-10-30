
from contextlib import contextmanager
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from commodity_finance_assistance.rag.chunker import DocumentChunk
from commodity_finance_assistance.rag.document_loader import Document
from commodity_finance_assistance.rag.store import RAGStore


class TestRAGEmbeddingStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "embedding_store.db"

        self.connection_patch = patch(
            "commodity_finance_assistance.rag.store.get_connection",
            new=self._get_connection,
        )
        self.connection_patch.start()

        self.store = RAGStore()
        self.store.initialize_schema()

        self.document = Document(
            document_id="doc-1",
            source="guide.txt",
            title="Commodity Guide",
            content="Gold futures basics",
            file_type="txt",
        )
        self.chunk = DocumentChunk(
            chunk_id="doc-1:0",
            document_id="doc-1",
            source="guide.txt",
            title="Commodity Guide",
            text="Gold futures basics",
            chunk_index=0,
            start_char=0,
            end_char=len(self.document.content),
        )
        self.store.save_document(self.document, [self.chunk])

    def tearDown(self):
        self.connection_patch.stop()
        self.temp_dir.cleanup()

    @contextmanager
    def _get_connection(self):
        connection = sqlite3.connect(str(self.db_path))
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

    def test_save_and_retrieve_embedding(self):
        vector = [0.1, 0.2, 0.3]

        saved = self.store.save_embeddings(
            {"doc-1:0": vector},
            model="test-embedding-model",
            dimensions=3,
        )

        self.assertEqual(saved, 1)

        result = self.store.get_embedding("doc-1:0")
        self.assertIsNotNone(result)
        self.assertEqual(result["model"], "test-embedding-model")
        self.assertEqual(result["dimensions"], 3)
        self.assertEqual(result["embedding"], vector)

    def test_get_embeddings_includes_chunk_source_and_text(self):
        self.store.save_embeddings(
            {"doc-1:0": [0.1, 0.2, 0.3]},
            model="test-embedding-model",
            dimensions=3,
        )

        results = self.store.get_embeddings(
            model="test-embedding-model"
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "doc-1:0")
        self.assertEqual(results[0]["text"], "Gold futures basics")
        self.assertEqual(results[0]["title"], "Commodity Guide")
        self.assertEqual(results[0]["source"], "guide.txt")

    def test_model_filter_excludes_other_models(self):
        self.store.save_embeddings(
            {"doc-1:0": [0.1, 0.2, 0.3]},
            model="model-a",
            dimensions=3,
        )

        self.assertEqual(len(self.store.get_embeddings(model="model-b")), 0)
        self.assertEqual(len(self.store.get_embeddings()), 1)

    def test_saving_again_updates_existing_vector(self):
        self.store.save_embeddings(
            {"doc-1:0": [0.1, 0.2, 0.3]},
            model="model-a",
            dimensions=3,
        )
        self.store.save_embeddings(
            {"doc-1:0": [0.4, 0.5, 0.6]},
            model="model-b",
            dimensions=3,
        )

        result = self.store.get_embedding("doc-1:0")
        self.assertEqual(result["model"], "model-b")
        self.assertEqual(result["embedding"], [0.4, 0.5, 0.6])
        self.assertEqual(len(self.store.get_embeddings()), 1)

    def test_rejects_wrong_dimensions(self):
        with self.assertRaisesRegex(ValueError, "Expected 3 dimensions"):
            self.store.save_embeddings(
                {"doc-1:0": [0.1, 0.2]},
                model="test-model",
                dimensions=3,
            )

    def test_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.store.save_embeddings(
                {"doc-1:0": [0.1, float("nan"), 0.3]},
                model="test-model",
                dimensions=3,
            )

    def test_rejects_unknown_chunk_id(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save_embeddings(
                {"missing-chunk": [0.1, 0.2, 0.3]},
                model="test-model",
                dimensions=3,
            )

    def test_deleting_document_cascades_to_embeddings(self):
        self.store.save_embeddings(
            {"doc-1:0": [0.1, 0.2, 0.3]},
            model="test-model",
            dimensions=3,
        )

        self.assertTrue(self.store.delete_document("doc-1"))
        self.assertIsNone(self.store.get_embedding("doc-1:0"))
        self.assertEqual(self.store.get_embeddings(), [])


if __name__ == "__main__":
    unittest.main()
