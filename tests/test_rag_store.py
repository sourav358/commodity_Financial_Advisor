
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from commodity_finance_assistance.rag.chunker import DocumentChunk
from commodity_finance_assistance.rag.document_loader import Document
from commodity_finance_assistance.rag.store import RAGStore


class TestRAGStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "rag_test.db"

        self.connection_patch = patch(
            "commodity_finance_assistance.rag.store.get_connection",
            new=self._get_connection,
        )
        self.connection_patch.start()

        self.store = RAGStore()
        self.store.initialize_schema()

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

    @staticmethod
    def make_document(
        document_id="doc-1",
        title="Commodity Guide",
        content="ABCDEFGHIJ",
    ):
        return Document(
            document_id=document_id,
            source="data/knowledge/commodity_guide.txt",
            title=title,
            content=content,
            file_type="txt",
        )

    @staticmethod
    def make_chunks(document, parts):
        return [
            DocumentChunk(
                chunk_id=f"{document.document_id}:{index}",
                document_id=document.document_id,
                source=document.source,
                title=document.title,
                text=text,
                chunk_index=index,
                start_char=start,
                end_char=end,
            )
            for index, (text, start, end) in enumerate(parts)
        ]

    def test_save_and_retrieve_document_and_chunks(self):
        document = self.make_document()
        chunks = self.make_chunks(
            document,
            [
                ("ABCDEF", 0, 6),
                ("EFGHIJ", 4, 10),
            ],
        )

        saved_count = self.store.save_document(document, chunks)

        self.assertEqual(saved_count, 2)

        saved_document = self.store.get_document(document.document_id)
        self.assertIsNotNone(saved_document)
        self.assertEqual(saved_document["title"], "Commodity Guide")
        self.assertEqual(saved_document["file_type"], "txt")

        saved_chunks = self.store.get_chunks(document.document_id)
        self.assertEqual(len(saved_chunks), 2)
        self.assertEqual(
            [chunk["chunk_index"] for chunk in saved_chunks],
            [0, 1],
        )
        self.assertEqual(
            [chunk["text"] for chunk in saved_chunks],
            ["ABCDEF", "EFGHIJ"],
        )
        self.assertEqual(
            saved_chunks[0]["source"],
            document.source,
        )

    def test_saving_document_again_replaces_old_chunks(self):
        document = self.make_document()
        original_chunks = self.make_chunks(
            document,
            [
                ("ABCDEF", 0, 6),
                ("EFGHIJ", 4, 10),
            ],
        )
        self.store.save_document(document, original_chunks)

        updated_document = self.make_document(title="Updated Guide")
        updated_chunks = self.make_chunks(
            updated_document,
            [("ABCDEFGHIJ", 0, 10)],
        )

        self.store.save_document(updated_document, updated_chunks)

        saved_chunks = self.store.get_chunks(document.document_id)
        self.assertEqual(len(saved_chunks), 1)
        self.assertEqual(saved_chunks[0]["text"], "ABCDEFGHIJ")
        self.assertEqual(
            self.store.get_document(document.document_id)["title"],
            "Updated Guide",
        )

    def test_missing_document_returns_none(self):
        self.assertIsNone(self.store.get_document("missing-document"))

    def test_missing_document_has_no_chunks(self):
        self.assertEqual(self.store.get_chunks("missing-document"), [])

    def test_rejects_chunks_from_another_document(self):
        document = self.make_document()
        other_document = self.make_document(document_id="other-doc")
        chunks = self.make_chunks(
            other_document,
            [("ABCDEF", 0, 6)],
        )

        with self.assertRaisesRegex(ValueError, "belong to"):
            self.store.save_document(document, chunks)

        self.assertIsNone(self.store.get_document(document.document_id))

    def test_rejects_empty_chunk_list(self):
        document = self.make_document()

        with self.assertRaisesRegex(ValueError, "At least one chunk"):
            self.store.save_document(document, [])

        self.assertIsNone(self.store.get_document(document.document_id))

    def test_deleting_document_also_deletes_chunks(self):
        document = self.make_document()
        chunks = self.make_chunks(
            document,
            [("ABCDEFGHIJ", 0, 10)],
        )
        self.store.save_document(document, chunks)

        self.assertTrue(self.store.delete_document(document.document_id))
        self.assertIsNone(self.store.get_document(document.document_id))
        self.assertEqual(self.store.get_chunks(document.document_id), [])

    def test_deleting_missing_document_returns_false(self):
        self.assertFalse(self.store.delete_document("missing-document"))

    def insert_raw_embedding(
        self,
        chunk_id,
        vector_json,
        dimensions=2,
        model="text-embedding-3-small",
    ):
        """Insert raw embedding data to simulate database corruption."""
        with self._get_connection() as connection:
            connection.execute(
                """
                INSERT INTO rag_embeddings (
                    chunk_id, model, dimensions, vector_json
                )
                VALUES (?, ?, ?, ?)
                """,
                (chunk_id, model, dimensions, vector_json),
            )

    def test_save_and_retrieve_embeddings(self):
        document = self.make_document()
        chunks = self.make_chunks(
            document,
            [("ABCDEFGHIJ", 0, 10)],
        )
        self.store.save_document(document, chunks)

        saved_count = self.store.save_embeddings(
            {"doc-1:0": [0.1, 0.2]},
            model="text-embedding-3-small",
            dimensions=2,
        )

        self.assertEqual(saved_count, 1)

        embeddings = self.store.get_embeddings(
            model="text-embedding-3-small"
        )

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0]["chunk_id"], "doc-1:0")
        self.assertEqual(embeddings[0]["embedding"], [0.1, 0.2])
        self.assertEqual(embeddings[0]["dimensions"], 2)

    def test_get_embedding_returns_validated_vector(self):
        document = self.make_document()
        chunks = self.make_chunks(
            document,
            [("ABCDEFGHIJ", 0, 10)],
        )
        self.store.save_document(document, chunks)

        self.store.save_embeddings(
            {"doc-1:0": [0.3, 0.4]},
            model="test-model",
            dimensions=2,
        )

        result = self.store.get_embedding("doc-1:0")

        self.assertIsNotNone(result)
        self.assertEqual(result["embedding"], [0.3, 0.4])

    def test_missing_embedding_returns_none(self):
        self.assertIsNone(
            self.store.get_embedding("missing-chunk")
        )

    def test_missing_embeddings_returns_empty_list(self):
        self.assertEqual(self.store.get_embeddings(), [])

    def test_corrupted_embedding_json_is_rejected(self):
        document = self.make_document()
        chunks = self.make_chunks(
            document,
            [("ABCDEFGHIJ", 0, 10)],
        )
        self.store.save_document(document, chunks)

        self.insert_raw_embedding(
            "doc-1:0",
            "{invalid-json",
        )

        with self.assertRaisesRegex(ValueError, "Corrupted embedding JSON"):
            self.store.get_embeddings()

    def test_embedding_dimension_mismatch_is_rejected(self):
        document = self.make_document()
        chunks = self.make_chunks(
            document,
            [("ABCDEFGHIJ", 0, 10)],
        )
        self.store.save_document(document, chunks)

        self.insert_raw_embedding(
            "doc-1:0",
            "[0.1, 0.2, 0.3]",
            dimensions=2,
        )

        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            self.store.get_embeddings()

    def test_non_finite_stored_embedding_is_rejected(self):
        document = self.make_document()
        chunks = self.make_chunks(
            document,
            [("ABCDEFGHIJ", 0, 10)],
        )
        self.store.save_document(document, chunks)

        self.insert_raw_embedding(
            "doc-1:0",
            "[0.1, NaN]",
        )

        with self.assertRaisesRegex(ValueError, "Invalid embedding values"):
            self.store.get_embeddings()

    def test_boolean_stored_embedding_is_rejected(self):
        document = self.make_document()
        chunks = self.make_chunks(
            document,
            [("ABCDEFGHIJ", 0, 10)],
        )
        self.store.save_document(document, chunks)

        self.insert_raw_embedding(
            "doc-1:0",
            "[0.1, true]",
        )

        with self.assertRaisesRegex(ValueError, "Invalid embedding values"):
            self.store.get_embeddings()

    def test_deleting_document_also_deletes_embeddings(self):
        document = self.make_document()
        chunks = self.make_chunks(
            document,
            [("ABCDEFGHIJ", 0, 10)],
        )
        self.store.save_document(document, chunks)

        self.store.save_embeddings(
            {"doc-1:0": [0.1, 0.2]},
            model="test-model",
            dimensions=2,
        )

        self.assertIsNotNone(
            self.store.get_embedding("doc-1:0")
        )

        self.assertTrue(
            self.store.delete_document(document.document_id)
        )

        self.assertIsNone(
            self.store.get_embedding("doc-1:0")
        )

    def test_replacing_chunks_removes_old_embeddings(self):
        document = self.make_document()
        original_chunks = self.make_chunks(
            document,
            [
                ("ABCDEF", 0, 6),
                ("EFGHIJ", 4, 10),
            ],
        )
        self.store.save_document(document, original_chunks)

        self.store.save_embeddings(
            {
                "doc-1:0": [0.1, 0.2],
                "doc-1:1": [0.3, 0.4],
            },
            model="test-model",
            dimensions=2,
        )

        updated_chunks = self.make_chunks(
            document,
            [("ABCDEFGHIJ", 0, 10)],
        )
        self.store.save_document(document, updated_chunks)

        self.assertIsNone(
            self.store.get_embedding("doc-1:0")
        )
        self.assertIsNone(
            self.store.get_embedding("doc-1:1")
        )
        self.assertEqual(
            self.store.get_embeddings(),
            [],
        )


if __name__ == "__main__":
    unittest.main()
