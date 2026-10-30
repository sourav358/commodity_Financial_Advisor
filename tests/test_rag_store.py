
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


if __name__ == "__main__":
    unittest.main()
