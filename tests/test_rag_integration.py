
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from commodity_finance_assistance.rag.ingestion import (
    RAGIngestionService,
)
from commodity_finance_assistance.rag.store import RAGStore


class TestRAGIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "integration_test.db"
        self.knowledge_dir = self.root / "knowledge"
        self.knowledge_dir.mkdir()

        self.connection_patch = patch(
            "commodity_finance_assistance.rag.store.get_connection",
            new=self._get_connection,
        )
        self.connection_patch.start()

        self.store = RAGStore()
        self.store.initialize_schema()

        self.service = RAGIngestionService(
            store=self.store,
            chunk_size=40,
            overlap=10,
        )

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

    def test_file_ingestion_persists_document_and_chunks(self):
        content = (
            "Gold is a commodity. Silver is another commodity. "
            "Futures contracts have expiry dates. "
            "Commodity risk requires careful monitoring."
        )
        file_path = self.knowledge_dir / "commodity_guide.txt"
        file_path.write_text(content, encoding="utf-8")

        result = self.service.ingest_file(file_path)

        document = self.store.get_document(result.document_id)
        chunks = self.store.get_chunks(result.document_id)

        self.assertIsNotNone(document)
        self.assertEqual(document["title"], "commodity_guide")
        self.assertEqual(document["file_type"], "txt")
        self.assertEqual(document["source"], str(file_path.resolve()))

        self.assertGreater(len(chunks), 1)
        self.assertEqual(result.chunk_count, len(chunks))
        self.assertEqual(
            [chunk["chunk_index"] for chunk in chunks],
            list(range(len(chunks))),
        )
        self.assertTrue(all(chunk["text"] for chunk in chunks))

    def test_directory_ingestion_persists_multiple_documents(self):
        first_file = self.knowledge_dir / "gold.txt"
        second_file = self.knowledge_dir / "silver.md"

        first_file.write_text(
            "Gold futures have contract expiry dates.",
            encoding="utf-8",
        )
        second_file.write_text(
            "Silver prices can be volatile.",
            encoding="utf-8",
        )

        results = self.service.ingest_directory(self.knowledge_dir)

        self.assertEqual(len(results), 2)

        for result in results:
            document = self.store.get_document(result.document_id)
            chunks = self.store.get_chunks(result.document_id)

            self.assertIsNotNone(document)
            self.assertGreater(len(chunks), 0)
            self.assertEqual(result.chunk_count, len(chunks))

        self.assertEqual(
            {result.title for result in results},
            {"gold", "silver"},
        )


if __name__ == "__main__":
    unittest.main()
