
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from commodity_finance_assistance.rag.chunker import DocumentChunk
from commodity_finance_assistance.rag.document_loader import Document
from commodity_finance_assistance.rag.ingestion import RAGIngestionService


class TestRAGIngestionService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.document = Document(
            document_id="doc-123",
            source=str(self.root / "guide.txt"),
            title="guide",
            content="Commodity trading basics",
            file_type="txt",
        )

        self.chunks = [
            DocumentChunk(
                chunk_id="doc-123:0",
                document_id="doc-123",
                source=self.document.source,
                title=self.document.title,
                text="Commodity trading basics",
                chunk_index=0,
                start_char=0,
                end_char=len(self.document.content),
            )
        ]

        self.store = Mock()
        self.store.save_document.return_value = 1

        self.service = RAGIngestionService(
            store=self.store,
            chunk_size=100,
            overlap=10,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("commodity_finance_assistance.rag.ingestion.chunk_document")
    @patch("commodity_finance_assistance.rag.ingestion.load_document")
    def test_ingest_file_loads_chunks_and_saves(
        self,
        mock_load_document,
        mock_chunk_document,
    ):
        mock_load_document.return_value = self.document
        mock_chunk_document.return_value = self.chunks

        result = self.service.ingest_file(self.document.source)

        self.assertEqual(result.document_id, "doc-123")
        self.assertEqual(result.title, "guide")
        self.assertEqual(result.chunk_count, 1)

        mock_load_document.assert_called_once_with(self.document.source)
        mock_chunk_document.assert_called_once_with(
            self.document,
            chunk_size=100,
            overlap=10,
        )
        self.store.save_document.assert_called_once_with(
            self.document,
            self.chunks,
        )

    @patch("commodity_finance_assistance.rag.ingestion.chunk_document")
    @patch("commodity_finance_assistance.rag.ingestion.load_document")
    def test_ingest_file_rejects_empty_chunks(
        self,
        mock_load_document,
        mock_chunk_document,
    ):
        mock_load_document.return_value = self.document
        mock_chunk_document.return_value = []

        with self.assertRaisesRegex(ValueError, "No chunks were generated"):
            self.service.ingest_file(self.document.source)

        self.store.save_document.assert_not_called()

    def test_ingest_directory_processes_supported_files(self):
        (self.root / "guide.txt").write_text(
            "Trading guide",
            encoding="utf-8",
        )
        (self.root / "notes.md").write_text(
            "Investor notes",
            encoding="utf-8",
        )
        (self.root / "ignore.csv").write_text(
            "unsupported",
            encoding="utf-8",
        )

        first_result = Mock(
            document_id="doc-1",
            source="guide.txt",
            title="guide",
            chunk_count=1,
        )
        second_result = Mock(
            document_id="doc-2",
            source="notes.md",
            title="notes",
            chunk_count=2,
        )

        with patch.object(
            self.service,
            "ingest_file",
            side_effect=[first_result, second_result],
        ) as mock_ingest:
            results = self.service.ingest_directory(self.root)

        self.assertEqual(len(results), 2)
        self.assertEqual(mock_ingest.call_count, 2)

        ingested_names = {
            Path(call.args[0]).name
            for call in mock_ingest.call_args_list
        }
        self.assertEqual(ingested_names, {"guide.txt", "notes.md"})

    def test_ingest_directory_rejects_missing_directory(self):
        missing = self.root / "does-not-exist"

        with self.assertRaises(NotADirectoryError):
            self.service.ingest_directory(missing)

    def test_constructor_validates_chunk_settings(self):
        with self.assertRaises(ValueError):
            RAGIngestionService(store=self.store, chunk_size=0)

        with self.assertRaises(ValueError):
            RAGIngestionService(
                store=self.store,
                chunk_size=100,
                overlap=100,
            )


if __name__ == "__main__":
    unittest.main()
