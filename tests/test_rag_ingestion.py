
import tempfile
import unittest
from pathlib import Path

from commodity_finance_assistance.rag.document_loader import (
    load_document,
    load_documents,
)
from commodity_finance_assistance.rag.chunker import chunk_document


class TestRAGIngestion(unittest.TestCase):
    def test_load_text_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gold_contract.txt"
            path.write_text(
                "Gold futures are standardized exchange-traded contracts.",
                encoding="utf-8",
            )

            document = load_document(path)

            self.assertEqual(document.title, "gold_contract")
            self.assertEqual(document.file_type, "txt")
            self.assertIn("Gold futures", document.content)
            self.assertTrue(document.document_id)

    def test_empty_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.txt"
            path.write_text("", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_document(path)

    def test_unsupported_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.csv"
            path.write_text("a,b", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_document(path)

    def test_chunking_preserves_content_and_overlap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            path.write_text(content, encoding="utf-8")
            document = load_document(path)

            chunks = chunk_document(
                document,
                chunk_size=10,
                overlap=3,
            )

            self.assertGreater(len(chunks), 1)
            self.assertEqual(chunks[0].text, "ABCDEFGHIJ")
            self.assertEqual(chunks[1].text, "HIJKLMNOPQ")
            for chunk in chunks:
                self.assertEqual(
                    content[chunk.start_char:chunk.end_char],
                    chunk.text,
                )

# The next chunk starts before the previous chunk ends.
            self.assertEqual(
                chunks[1].start_char,
                chunks[0].end_char - 3,
            )

    def test_invalid_chunk_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("Example content", encoding="utf-8")
            document = load_document(path)

            with self.assertRaises(ValueError):
                chunk_document(document, chunk_size=0)

            with self.assertRaises(ValueError):
                chunk_document(document, chunk_size=10, overlap=10)

    def test_load_documents_recursively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()

            (root / "one.txt").write_text("First document", encoding="utf-8")
            (nested / "two.md").write_text("Second document", encoding="utf-8")
            (root / "ignored.csv").write_text("a,b", encoding="utf-8")

            documents = load_documents(root)

            self.assertEqual(len(documents), 2)


if __name__ == "__main__":
    unittest.main()
