
from dataclasses import dataclass
from pathlib import Path
import hashlib

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class Document:
    """A source document with traceable metadata."""

    document_id: str
    source: str
    title: str
    content: str
    file_type: str


def load_document(path: str | Path) -> Document:
    """Load a PDF, text, or Markdown document."""
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {suffix}"
        )

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        text_parts = []

        for page in reader.pages:
            text_parts.append(page.extract_text() or "")

        content = "\n".join(text_parts)
    else:
        content = file_path.read_text(encoding="utf-8")

    content = content.strip()

    if not content:
        raise ValueError(
            f"No extractable text found in: {file_path.name}"
        )

    document_id = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return Document(
        document_id=document_id,
        source=str(file_path.resolve()),
        title=file_path.stem,
        content=content,
        file_type=suffix.lstrip("."),
    )


def load_documents(directory: str | Path) -> list[Document]:
    """Load supported files from a directory, recursively."""
    root = Path(directory)

    if not root.is_dir():
        raise NotADirectoryError(
            f"Knowledge directory not found: {root}"
        )

    documents = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        documents.append(load_document(path))

    return documents
