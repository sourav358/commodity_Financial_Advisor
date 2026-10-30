
from dataclasses import dataclass

from commodity_finance_assistance.rag.document_loader import Document


@dataclass(frozen=True)
class DocumentChunk:
    """A passage that can be indexed and retrieved."""

    chunk_id: str
    document_id: str
    source: str
    title: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_document(
    document: Document,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """Split document text into overlapping character windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    content = document.content
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))
        text = content[start:end].strip()

        if text:
            chunk_id = f"{document.document_id}:{chunk_index}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source=document.source,
                    title=document.title,
                    text=text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
            )

            chunk_index += 1

        if end == len(content):
            break

        start = end - overlap

    return chunks
