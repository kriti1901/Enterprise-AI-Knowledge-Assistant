"""Document parsing, cleaning and chunking for PDF, DOCX and TXT files."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UnsupportedFileError(ValueError):
    pass


@dataclass
class Page:
    number: int
    text: str


@dataclass
class TextChunk:
    index: int
    page: int
    text: str


def parse_file(path: Path) -> list[Page]:
    """Extract text per page. DOCX and TXT have no page concept, so they are a single page."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [Page(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix == ".docx":
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return [Page(1, "\n".join(parts))]
    if suffix == ".txt":
        return [Page(1, path.read_bytes().decode("utf-8", errors="replace"))]
    raise UnsupportedFileError(f"Unsupported file type: {suffix or 'unknown'}")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_repeated_lines(pages: list[Page], min_pages: int = 3, max_len: int = 80) -> list[Page]:
    """Drop short lines (headers/footers) that repeat on several pages."""
    if len(pages) < min_pages:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({line.strip() for line in page.text.splitlines() if 0 < len(line.strip()) <= max_len})
    repeated = {line for line, n in counts.items() if n >= min_pages}
    cleaned = []
    for page in pages:
        lines = [line for line in page.text.splitlines() if line.strip() not in repeated]
        cleaned.append(Page(page.number, "\n".join(lines)))
    return cleaned


def chunk_pages(pages: list[Page], chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[TextChunk] = []
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue
        for piece in splitter.split_text(cleaned):
            chunks.append(TextChunk(index=len(chunks), page=page.number, text=piece))
    return chunks


def process_file(path: Path, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    pages = remove_repeated_lines(parse_file(path))
    return chunk_pages(pages, chunk_size, chunk_overlap)
