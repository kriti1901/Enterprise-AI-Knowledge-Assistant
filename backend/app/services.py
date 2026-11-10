"""Document lifecycle: store file, parse, chunk, embed, index, and delete."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.cache import bump_index_version
from app.config import get_settings
from app.db import Chunk, Document
from app.ingestion import SUPPORTED_EXTENSIONS, UnsupportedFileError, process_file
from app.observability import emit_metric, log_event
from app.retrieval import delete_document_vectors, index_chunks

_MAGIC = {".pdf": b"%PDF", ".docx": b"PK\x03\x04"}


def _safe_name(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)[:200]


def validate_upload(filename: str, content: bytes) -> str:
    """Check extension, size and file signature. Returns the lower-case extension."""
    settings = get_settings()
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError("Only PDF, DOCX and TXT files are supported")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise UnsupportedFileError(f"File exceeds {settings.max_upload_mb} MB limit")
    if suffix in _MAGIC and not content.startswith(_MAGIC[suffix]):
        raise UnsupportedFileError(f"File content does not match its {suffix} extension")
    if suffix == ".txt" and b"\x00" in content[:4096]:
        raise UnsupportedFileError("Text file appears to be binary")
    return suffix


def ingest_document(
    session: Session,
    *,
    filename: str,
    content: bytes,
    title: str,
    department: str,
    uploaded_by: str,
) -> Document:
    settings = get_settings()
    validate_upload(filename, content)

    document = Document(
        title=title.strip()[:300] or filename,
        filename=filename[:300],
        department=department.strip(),
        uploaded_by=uploaded_by,
        status="pending",
    )
    session.add(document)
    session.commit()

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored = upload_dir / f"{document.id}_{_safe_name(filename)}"
    stored.write_bytes(content)

    try:
        chunks = process_file(stored, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise UnsupportedFileError("No extractable text found in the document")
        session.add_all(
            Chunk(
                document_id=document.id,
                department=document.department,
                chunk_index=c.index,
                page=c.page,
                text=c.text,
            )
            for c in chunks
        )
        index_chunks(document, chunks)
        document.chunk_count = len(chunks)
        document.status = "indexed"
        document.error = None
    except Exception as exc:  # record the failure on the document instead of losing it
        session.rollback()
        try:
            delete_document_vectors(document.id)  # drop any partially written vectors
        except Exception:
            pass
        document = session.get(Document, document.id)
        document.status = "failed"
        document.error = str(exc)[:1000]
        log_event("ingestion_failed", document_id=document.id, error=str(exc)[:200])
        emit_metric("IngestionFailures")
    session.commit()
    session.refresh(document)

    if document.status == "indexed":
        bump_index_version()
        emit_metric("DocumentsIndexed")
        log_event("ingestion_indexed", document_id=document.id, chunks=document.chunk_count)
    return document


def delete_document(session: Session, document: Document) -> None:
    delete_document_vectors(document.id)
    for path in Path(get_settings().upload_dir).glob(f"{document.id}_*"):
        path.unlink(missing_ok=True)
    session.delete(document)
    session.commit()
    bump_index_version()
    log_event("document_deleted", document_id=document.id)


def purge_upload_dir() -> None:
    """Helper for tests and local resets."""
    shutil.rmtree(get_settings().upload_dir, ignore_errors=True)
