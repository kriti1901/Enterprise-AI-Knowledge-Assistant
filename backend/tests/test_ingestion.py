from pathlib import Path

import pytest
from docx import Document as DocxDocument

from app.ingestion import (
    Page,
    UnsupportedFileError,
    chunk_pages,
    clean_text,
    parse_file,
    process_file,
    remove_repeated_lines,
)
from app.services import validate_upload


def test_clean_text_rejoins_hyphenated_words_and_collapses_whitespace():
    raw = "The commit-\nment   is   firm.\r\n\r\n\r\n\r\nNext  para"
    assert clean_text(raw) == "The commitment is firm.\n\nNext para"


def test_clean_text_normalizes_unicode():
    assert clean_text("ﬁnance") == "finance"


def test_repeated_header_and_footer_lines_are_removed():
    pages = [Page(i, f"ACME CONFIDENTIAL\nBody text for page {i}\nPage footer") for i in range(1, 5)]
    cleaned = remove_repeated_lines(pages)
    for page in cleaned:
        assert "ACME CONFIDENTIAL" not in page.text
        assert "Body text" in page.text


def test_chunks_keep_page_numbers_and_indexes():
    pages = [Page(1, "alpha " * 300), Page(2, "beta " * 300)]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 200 for c in chunks)


def test_txt_file_is_parsed_and_chunked(tmp_path: Path):
    path = tmp_path / "policy.txt"
    path.write_text("Employees receive 20 days of leave.\n" * 5, encoding="utf-8")
    chunks = process_file(path, chunk_size=100, chunk_overlap=10)
    assert chunks and chunks[0].page == 1
    assert "20 days" in chunks[0].text


def test_docx_file_includes_tables(tmp_path: Path):
    path = tmp_path / "budget.docx"
    doc = DocxDocument()
    doc.add_paragraph("Quarterly budget summary")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Travel"
    table.rows[0].cells[1].text = "15000"
    doc.save(path)
    text = parse_file(path)[0].text
    assert "Quarterly budget summary" in text
    assert "Travel | 15000" in text


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedFileError):
        validate_upload("malware.exe", b"MZ")


def test_pdf_extension_requires_pdf_signature():
    with pytest.raises(UnsupportedFileError):
        validate_upload("report.pdf", b"not really a pdf")
    assert validate_upload("report.pdf", b"%PDF-1.4 ...") == ".pdf"
