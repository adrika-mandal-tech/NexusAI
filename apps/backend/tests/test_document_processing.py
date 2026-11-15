"""Phase 4 Verification Tests — Document Processing Router, PyPDF, Docling, PaddleOCR, and Chunking."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from nexusai.core.ids import (
    compute_sha256_bytes,
    deterministic_chunk_id,
    deterministic_document_id,
    deterministic_entity_id,
)
from nexusai.infrastructure.parsers.chunker import ConfigurableChunker, ExtractedSection
from nexusai.infrastructure.parsers.file_inspector import (
    FileValidationError,
    validate_and_inspect_file,
)
from nexusai.infrastructure.parsers.router_parser import SmartDocumentRouter

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_FILES_DIR = REPO_ROOT / "Test_Files"


def test_deterministic_ids_are_idempotent_and_distinct() -> None:
    uid = uuid.UUID("00000000-0000-0000-0000-000000000000")
    digest = compute_sha256_bytes(b"hello nexusai")
    doc1 = deterministic_document_id(uid, digest)
    doc2 = deterministic_document_id(uid, digest)
    assert doc1 == doc2

    c1 = deterministic_chunk_id(doc1, 0, "Sample chunk text")
    c2 = deterministic_chunk_id(doc1, 0, "   Sample   chunk text ")
    c3 = deterministic_chunk_id(doc1, 1, "Sample chunk text")
    assert c1 == c2
    assert c1 != c3

    e1 = deterministic_entity_id("Technology", "CUDA Runtime")
    e2 = deterministic_entity_id("technology", "  cuda   runtime ")
    assert e1 == e2


def test_file_inspector_rejects_empty_and_spoofed_magic_bytes(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(FileValidationError, match="empty"):
        validate_and_inspect_file(empty)

    spoofed_pdf = tmp_path / "fake.pdf"
    spoofed_pdf.write_bytes(b"PK\x03\x04not_actually_a_pdf_archive")
    with pytest.raises(FileValidationError, match="magic-byte"):
        validate_and_inspect_file(spoofed_pdf)

    unsupported = tmp_path / "malware.exe"
    unsupported.write_bytes(b"MZ\x90\x00binary")
    with pytest.raises(FileValidationError, match="Unsupported"):
        validate_and_inspect_file(unsupported)


@pytest.mark.parametrize(
    "strategy",
    ["structure_aware", "table_aware", "recursive", "token_aware", "parent_child"],
)
def test_configurable_chunker_strategies_and_table_preservation(strategy: str) -> None:
    rows = "\n".join(f"| Product_{i} | Category_{i % 4} | ${i * 12}.00 |" for i in range(45))
    md_text = (
        "# Chapter 1: Architecture\n\n"
        "NexusAI combines vector retrieval and knowledge graphs for grounded answers.\n\n"
        "## Section 1.1: Inventory Table\n\n"
        "| Name | Category | Price |\n"
        "| --- | --- | --- |\n"
        f"{rows}\n"
    )
    doc_id = uuid.uuid4()
    chunker = ConfigurableChunker(strategy=strategy, chunk_size=350, chunk_overlap=40)
    chunks = chunker.chunk_sections(
        sections=[ExtractedSection(text=md_text, page_number=3, section_title="Chapter 1")],
        document_id=doc_id,
        filename="inventory.md",
    )
    assert len(chunks) >= 2
    assert all(c.page_number == 3 for c in chunks)
    # Re-running produces identical deterministic chunk IDs
    chunks_again = chunker.chunk_sections(
        sections=[ExtractedSection(text=md_text, page_number=3, section_title="Chapter 1")],
        document_id=doc_id,
        filename="inventory.md",
    )
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunks_again]

    if strategy in ("structure_aware", "table_aware"):
        table_chunks = [c for c in chunks if c.metadata.get("contains_table")]
        assert len(table_chunks) >= 2
        for tc in table_chunks:
            assert "| Name | Category | Price |" in tc.text

    if strategy == "parent_child":
        assert all("parent_chunk_id" in c.metadata and "parent_text" in c.metadata for c in chunks)
        assert any(len(c.metadata["parent_text"]) >= len(c.text) for c in chunks)


@pytest.mark.asyncio
async def test_router_parses_real_text_pdf_with_pypdf() -> None:
    pdf_path = TEST_FILES_DIR / "CUDA_C_Best_Practices_Guide.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample CUDA PDF not present")

    report = validate_and_inspect_file(pdf_path)
    assert report.recommended_parser == "pypdf"
    assert report.page_count > 10

    router = SmartDocumentRouter()
    doc_id = uuid.uuid4()
    parsed = await router.parse(str(pdf_path), doc_id, pdf_path.name, "pdf")
    assert len(parsed.chunks) > 20
    assert parsed.metadata["parser_used"].startswith("pypdf")
    assert any(c.page_number is not None and c.page_number >= 1 for c in parsed.chunks)
    assert any("CUDA" in c.text for c in parsed.chunks)


@pytest.mark.asyncio
async def test_router_parses_real_docx_with_docling() -> None:
    docx_path = TEST_FILES_DIR / "report.docx"
    if not docx_path.exists():
        pytest.skip("Sample report.docx not present")

    report = validate_and_inspect_file(docx_path)
    assert report.recommended_parser == "docling"

    router = SmartDocumentRouter()
    parsed = await router.parse(str(docx_path), uuid.uuid4(), docx_path.name, "docx")
    assert parsed.metadata["parser_used"] == "docling"
    assert len(parsed.chunks) >= 1
    assert sum(len(c.text) for c in parsed.chunks) > 50


@pytest.mark.asyncio
async def test_router_detects_scanned_pdf_and_image_using_paddleocr(tmp_path: Path) -> None:
    # Create a high-contrast synthetic scanned document image and a zero-text-layer scanned PDF
    img = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 60), "NEXUSAI SCANNED INVOICE 2026", fill="black")
    draw.text((40, 130), "Customer: Northwind Traders", fill="black")
    draw.text((40, 200), "Total Amount: 4850 USD", fill="black")

    img_path = tmp_path / "scanned_invoice.png"
    img.save(img_path, format="PNG")

    scanned_pdf_path = tmp_path / "scanned_invoice.pdf"
    img.save(scanned_pdf_path, format="PDF", resolution=150.0)

    router = SmartDocumentRouter()

    # 1. Image OCR
    parsed_img = await router.parse(str(img_path), uuid.uuid4(), img_path.name, "image")
    assert parsed_img.metadata["parser_used"] == "paddleocr"
    assert len(parsed_img.chunks) >= 1
    combined_img_text = " ".join(c.text for c in parsed_img.chunks).upper()
    assert "NEXUSAI" in combined_img_text or "NORTHWIND" in combined_img_text
    assert parsed_img.chunks[0].page_number == 1

    # 2. Scanned PDF auto-detection & PaddleOCR routing
    pdf_report = validate_and_inspect_file(scanned_pdf_path)
    assert pdf_report.recommended_parser == "paddleocr"

    parsed_pdf = await router.parse(str(scanned_pdf_path), uuid.uuid4(), scanned_pdf_path.name, "pdf")
    assert parsed_pdf.metadata["parser_used"] == "paddleocr"
    assert len(parsed_pdf.chunks) >= 1
    combined_pdf_text = " ".join(c.text for c in parsed_pdf.chunks).upper()
    assert "NORTHWIND" in combined_pdf_text or "4850" in combined_pdf_text
    assert parsed_pdf.chunks[0].page_number == 1
    assert parsed_pdf.metadata.get("average_ocr_confidence", 0.0) > 0.6

