"""Intelligent Document Processing Router combining PyPDF, Docling, PaddleOCR, and StructuredText.

Workflow:
  Upload -> File Validation & Magic Bytes -> Quality & Layout Inspection ->
  Optimal Primary Parser (PyPDF / Docling / PaddleOCR / StructuredText) ->
  Selective Hybrid Page OCR & Automatic Fallback on Poor Extraction ->
  Configurable Chunking & Deterministic IDs.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.ports.document_parser import DocumentParserPort, ParsedDocument
from nexusai.infrastructure.parsers.chunker import ConfigurableChunker, ExtractedSection
from nexusai.infrastructure.parsers.docling_parser import DoclingDocumentParser
from nexusai.infrastructure.parsers.file_inspector import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    FileInspectionReport,
    compute_text_quality_score,
    validate_and_inspect_file,
)
from nexusai.infrastructure.parsers.paddle_ocr_parser import PaddleOCRParser
from nexusai.infrastructure.parsers.pymupdf_parser import PyMuPDFParser
from nexusai.infrastructure.parsers.pypdf_parser import PyPDFParser

logger = get_logger(__name__)


class StructuredTextParser(DocumentParserPort):
    """Fast native parser for `.md`, `.txt`, `.json`, and document-RAG `.csv` files."""

    def __init__(self, chunker: ConfigurableChunker | None = None) -> None:
        self._chunker = chunker or ConfigurableChunker()

    def supported_types(self) -> list[str]:
        return [".txt", ".md", ".json", ".csv"]

    async def parse(
        self,
        file_path: str,
        document_id: UUID,
        filename: str,
        source_type: str,
    ) -> ParsedDocument:
        raw_text = await asyncio.to_thread(self._read_text_sync, file_path)
        ext = Path(filename).suffix.lower() or f".{source_type.lower()}"
        formatted_text = self._format_content(raw_text, ext, filename)

        sections = [
            ExtractedSection(
                text=formatted_text,
                page_number=1,
                section_title=Path(filename).stem,
                metadata={"parser_used": "structured_text"},
            )
        ]
        chunks = self._chunker.chunk_sections(
            sections=sections,
            document_id=document_id,
            filename=filename,
            base_metadata={"parser_used": "structured_text", "source_type": source_type},
        )
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            source_type=source_type,
            chunks=chunks,
            metadata={
                "parser_used": "structured_text",
                "total_pages": 1,
                "total_chunks": len(chunks),
                "quality_score": compute_text_quality_score(formatted_text),
            },
        )

    @staticmethod
    def _read_text_sync(file_path: str) -> str:
        raw_bytes = Path(file_path).read_bytes()
        if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
            try:
                return raw_bytes.decode("utf-16")
            except UnicodeDecodeError:
                pass
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return raw_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="replace")

    @staticmethod
    def _format_content(raw_text: str, ext: str, filename: str) -> str:
        if ext == ".json":
            try:
                parsed = json.loads(raw_text)
                pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                return f"# JSON Document: {filename}\n\n```json\n{pretty}\n```"
            except Exception:
                return raw_text

        if ext == ".csv":
            try:
                reader = list(csv.reader(io.StringIO(raw_text)))
                if len(reader) >= 2:
                    headers = [h.strip() or f"col_{i}" for i, h in enumerate(reader[0])]
                    total_data_rows = len(reader) - 1
                    row_cap = 500
                    lines = [
                        f"# CSV Table: {filename}",
                        f"Total rows: {total_data_rows} | Columns ({len(headers)}): {', '.join(headers)}",
                        "",
                        "| " + " | ".join(headers) + " |",
                        "| " + " | ".join(["---"] * len(headers)) + " |",
                    ]
                    for row in reader[1 : 1 + row_cap]:
                        padded = (row + [""] * len(headers))[: len(headers)]
                        escaped = [cell.replace("|", "\\|").replace("\n", " ").strip() for cell in padded]
                        lines.append("| " + " | ".join(escaped) + " |")
                    if total_data_rows > row_cap:
                        lines.append(
                            f"\n_(Indexed first {row_cap} of {total_data_rows} rows for vector search; full statistical profiling and SQL/graph analytics remain available.)_"
                        )
                    return "\n".join(lines)
            except Exception:
                pass

        return raw_text


class SmartDocumentRouter(DocumentParserPort):
    """Inspects uploaded files and routes across PyPDF, PyMuPDF, Docling, PaddleOCR, and StructuredText."""

    def __init__(
        self,
        chunker: ConfigurableChunker | None = None,
        pypdf_parser: PyPDFParser | None = None,
        pymupdf_parser: PyMuPDFParser | None = None,
        docling_parser: DoclingDocumentParser | None = None,
        ocr_parser: PaddleOCRParser | None = None,
        text_parser: StructuredTextParser | None = None,
    ) -> None:
        self._chunker = chunker or ConfigurableChunker()
        self._pypdf = pypdf_parser or PyPDFParser(chunker=self._chunker)
        self._pymupdf = pymupdf_parser or PyMuPDFParser(chunker=self._chunker)
        self._docling = docling_parser or DoclingDocumentParser(chunker=self._chunker, enable_pdf_ocr=False)
        self._ocr = ocr_parser or PaddleOCRParser(chunker=self._chunker)
        self._text = text_parser or StructuredTextParser(chunker=self._chunker)

    def supported_types(self) -> list[str]:
        return sorted(SUPPORTED_DOCUMENT_EXTENSIONS)

    async def parse(
        self,
        file_path: str,
        document_id: UUID,
        filename: str,
        source_type: str,
    ) -> ParsedDocument:
        """Inspect file, execute optimal parser, and apply selective OCR / fallback if needed."""
        report: FileInspectionReport = await asyncio.to_thread(
            validate_and_inspect_file,
            file_path,
            filename,
        )
        logger.info(
            "DocumentRouter selected '%s' for %s (%s)",
            report.recommended_parser,
            filename,
            report.reason,
        )

        attempted_parsers: list[str] = []
        parsed_doc: ParsedDocument | None = None

        if report.recommended_parser == "structured_text":
            attempted_parsers.append("structured_text")
            parsed_doc = await self._text.parse(file_path, document_id, filename, source_type)

        elif report.recommended_parser == "paddleocr":
            attempted_parsers.append("paddleocr")
            parsed_doc = await self._ocr.parse(file_path, document_id, filename, source_type)

        elif report.recommended_parser == "docling":
            parsed_doc = await self._run_docling_with_fallback(
                file_path, document_id, filename, source_type, report, attempted_parsers
            )

        elif report.recommended_parser == "pymupdf":
            parsed_doc = await self._run_pymupdf_with_fallback(
                file_path, document_id, filename, source_type, report, attempted_parsers
            )

        else:
            # Primary: PyPDF with selective page-level PaddleOCR + PyMuPDF/Docling/PaddleOCR fallback
            parsed_doc = await self._run_pypdf_with_fallback(
                file_path, document_id, filename, source_type, report, attempted_parsers
            )

        if not parsed_doc or not parsed_doc.chunks:
            raise ValueError(
                f"No extractable text found in '{filename}' after parsers {attempted_parsers}."
            )

        # Enrich document and chunk metadata with inspection & routing lineage
        parsed_doc.metadata.update(
            {
                "content_hash": report.content_hash,
                "file_size_bytes": report.file_size_bytes,
                "routing_reason": report.reason,
                "recommended_parser": report.recommended_parser,
                "attempted_parsers": attempted_parsers,
            }
        )
        for chunk in parsed_doc.chunks:
            chunk.metadata["file_content_hash"] = report.content_hash
            chunk.metadata.setdefault("content_hash", report.content_hash)
            chunk.metadata.setdefault("document_id", str(document_id))
            chunk.metadata.setdefault("chunk_id", str(chunk.chunk_id))
            chunk.metadata.setdefault("chunk_index", chunk.index)
            chunk.metadata.setdefault("page_number", chunk.page_number)
            chunk.metadata.setdefault("section_title", chunk.section_title)
            chunk.metadata.setdefault("filename", filename)
            chunk.metadata.setdefault("source_type", source_type)

        return parsed_doc

    async def _run_pypdf_with_fallback(
        self,
        file_path: str,
        document_id: UUID,
        filename: str,
        source_type: str,
        report: FileInspectionReport,
        attempted_parsers: list[str],
    ) -> ParsedDocument:
        settings = get_settings()
        attempted_parsers.append("pypdf")
        extracted = await self._pypdf.extract_pages(file_path, settings.OCR_MIN_CHARS_PER_PAGE)

        sections = list(extracted.sections)
        ocr_enriched_pages: list[int] = []

        # Selective per-page OCR for hybrid PDFs (mostly digital text, a few scanned/diagram pages)
        if extracted.low_density_pages and len(extracted.low_density_pages) <= max(5, extracted.total_pages // 2):
            try:
                attempted_parsers.append("paddleocr_selective_pages")
                ocr_sections, _ = await self._ocr.ocr_pdf_pages(file_path, extracted.low_density_pages)
                if ocr_sections:
                    ocr_by_page = {s.page_number: s for s in ocr_sections if s.page_number is not None}
                    ocr_enriched_pages = sorted(ocr_by_page.keys())
                    existing_pages = {s.page_number for s in sections}
                    sections = [
                        ocr_by_page.get(s.page_number, s) if s.page_number in ocr_by_page else s
                        for s in sections
                    ]
                    for pno, osec in ocr_by_page.items():
                        if pno not in existing_pages:
                            sections.append(osec)
                    sections.sort(key=lambda s: s.page_number or 0)
            except Exception as exc:
                logger.warning("Selective page OCR warning on %s: %s", filename, exc)

        combined_text = "\n\n".join(s.text for s in sections)
        quality = compute_text_quality_score(combined_text)

        # Fallback to PyMuPDF first (fast block/font recovery), then Docling/PaddleOCR if still degraded
        if quality < 0.45 or len(combined_text.strip()) < settings.OCR_MIN_CHARS_PER_PAGE:
            logger.info(
                "PyPDF extraction quality %.2f low for %s; falling back to PyMuPDF -> Docling -> PaddleOCR",
                quality,
                filename,
            )
            return await self._run_pymupdf_with_fallback(
                file_path, document_id, filename, source_type, report, attempted_parsers
            )

        parser_label = "pypdf+paddleocr" if ocr_enriched_pages else "pypdf"
        chunks = self._chunker.chunk_sections(
            sections=sections,
            document_id=document_id,
            filename=filename,
            base_metadata={"parser_used": parser_label, "source_type": source_type},
        )
        meta: dict[str, Any] = {
            **extracted.pdf_metadata,
            "parser_used": parser_label,
            "total_pages": extracted.total_pages,
            "total_chunks": len(chunks),
            "quality_score": quality,
            "ocr_pages_processed": ocr_enriched_pages,
        }
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            source_type=source_type,
            chunks=chunks,
            metadata=meta,
        )

    async def _run_pymupdf_with_fallback(
        self,
        file_path: str,
        document_id: UUID,
        filename: str,
        source_type: str,
        report: FileInspectionReport,
        attempted_parsers: list[str],
    ) -> ParsedDocument:
        settings = get_settings()
        attempted_parsers.append("pymupdf")
        try:
            extracted = await self._pymupdf.extract_pages(file_path, settings.OCR_MIN_CHARS_PER_PAGE)
            sections = list(extracted.sections)
            ocr_enriched_pages: list[int] = []

            if extracted.low_density_pages and len(extracted.low_density_pages) <= max(5, extracted.total_pages // 2):
                try:
                    attempted_parsers.append("paddleocr_selective_pages")
                    ocr_sections, _ = await self._ocr.ocr_pdf_pages(file_path, extracted.low_density_pages)
                    if ocr_sections:
                        ocr_by_page = {s.page_number: s for s in ocr_sections if s.page_number is not None}
                        ocr_enriched_pages = sorted(ocr_by_page.keys())
                        existing_pages = {s.page_number for s in sections}
                        sections = [
                            ocr_by_page.get(s.page_number, s) if s.page_number in ocr_by_page else s
                            for s in sections
                        ]
                        for pno, osec in ocr_by_page.items():
                            if pno not in existing_pages:
                                sections.append(osec)
                        sections.sort(key=lambda s: s.page_number or 0)
                except Exception as exc:
                    logger.warning("PyMuPDF selective page OCR warning on %s: %s", filename, exc)

            combined_text = "\n\n".join(s.text for s in sections)
            quality = compute_text_quality_score(combined_text)
            if quality >= 0.45 and len(combined_text.strip()) >= settings.OCR_MIN_CHARS_PER_PAGE:
                parser_label = "pymupdf+paddleocr" if ocr_enriched_pages else "pymupdf"
                chunks = self._chunker.chunk_sections(
                    sections=sections,
                    document_id=document_id,
                    filename=filename,
                    base_metadata={"parser_used": parser_label, "source_type": source_type},
                )
                return ParsedDocument(
                    document_id=document_id,
                    filename=filename,
                    source_type=source_type,
                    chunks=chunks,
                    metadata={
                        **extracted.pdf_metadata,
                        "parser_used": parser_label,
                        "total_pages": extracted.total_pages,
                        "total_chunks": len(chunks),
                        "quality_score": quality,
                        "ocr_pages_processed": ocr_enriched_pages,
                    },
                )
        except Exception as exc:
            logger.warning("PyMuPDF failed on %s (%s); escalating to Docling/PaddleOCR", filename, exc)

        return await self._run_docling_with_fallback(
            file_path, document_id, filename, source_type, report, attempted_parsers
        )

    async def _run_docling_with_fallback(
        self,
        file_path: str,
        document_id: UUID,
        filename: str,
        source_type: str,
        report: FileInspectionReport,
        attempted_parsers: list[str],
    ) -> ParsedDocument:
        settings = get_settings()
        attempted_parsers.append("docling")
        docling_err: Exception | None = None

        try:
            doc = await self._docling.parse(file_path, document_id, filename, source_type)
            total_chars = sum(len(c.text) for c in doc.chunks)
            quality = float(doc.metadata.get("quality_score", 0.0))
            if doc.chunks and total_chars >= settings.OCR_MIN_CHARS_PER_PAGE and quality >= 0.35:
                return doc
            logger.info(
                "Docling extracted sparse/low-quality content for %s (chars=%d, quality=%.2f); escalating fallback",
                filename,
                total_chars,
                quality,
            )
        except Exception as exc:
            docling_err = exc
            logger.warning("Docling failed on %s (%s); attempting fallback parser", filename, exc)

        # Fallback for PDFs: try PyMuPDF (if not yet attempted), PaddleOCR for scanned pages, or PyPDF fallback
        if report.extension == ".pdf":
            if "pymupdf" not in attempted_parsers:
                try:
                    attempted_parsers.append("pymupdf_fallback")
                    mupdf_doc = await self._pymupdf.parse(file_path, document_id, filename, source_type)
                    if mupdf_doc.chunks and sum(len(c.text) for c in mupdf_doc.chunks) >= settings.OCR_MIN_CHARS_PER_PAGE:
                        return mupdf_doc
                except Exception as mupdf_exc:
                    logger.warning("PyMuPDF fallback failed on %s: %s", filename, mupdf_exc)

            try:
                attempted_parsers.append("paddleocr")
                ocr_doc = await self._ocr.parse(file_path, document_id, filename, source_type)
                if ocr_doc.chunks:
                    return ocr_doc
            except Exception as ocr_exc:
                logger.warning("PaddleOCR fallback failed on %s: %s", filename, ocr_exc)

            attempted_parsers.append("pypdf_fallback")
            pypdf_doc = await self._pypdf.parse(file_path, document_id, filename, source_type)
            if pypdf_doc.chunks:
                return pypdf_doc

        if report.extension in (".txt", ".md", ".json", ".csv"):
            attempted_parsers.append("structured_text_fallback")
            try:
                return await self._text.parse(file_path, document_id, filename, source_type)
            except Exception as txt_exc:
                logger.warning("StructuredText fallback failed on %s: %s", filename, txt_exc)

        if docling_err is not None:
            raise RuntimeError(f"Docling failed to parse '{filename}': {docling_err}") from docling_err
        raise ValueError(f"Unable to extract text from '{filename}' via {attempted_parsers}")

