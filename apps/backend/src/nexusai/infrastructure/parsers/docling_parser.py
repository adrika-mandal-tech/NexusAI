"""Docling structured document parser implementation.

Handles layout-sensitive PDFs, DOCX, PPTX, XLSX, and HTML documents with table
structure preservation, async threadpool execution, and deterministic chunk IDs.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from nexusai.core.ids import deterministic_chunk_id
from nexusai.core.logging import get_logger
from nexusai.domain.ports.document_parser import DocumentParserPort, ParsedChunk, ParsedDocument
from nexusai.infrastructure.parsers.chunker import ConfigurableChunker, ExtractedSection, estimate_token_count
from nexusai.infrastructure.parsers.file_inspector import compute_text_quality_score

logger = get_logger(__name__)

_DOCLING_LOCK = threading.Lock()
_CONVERTER_NO_OCR: DocumentConverter | None = None
_CONVERTER_WITH_OCR: DocumentConverter | None = None


def _get_docling_converter(do_ocr: bool = False) -> DocumentConverter:
    """Lazily initialize singleton Docling DocumentConverter."""
    global _CONVERTER_NO_OCR, _CONVERTER_WITH_OCR
    if not do_ocr and _CONVERTER_NO_OCR is not None:
        return _CONVERTER_NO_OCR
    if do_ocr and _CONVERTER_WITH_OCR is not None:
        return _CONVERTER_WITH_OCR

    with _DOCLING_LOCK:
        if not do_ocr and _CONVERTER_NO_OCR is None:
            _CONVERTER_NO_OCR = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=PdfPipelineOptions(
                            do_ocr=False,
                            do_table_structure=True,
                        )
                    )
                }
            )
        elif do_ocr and _CONVERTER_WITH_OCR is None:
            _CONVERTER_WITH_OCR = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=PdfPipelineOptions(
                            do_ocr=True,
                            do_table_structure=True,
                        )
                    )
                }
            )
    return _CONVERTER_WITH_OCR if do_ocr else _CONVERTER_NO_OCR  # type: ignore[return-value]


class DoclingDocumentParser(DocumentParserPort):
    """Docling-based structured document parser with layout & table preservation."""

    def __init__(
        self,
        chunker: ConfigurableChunker | None = None,
        enable_pdf_ocr: bool = False,
    ) -> None:
        self._chunker = chunker or ConfigurableChunker()
        self._enable_pdf_ocr = enable_pdf_ocr

    def supported_types(self) -> list[str]:
        return [".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".csv", ".md", ".txt", ".html", ".htm"]

    async def parse(
        self,
        file_path: str,
        document_id: uuid.UUID,
        filename: str,
        source_type: str,
    ) -> ParsedDocument:
        """Parse document using Docling off the event loop and build deterministic chunks."""
        logger.info("Parsing document %s with Docling (pdf_ocr=%s)", filename, self._enable_pdf_ocr)
        return await asyncio.to_thread(
            self._parse_sync,
            file_path,
            document_id,
            filename,
            source_type,
        )

    def _parse_sync(
        self,
        file_path: str,
        document_id: uuid.UUID,
        filename: str,
        source_type: str,
    ) -> ParsedDocument:
        converter = _get_docling_converter(do_ocr=self._enable_pdf_ocr)
        result = converter.convert(file_path)
        doc = result.document

        markdown_text = doc.export_to_markdown() or ""
        total_pages = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1
        table_count = len(doc.tables) if hasattr(doc, "tables") and doc.tables else 0

        chunks = self._build_chunks_from_doc(doc, markdown_text, document_id, filename, source_type)
        quality = compute_text_quality_score(markdown_text)

        metadata = {
            "parser_used": "docling",
            "total_pages": total_pages,
            "tables_extracted": table_count,
            "total_chunks": len(chunks),
            "quality_score": quality,
        }

        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            source_type=source_type,
            chunks=chunks,
            metadata=metadata,
        )

    def _build_chunks_from_doc(
        self,
        doc: Any,
        markdown_text: str,
        document_id: uuid.UUID,
        filename: str,
        source_type: str,
    ) -> list[ParsedChunk]:
        """Extract page- and heading-aware chunks using Docling items + ConfigurableChunker."""
        # When non-default chunking strategy (e.g. parent_child or token_aware) is active,
        # extract structured sections from Docling items and pass through ConfigurableChunker.
        sections = self._extract_structured_sections(doc, markdown_text)
        if self._chunker.strategy in ("parent_child", "token_aware", "recursive", "table_aware"):
            return self._chunker.chunk_sections(
                sections=sections,
                document_id=document_id,
                filename=filename,
                base_metadata={"parser_used": "docling", "source_type": source_type},
            )

        # For structure_aware, try Docling's HybridChunker with offline-safe fallback
        try:
            from docling.chunking import HybridChunker

            hybrid = HybridChunker(
                max_tokens=max(256, self._chunker.chunk_size // 3),
                merge_peers=True,
            )
            doc_chunks = list(hybrid.chunk(doc))
            if doc_chunks:
                from nexusai.core.ids import compute_sha256_bytes

                chunks: list[ParsedChunk] = []
                seen_hashes: set[str] = set()
                for dchunk in doc_chunks:
                    txt = (dchunk.text or "").strip()
                    if not txt:
                        continue
                    c_hash = compute_sha256_bytes(txt.encode("utf-8"))
                    if c_hash in seen_hashes:
                        continue
                    seen_hashes.add(c_hash)
                    c_idx = len(chunks)
                    cid = deterministic_chunk_id(document_id, c_idx, txt)
                    page_num = self._get_chunk_page(dchunk)
                    section = self._get_chunk_section(dchunk)
                    chunks.append(
                        ParsedChunk(
                            chunk_id=cid,
                            document_id=document_id,
                            index=c_idx,
                            text=txt,
                            page_number=page_num,
                            section_title=section,
                            metadata={
                                "document_id": str(document_id),
                                "chunk_id": str(cid),
                                "chunk_index": c_idx,
                                "filename": filename,
                                "page_number": page_num,
                                "section_title": section,
                                "content_hash": c_hash,
                                "parser_used": "docling",
                                "source_type": source_type,
                                "chunk_strategy": "docling_hybrid",
                                "token_count": estimate_token_count(txt),
                                "char_count": len(txt),
                            },
                        )
                    )
                if chunks:
                    return chunks
        except Exception as exc:
            logger.debug("Docling HybridChunker unavailable/skipped (%s); using ConfigurableChunker", exc)

        return self._chunker.chunk_sections(
            sections=sections,
            document_id=document_id,
            filename=filename,
            base_metadata={"parser_used": "docling", "source_type": source_type},
        )

    def _extract_structured_sections(self, doc: Any, markdown_text: str) -> list[ExtractedSection]:
        """Group Docling document items by page & heading while preserving Markdown tables."""
        sections: list[ExtractedSection] = []
        try:
            if hasattr(doc, "iterate_items"):
                current_page: int | None = None
                current_heading: str | None = None
                buf: list[str] = []

                def flush() -> None:
                    nonlocal buf
                    if buf:
                        joined = "\n\n".join(buf).strip()
                        if joined:
                            sections.append(
                                ExtractedSection(
                                    text=joined,
                                    page_number=current_page,
                                    section_title=current_heading,
                                    metadata={"parser_used": "docling"},
                                )
                            )
                        buf = []

                for item, _level in doc.iterate_items():
                    page_no = None
                    if hasattr(item, "prov") and item.prov:
                        page_no = getattr(item.prov[0], "page_no", None)

                    label_str = str(getattr(item, "label", "")).lower()
                    if "section_header" in label_str or "title" in label_str:
                        heading_txt = (getattr(item, "text", "") or "").strip()
                        if heading_txt:
                            flush()
                            current_heading = heading_txt
                            if page_no is not None:
                                current_page = page_no
                            buf.append(f"## {heading_txt}")
                            continue

                    if page_no is not None and current_page is not None and page_no != current_page:
                        flush()
                    if page_no is not None:
                        current_page = page_no

                    if hasattr(item, "export_to_markdown"):
                        try:
                            tbl_md = item.export_to_markdown(doc=doc).strip()
                        except TypeError:
                            tbl_md = item.export_to_markdown().strip()
                        if tbl_md:
                            buf.append(tbl_md)
                            continue

                    item_txt = (getattr(item, "text", "") or "").strip()
                    if item_txt:
                        buf.append(item_txt)

                flush()
        except Exception as exc:
            logger.debug("iterate_items fallback to full markdown (%s)", exc)

        if not sections and markdown_text.strip():
            sections.append(
                ExtractedSection(
                    text=markdown_text,
                    page_number=1,
                    section_title=None,
                    metadata={"parser_used": "docling"},
                )
            )
        return sections

    @staticmethod
    def _get_chunk_page(chunk: Any) -> int | None:
        try:
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        return item.prov[0].page_no
        except Exception:
            pass
        return None

    @staticmethod
    def _get_chunk_section(chunk: Any) -> str | None:
        try:
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "headings"):
                headings = chunk.meta.headings
                if headings:
                    return " > ".join(headings)
        except Exception:
            pass
        return None
