"""PyMuPDF (`pymupdf` / `fitz`) high-speed layout- and block-preserving PDF parser.

Extracts per-page text blocks sorted in natural reading order (`(y0, x0)`),
infers headings from font-size spans relative to body text, detects embedded
tables (`page.find_tables()` when available), and flags low-density scanned
pages for selective PaddleOCR enrichment.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore[no-redef]

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.ports.document_parser import DocumentParserPort, ParsedDocument
from nexusai.infrastructure.parsers.chunker import (
    ConfigurableChunker,
    ExtractedSection,
    normalize_extracted_text,
)
from nexusai.infrastructure.parsers.file_inspector import compute_text_quality_score

logger = get_logger(__name__)

_HEADING_PATTERN = re.compile(
    r"^(?:(?:Chapter|Section|Appendix|Part)\s+\w+[:.\s-]*.*|\d+(?:\.\d+){0,3}\s+[A-Z][^\n]{2,90}|[A-Z][A-Z0-9\s,&/:-]{4,70})$"
)


@dataclass
class PyMuPDFExtractionResult:
    """Page-level PyMuPDF extraction result prior to chunking."""

    sections: list[ExtractedSection]
    low_density_pages: list[int]
    total_pages: int
    pdf_metadata: dict[str, Any]
    quality_score: float


class PyMuPDFParser(DocumentParserPort):
    """High-accuracy, ultra-low-latency PDF extractor powered by `pymupdf`."""

    def __init__(self, chunker: ConfigurableChunker | None = None) -> None:
        self._chunker = chunker or ConfigurableChunker()

    def supported_types(self) -> list[str]:
        return [".pdf"]

    async def extract_pages(
        self,
        file_path: str,
        min_chars_per_page: int | None = None,
    ) -> PyMuPDFExtractionResult:
        """Extract layout-ordered page text, tables, and headings via PyMuPDF."""
        settings = get_settings()
        threshold = (
            min_chars_per_page
            if min_chars_per_page is not None
            else settings.OCR_MIN_CHARS_PER_PAGE
        )
        return await asyncio.to_thread(self._extract_pages_sync, file_path, threshold)

    def _extract_pages_sync(
        self,
        file_path: str,
        min_chars_per_page: int,
    ) -> PyMuPDFExtractionResult:
        doc = pymupdf.open(file_path)
        try:
            total_pages = len(doc)
            raw_meta = doc.metadata or {}
            pdf_meta: dict[str, Any] = {}
            for out_k, src_k in (
                ("pdf_title", "title"),
                ("pdf_author", "author"),
                ("pdf_subject", "subject"),
            ):
                val = raw_meta.get(src_k)
                if val and str(val).strip():
                    pdf_meta[out_k] = str(val).strip()

            sections: list[ExtractedSection] = []
            low_density_pages: list[int] = []
            all_texts: list[str] = []
            current_heading: str | None = pdf_meta.get("pdf_title")

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_no = page_idx + 1

                page_md, inferred_heading, has_table = self._extract_single_page(page)
                cleaned = normalize_extracted_text(page_md)

                if len(cleaned) < min_chars_per_page:
                    low_density_pages.append(page_no)

                if cleaned:
                    if inferred_heading:
                        current_heading = inferred_heading
                    all_texts.append(cleaned)
                    sections.append(
                        ExtractedSection(
                            text=cleaned,
                            page_number=page_no,
                            section_title=current_heading,
                            metadata={
                                "parser_used": "pymupdf",
                                "contains_table": has_table,
                            },
                        )
                    )

            quality = (
                compute_text_quality_score("\n\n".join(all_texts)) if all_texts else 0.0
            )
            return PyMuPDFExtractionResult(
                sections=sections,
                low_density_pages=low_density_pages,
                total_pages=total_pages,
                pdf_metadata=pdf_meta,
                quality_score=quality,
            )
        finally:
            doc.close()

    @staticmethod
    def _extract_single_page(page: Any) -> tuple[str, str | None, bool]:
        """Extract reading-order blocks, detect font-size headings, and render tables."""
        has_table = False
        table_markdown_blocks: list[str] = []

        find_tables_fn = getattr(page, "find_tables", None)
        if callable(find_tables_fn):
            try:
                finder = find_tables_fn()
                tables = getattr(finder, "tables", None) or []
                for tbl in tables[:6]:
                    md = tbl.to_markdown() if hasattr(tbl, "to_markdown") else ""
                    if md and "|" in md and len(md.strip()) >= 20:
                        has_table = True
                        table_markdown_blocks.append(md.strip())
            except Exception:
                pass

        # Extract span-level dictionary to detect font sizes for accurate heading promotion
        inferred_heading: str | None = None
        rendered_lines: list[str] = []

        try:
            page_dict = page.get_text("dict", sort=True)
            blocks = page_dict.get("blocks", [])
            span_sizes: list[float] = []
            for blk in blocks:
                if blk.get("type") != 0:
                    continue
                for ln in blk.get("lines", []):
                    for sp in ln.get("spans", []):
                        txt = (sp.get("text") or "").strip()
                        sz = float(sp.get("size") or 0.0)
                        if txt and sz > 0:
                            span_sizes.append(sz)

            median_size = 11.0
            if span_sizes:
                sorted_sizes = sorted(span_sizes)
                median_size = sorted_sizes[len(sorted_sizes) // 2]

            for blk in blocks:
                if blk.get("type") != 0:
                    continue
                block_lines: list[str] = []
                for ln in blk.get("lines", []):
                    spans = ln.get("spans", [])
                    line_text = "".join(sp.get("text") or "" for sp in spans).strip()
                    if not line_text:
                        continue
                    max_span_sz = max(
                        (float(sp.get("size") or 0.0) for sp in spans),
                        default=median_size,
                    )
                    is_heading = (
                        len(line_text) <= 95
                        and not line_text.isdigit()
                        and (
                            max_span_sz >= median_size * 1.22
                            or bool(_HEADING_PATTERN.match(line_text))
                        )
                    )
                    if is_heading:
                        if inferred_heading is None:
                            inferred_heading = line_text
                        if not line_text.startswith("#"):
                            block_lines.append(f"## {line_text}")
                        else:
                            block_lines.append(line_text)
                    else:
                        block_lines.append(line_text)
                if block_lines:
                    rendered_lines.append("\n".join(block_lines))
        except Exception:
            fallback_txt = (page.get_text("text", sort=True) or "").strip()
            if fallback_txt:
                rendered_lines.append(fallback_txt)

        if table_markdown_blocks:
            rendered_lines.extend(table_markdown_blocks)

        return "\n\n".join(rendered_lines), inferred_heading, has_table

    async def parse(
        self,
        file_path: str,
        document_id: UUID,
        filename: str,
        source_type: str,
    ) -> ParsedDocument:
        """Parse a PDF with PyMuPDF into deterministic structure-aware chunks."""
        logger.info("Parsing %s with PyMuPDFParser", filename)
        extracted = await self.extract_pages(file_path)

        doc_meta = {
            **extracted.pdf_metadata,
            "parser_used": "pymupdf",
            "total_pages": extracted.total_pages,
            "low_density_pages": extracted.low_density_pages,
            "quality_score": extracted.quality_score,
        }
        chunks = self._chunker.chunk_sections(
            sections=extracted.sections,
            document_id=document_id,
            filename=filename,
            base_metadata={"parser_used": "pymupdf", "source_type": source_type},
        )
        doc_meta["total_chunks"] = len(chunks)

        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            source_type=source_type,
            chunks=chunks,
            metadata=doc_meta,
        )

