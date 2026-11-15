"""PyPDF document parser for clean digital text-based PDFs and lightweight fallback."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pypdf import PdfReader

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.ports.document_parser import DocumentParserPort, ParsedDocument
from nexusai.infrastructure.parsers.chunker import ConfigurableChunker, ExtractedSection, normalize_extracted_text
from nexusai.infrastructure.parsers.file_inspector import compute_text_quality_score

logger = get_logger(__name__)

_SECTION_LINE_RE = re.compile(
    r"^(?:(?:Chapter|Section|Appendix)\s+\w+[:.\s-]*.*|\d+(?:\.\d+){0,3}\s+[A-Z][^\n]{2,80}|[A-Z][A-Z0-9\s,&/-]{4,60})$"
)


@dataclass
class PyPDFExtractionResult:
    """Raw page-level extraction output before chunking (allows selective page OCR enrichment)."""

    sections: list[ExtractedSection]
    low_density_pages: list[int]
    total_pages: int
    pdf_metadata: dict[str, Any]
    quality_score: float


class PyPDFParser(DocumentParserPort):
    """Lightweight page-preserving PDF extractor powered by `pypdf`."""

    def __init__(self, chunker: ConfigurableChunker | None = None) -> None:
        self._chunker = chunker or ConfigurableChunker()

    def supported_types(self) -> list[str]:
        return [".pdf"]

    async def extract_pages(
        self,
        file_path: str,
        min_chars_per_page: int | None = None,
    ) -> PyPDFExtractionResult:
        """Extract per-page text and identify pages needing OCR enrichment."""
        settings = get_settings()
        threshold = min_chars_per_page if min_chars_per_page is not None else settings.OCR_MIN_CHARS_PER_PAGE
        return await asyncio.to_thread(self._extract_pages_sync, file_path, threshold)

    def _extract_pages_sync(self, file_path: str, min_chars_per_page: int) -> PyPDFExtractionResult:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        raw_meta = reader.metadata or {}
        pdf_meta: dict[str, Any] = {}
        for key, attr in (("pdf_title", "title"), ("pdf_author", "author"), ("pdf_subject", "subject")):
            val = getattr(raw_meta, attr, None)
            if val:
                pdf_meta[key] = str(val).strip()

        sections: list[ExtractedSection] = []
        low_density_pages: list[int] = []
        all_texts: list[str] = []
        last_heading: str | None = pdf_meta.get("pdf_title")

        for idx, page in enumerate(reader.pages):
            page_no = idx + 1
            raw_text = page.extract_text() or ""
            cleaned = normalize_extracted_text(raw_text)

            if len(cleaned) < min_chars_per_page:
                low_density_pages.append(page_no)

            if cleaned:
                detected_heading = self._detect_page_heading(cleaned)
                if detected_heading:
                    last_heading = detected_heading
                all_texts.append(cleaned)
                sections.append(
                    ExtractedSection(
                        text=cleaned,
                        page_number=page_no,
                        section_title=last_heading,
                        metadata={"parser_used": "pypdf"},
                    )
                )

        quality = compute_text_quality_score("\n\n".join(all_texts)) if all_texts else 0.0
        return PyPDFExtractionResult(
            sections=sections,
            low_density_pages=low_density_pages,
            total_pages=total_pages,
            pdf_metadata=pdf_meta,
            quality_score=quality,
        )

    async def parse(
        self,
        file_path: str,
        document_id: UUID,
        filename: str,
        source_type: str,
    ) -> ParsedDocument:
        """Parse a text-based PDF into deterministic chunks preserving page numbers."""
        logger.info("Parsing %s with PyPDFParser", filename)
        extracted = await self.extract_pages(file_path)

        doc_meta = {
            **extracted.pdf_metadata,
            "parser_used": "pypdf",
            "total_pages": extracted.total_pages,
            "low_density_pages": extracted.low_density_pages,
            "quality_score": extracted.quality_score,
        }
        chunks = self._chunker.chunk_sections(
            sections=extracted.sections,
            document_id=document_id,
            filename=filename,
            base_metadata={"parser_used": "pypdf", "source_type": source_type},
        )
        doc_meta["total_chunks"] = len(chunks)

        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            source_type=source_type,
            chunks=chunks,
            metadata=doc_meta,
        )

    @staticmethod
    def _detect_page_heading(page_text: str) -> str | None:
        """Infer top-level section title from the first few non-empty lines of a page."""
        for line in page_text.split("\n")[:6]:
            candidate = line.strip()
            if not candidate or candidate.isdigit() or len(candidate) > 90:
                continue
            if _SECTION_LINE_RE.match(candidate):
                return candidate
        return None

