"""File validation, magic-byte inspection, and document quality detection.

Inspects uploaded files before selecting PyPDF, Docling, PaddleOCR, or
lightweight structured text parsing so NexusAI avoids running every parser on
every document while catching scanned pages and table-heavy layouts accurately.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nexusai.core.config import get_settings
from nexusai.core.ids import compute_sha256_file
from nexusai.core.logging import get_logger

logger = get_logger(__name__)

ParserChoice = Literal["pypdf", "pymupdf", "docling", "paddleocr", "structured_text"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
OFFICE_OR_HTML_EXTENSIONS = {".docx", ".doc", ".pptx", ".xlsx", ".html", ".htm"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
SUPPORTED_DOCUMENT_EXTENSIONS = (
    {".pdf"} | IMAGE_EXTENSIONS | OFFICE_OR_HTML_EXTENSIONS | TEXT_EXTENSIONS
)

# Per-document-type upload size limits (bytes), capped by global DOC_MAX_UPLOAD_BYTES
DOCUMENT_TYPE_MAX_BYTES: dict[str, int] = {
    ".pdf": 100 * 1024 * 1024,   # 100 MB
    ".csv": 100 * 1024 * 1024,   # 100 MB
    ".docx": 50 * 1024 * 1024,   # 50 MB
    ".doc": 50 * 1024 * 1024,    # 50 MB
    ".pptx": 50 * 1024 * 1024,   # 50 MB
    ".xlsx": 100 * 1024 * 1024,  # 100 MB
    ".html": 25 * 1024 * 1024,   # 25 MB
    ".htm": 25 * 1024 * 1024,    # 25 MB
    ".txt": 25 * 1024 * 1024,    # 25 MB
    ".md": 25 * 1024 * 1024,     # 25 MB
    ".json": 25 * 1024 * 1024,   # 25 MB
    ".png": 25 * 1024 * 1024,    # 25 MB
    ".jpg": 25 * 1024 * 1024,    # 25 MB
    ".jpeg": 25 * 1024 * 1024,   # 25 MB
    ".tiff": 25 * 1024 * 1024,   # 25 MB
    ".tif": 25 * 1024 * 1024,    # 25 MB
    ".bmp": 25 * 1024 * 1024,    # 25 MB
    ".webp": 25 * 1024 * 1024,   # 25 MB
}


def get_max_upload_bytes_for_extension(ext: str, global_max_bytes: int | None = None) -> int:
    """Return enforced upload size limit in bytes for a specific file extension."""
    settings = get_settings()
    cap = global_max_bytes if global_max_bytes is not None else settings.DOC_MAX_UPLOAD_BYTES
    per_type = DOCUMENT_TYPE_MAX_BYTES.get(ext.lower(), cap)
    return min(cap, per_type) if global_max_bytes is None else global_max_bytes

_MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".docx": (b"PK\x03\x04",),
    ".pptx": (b"PK\x03\x04",),
    ".xlsx": (b"PK\x03\x04",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".bmp": (b"BM",),
    ".webp": (b"RIFF",),
}

_DANGEROUS_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x7fELF", "ELF executable"),
    (b"MZ", "PE/Windows executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O binary"),
    (b"\xfe\xed\xfa\xcf", "Mach-O binary"),
)

_TABLE_LINE_RE = re.compile(r"(?:\|.+\|.+\|)|(?:\S+\s{3,}\S+\s{3,}\S+)")
_CID_GIBBERISH_RE = re.compile(r"\(cid:\d+\)|/uni[0-9A-Fa-f]{4}")


class FileValidationError(ValueError):
    """Raised when an uploaded file fails security, type, or size validation."""


@dataclass
class FileInspectionReport:
    """Structured inspection result driving the Document Processing Router."""

    file_path: str
    filename: str
    extension: str
    file_size_bytes: int
    content_hash: str
    recommended_parser: ParserChoice
    reason: str
    page_count: int = 1
    avg_chars_per_page: float = 0.0
    text_page_ratio: float = 1.0
    scanned_page_numbers: list[int] = field(default_factory=list)
    has_tables_or_complex_layout: bool = False
    estimated_quality: float = 1.0
    max_allowed_bytes: int = 100 * 1024 * 1024


def compute_text_quality_score(text: str) -> float:
    """Score extracted text quality in [0.0, 1.0] to trigger fallback parsers."""
    stripped = (text or "").strip()
    if not stripped:
        return 0.0

    total_len = len(stripped)
    printable = sum(1 for ch in stripped if ch.isprintable() or ch in "\n\r\t")
    printable_ratio = printable / max(1, total_len)

    alpha_num = sum(1 for ch in stripped if ch.isalnum())
    alnum_ratio = alpha_num / max(1, total_len)

    cid_matches = len(_CID_GIBBERISH_RE.findall(stripped))
    cid_penalty = min(0.6, cid_matches * 0.05)

    words = stripped.split()
    avg_word_len = sum(len(w) for w in words) / max(1, len(words))
    word_len_penalty = 0.25 if (avg_word_len < 1.8 or avg_word_len > 28.0) else 0.0

    raw_score = (0.5 * printable_ratio) + (0.5 * min(1.0, alnum_ratio / 0.55))
    return round(max(0.0, min(1.0, raw_score - cid_penalty - word_len_penalty)), 3)


def validate_and_inspect_file(
    file_path: str | Path,
    filename: str | None = None,
    max_bytes: int | None = None,
    prefer_structured_pdf: bool = False,
    prefer_pymupdf: bool = False,
) -> FileInspectionReport:
    """Validate file safety/type and determine optimal primary parser."""
    settings = get_settings()
    path = Path(file_path)
    display_name = filename or path.name

    if not path.exists() or not path.is_file():
        raise FileValidationError(f"Uploaded file does not exist: {display_name}")

    size = os.path.getsize(path)
    if size == 0:
        raise FileValidationError(f"Uploaded file '{display_name}' is empty (0 bytes).")

    ext = Path(display_name).suffix.lower() or path.suffix.lower()
    if ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file extension '{ext}'. Allowed: {sorted(SUPPORTED_DOCUMENT_EXTENSIONS)}"
        )

    limit = get_max_upload_bytes_for_extension(ext, max_bytes)
    if size > limit:
        raise FileValidationError(
            f"Uploaded file '{display_name}' ({size:,} bytes) exceeds maximum allowed size for '{ext}' ({limit:,} bytes)."
        )

    with open(path, "rb") as fh:
        header = fh.read(4096)

    for sig, label in _DANGEROUS_SIGNATURES:
        if header.startswith(sig):
            raise FileValidationError(f"Rejected binary file disguised as '{ext}' ({label}).")

    expected_sigs = _MAGIC_SIGNATURES.get(ext)
    if expected_sigs and not any(header.startswith(sig) for sig in expected_sigs):
        # Allow minor leading whitespace before %PDF- per PDF spec
        if ext == ".pdf" and b"%PDF-" in header[:1024]:
            pass
        else:
            raise FileValidationError(
                f"File '{display_name}' failed magic-byte signature validation for '{ext}'."
            )

    if ext in TEXT_EXTENSIONS:
        is_utf16_bom = header.startswith((b"\xff\xfe", b"\xfe\xff"))
        if header.startswith(b"PK\x03\x04") or (b"\x00" in header and not is_utf16_bom):
            raise FileValidationError(
                f"Text file '{display_name}' contains binary or compressed archive bytes."
            )

    content_hash = compute_sha256_file(path)

    # 1. Image formats -> PaddleOCR directly
    if ext in IMAGE_EXTENSIONS:
        return FileInspectionReport(
            file_path=str(path),
            filename=display_name,
            extension=ext,
            file_size_bytes=size,
            content_hash=content_hash,
            recommended_parser="paddleocr",
            reason=f"Image format ({ext}) requires PaddleOCR text extraction.",
            page_count=1,
            text_page_ratio=0.0,
            scanned_page_numbers=[1],
        )

    # 2. Structured plain text / Markdown / JSON / CSV -> lightweight structured parser
    if ext in TEXT_EXTENSIONS:
        return FileInspectionReport(
            file_path=str(path),
            filename=display_name,
            extension=ext,
            file_size_bytes=size,
            content_hash=content_hash,
            recommended_parser="structured_text",
            reason=f"Plain/structured text format ({ext}) routed to native structured parser.",
            page_count=1,
        )

    # 3. Office & HTML documents -> verify archive integrity for OOXML, then route to Docling
    if ext in {".docx", ".pptx", ".xlsx"}:
        import zipfile

        if not zipfile.is_zipfile(path):
            raise FileValidationError(
                f"Corrupted Office archive '{display_name}' (invalid ZIP structure for '{ext}')."
            )
        try:
            with zipfile.ZipFile(path, "r") as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    raise FileValidationError(
                        f"Corrupted member '{bad_member}' inside Office document '{display_name}'."
                    )
        except FileValidationError:
            raise
        except Exception as exc:
            raise FileValidationError(
                f"Corrupted Office document '{display_name}': {exc}"
            ) from exc

    if ext in OFFICE_OR_HTML_EXTENSIONS:
        return FileInspectionReport(
            file_path=str(path),
            filename=display_name,
            extension=ext,
            file_size_bytes=size,
            content_hash=content_hash,
            recommended_parser="docling",
            reason=f"Structured Office/HTML format ({ext}) routed to Docling.",
            page_count=1,
            has_tables_or_complex_layout=True,
        )

    # 4. PDF inspection: sample pages via pypdf to distinguish clean text PDF,
    #    PyMuPDF block-layout PDF, Docling table-heavy PDF, hybrid PDF, or scanned image PDF.
    return _inspect_pdf(
        path=path,
        display_name=display_name,
        size=size,
        content_hash=content_hash,
        min_chars_per_page=settings.OCR_MIN_CHARS_PER_PAGE,
        prefer_structured_pdf=prefer_structured_pdf,
        prefer_pymupdf=prefer_pymupdf,
        max_allowed_bytes=limit,
    )


def _inspect_pdf(
    path: Path,
    display_name: str,
    size: int,
    content_hash: str,
    min_chars_per_page: int,
    prefer_structured_pdf: bool = False,
    prefer_pymupdf: bool = False,
    max_allowed_bytes: int = 100 * 1024 * 1024,
) -> FileInspectionReport:
    """Fast PDF pre-flight inspection using pypdf without running heavy ML/OCR."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise FileValidationError(f"PDF '{display_name}' contains 0 pages.")

        # Inspect all pages for small/medium PDFs; sample up to 24 evenly spaced pages for huge PDFs
        if total_pages <= 24:
            sample_indices = list(range(total_pages))
        else:
            step = max(1, total_pages // 24)
            sample_indices = list(range(0, total_pages, step))[:24]

        char_counts: list[int] = []
        scanned_pages: list[int] = []
        table_pages = 0
        sample_texts: list[str] = []

        for idx in sample_indices:
            page = reader.pages[idx]
            txt = (page.extract_text() or "").strip()
            char_counts.append(len(txt))
            if len(txt) < min_chars_per_page:
                scanned_pages.append(idx + 1)
            else:
                sample_texts.append(txt[:1500])
                table_lines = len(_TABLE_LINE_RE.findall(txt))
                if table_lines >= 4:
                    table_pages += 1

        sampled_count = max(1, len(sample_indices))
        avg_chars = sum(char_counts) / sampled_count
        text_pages = sum(1 for c in char_counts if c >= min_chars_per_page)
        text_ratio = text_pages / sampled_count
        quality = compute_text_quality_score("\n".join(sample_texts)) if sample_texts else 0.0
        has_tables = (table_pages / sampled_count) >= 0.35

        if text_ratio < 0.35 or avg_chars < min_chars_per_page:
            return FileInspectionReport(
                file_path=str(path),
                filename=display_name,
                extension=".pdf",
                file_size_bytes=size,
                content_hash=content_hash,
                recommended_parser="paddleocr",
                reason=(
                    f"Scanned/image PDF detected (text_page_ratio={text_ratio:.2f}, "
                    f"avg_chars={avg_chars:.0f} < {min_chars_per_page}); routing to PaddleOCR."
                ),
                page_count=total_pages,
                avg_chars_per_page=round(avg_chars, 1),
                text_page_ratio=round(text_ratio, 3),
                scanned_page_numbers=scanned_pages,
                has_tables_or_complex_layout=has_tables,
                estimated_quality=quality,
            )

        if prefer_structured_pdf or quality < 0.50 or (has_tables and total_pages <= 40):
            return FileInspectionReport(
                file_path=str(path),
                filename=display_name,
                extension=".pdf",
                file_size_bytes=size,
                content_hash=content_hash,
                recommended_parser="docling",
                reason=(
                    f"Layout/table-sensitive or low-font-quality PDF "
                    f"(has_tables={has_tables}, quality={quality:.2f}); routing to Docling."
                ),
                page_count=total_pages,
                avg_chars_per_page=round(avg_chars, 1),
                text_page_ratio=round(text_ratio, 3),
                scanned_page_numbers=scanned_pages,
                has_tables_or_complex_layout=True,
                estimated_quality=quality,
            )

        if prefer_pymupdf or (0.50 <= quality < 0.72 and not prefer_structured_pdf):
            return FileInspectionReport(
                file_path=str(path),
                filename=display_name,
                extension=".pdf",
                file_size_bytes=size,
                content_hash=content_hash,
                recommended_parser="pymupdf",
                reason=(
                    f"Multi-block or custom-font digital PDF (pages={total_pages}, "
                    f"quality={quality:.2f}); routing to PyMuPDF block/table extractor."
                ),
                page_count=total_pages,
                avg_chars_per_page=round(avg_chars, 1),
                text_page_ratio=round(text_ratio, 3),
                scanned_page_numbers=scanned_pages,
                has_tables_or_complex_layout=has_tables,
                estimated_quality=quality,
                max_allowed_bytes=max_allowed_bytes,
            )

        return FileInspectionReport(
            file_path=str(path),
            filename=display_name,
            extension=".pdf",
            file_size_bytes=size,
            content_hash=content_hash,
            recommended_parser="pypdf",
            reason=(
                f"Clean digital text PDF (pages={total_pages}, text_page_ratio={text_ratio:.2f}, "
                f"quality={quality:.2f}); routing to PyPDF with PyMuPDF/PaddleOCR fallback."
            ),
            page_count=total_pages,
            avg_chars_per_page=round(avg_chars, 1),
            text_page_ratio=round(text_ratio, 3),
            scanned_page_numbers=scanned_pages,
            has_tables_or_complex_layout=has_tables,
            estimated_quality=quality,
            max_allowed_bytes=max_allowed_bytes,
        )

    except FileValidationError:
        raise
    except Exception as exc:
        from pypdf.errors import PdfReadError, PdfStreamError

        if isinstance(exc, (PdfReadError, PdfStreamError)):
            raise FileValidationError(
                f"Corrupted or unreadable PDF '{display_name}': {exc}"
            ) from exc
        logger.warning("PDF pre-inspection warning on %s (%s); defaulting to Docling", display_name, exc)
        return FileInspectionReport(
            file_path=str(path),
            filename=display_name,
            extension=".pdf",
            file_size_bytes=size,
            content_hash=content_hash,
            recommended_parser="docling",
            reason=f"PyPDF pre-inspection encountered unusual structure ({exc}); escalating to Docling.",
            page_count=1,
            estimated_quality=0.5,
        )

