"""PaddleOCR document parser for scanned PDFs and image-based documents.

Uses native `paddleocr` (`PaddleOCR`) when `paddlepaddle` is installed, and
seamlessly executes PaddleOCR PP-OCRv4/v6 models via `RapidOCR` (PyTorch /
ONNX Runtime) + `pypdfium2` on platforms without `paddlepaddle` wheels (such as
Python 3.14 macOS arm64).

Preserves 1-based page numbers, bounding-box positional layout, tabular column
alignment (` | `), and OCR confidence scores.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
from PIL import Image

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.ports.document_parser import DocumentParserPort, ParsedDocument
from nexusai.infrastructure.parsers.chunker import ConfigurableChunker, ExtractedSection
from nexusai.infrastructure.parsers.file_inspector import IMAGE_EXTENSIONS

logger = get_logger(__name__)

_ENGINE_LOCK = threading.Lock()
_OCR_BACKEND_INSTANCE: Any = None
_OCR_BACKEND_KIND: str = "uninitialized"


@dataclass
class OCRBox:
    """Single detected text box with 4-corner coordinates and confidence."""

    text: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def height(self) -> float:
        return max(1.0, self.y_max - self.y_min)

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2.0


def _get_ocr_backend() -> tuple[Any, str]:
    """Lazily initialize singleton PaddleOCR or RapidOCR PP-OCRv6 engine."""
    global _OCR_BACKEND_INSTANCE, _OCR_BACKEND_KIND
    if _OCR_BACKEND_INSTANCE is not None:
        return _OCR_BACKEND_INSTANCE, _OCR_BACKEND_KIND

    with _ENGINE_LOCK:
        if _OCR_BACKEND_INSTANCE is not None:
            return _OCR_BACKEND_INSTANCE, _OCR_BACKEND_KIND

        settings = get_settings()
        preferred = (settings.OCR_ENGINE or "auto").lower()

        # 1. Try native paddleocr when requested or in auto mode
        if preferred in ("auto", "paddleocr"):
            try:
                import paddle  # noqa: F401
                from paddleocr import PaddleOCR

                _OCR_BACKEND_INSTANCE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                _OCR_BACKEND_KIND = "paddleocr"
                logger.info("Initialized native PaddleOCR backend")
                return _OCR_BACKEND_INSTANCE, _OCR_BACKEND_KIND
            except Exception as exc:
                if preferred == "paddleocr":
                    logger.warning("Native paddleocr unavailable (%s); trying RapidOCR PP-OCR runtime", exc)

        # 2. RapidOCR (official PaddleOCR PP-OCRv4/v6 weights via ONNXRuntime or PyTorch)
        try:
            try:
                from rapidocr_onnxruntime import RapidOCR as RapidOCROnnx

                _OCR_BACKEND_INSTANCE = RapidOCROnnx()
                _OCR_BACKEND_KIND = "paddleocr_rapidocr_onnxruntime"
                logger.info("Initialized PaddleOCR (rapidocr_onnxruntime) backend")
            except Exception:
                from rapidocr import RapidOCR
                try:
                    import onnxruntime  # noqa: F401

                    _OCR_BACKEND_INSTANCE = RapidOCR()
                    _OCR_BACKEND_KIND = "paddleocr_rapidocr_onnx"
                    logger.info("Initialized PaddleOCR (RapidOCR ONNXRuntime) backend")
                except Exception:
                    from rapidocr.inference_engine.base import EngineType

                    _OCR_BACKEND_INSTANCE = RapidOCR(
                        params={
                            "Det.engine_type": EngineType.TORCH,
                            "Cls.engine_type": EngineType.TORCH,
                            "Rec.engine_type": EngineType.TORCH,
                        }
                    )
                    _OCR_BACKEND_KIND = "paddleocr_rapidocr_torch"
                    logger.info("Initialized PaddleOCR (RapidOCR PyTorch PP-OCRv6) backend")
        except Exception as exc:
            logger.warning("No OCR backend engine could be loaded (%s)", exc)
            _OCR_BACKEND_INSTANCE = None
            _OCR_BACKEND_KIND = "unavailable"

        return _OCR_BACKEND_INSTANCE, _OCR_BACKEND_KIND


def _reconstruct_layout_from_boxes(boxes: list[OCRBox]) -> tuple[str, bool]:
    """Group OCR bounding boxes into horizontal lines and detect tabular columns."""
    if not boxes:
        return "", False

    sorted_by_y = sorted(boxes, key=lambda b: (b.y_center, b.x_min))
    lines: list[list[OCRBox]] = []

    for box in sorted_by_y:
        if not lines:
            lines.append([box])
            continue
        last_line = lines[-1]
        line_y = sum(b.y_center for b in last_line) / len(last_line)
        avg_h = sum(b.height for b in last_line) / len(last_line)
        if abs(box.y_center - line_y) <= max(8.0, avg_h * 0.6):
            last_line.append(box)
        else:
            lines.append([box])

    rendered_lines: list[str] = []
    multi_col_lines = 0

    for line_boxes in lines:
        line_boxes.sort(key=lambda b: b.x_min)
        if len(line_boxes) == 1:
            rendered_lines.append(line_boxes[0].text)
            continue

        avg_h = sum(b.height for b in line_boxes) / len(line_boxes)
        col_gap_threshold = max(24.0, avg_h * 1.6)
        gaps = [
            line_boxes[i].x_min - line_boxes[i - 1].x_max
            for i in range(1, len(line_boxes))
        ]
        has_columns = any(g >= col_gap_threshold for g in gaps)
        if has_columns:
            multi_col_lines += 1
            parts: list[str] = [line_boxes[0].text]
            for i in range(1, len(line_boxes)):
                sep = " | " if gaps[i - 1] >= col_gap_threshold else " "
                parts.append(sep + line_boxes[i].text)
            rendered_lines.append("| " + "".join(parts) + " |")
        else:
            rendered_lines.append(" ".join(b.text for b in line_boxes))

    # Insert Markdown table separator after first multi-column row if >=2 consecutive table rows
    is_tabular = multi_col_lines >= 2
    if is_tabular:
        final_lines: list[str] = []
        inserted_sep = False
        for ln in rendered_lines:
            final_lines.append(ln)
            if not inserted_sep and ln.startswith("| ") and ln.endswith(" |"):
                col_count = max(2, ln.count("|") - 1)
                final_lines.append("| " + " | ".join(["---"] * col_count) + " |")
                inserted_sep = True
            elif not (ln.startswith("| ") and ln.endswith(" |")):
                inserted_sep = False
        return "\n".join(final_lines), True

    return "\n".join(rendered_lines), False


class PaddleOCRParser(DocumentParserPort):
    """OCR parser for scanned PDFs, hybrid PDF pages, and image documents."""

    def __init__(self, chunker: ConfigurableChunker | None = None, render_scale: float = 2.0) -> None:
        self._chunker = chunker or ConfigurableChunker()
        self._render_scale = render_scale

    def supported_types(self) -> list[str]:
        return [".pdf", *sorted(IMAGE_EXTENSIONS)]

    def _ocr_rgb_array_sync(self, rgb_array: np.ndarray) -> tuple[str, float, list[dict[str, Any]], bool, str]:
        """Run OCR on a single RGB numpy image array and return layout-preserved text + metadata."""
        engine, backend_kind = _get_ocr_backend()
        if engine is None:
            return "", 0.0, [], False, "unavailable"

        ocr_boxes: list[OCRBox] = []

        if backend_kind == "paddleocr":
            bgr = rgb_array[:, :, ::-1].copy()
            raw = engine.ocr(bgr, cls=True)
            lines = raw[0] if (raw and raw[0]) else []
            for item in lines:
                if not item or len(item) < 2:
                    continue
                coords, (txt, conf) = item[0], item[1]
                txt_str = (txt or "").strip()
                if not txt_str:
                    continue
                xs = [float(pt[0]) for pt in coords]
                ys = [float(pt[1]) for pt in coords]
                ocr_boxes.append(
                    OCRBox(
                        text=txt_str,
                        confidence=float(conf),
                        x_min=min(xs),
                        y_min=min(ys),
                        x_max=max(xs),
                        y_max=max(ys),
                    )
                )
        elif backend_kind == "paddleocr_rapidocr_onnxruntime":
            result_tuple = engine(rgb_array)
            items = result_tuple[0] if (isinstance(result_tuple, tuple) and result_tuple) else result_tuple
            for item in items or []:
                if not item or len(item) < 3:
                    continue
                pts, txt, conf = item[0], item[1], item[2]
                txt_str = (str(txt) or "").strip()
                if not txt_str:
                    continue
                xs = [float(pt[0]) for pt in pts]
                ys = [float(pt[1]) for pt in pts]
                ocr_boxes.append(
                    OCRBox(
                        text=txt_str,
                        confidence=float(conf),
                        x_min=min(xs),
                        y_min=min(ys),
                        x_max=max(xs),
                        y_max=max(ys),
                    )
                )
        else:
            out = engine(rgb_array)
            txts = getattr(out, "txts", None) or ()
            scores = getattr(out, "scores", None) or ()
            boxes = getattr(out, "boxes", None)
            for idx, txt in enumerate(txts):
                txt_str = (txt or "").strip()
                if not txt_str:
                    continue
                conf = float(scores[idx]) if idx < len(scores) else 0.0
                if boxes is not None and idx < len(boxes):
                    pts = boxes[idx]
                    xs = [float(pt[0]) for pt in pts]
                    ys = [float(pt[1]) for pt in pts]
                    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                else:
                    x_min, y_min, x_max, y_max = 0.0, float(idx * 20), 100.0, float(idx * 20 + 16)
                ocr_boxes.append(
                    OCRBox(
                        text=txt_str,
                        confidence=conf,
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    )
                )

        if not ocr_boxes:
            return "", 0.0, [], False, backend_kind

        text, is_tabular = _reconstruct_layout_from_boxes(ocr_boxes)
        avg_conf = round(sum(b.confidence for b in ocr_boxes) / len(ocr_boxes), 4)
        sample_positions = [
            {
                "text": b.text[:80],
                "bbox": [round(b.x_min, 1), round(b.y_min, 1), round(b.x_max, 1), round(b.y_max, 1)],
                "confidence": round(b.confidence, 4),
            }
            for b in ocr_boxes[:25]
        ]
        return text, avg_conf, sample_positions, is_tabular, backend_kind

    async def ocr_pdf_pages(
        self,
        file_path: str,
        page_numbers: list[int] | None = None,
    ) -> tuple[list[ExtractedSection], dict[str, Any]]:
        """Render and OCR all or selected 1-based pages of a PDF."""
        return await asyncio.to_thread(self._ocr_pdf_pages_sync, file_path, page_numbers)

    def _ocr_pdf_pages_sync(
        self,
        file_path: str,
        page_numbers: list[int] | None = None,
    ) -> tuple[list[ExtractedSection], dict[str, Any]]:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(file_path)
        total_pages = len(pdf)
        target_pages = (
            [p for p in page_numbers if 1 <= p <= total_pages]
            if page_numbers is not None
            else list(range(1, total_pages + 1))
        )

        sections: list[ExtractedSection] = []
        confidences: list[float] = []
        backend_used = "paddleocr"

        for page_no in target_pages:
            page = pdf[page_no - 1]
            bitmap = page.render(scale=self._render_scale)
            pil_image = bitmap.to_pil().convert("RGB")
            rgb_arr = np.array(pil_image)

            text, avg_conf, bboxes, is_tabular, backend_used = self._ocr_rgb_array_sync(rgb_arr)
            if text.strip():
                confidences.append(avg_conf)
                sections.append(
                    ExtractedSection(
                        text=text,
                        page_number=page_no,
                        section_title=None,
                        metadata={
                            "parser_used": "paddleocr",
                            "ocr_runtime": backend_used,
                            "ocr_confidence": avg_conf,
                            "contains_table": is_tabular,
                            "positional_boxes_sample": bboxes,
                        },
                    )
                )

        pdf.close()
        overall_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        summary = {
            "parser_used": "paddleocr",
            "ocr_runtime": backend_used,
            "total_pages": total_pages,
            "ocr_pages_processed": target_pages,
            "average_ocr_confidence": overall_conf,
        }
        return sections, summary

    async def parse(
        self,
        file_path: str,
        document_id: UUID,
        filename: str,
        source_type: str,
    ) -> ParsedDocument:
        """Parse an image or scanned PDF with PaddleOCR into deterministic chunks."""
        logger.info("Parsing %s with PaddleOCRParser", filename)
        lower = filename.lower()

        if any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS) or source_type.lower() == "image":
            section, summary = await asyncio.to_thread(self._ocr_image_file_sync, file_path)
            sections = [section] if section.text.strip() else []
        else:
            sections, summary = await self.ocr_pdf_pages(file_path)

        chunks = self._chunker.chunk_sections(
            sections=sections,
            document_id=document_id,
            filename=filename,
            base_metadata={"parser_used": "paddleocr", "source_type": source_type},
        )
        summary["total_chunks"] = len(chunks)

        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            source_type=source_type,
            chunks=chunks,
            metadata=summary,
        )

    def _ocr_image_file_sync(self, file_path: str) -> tuple[ExtractedSection, dict[str, Any]]:
        with Image.open(file_path) as img:
            rgb_arr = np.array(img.convert("RGB"))
        text, avg_conf, bboxes, is_tabular, backend_used = self._ocr_rgb_array_sync(rgb_arr)
        section = ExtractedSection(
            text=text,
            page_number=1,
            section_title=None,
            metadata={
                "parser_used": "paddleocr",
                "ocr_runtime": backend_used,
                "ocr_confidence": avg_conf,
                "contains_table": is_tabular,
                "positional_boxes_sample": bboxes,
            },
        )
        summary = {
            "parser_used": "paddleocr",
            "ocr_runtime": backend_used,
            "total_pages": 1,
            "ocr_pages_processed": [1],
            "average_ocr_confidence": avg_conf,
        }
        return section, summary

