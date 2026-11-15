"""Configurable document chunking strategies with deterministic chunk IDs.

Supports:
- `structure_aware`: Heading- and page-boundary-aware splitting preserving intact tables/code blocks.
- `table_aware`: Repeats Markdown table headers across split slices of oversized tables.
- `recursive`: Recursive character splitting via LangChain `RecursiveCharacterTextSplitter`.
- `token_aware`: Token-window splitting with configurable token size and overlap.
- `parent_child`: Hierarchical small-to-big chunking (child search chunks linked to parent context blocks).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from nexusai.core.config import get_settings
from nexusai.core.ids import (
    compute_sha256_bytes,
    deterministic_chunk_id,
    deterministic_parent_chunk_id,
)
from nexusai.domain.ports.document_parser import ParsedChunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WHITESPACE_CLEAN_RE = re.compile(r"\n{3,}")


@dataclass
class ExtractedSection:
    """Intermediate structured block produced by parsers prior to chunking."""

    text: str
    page_number: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_extracted_text(text: str) -> str:
    """Clean extraction artifacts while preserving Markdown structure and tables."""
    if not text:
        return ""
    cleaned = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    # Rejoin hyphenated line breaks in prose (avoid touching Markdown table separators `|---|`)
    cleaned = re.sub(r"([a-zA-Z])-\n([a-z])", r"\1\2", cleaned)
    cleaned = _WHITESPACE_CLEAN_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def estimate_token_count(text: str) -> int:
    """Offline-safe subword token count approximation (~1.25x word/punct tokens)."""
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


def Path_ext(filename: str) -> str:
    """Extract normalized lowercase source_type extension without leading dot."""
    if "." in filename:
        return filename.rsplit(".", 1)[-1].lower()
    return "txt"


class ConfigurableChunker:
    """Configurable chunking engine producing deterministic `ParsedChunk` objects."""

    def __init__(
        self,
        strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self.strategy = (strategy or settings.CHUNK_STRATEGY or "structure_aware").lower()
        self.chunk_size = max(128, chunk_size or settings.CHUNK_SIZE or 800)
        default_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else (settings.CHUNK_OVERLAP if settings.CHUNK_OVERLAP is not None else max(16, int(round(self.chunk_size * 0.10))))
        )
        # If chunk_size was explicitly overridden without chunk_overlap, maintain ~10% overlap ratio
        if chunk_size is not None and chunk_overlap is None:
            default_overlap = max(16, int(round(self.chunk_size * 0.10)))
        self.chunk_overlap = max(0, min(self.chunk_size // 2, default_overlap))
        self.overlap_ratio = round(self.chunk_overlap / max(1, self.chunk_size), 4)
        self._recursive = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_sections(
        self,
        sections: list[ExtractedSection],
        document_id: uuid.UUID,
        filename: str,
        base_metadata: dict[str, Any] | None = None,
    ) -> list[ParsedChunk]:
        """Chunk structured page/heading sections according to configured strategy."""
        shared_meta = dict(base_metadata or {})
        shared_meta.setdefault("filename", filename)
        shared_meta["chunk_strategy"] = self.strategy
        shared_meta["chunk_size"] = self.chunk_size
        shared_meta["chunk_overlap"] = self.chunk_overlap
        shared_meta["chunk_overlap_ratio"] = self.overlap_ratio

        if self.strategy == "parent_child":
            return self._chunk_parent_child(sections, document_id, filename, shared_meta)

        raw_pieces: list[tuple[str, int | None, str | None, dict[str, Any]]] = []

        for sec in sections:
            cleaned = normalize_extracted_text(sec.text)
            if not cleaned:
                continue

            if self.strategy == "recursive":
                for piece in self._recursive.split_text(cleaned):
                    if piece.strip():
                        raw_pieces.append((piece.strip(), sec.page_number, sec.section_title, dict(sec.metadata)))
            elif self.strategy == "token_aware":
                for piece in self._split_by_tokens(cleaned, self.chunk_size // 4 or 180, self.chunk_overlap // 4 or 30):
                    if piece.strip():
                        raw_pieces.append((piece.strip(), sec.page_number, sec.section_title, dict(sec.metadata)))
            elif self.strategy == "table_aware":
                for piece, sub_heading, is_tbl in self._split_structure_and_tables(cleaned, sec.section_title, preserve_table_headers=True):
                    meta = {**sec.metadata, "contains_table": is_tbl}
                    raw_pieces.append((piece, sec.page_number, sub_heading, meta))
            else:
                # Default: structure_aware (also table-safe!)
                for piece, sub_heading, is_tbl in self._split_structure_and_tables(cleaned, sec.section_title, preserve_table_headers=True):
                    meta = {**sec.metadata, "contains_table": is_tbl}
                    raw_pieces.append((piece, sec.page_number, sub_heading, meta))

        chunks: list[ParsedChunk] = []
        seen_hashes: set[str] = set()
        source_type_val = str(shared_meta.get("source_type") or Path_ext(filename))

        for piece_text, page_no, heading, piece_meta in raw_pieces:
            normalized_piece = piece_text.strip()
            if not normalized_piece:
                continue
            chunk_hash = compute_sha256_bytes(normalized_piece.encode("utf-8"))
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)

            idx = len(chunks)
            cid = deterministic_chunk_id(document_id, idx, normalized_piece)
            merged_meta = {
                **shared_meta,
                **piece_meta,
                "document_id": str(document_id),
                "chunk_id": str(cid),
                "chunk_index": idx,
                "filename": filename,
                "page_number": page_no,
                "section_title": heading,
                "content_hash": chunk_hash,
                "source_type": source_type_val,
                "token_count": estimate_token_count(normalized_piece),
                "char_count": len(normalized_piece),
            }
            chunks.append(
                ParsedChunk(
                    chunk_id=cid,
                    document_id=document_id,
                    index=idx,
                    text=normalized_piece,
                    page_number=page_no,
                    section_title=heading,
                    metadata=merged_meta,
                )
            )
        return chunks

    def _chunk_parent_child(
        self,
        sections: list[ExtractedSection],
        document_id: uuid.UUID,
        filename: str,
        shared_meta: dict[str, Any],
    ) -> list[ParsedChunk]:
        """Hierarchical small-to-big chunking: child chunks reference expanded parent text."""
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self.chunk_size * 2.5),
            chunk_overlap=self.chunk_overlap * 2,
            separators=["\n\n", "\n", ". ", " "],
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(200, self.chunk_size // 2),
            chunk_overlap=max(30, self.chunk_overlap // 2),
            separators=["\n\n", "\n", ". ", " "],
        )

        chunks: list[ParsedChunk] = []
        seen_hashes: set[str] = set()
        source_type_val = str(shared_meta.get("source_type") or Path_ext(filename))
        child_global_idx = 0
        parent_idx = 0

        for sec in sections:
            cleaned = normalize_extracted_text(sec.text)
            if not cleaned:
                continue
            parent_texts = parent_splitter.split_text(cleaned)
            for p_text in parent_texts:
                p_text = p_text.strip()
                if not p_text:
                    continue
                parent_id = deterministic_parent_chunk_id(document_id, parent_idx)
                parent_idx += 1

                child_texts = child_splitter.split_text(p_text) or [p_text]
                for c_text in child_texts:
                    c_text = c_text.strip()
                    if not c_text:
                        continue
                    chunk_hash = compute_sha256_bytes(c_text.encode("utf-8"))
                    if chunk_hash in seen_hashes:
                        continue
                    seen_hashes.add(chunk_hash)
                    cid = deterministic_chunk_id(document_id, child_global_idx, c_text)
                    meta = {
                        **shared_meta,
                        **sec.metadata,
                        "document_id": str(document_id),
                        "chunk_id": str(cid),
                        "chunk_index": child_global_idx,
                        "filename": filename,
                        "page_number": sec.page_number,
                        "section_title": sec.section_title,
                        "content_hash": chunk_hash,
                        "source_type": source_type_val,
                        "parent_chunk_id": str(parent_id),
                        "parent_text": p_text,
                        "token_count": estimate_token_count(c_text),
                        "char_count": len(c_text),
                    }
                    chunks.append(
                        ParsedChunk(
                            chunk_id=cid,
                            document_id=document_id,
                            index=child_global_idx,
                            text=c_text,
                            page_number=sec.page_number,
                            section_title=sec.section_title,
                            metadata=meta,
                        )
                    )
                    child_global_idx += 1

        return chunks

    def _split_structure_and_tables(
        self,
        text: str,
        initial_heading: str | None,
        preserve_table_headers: bool = True,
    ) -> list[tuple[str, str | None, bool]]:
        """Split text by Markdown headings while keeping Markdown tables intact or header-prefixed."""
        lines = text.split("\n")
        current_heading = initial_heading
        blocks: list[tuple[list[str], str | None, bool]] = []
        buf: list[str] = []
        in_table = False

        def flush() -> None:
            nonlocal buf
            if buf:
                joined = "\n".join(buf).strip()
                if joined:
                    blocks.append((list(buf), current_heading, in_table))
                buf = []

        for line in lines:
            stripped = line.strip()
            heading_match = _HEADING_RE.match(stripped)
            is_table_line = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

            if heading_match:
                flush()
                in_table = False
                current_heading = heading_match.group(2).strip()
                buf.append(line)
                continue

            if is_table_line != in_table:
                flush()
                in_table = is_table_line

            buf.append(line)

        flush()

        results: list[tuple[str, str | None, bool]] = []
        for block_lines, heading, is_tbl in blocks:
            block_text = "\n".join(block_lines).strip()
            if not block_text:
                continue

            if is_tbl and preserve_table_headers:
                if len(block_text) <= self.chunk_size * 1.4:
                    results.append((block_text, heading, True))
                else:
                    results.extend(self._split_large_table(block_lines, heading))
            else:
                if len(block_text) <= self.chunk_size:
                    results.append((block_text, heading, False))
                else:
                    for sub in self._recursive.split_text(block_text):
                        if sub.strip():
                            results.append((sub.strip(), heading, False))

        return results

    def _split_large_table(
        self,
        table_lines: list[str],
        heading: str | None,
    ) -> list[tuple[str, str | None, bool]]:
        """Split an oversized Markdown table while repeating header + separator rows."""
        if len(table_lines) <= 3:
            return [("\n".join(table_lines), heading, True)]

        header_rows = table_lines[:2] if "---" in table_lines[1] else table_lines[:1]
        data_rows = table_lines[len(header_rows) :]
        header_prefix = "\n".join(header_rows)

        slices: list[tuple[str, str | None, bool]] = []
        current_rows: list[str] = []
        current_len = len(header_prefix)

        for row in data_rows:
            if current_rows and (current_len + len(row) + 1) > self.chunk_size:
                slices.append((header_prefix + "\n" + "\n".join(current_rows), heading, True))
                current_rows = [row]
                current_len = len(header_prefix) + len(row) + 1
            else:
                current_rows.append(row)
                current_len += len(row) + 1

        if current_rows:
            slices.append((header_prefix + "\n" + "\n".join(current_rows), heading, True))

        return slices

    @staticmethod
    def _split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        """Offline-safe token sliding window over whitespace-delimited words."""
        words = text.split()
        if not words:
            return []
        if len(words) <= max_tokens:
            return [text]

        step = max(1, max_tokens - overlap_tokens)
        out: list[str] = []
        idx = 0
        while idx < len(words):
            end = min(len(words), idx + max_tokens)
            out.append(" ".join(words[idx:end]))
            if end >= len(words):
                break
            idx += step
        return out

