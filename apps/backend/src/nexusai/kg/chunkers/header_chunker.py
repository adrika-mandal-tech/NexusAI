"""Header-based chunking for structured documents (Markdown, HTML, reST)."""

from __future__ import annotations

from typing import Optional

from nexusai.kg.chunkers.chunker_base import BaseChunker, Chunk


class HeaderChunker(BaseChunker):
    """Chunk by Markdown/HTML headers, keeping sections whole."""

    def chunk(self, text: str, source_title: str, source_file: str | None = None, source_type: str | None = None) -> list[Chunk]:

        lines = text.splitlines()
        if not lines:
            return []

        # Detect header markers and record their line indices
        headers: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("##") or stripped.startswith("###") or stripped.startswith("####") or stripped.startswith("#####") or stripped.startswith("######"):
                header_text = stripped.lstrip("#").strip()
                headers.append((i, header_text))

        # Build chunks: between consecutive headers
        chunk_starts = [0] + [h[0] for h in headers] + [len(lines)]
        chunks: list[Chunk] = []
        index = 0

        for k in range(len(chunk_starts) - 1):
            start_idx = chunk_starts[k]
            end_idx = chunk_starts[k + 1]
            if end_idx <= start_idx:
                continue

            section_lines = lines[start_idx:end_idx]
            section_text = "\n".join(section_lines).strip()

            # Skip tiny sections or merge with next
            if len(section_text) < self.min_chunk_size:
                continue

            if len(section_text) > self.chunk_size * 2:
                words = section_text.split()
                section_text = " ".join(words[: self.chunk_size])

            header = self._header_for_line(lines, start_idx)
            chunks.append(
                Chunk(
                    id=self._new_uuid(),
                    index=index,
                    text=section_text,
                    section_title=header,
                    source_title=source_title,
                    source_file=source_file,
                    source_type=source_type,
                    metadata={},
                )
            )
            index += 1

        return chunks

    @staticmethod
    def _header_for_line(lines: list[str], line_idx: int) -> Optional[str]:
        for i in range(line_idx, -1, -1):
            stripped = lines[i].strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return None

    @staticmethod
    def _new_uuid() -> __import__("uuid").UUID:
        return __import__("uuid").uuid4()
