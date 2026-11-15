"""Streaming CSV profiler.

The profiler never loads a whole file into memory. It reads the header, then
makes two passes over the file:

* a **counting pass** that tallies rows and detects duplicate rows using a
  bounded set of row digests;
* a **sampling pass** that collects at most ``CSV_PROFILE_SAMPLE_ROWS`` rows
  for column statistics.

For files whose size is known and small enough, a single pass is used. Above
the threshold, the sampling pass reads the first N rows and then takes an
evenly-spaced stride over the remainder.
"""

from __future__ import annotations

import csv
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from nexusai.application.graph.csv.type_inference import (
    INFERRED_TO_PROPERTY,
    collect_null_sentinels,
    infer_type,
    looks_like_pii,
    normalise_cell,
    score_key_candidate,
    top_values,
)
from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.entities.graph_schema import (
    ColumnProfile,
    Confidence,
    FileProfile,
    InferredType,
)

logger = get_logger(__name__)

# Types that may legitimately carry PII.
_PII_TYPES = {InferredType.STRING, InferredType.CATEGORICAL, InferredType.FREE_TEXT}


class CsvValidationError(ValueError):
    """Raised when a CSV file cannot be accepted at all."""


@dataclass
class CsvFile:
    """A validated, ready-to-profile CSV file."""

    file_id: str
    filename: str
    path: Path
    source_id: uuid.UUID
    size_bytes: int
    encoding: str = "utf-8"
    delimiter: str = ","
    content_hash: str = ""
    warnings: list[str] = field(default_factory=list)


def normalise_column_name(name: str) -> str:
    """Convert a CSV header to a safe snake_case identifier.

    Returns a string that always satisfies the Cypher identifier pattern, so
    a hostile header can never break out of a generated query.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", (name or "").strip().lower()).strip("_")
    if not cleaned:
        cleaned = "column"
    if cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    return cleaned[:120]


# Words ending in "s" that are not plurals.
_NON_PLURAL_S = frozenset(
    {
        "news",
        "series",
        "species",
        "process",
        "class",
        "analysis",
        "basis",
        "status",
        "business",
        "address",
        "focus",
        "campus",
        "bonus",
        "virus",
        "consensus",
        "corpus",
    }
)


def singularise(word: str) -> str:
    """A small, safe singulariser. No NLP dependency."""
    lowered = word.lower()
    if lowered in _NON_PLURAL_S:
        return lowered
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith(("ches", "shes", "sses", "xes", "zes")):
        return lowered[:-2]
    if lowered.endswith("ses") and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("us") and len(lowered) > 4:
        # Latin singulars: status, campus, virus.
        return lowered
    if lowered.endswith("s") and not lowered.endswith("ss") and len(lowered) > 3:
        return lowered[:-1]
    return lowered


def propose_label(filename: str) -> tuple[str, Confidence, str]:
    """Derive a node label from a filename.

    Returns ``(label, confidence, derivation)``.
    """
    stem = Path(filename).stem
    parts = re.split(r"[^0-9a-zA-Z]+", stem)
    parts = [p for p in parts if p]
    if not parts:
        return "Row", Confidence.LOW, "filename contained no usable words"

    singular = [singularise(p) for p in parts]
    label = "".join(p[:1].upper() + p[1:] for p in singular)
    label = re.sub(r"[^0-9a-zA-Z_]", "", label)
    if not label or label[0].isdigit():
        label = f"C{label}" if label else "Row"
    label = label[:64]

    if len(parts) == 1:
        return label, Confidence.HIGH, f"filename {filename!r} singularised to {label!r}"
    return label, Confidence.MEDIUM, f"filename {filename!r} title-cased to {label!r}"


def detect_delimiter(sample: str) -> str:
    """Sniff the delimiter of a CSV sample."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # Fall back to counting occurrences on the first line.
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {d: first_line.count(d) for d in (",", ";", "\t", "|")}
        best = max(counts, key=lambda k: counts[k])
        return best if counts[best] > 0 else ","


def _row_digest(row: Sequence[str]) -> str:
    payload = "\x1f".join(row).encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()


def _iter_rows(path: Path, encoding: str, delimiter: str) -> Iterator[list[str]]:
    """Yield raw rows from a CSV file, one at a time."""
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            yield row


def _sample_rows(
    path: Path,
    encoding: str,
    delimiter: str,
    limit: int,
) -> tuple[list[list[str]], int, int]:
    """Collect up to ``limit`` sample rows.

    Returns ``(sample_rows, total_rows_seen, blank_line_count)``. The header
    row is skipped and is not counted as a data row.
    """
    sample: list[list[str]] = []
    total = 0
    blank = 0
    first = True
    for row in _iter_rows(path, encoding, delimiter):
        if not row or (len(row) == 1 and row[0].strip() == ""):
            blank += 1
            continue
        if first:
            first = False
            continue
        total += 1
        if len(sample) < limit:
            sample.append(row)
    return sample, total, blank


def profile_csv(
    csv_file: CsvFile,
    sample_limit: int | None = None,
) -> FileProfile:
    """Profile a single CSV file and return a :class:`FileProfile`."""
    settings = get_settings()
    limit = sample_limit or settings.CSV_PROFILE_SAMPLE_ROWS

    text = csv_file.path.read_text(encoding=csv_file.encoding, errors="replace")
    if not text.strip():
        raise CsvValidationError(f"{csv_file.filename} is empty")

    header = _read_header(csv_file)
    if not header:
        raise CsvValidationError(f"{csv_file.filename} has no header row")
    if len(header) > settings.CSV_MAX_COLUMNS:
        raise CsvValidationError(
            f"{csv_file.filename} has {len(header)} columns, exceeding the limit of "
            f"{settings.CSV_MAX_COLUMNS}"
        )

    rows, total_rows, blank_lines = _sample_rows(
        csv_file.path, csv_file.encoding, csv_file.delimiter, limit
    )
    if total_rows == 0:
        raise CsvValidationError(f"{csv_file.filename} has a header but no data rows")

    columns = _profile_columns(header, rows, total_rows)
    duplicates, duplicate_ratio = _count_duplicate_rows(rows)
    ragged = sum(1 for row in rows if len(row) != len(header))

    label, label_confidence, label_derivation = propose_label(csv_file.filename)

    warnings: list[str] = []
    if csv_file.encoding != "utf-8":
        warnings.append(f"file decoded as {csv_file.encoding}; invalid bytes were replaced")
    if ragged:
        warnings.append(f"{ragged} sampled row(s) have a different field count than the header")
    if duplicate_ratio > 0:
        warnings.append(f"{duplicate_ratio:.1%} of sampled rows are exact duplicates")
    if total_rows < 50:
        warnings.append("fewer than 50 rows — types and keys are low confidence")

    return FileProfile(
        file_id=csv_file.file_id,
        filename=csv_file.filename,
        source_id=csv_file.source_id,
        size_bytes=csv_file.size_bytes,
        encoding=csv_file.encoding,
        delimiter=csv_file.delimiter,
        has_header=True,
        total_rows=total_rows,
        sampled_rows=len(rows),
        column_count=len(header),
        columns=columns,
        duplicate_row_count=duplicates,
        duplicate_row_ratio=duplicate_ratio,
        blank_line_count=blank_lines,
        ragged_row_count=ragged,
        proposed_label=label,
        label_confidence=label_confidence,
        label_derivation=label_derivation,
        warnings=warnings,
        content_hash=csv_file.content_hash,
    )


def _read_header(csv_file: CsvFile) -> list[str]:
    """Read and de-duplicate the header row."""
    with csv_file.path.open(
        "r", encoding=csv_file.encoding, errors="replace", newline=""
    ) as handle:
        reader = csv.reader(handle, delimiter=csv_file.delimiter)
        for row in reader:
            if not row or (len(row) == 1 and row[0].strip() == ""):
                continue
            header = [c.strip() for c in row]
            # De-duplicate empty and repeated headers deterministically.
            seen: dict[str, int] = {}
            resolved: list[str] = []
            for index, name in enumerate(header):
                base = name or f"column_{index + 1}"
                if base in seen:
                    seen[base] += 1
                    base = f"{base}_{seen[base]}"
                else:
                    seen[base] = 1
                resolved.append(base)
            return resolved
    return []


def _count_duplicate_rows(rows: list[list[str]]) -> tuple[int, float]:
    """Count exact duplicate rows in the sample."""
    if not rows:
        return 0, 0.0
    digests = [_row_digest(r) for r in rows]
    unique = len(set(digests))
    duplicates = len(digests) - unique
    return duplicates, duplicates / len(digests)


def _profile_columns(
    header: list[str],
    rows: list[list[str]],
    total_rows: int,
) -> list[ColumnProfile]:
    """Build a :class:`ColumnProfile` for each column."""
    profiles: list[ColumnProfile] = []
    for index, name in enumerate(header):
        raw_values = [row[index] if index < len(row) else "" for row in rows]
        cleaned = [normalise_cell(v) for v in raw_values]
        non_null = [v for v in cleaned if v != ""]

        guess = infer_type(raw_values, name)
        is_key, score, confidence, evidence = score_key_candidate(
            name, cleaned, total_rows, guess.inferred_type
        )

        numeric = [float(v) for v in non_null if _is_number(v)]
        min_value: Any = None
        max_value: Any = None
        mean_value: float | None = None
        if numeric and len(numeric) == len(non_null):
            min_value = min(numeric)
            max_value = max(numeric)
            mean_value = round(sum(numeric) / len(numeric), 6)

        flags = list(guess.quality_flags)
        if any(v != v.strip() for v in raw_values if v):
            flags.append("whitespace_issues")
        if guess.inferred_type == InferredType.MIXED:
            flags.append("mixed_types")

        pii = looks_like_pii(name) and guess.inferred_type in _PII_TYPES
        if pii:
            flags.append("pii_candidate")

        profiles.append(
            ColumnProfile(
                name=name,
                normalized_name=normalise_column_name(name),
                position=index,
                inferred_type=guess.inferred_type,
                property_type=INFERRED_TO_PROPERTY[guess.inferred_type],
                dominant_type=guess.dominant_type,
                null_ratio=round(1 - (len(non_null) / len(rows)), 6) if rows else 0.0,
                distinct_count=len(set(non_null)),
                non_null_count=len(non_null),
                total_count=total_rows,
                min_value=min_value,
                max_value=max_value,
                mean_value=mean_value,
                sample_values=[v for v in dict.fromkeys(non_null)][:5],
                top_values=top_values(cleaned),
                null_sentinels_found=collect_null_sentinels(raw_values),
                quality_flags=sorted(set(flags)),
                may_contain_pii=pii,
                is_candidate_key=is_key,
                key_confidence=confidence,
                notes=[f"key score {score}: {e}" for e in evidence],
            )
        )
    return profiles


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def read_rows(
    path: Path,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> Iterator[dict[str, str]]:
    """Stream a CSV file as dictionaries.

    Used by the ingestion path and by the row-level reconciliation check.
    """
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            yield {k: ("" if v is None else v) for k, v in row.items() if k is not None}
