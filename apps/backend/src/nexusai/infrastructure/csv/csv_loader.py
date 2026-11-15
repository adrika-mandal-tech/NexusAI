"""Safe access to uploaded CSV files.

Upload validation happens here: extension, size, encoding, delimiter and
row/column limits. The file is copied into the configured upload directory so
that profiling, ingestion and the row-level reconciliation check can all read
the same bytes later.
"""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "text/plain",
    "text/tab-separated-values",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/octet-stream",
    "",
}

# A file that starts with these bytes is not a CSV — it is a zip (xlsx) or
# another container. Rejecting it here stops a zip bomb reaching the CSV path.
ZIP_MAGIC = b"PK\x03\x04"
EXCEL_MAGIC = b"\xd0\xcf\x11\xe0"


class CsvUploadError(ValueError):
    """Raised when an uploaded file cannot be accepted."""


@dataclass
class StoredCsv:
    """A CSV that has been validated and written to the upload directory."""

    file_id: str
    filename: str
    path: Path
    source_id: uuid.UUID
    size_bytes: int
    encoding: str = "utf-8"
    delimiter: str = ","
    content_hash: str = ""
    warnings: list[str] = field(default_factory=list)


def _detect_encoding(raw: bytes) -> tuple[str, bool]:
    """Detect the encoding of the file.

    Returns ``(encoding, is_fallback)``. A UTF-8 BOM is stripped by using
    ``utf-8-sig``. Latin-1 is the last resort because it never fails — the
    fallback is reported so the user knows the bytes were reinterpreted.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", False
    try:
        raw.decode("utf-8")
        return "utf-8", False
    except UnicodeDecodeError:
        return "latin-1", True


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {d: first_line.count(d) for d in (",", ";", "\t", "|")}
        best = max(counts, key=lambda k: counts[k])
        return best if counts[best] > 0 else ","


def upload_directory() -> Path:
    """The directory where uploaded CSVs are stored."""
    settings = get_settings()
    path = Path(settings.UPLOAD_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_csv(
    content: bytes,
    filename: str,
    source_id: uuid.UUID,
    content_type: str = "",
) -> StoredCsv:
    """Validate and store an uploaded CSV.

    Raises :class:`CsvUploadError` when the file cannot be accepted.
    """
    settings = get_settings()

    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise CsvUploadError(
            f"Unsupported file type {suffix or '(none)'}. Only .csv, .tsv, .xlsx, and .xls are accepted."
        )
    if not content:
        raise CsvUploadError("The file is empty.")
    if len(content) > settings.CSV_MAX_UPLOAD_BYTES:
        raise CsvUploadError(
            f"The file is {len(content) // (1024 * 1024)} MB, exceeding the limit of "
            f"{settings.CSV_MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
            df = pd.read_excel(io.BytesIO(content))
            out_buf = io.StringIO()
            df.to_csv(out_buf, index=False)
            content = out_buf.getvalue().encode("utf-8")
            filename = Path(filename).stem + ".csv"
            content_type = "text/csv"
            suffix = ".csv"
        except Exception as exc:
            raise CsvUploadError(f"Could not convert Excel spreadsheet: {exc}") from exc

    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise CsvUploadError(f"Unsupported content type {content_type!r}.")
    if content.startswith(ZIP_MAGIC) or content.startswith(EXCEL_MAGIC):
        raise CsvUploadError(
            "This file is not a valid CSV format. Please upload standard CSV/XLSX."
        )

    encoding, is_fallback = _detect_encoding(content)
    try:
        text = content.decode(encoding, errors="replace")
    except LookupError:
        encoding, is_fallback = "latin-1", True
        text = content.decode(encoding, errors="replace")

    warnings: list[str] = []
    if is_fallback:
        warnings.append(
            "The file is not valid UTF-8; it was decoded as latin-1, which may have "
            "changed some characters."
        )
    if not text.strip():
        raise CsvUploadError("The file contains no data.")

    delimiter = _sniff_delimiter(text[:8192])
    if delimiter != ",":
        warnings.append(f"The delimiter was detected as {delimiter!r} rather than a comma.")

    # Row and column limits.
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        raise CsvUploadError("The file has no header row.") from None
    if len(header) > settings.CSV_MAX_COLUMNS:
        raise CsvUploadError(
            f"The file has {len(header)} columns, exceeding the limit of "
            f"{settings.CSV_MAX_COLUMNS}."
        )
    row_count = 0
    for _ in reader:
        row_count += 1
        if row_count > settings.CSV_MAX_ROWS:
            raise CsvUploadError(
                f"The file has more than {settings.CSV_MAX_ROWS} rows."
            )

    file_id = uuid.uuid4().hex
    path = upload_directory() / f"{file_id}{suffix}"
    path.write_bytes(content)

    return StoredCsv(
        file_id=file_id,
        filename=filename or f"{file_id}.csv",
        path=path,
        source_id=source_id,
        size_bytes=len(content),
        encoding=encoding,
        delimiter=delimiter,
        content_hash=hashlib.sha256(content).hexdigest(),
        warnings=warnings,
    )


def remove_stored_csv(file_id: str) -> None:
    """Delete a stored upload. Best effort."""
    for path in upload_directory().glob(f"{file_id}.*"):
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Could not remove uploaded file %s: %s", path, exc)


def copy_to(source: Path, destination: Path) -> None:
    """Copy a stored upload elsewhere."""
    shutil.copy2(source, destination)
