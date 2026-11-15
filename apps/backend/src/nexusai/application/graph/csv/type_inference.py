"""Deterministic type inference and key detection for CSV columns.

Everything here is pure Python: no LLM, no database, no network. That is what
makes the resulting proposal trustworthy enough to gate a write.

The type ladder is evaluated top-down and the first matching rule wins. Every
rule that was attempted is recorded so the review UI can explain the outcome.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from nexusai.domain.entities.graph_schema import (
    Confidence,
    InferredType,
    PropertyType,
)

# The Neo4j property type that each inferred type is written as.
INFERRED_TO_PROPERTY: dict[InferredType, PropertyType] = {
    InferredType.INTEGER: PropertyType.INTEGER,
    InferredType.FLOAT: PropertyType.FLOAT,
    InferredType.BOOLEAN: PropertyType.BOOLEAN,
    InferredType.DATE: PropertyType.DATE,
    InferredType.DATETIME: PropertyType.DATETIME,
    InferredType.TIMESTAMP: PropertyType.DATETIME,
    InferredType.STRING: PropertyType.STRING,
    InferredType.CATEGORICAL: PropertyType.STRING,
    InferredType.FREE_TEXT: PropertyType.STRING,
    InferredType.EMPTY: PropertyType.STRING,
    InferredType.MIXED: PropertyType.STRING,
}

# ── Sentinels ────────────────────────────────────────────────────────────────

NULL_SENTINELS: frozenset[str] = frozenset(
    {
        "",
        "-",
        "--",
        "?",
        "n/a",
        "na",
        "n.a.",
        "null",
        "none",
        "nil",
        "unknown",
        "(unknown)",
        "#n/a",
        "#value!",
        "nan",
    }
)

TRUE_VALUES: frozenset[str] = frozenset({"true", "t", "yes", "y", "1"})
FALSE_VALUES: frozenset[str] = frozenset({"false", "f", "no", "n", "0"})

# ── Date formats ─────────────────────────────────────────────────────────────

DATE_FORMATS: tuple[tuple[str, str], ...] = (
    ("%Y-%m-%d", "iso_date"),
    ("%Y/%m/%d", "slash_year_first"),
    ("%d/%m/%Y", "slash_day_first"),
    ("%m/%d/%Y", "slash_month_first"),
    ("%d-%m-%Y", "dash_day_first"),
    ("%d.%m.%Y", "dot_day_first"),
    ("%d %b %Y", "day_mon_year"),
    ("%d %B %Y", "day_longmon_year"),
    ("%b %d, %Y", "mon_day_year"),
    ("%B %d, %Y", "longmon_day_year"),
    ("%Y%m%d", "compact_date"),
)

DATETIME_FORMATS: tuple[tuple[str, str], ...] = (
    ("%Y-%m-%dT%H:%M:%S", "iso_datetime"),
    ("%Y-%m-%d %H:%M:%S", "space_datetime"),
    ("%Y-%m-%d %H:%M", "space_datetime_minute"),
    ("%Y-%m-%dT%H:%M:%S.%f", "iso_datetime_micro"),
    ("%Y-%m-%dT%H:%M:%S%z", "iso_datetime_tz"),
    ("%Y-%m-%d %H:%M:%S%z", "space_datetime_tz"),
    ("%d/%m/%Y %H:%M:%S", "slash_day_first_datetime"),
    ("%m/%d/%Y %H:%M:%S", "slash_month_first_datetime"),
)

_EPOCH_SECONDS_MIN = 100_000_000  # 1973-03-03
_EPOCH_SECONDS_MAX = 4_102_444_800  # 2100-01-01

# ── Key-name patterns ───────────────────────────────────────────────────────

KEY_NAME_RE = re.compile(
    r"(^id$|_id$|[a-z](?:ID|Id)$|^uuid$|^guid$|_key$|[a-z](?:Key|KEY)$|^pk$|_pk$|code$|sku$|isbn$|email$|_no$|_number$)"
)
FK_NAME_RE = re.compile(r"(_id$|^id$|_fk$|^fk$)", re.IGNORECASE)
PII_NAME_RE = re.compile(
    r"(email|e_mail|mail|phone|mobile|s sn|ssn|social|address|postcode|zip|"
    r"password|passwd|secret|token|api_key|credit|card|cvv|iban|tax_id|"
    r"first_name|last_name|full_name|date_of_birth|dob)",
    re.IGNORECASE,
)
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# A column whose name is temporal, used to tell an epoch timestamp from a
# plain integer.
_TEMPORAL_NAME_RE = re.compile(
    r"(date|time|timestamp|created|updated|modified|deleted|epoch|_at$|at_)",
    re.IGNORECASE,
)


@dataclass
class TypeGuess:
    """The outcome of the type ladder for one column."""

    inferred_type: InferredType
    matched_rule: str
    quality_flags: list[str] = field(default_factory=list)
    possible_formats: list[str] = field(default_factory=list)
    dominant_type: InferredType | None = None
    """For a MIXED column, the type most of its values fit.

    A column of mostly-dates with one bad value is still a date column; using
    the dominant type keeps the good values typed correctly, and the
    unconvertible values are reported rather than silently stored as text.
    """


def normalise_cell(raw: str | None) -> str:
    """Trim and collapse whitespace; map null sentinels to the empty string."""
    if raw is None:
        return ""
    value = raw.strip()
    if value.lower() in NULL_SENTINELS:
        return ""
    return value


def looks_like_pii(column_name: str) -> bool:
    """Whether a column name suggests personally identifiable information."""
    return bool(PII_NAME_RE.search(column_name))


def _try_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _try_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _parse_with(value: str, fmt: str) -> date | datetime | None:
    try:
        return datetime.strptime(value, fmt)
    except (TypeError, ValueError):
        return None


def infer_type(values: list[str], column_name: str = "") -> TypeGuess:
    """Infer the type of a column from its non-null values.

    ``values`` should be the *raw* cell strings; normalisation happens here.
    """
    cleaned = [normalise_cell(v) for v in values]
    non_null = [v for v in cleaned if v != ""]

    if not non_null:
        return TypeGuess(InferredType.EMPTY, "all_values_empty")

    total = len(non_null)
    # A type must cover essentially every value. A minority of text in a
    # numeric column is a data-quality problem, not a numeric column.
    strict = total >= 20
    ratio = 0.95 if strict else 1.0

    # Boolean
    lowered = {v.lower() for v in non_null}
    if lowered <= (TRUE_VALUES | FALSE_VALUES):
        # A column of only 0/1 that is also a unique candidate is far more
        # likely to be an identifier than a boolean flag.
        unique_ratio = len(set(non_null)) / total
        if not (unique_ratio == 1.0 and total > 1 and column_name.lower() in ("id", "key")):
            return TypeGuess(InferredType.BOOLEAN, "boolean_values")

    # Epoch timestamps are checked before plain integers, because an epoch
    # value is an integer. A temporal column name is required so that a
    # numeric identifier column is not mistaken for a timestamp.
    if _TEMPORAL_NAME_RE.search(column_name):
        epoch_hits = 0
        for v in non_null:
            parsed = _try_int(v)
            if parsed is not None and _EPOCH_SECONDS_MIN <= parsed <= _EPOCH_SECONDS_MAX:
                epoch_hits += 1
        if epoch_hits / total >= ratio:
            return TypeGuess(InferredType.TIMESTAMP, "epoch_seconds")

    # Integer
    int_hits = sum(1 for v in non_null if _try_int(v) is not None)
    if int_hits / total >= ratio:
        leading_zero = any(len(v) > 1 and v.startswith("0") and v.isdigit() for v in non_null)
        guess = TypeGuess(InferredType.INTEGER, "integer_values")
        if leading_zero:
            guess.quality_flags.append("leading_zero_ids")
        return guess

    # Float
    float_hits = sum(1 for v in non_null if _try_float(v) is not None)
    if float_hits / total >= ratio:
        return TypeGuess(InferredType.FLOAT, "float_values")

    # Date / datetime
    date_formats_hit: dict[str, int] = {}
    for fmt, name in DATE_FORMATS:
        hits = sum(1 for v in non_null if _parse_with(v, fmt) is not None)
        if hits / total >= ratio:
            date_formats_hit[name] = hits
    if date_formats_hit:
        names = sorted(date_formats_hit)
        guess = TypeGuess(InferredType.DATE, "date_values", possible_formats=names)
        if len(names) > 1:
            # Two or more formats parse every value — the day/month order is
            # genuinely ambiguous and must not be guessed.
            guess.quality_flags.append("ambiguous_date_format")
        return guess

    datetime_formats_hit: dict[str, int] = {}
    for fmt, name in DATETIME_FORMATS:
        hits = sum(1 for v in non_null if _parse_with(v, fmt) is not None)
        if hits / total >= ratio:
            datetime_formats_hit[name] = hits
    if datetime_formats_hit:
        names = sorted(datetime_formats_hit)
        return TypeGuess(
            InferredType.DATETIME, "datetime_values", possible_formats=names
        )

    # Epoch timestamps
    epoch_hits = 0
    for v in non_null:
        parsed = _try_int(v)
        if parsed is not None and _EPOCH_SECONDS_MIN <= parsed <= _EPOCH_SECONDS_MAX:
            epoch_hits += 1
    if epoch_hits / total >= ratio and _TEMPORAL_NAME_RE.search(column_name):
        return TypeGuess(InferredType.TIMESTAMP, "epoch_seconds")

    # A type that covers a clear majority but not every value: the column is
    # mixed. The dominant type is kept so that the good values are still
    # stored with the right type and the bad ones are reported.
    dominant = _dominant_type(non_null, total)
    if dominant is not None:
        return TypeGuess(
            InferredType.MIXED,
            f"mixed_with_{dominant.value}",
            quality_flags=["mixed_types"],
            dominant_type=dominant,
        )

    # Categorical — a column whose every value is distinct is an identifier
    # or free text, never a category. The column name is what separates a
    # low-cardinality attribute ("country") from an identifier ("customer_id")
    # when the file is too small for cardinality to tell them apart.
    distinct = len(set(non_null))
    distinct_ratio = distinct / total
    # Note: case-insensitive check for snake_case/exact key names + explicit camelCase `customerID`/`orderId`
    identifier_like = bool(
        KEY_NAME_RE.search(column_name)
        or re.search(r"(^id$|_id$|^uuid$|^guid$|_key$|^pk$|_pk$|code$|sku$|isbn$|email$|_no$|_number$)", column_name, re.IGNORECASE)
    )
    if distinct <= 50 and (distinct <= 10 or distinct_ratio <= 0.05):
        if identifier_like:
            return TypeGuess(InferredType.STRING, "identifier_like_strings")
        return TypeGuess(InferredType.CATEGORICAL, "low_cardinality")

    # All values distinct: an identifier-like string, or free text.
    avg_length = sum(len(v) for v in non_null) / total
    if distinct == total:
        if avg_length <= 64:
            return TypeGuess(InferredType.STRING, "unique_identifier_strings")
        return TypeGuess(InferredType.FREE_TEXT, "unique_long_strings")

    if identifier_like and avg_length <= 64:
        return TypeGuess(InferredType.STRING, "repeated_identifier_strings")

    return TypeGuess(InferredType.FREE_TEXT, "free_text")


def score_key_candidate(
    column: str,
    values: list[str],
    total_rows: int,
    inferred_type: InferredType | None = None,
) -> tuple[bool, int, Confidence, list[str]]:
    """Score a column as a primary-key candidate.

    Returns ``(is_candidate, score, confidence, evidence)``.

    Uniqueness alone is not sufficient. A date column with one row per day and
    a float column with one row per measurement are both "unique" by accident;
    neither is a useful business key, so those types are penalised.
    """
    evidence: list[str] = []
    score = 0

    non_null = [v for v in values if v != ""]
    distinct = len(set(non_null))

    if not non_null:
        return False, 0, Confidence.LOW, ["column is entirely empty"]

    if distinct == len(non_null):
        score += 40
        evidence.append(f"{distinct}/{len(non_null)} distinct values")
    else:
        score -= 50
        evidence.append(f"only {distinct} distinct values across {len(non_null)} rows — not unique")

    if len(non_null) < total_rows:
        score -= 60
        evidence.append("contains null values")

    if KEY_NAME_RE.search(column) or re.search(
        r"(^id$|_id$|^uuid$|^guid$|_key$|^pk$|_pk$|code$|sku$|isbn$|email$|_no$|_number$)",
        column,
        re.IGNORECASE,
    ):
        score += 25
        evidence.append("column name matches a key-name pattern")
    if FK_NAME_RE.search(column):
        # A weak signal only; the real FK test is value containment.
        score -= 5
        evidence.append("column name looks like a foreign key")

    if any(UUID_RE.match(v) for v in non_null):
        score += 15
        evidence.append("values are UUID-shaped")
    if any(EMAIL_RE.match(v) for v in non_null):
        score += 15
        evidence.append("values are email-shaped")

    # A pure integer key that is a dense 1..N run is a surrogate, not a
    # business key — still a valid identity, just less meaningful.
    ints = [_try_int(v) for v in non_null]
    if all(i is not None for i in ints) and ints:
        ordered = sorted(i for i in ints if i is not None)
        if ordered == list(range(ordered[0], ordered[0] + len(ordered))):
            score += 10
            evidence.append("sequential integer values — likely a surrogate key")
        else:
            score += 10
            evidence.append("non-sequential integer values")
    else:
        score += 10
        evidence.append("string-typed values")

    # Types that are unique by accident rather than by design.
    if inferred_type in (InferredType.DATE, InferredType.DATETIME, InferredType.TIMESTAMP):
        score -= 45
        evidence.append("column is a date/timestamp — uniqueness is likely coincidental")
    elif inferred_type == InferredType.FLOAT:
        score -= 35
        evidence.append("column is a float — uniqueness is likely coincidental")
    elif inferred_type == InferredType.CATEGORICAL:
        score -= 30
        evidence.append("column is categorical — a category is not a key")

    # In a very small file every column is unique by accident. A column with
    # no key-like name, and no UUID/email shape, is not a business key just
    # because the file has three rows.
    if total_rows < 20 and not KEY_NAME_RE.search(column) and not any(
        UUID_RE.match(v) for v in non_null
    ) and not any(EMAIL_RE.match(v) for v in non_null):
        score -= 20
        evidence.append(
            "small file and the column name is not identifier-like — "
            "uniqueness is likely coincidental"
        )

    if total_rows < 50:
        evidence.append("small sample — key uniqueness must be re-validated after ingestion")

    if score >= 60:
        confidence = Confidence.HIGH
    elif score >= 40:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    if total_rows < 50 and confidence == Confidence.HIGH:
        confidence = Confidence.MEDIUM

    is_candidate = score >= 40 and distinct == len(non_null) and len(non_null) == total_rows
    return is_candidate, score, confidence, evidence


def _dominant_type(non_null: list[str], total: int) -> InferredType | None:
    """The type that most of a mixed column's values fit.

    Used so that a column of mostly-dates with one bad value still stores its
    good values as dates.
    """
    scores: dict[InferredType, int] = {}

    def score(kind: InferredType, predicate: Any) -> None:
        hits = sum(1 for v in non_null if predicate(v))
        if hits:
            scores[kind] = hits

    score(
        InferredType.DATETIME,
        lambda v: any(_parse_with(v, fmt) is not None for fmt, _ in DATETIME_FORMATS),
    )
    score(
        InferredType.DATE,
        lambda v: any(_parse_with(v, fmt) is not None for fmt, _ in DATE_FORMATS),
    )
    score(InferredType.INTEGER, lambda v: _try_int(v) is not None)
    score(InferredType.FLOAT, lambda v: _try_float(v) is not None)

    if not scores:
        return None
    best = max(scores, key=lambda k: scores[k])
    # The dominant type must cover a clear majority, otherwise there is no
    # meaningful dominant type.
    if scores[best] / total < 0.6:
        return None
    return best


def best_key_column(columns: list[Any]) -> Any | None:
    """The column that should serve as a file's identity.

    A column that is *also* a candidate foreign key is perfectly normal — the
    question is whether it is the file's *own* identity. This picks the
    identity the same way the proposal does, so the two agree:

    * an ``_id``-style name wins over a natural key such as an email;
    * then the higher-cardinality column;
    * then the earlier column, which in practice is where a file lists its
      own primary key.
    """
    candidates = [c for c in columns if getattr(c, "is_candidate_key", False)]
    if not candidates:
        return None

    def rank(column: Any) -> tuple[int, int, int]:
        name = str(column.name).lower()
        id_like = 1 if name.endswith("_id") or name == "id" else 0
        return (id_like, int(getattr(column, "distinct_count", 0)), -int(column.position))

    candidates.sort(key=rank, reverse=True)
    return candidates[0]


def collect_null_sentinels(values: list[str]) -> list[str]:
    found: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        stripped = raw.strip()
        if stripped and stripped.lower() in NULL_SENTINELS:
            found.add(stripped)
    return sorted(found)


def top_values(values: list[str], limit: int = 10) -> list[list[Any]]:
    """The most frequent non-null values, as ``[value, count]`` pairs."""
    counter = Counter(v for v in values if v != "")
    return [[value, count] for value, count in counter.most_common(limit)]


def distinct_values(values: list[str]) -> set[str]:
    """The distinct non-null values of a column."""
    return {v for v in values if v != ""}
