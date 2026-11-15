"""Domain entities for the CSV-to-knowledge-graph subsystem.

The central object is :class:`GraphSchemaProposal` — a complete, reviewable,
serialisable description of the graph that a set of CSV files *should*
produce. It is produced by deterministic inference (never by the LLM) and it
is the only thing that authorises a write to Neo4j.

Keeping this in the domain layer means the proposal can be stored, versioned,
edited and rejected without any graph state, and that the ingestion path and
the query path can both be grounded in the same model.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Neo4j identifiers (labels, relationship types, property keys) cannot be
# parameterised in Cypher, so they must be interpolated. Every identifier that
# reaches a query passes through this pattern first.
#
# ``\Z`` is used rather than ``$`` on purpose: ``$`` also matches just before
# a trailing newline, which would let "Label\n" through and put a newline into
# a generated query.
IDENTIFIER_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]{0,127}\Z")


class UnsafeIdentifierError(ValueError):
    """Raised when a value cannot be safely used as a Cypher identifier."""


def validate_identifier(value: str, kind: str = "identifier") -> str:
    """Return ``value`` if it is a safe Cypher identifier, else raise.

    This is a security control, not hygiene: an identifier that reaches a
    query is interpolated into the query text.
    """
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise UnsafeIdentifierError(f"Rejected {kind}: {value!r}")
    return value


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InferredType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    STRING = "string"
    CATEGORICAL = "categorical"
    FREE_TEXT = "free_text"
    EMPTY = "empty"
    MIXED = "mixed"


class PropertyType(str, Enum):
    """Neo4j property types that can be written."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LOCAL_DATETIME = "local_datetime"
    TIME = "time"
    LIST_STRING = "list[string]"
    LIST_INTEGER = "list[integer]"
    LIST_FLOAT = "list[float]"


class Normalisation(str, Enum):
    TRIM = "trim"
    TRIM_LOWER = "trim_lower"
    TRIM_UPPER = "trim_upper"
    WHITESPACE_COLLAPSE = "whitespace_collapse"
    STRIP_CURRENCY = "strip_currency"
    STRIP_PERCENT = "strip_percent"
    STRIP_THOUSANDS = "strip_thousands"
    PARSE_DATE = "parse_date"
    PARSE_DATETIME = "parse_datetime"
    LEADING_ZERO_STRING = "leading_zero_string"


class EdgeIdentity(str, Enum):
    """What makes an edge unique.

    ``ENDPOINTS`` is the classic many-to-one edge: ``(a)-[:REL]->(b)``.
    ``ENDPOINTS_PLUS_KEY`` additionally includes a business key property so
    that distinct rows of the same kind stay distinct edges.
    """

    ENDPOINTS = "endpoints"
    ENDPOINTS_PLUS_KEY = "endpoints+key"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class ProposalStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    INGESTED = "ingested"


class IngestionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Column level ─────────────────────────────────────────────────────────────


class ColumnProfile(BaseModel):
    """Everything observed about one CSV column."""

    name: str
    normalized_name: str
    position: int
    inferred_type: InferredType = InferredType.STRING
    property_type: PropertyType | None = None
    dominant_type: InferredType | None = None
    null_ratio: float = 0.0
    distinct_count: int = 0
    distinct_count_is_estimate: bool = False
    non_null_count: int = 0
    total_count: int = 0
    min_value: Any = None
    max_value: Any = None
    mean_value: float | None = None
    sample_values: list[Any] = Field(default_factory=list)
    top_values: list[list[Any]] = Field(default_factory=list)
    null_sentinels_found: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    may_contain_pii: bool = False
    is_candidate_key: bool = False
    key_confidence: Confidence = Confidence.LOW
    is_candidate_fk: bool = False
    fk_target_file: str | None = None
    fk_target_column: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("normalized_name")
    @classmethod
    def _safe_normalized_name(cls, v: str) -> str:
        return validate_identifier(v, "column name") if v else v


# ── File level ───────────────────────────────────────────────────────────────


class FileProfile(BaseModel):
    """Profiling result for one uploaded CSV."""

    file_id: str
    filename: str
    source_id: uuid.UUID
    size_bytes: int = 0
    encoding: str = "utf-8"
    delimiter: str = ","
    has_header: bool = True
    total_rows: int = 0
    sampled_rows: int = 0
    column_count: int = 0
    columns: list[ColumnProfile] = Field(default_factory=list)
    duplicate_row_count: int = 0
    duplicate_row_ratio: float = 0.0
    blank_line_count: int = 0
    ragged_row_count: int = 0
    proposed_label: str = ""
    label_confidence: Confidence = Confidence.LOW
    label_derivation: str = ""
    warnings: list[str] = Field(default_factory=list)
    content_hash: str = ""

    def column(self, name: str) -> ColumnProfile | None:
        for col in self.columns:
            if col.name == name:
                return col
        return None


# ── Graph definition level ───────────────────────────────────────────────────


class PropertyMapping(BaseModel):
    """How one CSV column becomes one node/edge property."""

    source_column: str
    target_property: str
    property_type: PropertyType = PropertyType.STRING
    normalisation: list[Normalisation] = Field(default_factory=list)
    required: bool = False
    nullable: bool = True
    conversion_error_policy: Literal["reject_row", "null_value", "keep_string"] = "reject_row"
    confidence: Confidence = Confidence.MEDIUM
    notes: list[str] = Field(default_factory=list)

    @field_validator("target_property")
    @classmethod
    def _safe_property_name(cls, v: str) -> str:
        return validate_identifier(v, "property name")


class KeyDefinition(BaseModel):
    """A uniqueness key on a node definition."""

    name: str
    columns: list[str]
    is_primary: bool = True
    confidence: Confidence = Confidence.MEDIUM
    evidence: str = ""


class NodeDefinition(BaseModel):
    """A proposed node label and its properties."""

    label: str
    source_file: str
    source_id: uuid.UUID
    identity_columns: list[str] = Field(default_factory=list)
    key: KeyDefinition
    property_mappings: list[PropertyMapping] = Field(default_factory=list)
    properties: dict[str, PropertyType] = Field(default_factory=dict)
    required_properties: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    expected_node_count: int = 0
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("label")
    @classmethod
    def _valid_label(cls, v: str) -> str:
        return validate_identifier(v, "label")

    @model_validator(mode="after")
    def _key_present(self) -> "NodeDefinition":
        if not self.identity_columns:
            raise ValueError(f"Node {self.label!r} has no identity column")
        if not self.key.columns:
            raise ValueError(f"Node {self.label!r} has no key definition")
        return self


class RelationshipDefinition(BaseModel):
    """A proposed relationship type between two node labels."""

    relationship_type: str
    source_label: str
    target_label: str
    source_file: str
    source_id: uuid.UUID
    source_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    direction: Literal["outbound", "inbound"] = "outbound"
    edge_identity: EdgeIdentity = EdgeIdentity.ENDPOINTS
    identity_key_columns: list[str] = Field(default_factory=list)
    property_mappings: list[PropertyMapping] = Field(default_factory=list)
    properties: dict[str, PropertyType] = Field(default_factory=dict)
    cardinality: Literal["one_to_one", "many_to_one", "many_to_many"] = "many_to_one"
    match_ratio: float = 0.0
    unmatched_value_count: int = 0
    confidence: Confidence = Confidence.MEDIUM
    expected_relationship_count: int = 0
    requires_direction_confirmation: bool = False
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("relationship_type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        return validate_identifier(v, "relationship type")


# ── Constraints / indexes / validation ──────────────────────────────────────


class ConstraintDefinition(BaseModel):
    statement: str
    label: str
    properties: list[str] = Field(default_factory=list)
    kind: Literal["uniqueness", "node_key", "existence"] = "uniqueness"
    rationale: str = ""


class IndexDefinition(BaseModel):
    statement: str
    label: str
    properties: list[str] = Field(default_factory=list)
    kind: Literal["range", "text", "fulltext", "point"] = "range"
    rationale: str = ""


class ValidationRule(BaseModel):
    rule_id: str
    name: str
    scope: Literal["row", "node", "relationship", "graph"] = "row"
    applies_to: str
    column: str | None = None
    condition: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True
    error_message: str = ""


# ── The proposal ─────────────────────────────────────────────────────────────


class GraphSchemaProposal(BaseModel):
    """The complete, reviewable, serialisable graph schema for one upload set."""

    proposal_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    version: int = 1
    status: ProposalStatus = ProposalStatus.PENDING_REVIEW
    created_at: str = Field(default_factory=_utcnow_iso)
    created_by: str = ""
    approved_at: str | None = None
    approved_by: str | None = None
    rejected_at: str | None = None
    rejection_reason: str | None = None

    source_ids: list[uuid.UUID] = Field(default_factory=list)
    files: list[FileProfile] = Field(default_factory=list)

    nodes: list[NodeDefinition] = Field(default_factory=list)
    relationships: list[RelationshipDefinition] = Field(default_factory=list)

    constraints: list[ConstraintDefinition] = Field(default_factory=list)
    indexes: list[IndexDefinition] = Field(default_factory=list)
    validation_rules: list[ValidationRule] = Field(default_factory=list)

    overall_confidence: Confidence = Confidence.MEDIUM
    ambiguities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    requires_approval: bool = True
    notes: str = ""

    @model_validator(mode="after")
    def _check_relationship_endpoints(self) -> "GraphSchemaProposal":
        """Every relationship must reference a defined node label."""
        labels = {n.label for n in self.nodes}
        for rel in self.relationships:
            if rel.source_label not in labels:
                raise ValueError(
                    f"Relationship {rel.relationship_type!r} references unknown label "
                    f"{rel.source_label!r}"
                )
            if rel.target_label not in labels:
                raise ValueError(
                    f"Relationship {rel.relationship_type!r} references unknown label "
                    f"{rel.target_label!r}"
                )
        return self

    def node(self, label: str) -> NodeDefinition | None:
        for node in self.nodes:
            if node.label == label:
                return node
        return None

    def file(self, filename: str) -> FileProfile | None:
        for profile in self.files:
            if profile.filename == filename:
                return profile
        return None


# ── Ingestion reporting ─────────────────────────────────────────────────────


class RejectedRow(BaseModel):
    file: str
    row: int
    column: str | None = None
    raw_value: str | None = None
    rule_id: str = ""
    message: str = ""
    action: str = "rejected"


class FileIngestionResult(BaseModel):
    file_id: str
    filename: str
    rows_read: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0
    nodes_created: int = 0
    nodes_matched: int = 0
    relationships_created: int = 0
    relationships_matched: int = 0
    unmatched_relationship_rows: int = 0
    duplicate_rows_skipped: int = 0
    conversion_failures: list[dict[str, Any]] = Field(default_factory=list)
    duplicate_candidates: list[dict[str, Any]] = Field(default_factory=list)
    rejected_rows: list[RejectedRow] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ValidationCheck(BaseModel):
    check_id: str
    status: Literal["pass", "warn", "fail", "skipped"] = "pass"
    expected: Any = None
    actual: Any = None
    detail: str = ""


class ValidationReport(BaseModel):
    checks: list[ValidationCheck] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def failed(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warned(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "warn"]


class IngestionReport(BaseModel):
    """The full result of one ingestion run."""

    job_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    proposal_id: uuid.UUID
    status: IngestionStatus = IngestionStatus.PENDING
    dry_run: bool = False
    started_at: str = Field(default_factory=_utcnow_iso)
    finished_at: str | None = None
    duration_seconds: float = 0.0
    files: list[FileIngestionResult] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    unresolved_mappings: list[str] = Field(default_factory=list)
    statements_executed: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


__all__ = [
    "IDENTIFIER_RE",
    "Confidence",
    "EdgeIdentity",
    "FileIngestionResult",
    "FileProfile",
    "GraphSchemaProposal",
    "InferredType",
    "IngestionReport",
    "IngestionStatus",
    "KeyDefinition",
    "NodeDefinition",
    "Normalisation",
    "PropertyMapping",
    "PropertyType",
    "ProposalStatus",
    "RejectedRow",
    "RelationshipDefinition",
    "UnsafeIdentifierError",
    "ValidationCheck",
    "ValidationReport",
    "ValidationRule",
    "ValidationSeverity",
    "validate_identifier",
]
