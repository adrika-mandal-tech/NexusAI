"""CSV-to-knowledge-graph subsystem.

The pipeline is: upload → validate → profile → discover relationships →
propose a schema → human approval → generate a Cypher plan → transactional
ingestion → validation → statistics → visualisation → query.

Inference is deterministic and separate from writes. The LLM is not in the
structural path: it may only *suggest* names, and every name is validated
before it can reach a query.
"""

from nexusai.application.graph.csv.profiler import (
    CsvFile,
    CsvValidationError,
    detect_delimiter,
    normalise_column_name,
    profile_csv,
    propose_label,
    read_rows,
    singularise,
)
from nexusai.application.graph.csv.type_inference import (
    infer_type,
    normalise_cell,
    score_key_candidate,
)

__all__ = [
    "CsvFile",
    "CsvValidationError",
    "detect_delimiter",
    "infer_type",
    "normalise_cell",
    "normalise_column_name",
    "propose_label",
    "profile_csv",
    "read_rows",
    "score_key_candidate",
    "singularise",
]
