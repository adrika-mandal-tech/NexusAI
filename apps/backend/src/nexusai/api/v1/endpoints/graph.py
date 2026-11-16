"""CSV-to-knowledge-graph API endpoints.

The workflow is deliberately multi-step: nothing is written to Neo4j until a
proposal has been reviewed and approved.

    POST   /graph/csv/upload                 upload CSVs -> a schema proposal
    GET    /graph/csv/proposal/{id}          the reviewable proposal
    PUT    /graph/csv/proposal/{id}          edit the proposal
    POST   /graph/csv/proposal/{id}/approve  approve it
    POST   /graph/csv/proposal/{id}/reject   reject it
    GET    /graph/csv/plan/{id}              the Cypher plan, before execution
    POST   /graph/csv/ingest                 run the ingestion
    GET    /graph/csv/jobs/{id}              the ingestion report
    GET    /graph/csv/stats                  per-file graph statistics
    GET    /graph/csv/viz                    nodes and edges for the graph view
    POST   /graph/csv/ask                    a natural-language question
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from nexusai.api.v1.deps import get_current_user
from nexusai.application.graph.csv.cypher_planner import render_plan_for_review
from nexusai.application.graph.csv.graph_qa import GraphQaService
from nexusai.application.graph.csv.service import (
    CsvGraphService,
    ProposalNotApproved,
    ProposalNotFound,
)
from nexusai.application.graph.csv.statistics import (
    collect_statistics,
    collect_visualisation,
)
from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.infrastructure.csv.csv_loader import CsvUploadError, store_csv
from nexusai.infrastructure.graph_stores.neo4j_readonly_store import (
    Neo4jReadOnlyStore,
    ReadOnlyStoreUnavailable,
)
from nexusai.infrastructure.graph_stores.neo4j_store import Neo4jGraphStore
from nexusai.infrastructure.llm import GeminiLLMClient, GroqLLMClient

logger = get_logger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


def _get_service() -> CsvGraphService:
    return CsvGraphService(Neo4jGraphStore())


def _get_read_only_store() -> Neo4jReadOnlyStore:
    try:
        return Neo4jReadOnlyStore()
    except ReadOnlyStoreUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


def _resolve_files(proposal: Any) -> dict[str, tuple[Path, str, str]]:
    """Locate the stored uploads for a proposal's files."""
    settings = get_settings()
    upload_dir = Path(settings.UPLOAD_DIR)
    files: dict[str, tuple[Path, str, str]] = {}
    for profile in proposal.files:
        path = upload_dir / f"{profile.file_id}.csv"
        if not path.exists():
            alt = next(iter(upload_dir.glob(f"{profile.file_id}.*")), None)
            if alt is None:
                raise HTTPException(
                    status_code=status.HTTP_410_GONE,
                    detail=(
                        f"The uploaded file for {profile.filename} is no longer available. "
                        "Upload the CSVs again."
                    ),
                )
            path = alt
        files[profile.filename] = (path, profile.encoding, profile.delimiter)
    return files


@router.post("/csv/upload", status_code=201)
async def upload_csvs(
    files: list[UploadFile] = File(...),
    include_pii: bool = Form(False),
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """Upload one or more CSVs and produce a reviewable schema proposal."""
    settings = get_settings()
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > settings.CSV_MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"At most {settings.CSV_MAX_FILES} files can be uploaded at once.",
        )

    source_id = uuid.uuid4()
    stored = []
    for upload in files:
        content = await upload.read()
        try:
            stored.append(
                store_csv(
                    content=content,
                    filename=upload.filename or "upload.csv",
                    source_id=source_id,
                    content_type=upload.content_type or "",
                )
            )
        except CsvUploadError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    service = _get_service()
    try:
        proposal = await service.create_proposal(
            stored, include_pii=include_pii, created_by=str(user_id)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "proposal_id": str(proposal.proposal_id),
        "source_id": str(source_id),
        "status": proposal.status.value,
        "overall_confidence": proposal.overall_confidence.value,
        "files": [p.model_dump(mode="json") for p in proposal.files],
        "nodes": [n.model_dump(mode="json") for n in proposal.nodes],
        "relationships": [r.model_dump(mode="json") for r in proposal.relationships],
        "constraints": [c.model_dump(mode="json") for c in proposal.constraints],
        "indexes": [i.model_dump(mode="json") for i in proposal.indexes],
        "validation_rules": [v.model_dump(mode="json") for v in proposal.validation_rules],
        "ambiguities": proposal.ambiguities,
        "warnings": proposal.warnings,
        "requires_approval": proposal.requires_approval,
    }


@router.get("/csv/proposal/{proposal_id}")
async def get_proposal(
    proposal_id: str,
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """The full, reviewable schema proposal."""
    service = _get_service()
    try:
        proposal = await service.load_proposal(proposal_id)
    except ProposalNotFound as exc:
        raise HTTPException(status_code=404, detail="Proposal not found.") from exc
    return proposal.model_dump(mode="json")


@router.put("/csv/proposal/{proposal_id}")
async def update_proposal(
    proposal_id: str,
    body: dict[str, Any],
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """Edit a proposal and store it as a new version."""
    service = _get_service()
    try:
        proposal = await service.load_proposal(proposal_id)
    except ProposalNotFound as exc:
        raise HTTPException(status_code=404, detail="Proposal not found.") from exc

    if "nodes" in body:
        for index, node in enumerate(body["nodes"]):
            if index < len(proposal.nodes):
                for key, value in node.items():
                    if hasattr(proposal.nodes[index], key):
                        setattr(proposal.nodes[index], key, value)
    if "relationships" in body:
        for index, rel in enumerate(body["relationships"]):
            if index < len(proposal.relationships):
                for key, value in rel.items():
                    if hasattr(proposal.relationships[index], key):
                        setattr(proposal.relationships[index], key, value)

    try:
        updated = await service.update_proposal(proposal)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return updated.model_dump(mode="json")


@router.post("/csv/proposal/{proposal_id}/approve")
async def approve_proposal(
    proposal_id: str,
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """Approve a proposal. This is the only thing that authorises a write."""
    service = _get_service()
    try:
        proposal = await service.approve_proposal(proposal_id, str(user_id))
    except ProposalNotFound as exc:
        raise HTTPException(status_code=404, detail="Proposal not found.") from exc
    except ProposalNotApproved as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return proposal.model_dump(mode="json")


@router.post("/csv/proposal/{proposal_id}/reject")
async def reject_proposal(
    proposal_id: str,
    body: dict[str, Any],
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """Reject a proposal."""
    service = _get_service()
    try:
        proposal = await service.reject_proposal(
            proposal_id, str(body.get("reason", "")), str(user_id)
        )
    except ProposalNotFound as exc:
        raise HTTPException(status_code=404, detail="Proposal not found.") from exc
    return proposal.model_dump(mode="json")


@router.get("/csv/plan/{proposal_id}")
async def get_plan(
    proposal_id: str,
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """The Cypher plan for an approved proposal, before it is executed."""
    service = _get_service()
    try:
        proposal = await service.load_proposal(proposal_id)
    except ProposalNotFound as exc:
        raise HTTPException(status_code=404, detail="Proposal not found.") from exc
    try:
        plan = service.build_plan_for(proposal)
    except ProposalNotApproved as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "proposal_id": proposal_id,
        "summary": plan.summary(),
        "statements": plan.all_statements(),
        "cypher": render_plan_for_review(plan),
    }


@router.post("/csv/ingest")
async def ingest(
    body: dict[str, Any],
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """Run the ingestion for an approved proposal."""
    proposal_id = body.get("proposal_id")
    if not proposal_id:
        raise HTTPException(status_code=400, detail="proposal_id is required.")
    dry_run = bool(body.get("dry_run", False))

    service = _get_service()
    try:
        proposal = await service.load_proposal(proposal_id)
    except ProposalNotFound as exc:
        raise HTTPException(status_code=404, detail="Proposal not found.") from exc

    files = _resolve_files(proposal)
    try:
        report = await service.ingest(proposal, files, dry_run=dry_run)
    except ProposalNotApproved as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await _store_report(report)
    return report.model_dump(mode="json")


async def _store_report(report: Any) -> None:
    """Attach the report to the job node so it can be fetched later."""
    store = Neo4jGraphStore()
    try:
        await store.run_write(
            "MATCH (j:IngestionJob {id: $job_id}) SET j.report = $report",
            {
                "job_id": str(report.job_id),
                # A Neo4j property holds primitives only, so the report is
                # stored as JSON.
                "report": json.dumps(report.model_dump(mode="json")),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist the ingestion report: %s", exc)


@router.get("/csv/jobs/{job_id}")
async def get_job(
    job_id: str,
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """The ingestion report for a job."""
    service = _get_service()
    report = await service.load_report(job_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return report


@router.get("/csv/stats")
async def get_stats(
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """Per-file graph statistics."""
    return await collect_statistics(Neo4jGraphStore())


@router.get("/csv/viz")
async def get_visualisation(
    node_limit: int = Query(500, ge=1, le=2000),
    edge_limit: int = Query(1000, ge=1, le=5000),
    source_file: str | None = Query(None),
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """Nodes and edges for the graph view strictly for uploaded dataset spreadsheets."""
    return await collect_visualisation(
        Neo4jGraphStore(),
        node_limit=node_limit,
        edge_limit=edge_limit,
        source_file=source_file,
    )


@router.post("/csv/ask")
async def ask(
    body: dict[str, Any],
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """Answer a natural-language question from the graph."""
    question = str(body.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required.")

    provider = str(body.get("provider") or get_settings().LLM_PROVIDER or "gemini").lower()
    llm = GroqLLMClient() if provider == "groq" else GeminiLLMClient()

    service = GraphQaService(read_only_store=_get_read_only_store(), llm=llm)
    result = await service.answer(question)
    return result.to_dict()


@router.get("/csv/files")
async def list_ingested_files(
    user_id: uuid.UUID = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """List all ingested spreadsheet files with node & edge counts."""
    store = Neo4jGraphStore()
    records = await store.run_read(
        "MATCH (n) WHERE n.source_file IS NOT NULL "
        "RETURN DISTINCT n.source_file AS filename, count(n) AS node_count, "
        "collect(DISTINCT labels(n)[0])[0..5] AS sample_labels"
    )
    return [
        {
            "filename": str(r.get("filename")),
            "node_count": int(r.get("node_count") or 0),
            "sample_labels": r.get("sample_labels") or [],
        }
        for r in records
        if r.get("filename")
    ]


@router.delete("/csv/files/{filename}")
async def delete_ingested_file(
    filename: str,
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """Delete all nodes, relationships, and metadata for a previously uploaded spreadsheet file."""
    store = Neo4jGraphStore()
    await store.run_write(
        "MATCH (n) WHERE n.source_file = $filename DETACH DELETE n",
        {"filename": filename},
    )
    await store.run_write(
        "MATCH (f:SourceFile {filename: $filename}) DETACH DELETE f",
        {"filename": filename},
    )
    return {"status": "ok", "deleted_file": filename}


@router.post("/csv/clear")
async def clear_all_csv_datasets(
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """Clear all previously ingested spreadsheet data to reset the graph."""
    store = Neo4jGraphStore()
    await store.run_write(
        "MATCH (n) WHERE n.kind = 'csv_ingest' OR n.source_file IS NOT NULL DETACH DELETE n"
    )
    await store.run_write("MATCH (f:SourceFile) DETACH DELETE f")
    await store.run_write("MATCH (p:GraphSchemaProposal) DETACH DELETE p")
    await store.run_write("MATCH (j:IngestionJob) DETACH DELETE j")
    return {"status": "ok", "message": "All CSV/Excel spreadsheet graph datasets cleared."}
