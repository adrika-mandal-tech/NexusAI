# NexusAI — Single-Agent Hybrid RAG Knowledge Assistant

NexusAI is a production-oriented, source-grounded **single-agent Hybrid Retrieval-Augmented Generation (RAG)** knowledge assistant combining intelligent document processing (**Docling**, **PyPDF**, **PaddleOCR / PP-OCRv6**), **CSV-to-Knowledge-Graph** construction, **BAAI/bge-m3** dense + lexical sparse embeddings in **Qdrant**, **Neo4j** entity/graph traversal, **Reciprocal Rank Fusion (RRF) + Calibrated Reranking**, **Tavily** external web research, **LangChain** single-agent tool orchestration, **Gemini / OpenLM** grounded synthesis, and a **Next.js 14** workspace UI.

---

## Live Running URLs

- **Next.js Frontend UI**: `http://localhost:3001`
  - Workspace & Ingestion (`/work`): `http://localhost:3001/work`
  - Grounded Hybrid RAG Chat (`/chat`): `http://localhost:3001/chat`
  - Knowledge Graph & CSV-to-KG Workbench (`/graph`): `http://localhost:3001/graph`
  - Multi-Channel Search & Web Research (`/research`): `http://localhost:3001/research`
  - Observability & Comparative RAG Evaluation (`/evaluation`): `http://localhost:3001/evaluation`
  - Runtime Settings & Dependency Health (`/settings`): `http://localhost:3001/settings`
- **FastAPI Backend API**: `http://localhost:8000`
  - OpenAPI / Swagger Docs: `http://localhost:8000/docs`
  - Deep Dependency Readiness: `http://localhost:8000/api/v1/health/dependencies`

> Note: Port `3001` is used for the Next.js frontend when port `3000` is occupied by local MCP tooling (`code-assist-mcp`). `NEXT_PUBLIC_API_BASE_URL` in `apps/frontend/.env.local` points directly to `http://localhost:8000`.

---

## Quick Start

### 1. Start Infrastructure (Qdrant, Neo4j, PostgreSQL, Redis)
Ensure Qdrant (`localhost:6333`), Neo4j (`localhost:7687`), and PostgreSQL (`localhost:5432/nexusai`) are running (see [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md)).

### 2. Run Backend (`apps/backend`)
```bash
cd apps/backend
.venv/bin/uvicorn nexusai.main:create_app --factory --host 0.0.0.0 --port 8000
```

### 3. Run Frontend (`apps/frontend`)
```bash
cd apps/frontend
npm run dev -- -p 3001
```

### 4. Run Full Automated Test Suite (`198` Backend Tests + Frontend Build)
```bash
cd apps/backend && .venv/bin/pytest -v
cd apps/frontend && npx tsc --noEmit && npm run build
```

---

## Architecture & Core Modules

### 1. Hybrid RAG Pipeline & Multi-Channel Retrieval
- **Dense Vector Search**: Powered by `BAAI/bge-m3` (1024-dimensional embeddings) indexed in **Qdrant**.
- **Sparse / Lexical Search**: BM25 inverted index for exact keyword, acronym (e.g., SIMT, ECC), and identifier matching.
- **Neo4j Knowledge Graph Traversal**: Extracts entity nodes and relationships to enrich context with relational facts.
- **Reciprocal Rank Fusion (RRF) & Calibrated Reranking**: Merges candidates across channels and scores them before context assembly.

### 2. Knowledge Graph Studio (`/graph`)
- **Strictly Dataset-Specific**: Upload CSV/XLSX spreadsheets to generate authentic Neo4j nodes, edges, and properties.
- **Interactive Neo4j Browser Layout**: Visual legend with colored buttons describing entity types, relationship badges, and attribute breakdown.
- **Dynamic Motion & Compact Cypher Console**: Seamless drag, pan, zoom, and query with real-time Cypher execution.

### 3. RAG Observability, Diagnostics & Evaluation (`/evaluation`)
- **Live Telemetry**: Tracks request counts, average latency, P95 latency, and token consumption via ring buffer.
- **Empirical RAG Benchmark**: Benchmarks `dense_only` vs `hybrid_rrf` vs `hybrid_reranked` across 7 key metrics:
  - *Context Precision*, *Context Recall*, *Retrieval Relevance*, *Faithfulness*, *Citation Correctness*, *Abstention Accuracy*, and *Latency*.
- **Direct Retrieval Channel Inspector**: Test queries across isolated channels (Dense, Sparse, Graph, Web) to inspect raw candidate ranks and fusion scores.

### 4. Interactive Data Analytics Chat (`/chat`)
- In-chat data analyst agent featuring automatic statistical KPI extraction, correlation detection, and interactive SVG charts with Plotly/Seaborn palette toggles.
