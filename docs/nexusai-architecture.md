# NexusAI — Enterprise Agentic Knowledge Platform
## Production Architecture Document (v1.0)

---

## 1. Design principles

1. **Clean / hexagonal architecture** — domain logic never imports a framework. FastAPI, LangChain, Qdrant, Neo4j, Gemini are all *adapters* behind interfaces (ports). You can swap Qdrant for Weaviate or Gemini for Claude without touching business logic.
2. **Modular monorepo** — one repo, clearly separated apps and packages, independently deployable.
3. **API-first** — every capability is a versioned REST endpoint before it's a UI feature. OpenAPI schema is auto-generated and is the contract between frontend and backend.
4. **Async-first backend** — FastAPI async endpoints for I/O-bound work; Celery workers for anything that takes more than ~2 seconds (parsing, embedding, graph building).
5. **Stateless services, stateful stores** — API and worker processes hold no state between requests; all state lives in Postgres, Redis, Qdrant, Neo4j, or object storage. This is what lets you horizontally scale on Railway with zero code changes.
6. **Observability from day one** — structured JSON logs + LangSmith tracing on every LLM/agent call, not bolted on later.
7. **Security by default** — JWT auth, RBAC, input validation at the edge, secrets only via environment variables, least-privilege DB roles.
8. **12-factor config** — nothing environment-specific is hardcoded; everything comes from `.env` / platform env vars.
9. **Progressive delivery** — every milestone below is a working, demoable system. Nothing is "half-built and unrunnable" between milestones.

---

## 2. Layered architecture (backend)

```
interface/          → FastAPI routers, Pydantic request/response schemas, dependency wiring
application/         → use cases / services (IngestDocumentUseCase, QueryKnowledgeBaseUseCase, ...)
domain/              → entities (Document, Chunk, Conversation, Message, User) + port interfaces
                        (VectorStorePort, GraphStorePort, LLMPort, EmbeddingPort, RerankerPort)
infrastructure/      → concrete adapters: QdrantVectorStore, Neo4jGraphStore, GeminiLLMClient,
                        BGEEmbeddingClient, PostgresRepository, DoclingParser, PaddleOCRAdapter,
                        TavilyClient, MCPClientAdapter, S3ObjectStore
workers/             → Celery tasks (async ingestion, embedding, graph construction)
core/                → config (Pydantic Settings), logging, security, exception handlers
```

Dependency direction always points **inward**: `interface → application → domain ← infrastructure`. The domain layer defines the ports; infrastructure implements them; application orchestrates them via dependency injection (FastAPI `Depends`).

---

## 3. Data flow — RAG query (read path)

1. User sends a question from the chat UI → `POST /api/v1/chat/stream` (SSE).
2. Auth middleware validates the JWT, resolves `user_id` + `tenant/workspace_id`.
3. Conversation history is loaded (Redis for the active session, Postgres for durable history).
4. **Query rewriter** (LLM call) reformulates the question using conversation context — resolves pronouns, expands abbreviations.
5. **Router** classifies the query (needs graph reasoning? needs live web data? pure lexical/semantic lookup?) and picks a retrieval strategy.
6. **Retrieval**, run in parallel where applicable:
   - Qdrant hybrid search — dense (BGE-M3 dense vectors) + sparse (BGE-M3 lexical weights), fused with Reciprocal Rank Fusion.
   - Neo4j GraphRAG traversal — entity-anchored subgraph + community summaries, when the router flags the query as relationship-heavy.
   - MCP tool calls (e.g. Tavily web search) when the router flags the query as needing live/external data.
7. **Reranker** (BGE reranker cross-encoder) reorders the merged candidate set.
8. **Context compressor** trims to the token budget, keeping only passages that support the answer.
9. **Gemini 2.5 Flash** generates the answer with inline citations, streamed token-by-token over SSE.
10. LangSmith traces every step of 4–9 as a single run tree. The final answer, retrieved chunk IDs, and citations are persisted to Postgres.

## 4. Data flow — ingestion (write path)

1. User uploads a file, submits a URL, or points at a GitHub repo → `POST /api/v1/sources`.
2. Raw file goes to object storage (MinIO); a `KnowledgeSource` row is created in Postgres with `status=PENDING`.
3. A Celery task is enqueued on Redis.
4. Worker picks the right adapter: **Docling** for PDF/DOCX/CSV structural parsing, **PaddleOCR** as a fallback for scanned/image-only pages, a crawler for websites, a GitHub loader (clone + tree-walk) for repos.
5. Text is chunked (recursive/semantic splitter).
6. Chunks are embedded via **BGE-M3** (dense + sparse output in a single pass) and upserted into Qdrant as named vectors.
7. In parallel, an LLM extraction step pulls entities/relationships from the same chunks and upserts nodes/edges into **Neo4j** (GraphRAG construction — see Milestone 6 for community detection and summarization).
8. `KnowledgeSource.status` moves to `READY` (or `FAILED` with an error reason); the frontend is notified via polling or SSE.

---

## 5. Key architecture decisions

| Concern | Decision | Rationale |
|---|---|---|
| Auth | Custom JWT (access + refresh) with Postgres users + Redis refresh-token store | Self-hosted, no vendor lock-in, fits Docker/Railway deployment; OAuth/SSO can be layered on later |
| Task queue | Celery + Redis broker | Mature, well-documented, works cleanly with FastAPI; Flower gives free monitoring |
| Object storage | MinIO (S3-compatible) | Runs in Docker Compose locally, swaps to real S3 in production with zero code change |
| Hybrid search | BGE-M3 native dense + sparse vectors stored in Qdrant, fused with RRF | BGE-M3 produces both in one forward pass — no separate BM25 engine needed |
| Reranker | BGE-reranker-v2-m3 | Same model family as embeddings, self-hostable, strong multilingual performance |
| GraphRAG | `neo4j-graphrag-python` + custom LLM extraction/community-summarization pipeline | Official Neo4j tooling, avoids vendoring Microsoft GraphRAG's opinionated storage layer |
| MCP | Official `mcp` Python SDK — NexusAI exposes an MCP server (internal tools) and an MCP client (Tavily + others) | Matches the explicit MCP tool-calling requirement, keeps tools swappable |
| LLM | Gemini 2.5 Flash via `langchain-google-genai` | Required by spec; wrapped behind `LLMPort` so it's swappable |
| Embeddings serving | In-process `FlagEmbedding`/`sentence-transformers` for early milestones → optional dedicated TEI (text-embeddings-inference) container for production scale | Keeps early milestones simple to run; documented upgrade path |
| Chat streaming | Server-Sent Events from FastAPI, consumed via Vercel AI SDK on the frontend | Simpler than WebSockets for one-directional token streaming, first-class Next.js support |

---

## 6. Monorepo structure (top level)

```
nexusai/
├── apps/
│   ├── frontend/                 # Next.js 14+ (App Router), TypeScript, Tailwind, ShadCN
│   └── backend/                  # FastAPI, Python 3.11+, Celery workers
├── infra/
│   ├── docker/                   # per-service Dockerfiles referenced by docker-compose
│   └── railway/                  # railway.json / service configs
├── docs/
│   ├── architecture.md           # this document
│   └── milestones/               # one spec file per milestone
├── scripts/                      # dev convenience scripts (bootstrap.sh, seed.py, ...)
├── .github/workflows/            # CI: lint, test, build, deploy
├── docker-compose.yml            # orchestrates all services for local dev
├── .env.example
└── README.md
```

Each milestone below only adds the services/files it actually needs to `docker-compose.yml` — Qdrant arrives at Milestone 4, Neo4j at Milestone 6, MinIO at Milestone 3 — so the compose file never has dead, unconfigured services sitting in it.

---

## 7. Cross-cutting concerns

**Security**: JWT access tokens (short-lived, ~15 min) + refresh tokens (Redis-backed, revocable); RBAC roles (`admin`, `member`, `viewer`) enforced via FastAPI dependencies; Pydantic validates every request body; secrets never committed, `.env.example` documents required keys only; non-root Docker users; CORS locked to known frontend origins.

**Scalability**: API and worker containers are stateless and horizontally scalable behind Railway's load balancer; Celery concurrency is tuned per queue (ingestion vs. quick jobs); Qdrant and Neo4j can move to managed services (Qdrant Cloud, Neo4j Aura) at production scale without code changes, since both sit behind ports.

**Observability**: structured JSON logs with request IDs (correlatable across API → worker → LLM call); LangSmith traces every chain/agent run; `/health`, `/health/db`, `/health/redis`, `/health/qdrant`, `/health/neo4j` endpoints for uptime checks and container orchestration.

**Testing strategy**: unit tests for domain/application logic (no I/O, fast), integration tests against real Postgres/Redis/Qdrant/Neo4j via `docker-compose.test.yml` or testcontainers, contract tests on the OpenAPI schema, and a smoke-test checklist per milestone (included in every milestone spec).

---

## 8. Milestone roadmap

| # | Milestone | What it proves works |
|---|---|---|
| 1 | **Foundation & core infrastructure** | Monorepo, Docker Compose, FastAPI + Next.js talking to each other, Postgres/Redis health checks, CI skeleton |
| 2 | **Authentication & user management** | Register/login/refresh, JWT + RBAC, protected routes, auth UI |
| 3 | **Document ingestion (core)** | Upload API, object storage, Docling parsing (PDF/DOCX/CSV), PaddleOCR fallback, chunking, Celery pipeline |
| 4 | **Vector search & basic RAG** | BGE-M3 embeddings, Qdrant indexing, semantic search API, first end-to-end RAG chat with Gemini 2.5 Flash |
| 5 | **Hybrid search & query enhancement** | BGE-M3 sparse vectors + RRF fusion, query rewriting, reranking, context compression |
| 6 | **Knowledge graph & GraphRAG** | Neo4j setup, entity/relationship extraction, community detection, graph-augmented retrieval |
| 7 | **Website, GitHub & web search ingestion** | Crawler service, GitHub repo loader, Tavily live web search as a retrieval source |
| 8 | **Agentic layer & MCP tool calling** | LangGraph agent orchestration, MCP server + client, multi-step tool use |
| 9 | **Conversation memory, citations & observability** | Durable multi-turn memory, inline source citations, full LangSmith tracing, SSE streaming polish |
| 10 | **Full frontend application** | Chat UI, knowledge source manager, graph explorer, citation viewer, responsive ShadCN UI |
| 11 | **Production hardening & deployment** | Rate limiting, security review, production Docker images, Railway (backend/workers/DBs) + Vercel (frontend), CI/CD |

Each milestone is independently runnable via `docker compose up` at that point in the build — nothing later is required to demo what came before.

Full detail (folder structure, diagram, file list, APIs, DB schema, env vars, install steps, packages, best practices, common bugs, testing checklist) is provided one milestone at a time, starting with Milestone 1 in `nexusai-milestone-01-foundation.md`.
