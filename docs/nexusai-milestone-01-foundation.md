# Milestone 1 — Foundation & Core Infrastructure

**Goal**: a running skeleton. Docker Compose brings up Postgres, Redis, a FastAPI backend, and a Next.js frontend that can prove they can all talk to each other. No business logic yet — this milestone exists so every later milestone has solid ground to build on.

**Definition of done**: `docker compose up`, open the frontend, see a live status page showing "API: healthy", "Database: healthy", "Cache: healthy", pulled from the backend in real time.

---

## 1. Architecture diagram (this milestone only)

Only a subset of the full system is live at this point:

```
┌─────────────────────────┐        ┌─────────────────────────┐
│  Next.js frontend        │ HTTP   │  FastAPI backend         │
│  app/page.tsx status page│──────▶│  /api/v1/health/*         │
└─────────────────────────┘        └───────────┬──────┬───────┘
                                                │      │
                                     asyncpg    │      │  redis-py
                                                ▼      ▼
                                     ┌──────────────┐ ┌──────────────┐
                                     │  Postgres 16  │ │  Redis 7     │
                                     └──────────────┘ └──────────────┘
```

Everything else in the full architecture diagram (Qdrant, Neo4j, Celery workers, Gemini, Tavily, LangSmith) is intentionally **not** wired up yet — it arrives in later milestones.

---

## 2. Folder structure

```
nexusai/
├── apps/
│   ├── backend/
│   │   ├── src/nexusai/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── config.py
│   │   │   │   └── logging.py
│   │   │   ├── api/
│   │   │   │   ├── __init__.py
│   │   │   │   └── v1/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── router.py
│   │   │   │       └── endpoints/
│   │   │   │           ├── __init__.py
│   │   │   │           └── health.py
│   │   │   └── infrastructure/
│   │   │       ├── __init__.py
│   │   │       └── db/
│   │   │           ├── __init__.py
│   │   │           └── session.py
│   │   ├── migrations/
│   │   │   ├── env.py
│   │   │   └── versions/          # empty at this milestone
│   │   ├── tests/
│   │   │   ├── conftest.py
│   │   │   └── test_health.py
│   │   ├── alembic.ini
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   └── .env.example
│   └── frontend/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx
│       │   └── globals.css
│       ├── components/ui/         # ShadCN primitives (button, card, badge)
│       ├── lib/
│       │   ├── api-client.ts
│       │   └── utils.ts
│       ├── package.json
│       ├── tsconfig.json
│       ├── tailwind.config.ts
│       ├── next.config.mjs
│       ├── Dockerfile
│       └── .env.local.example
├── .github/workflows/ci.yml
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 3. File list with purpose

| File | Purpose |
|---|---|
| `apps/backend/src/nexusai/main.py` | FastAPI app factory, mounts the v1 router, registers CORS + exception handlers |
| `apps/backend/src/nexusai/core/config.py` | `pydantic-settings` `Settings` class reading all env vars once, cached with `lru_cache` |
| `apps/backend/src/nexusai/core/logging.py` | Structured JSON logging config with request-ID injection |
| `apps/backend/src/nexusai/api/v1/router.py` | Aggregates endpoint routers under `/api/v1` |
| `apps/backend/src/nexusai/api/v1/endpoints/health.py` | `/health`, `/health/db`, `/health/redis` |
| `apps/backend/src/nexusai/infrastructure/db/session.py` | Async SQLAlchemy engine + `AsyncSession` dependency |
| `apps/backend/migrations/env.py` | Alembic async-aware migration environment |
| `apps/backend/tests/test_health.py` | Verifies all three health endpoints |
| `apps/frontend/app/page.tsx` | Status dashboard — polls backend health endpoints, renders ShadCN `Card`/`Badge` |
| `apps/frontend/lib/api-client.ts` | Typed `fetch` wrapper reading `NEXT_PUBLIC_API_BASE_URL` |
| `docker-compose.yml` | Orchestrates `postgres`, `redis`, `backend`, `frontend` with a shared network |
| `.github/workflows/ci.yml` | Lints (`ruff`, `eslint`) and runs backend tests on every push |

---

## 4. APIs

| Method | Path | Description | Response |
|---|---|---|---|
| GET | `/api/v1/health` | Liveness — process is up | `{"status": "ok"}` |
| GET | `/api/v1/health/db` | Runs `SELECT 1` against Postgres | `{"status": "ok", "latency_ms": 4.2}` |
| GET | `/api/v1/health/redis` | Runs `PING` against Redis | `{"status": "ok", "latency_ms": 1.1}` |

All responses are `200` on success, `503` with `{"status": "error", "detail": "..."}` on failure — so the frontend status page and any orchestrator health check can rely on the status code alone.

---

## 5. Database schema

No domain schema yet by design — that starts in Milestone 2 with the `users` table. This milestone's only job is to prove Alembic + async SQLAlchemy + Postgres work together end to end:

- Alembic is initialized (`alembic init -t async migrations`) and produces a baseline (empty) migration.
- `alembic upgrade head` on a fresh database creates only Alembic's own `alembic_version` tracking table.
- `GET /api/v1/health/db` proves connectivity by running a raw `SELECT 1`, not by querying a domain table.

---

## 6. Environment variables

**Backend** (`apps/backend/.env.example`):
```
ENVIRONMENT=development
APP_NAME=nexusai-backend
API_V1_PREFIX=/api/v1
LOG_LEVEL=INFO

POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=nexusai
POSTGRES_PASSWORD=changeme
POSTGRES_DB=nexusai
DATABASE_URL=postgresql+asyncpg://nexusai:changeme@postgres:5432/nexusai

REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL=redis://redis:6379/0

BACKEND_CORS_ORIGINS=["http://localhost:3000"]
```

**Frontend** (`apps/frontend/.env.local.example`):
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## 7. Installation steps

**Option A — Docker (recommended)**
```bash
git clone <repo-url> nexusai && cd nexusai
cp .env.example .env
cp apps/backend/.env.example apps/backend/.env
cp apps/frontend/.env.local.example apps/frontend/.env.local
docker compose up --build
# backend:  http://localhost:8000/docs
# frontend: http://localhost:3000
```

**Option B — local dev (no Docker for app code, Docker only for DB/cache)**
```bash
docker compose up -d postgres redis

# backend
cd apps/backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
uvicorn nexusai.main:app --reload --port 8000

# frontend (new terminal)
cd apps/frontend
npm install
npm run dev
```

---

## 8. Required packages

**Backend** (`pyproject.toml`):
```
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.8
pydantic-settings>=2.4
sqlalchemy[asyncio]>=2.0
asyncpg>=0.29
alembic>=1.13
redis>=5.0
httpx>=0.27
python-dotenv>=1.0

# dev
pytest>=8.0
pytest-asyncio>=0.24
ruff>=0.6
```

**Frontend** (`package.json`):
```
next@^14
react@^18
react-dom@^18
typescript@^5
tailwindcss@^3
postcss
autoprefixer
class-variance-authority
clsx
tailwind-merge
lucide-react
eslint
eslint-config-next
```

---

## 9. Best practices

- Keep `core/config.py` as the **only** place that reads `os.environ` — everything else imports the `Settings` object.
- Use async SQLAlchemy consistently: `asyncpg` driver, `AsyncSession`, `await` on every DB call. Mixing sync and async drivers is the #1 source of bugs at this stage (see below).
- Give every request a correlation ID (middleware) and include it in every log line — this pays off enormously once Celery workers and LLM calls enter the picture in later milestones.
- Multi-stage Dockerfiles: builder stage installs dependencies, runtime stage copies only what's needed — keeps images small and out of dependency-resolution surprises.
- `.dockerignore` and `.gitignore` from the start (`.env`, `__pycache__`, `node_modules`, `.venv`).
- Run containers as a non-root user in both Dockerfiles.
- Restrict `BACKEND_CORS_ORIGINS` to known frontend origins even in dev — catching CORS misconfiguration early is cheaper than debugging it in Milestone 10.
- Pin major versions of all dependencies; let CI catch breakage from minor/patch bumps rather than discovering it mid-build.

---

## 10. Common bugs to watch for

| Symptom | Cause | Fix |
|---|---|---|
| `MissingGreenlet` error on any DB call | Mixed sync/async SQLAlchemy usage, or using `psycopg2` with an async engine | Use `asyncpg` driver and `AsyncSession` everywhere; never call sync SQLAlchemy methods inside an async route |
| Backend can't reach Postgres/Redis inside Docker | Using `localhost` instead of the Compose service name (`postgres`, `redis`) in `DATABASE_URL`/`REDIS_URL` | Service names are the hostnames on the Compose network — use them, not `localhost` |
| Frontend `fetch` to backend fails with CORS error | `BACKEND_CORS_ORIGINS` doesn't include the frontend's actual origin | Add the exact scheme+host+port (`http://localhost:3000`) to the CORS list |
| `NEXT_PUBLIC_API_BASE_URL` is `undefined` in the browser | Env var missing the `NEXT_PUBLIC_` prefix, or `.env.local` not present at build time | Next.js only exposes client-side env vars prefixed `NEXT_PUBLIC_`; restart `next dev` after adding one |
| `alembic upgrade head` no-ops or errors on a fresh DB | `alembic.ini`/`env.py` pointing at the wrong `DATABASE_URL`, or migrations generated against sync engine | Confirm `env.py` reads the async URL and uses `run_sync` for autogenerate |
| Port already in use on `docker compose up` | Another local Postgres/Redis instance already bound to 5432/6379 | Change the host-side port mapping in `docker-compose.yml` (e.g. `5433:5432`) or stop the conflicting local service |
| Health check passes locally but fails in CI | CI runner doesn't wait for Postgres/Redis containers to be ready before tests start | Add a `depends_on` with `condition: service_healthy` plus Compose healthchecks, or a wait-for-it retry loop in the test fixture |

---

## 11. Testing checklist

- [ ] `docker compose up --build` succeeds with no errors
- [ ] `GET /api/v1/health` returns `200 {"status": "ok"}`
- [ ] `GET /api/v1/health/db` returns `200` with a real latency value
- [ ] `GET /api/v1/health/redis` returns `200` with a real latency value
- [ ] Stopping the `postgres` container makes `/health/db` return `503`, not a 500 or a hang
- [ ] Frontend status page loads at `http://localhost:3000` and reflects live backend health (auto-refresh, not just on load)
- [ ] `alembic upgrade head` runs cleanly against a freshly created database
- [ ] `pytest` passes locally and in CI
- [ ] `ruff check .` and `npm run lint` both pass with zero errors
- [ ] `docker compose down && docker compose up` — containers come back up cleanly (no orphaned volumes/state issues)
- [ ] No `.env` files are committed to git; `.env.example` files are up to date with every var actually used
- [ ] Backend hot-reloads on code change in dev mode; frontend hot-reloads via Next.js Fast Refresh

---

**Next**: once this milestone checks out on your end, confirm and I'll generate **Milestone 2 — Authentication & User Management** (JWT auth, `users` table, RBAC, register/login/refresh endpoints, protected-route middleware, and the frontend auth pages).
