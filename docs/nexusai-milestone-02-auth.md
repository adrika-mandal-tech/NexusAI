# Milestone 2 — Authentication & User Management

**Goal**: a complete, working auth layer on top of Milestone 1's foundation — register, login, refresh-token rotation, logout (revocation), protected routes, RBAC, a profile-editing UI, and admin user management.

**Definition of done**: `docker compose up`, seed an admin, register a second user from the UI, log in, edit your profile, log out (token revoked server-side), and as an admin list/role other users — all without 500s.

---

## 1. Architecture diagram (this milestone only)

```
┌─────────────────────────┐   HTTPS   ┌─────────────────────────┐
│  Next.js frontend        │─────────▶│  FastAPI backend         │
│  /login /register        │  JWT Bearer│  /api/v1/auth/*          │
│  /settings (profile)     │◀─────────│  AuthService             │
└─────────────────────────┘           └────────┬──────────┬──────┘
                                                │          │
                                   asyncpg      │          │  aioredis
                                                ▼          ▼
                                        ┌──────────────┐ ┌──────────────┐
                                        │  Postgres 16  │ │  Redis 7     │
                                        │  users table  │ │ refresh set  │
                                        └──────────────┘ └──────────────┘
```

## 2. Folder structure (changes from M1)

```
apps/backend/src/nexusai/
├── api/v1/
│   ├── endpoints/auth.py          # register/login/refresh/logout/me(PATCH)+admin
│   └── schemas/auth.py            # + UpdateProfileRequest, RoleUpdateRequest, UserListResponse
├── application/services/auth_service.py  # + update_profile, list_users, set_role, revoke_all_for_user
├── domain/ports/user_repository.py      # + list_users
├── infrastructure/repositories/user_repository.py  # + list_users (paginated)
└── core/security.py                    # refresh tokens now carry a jti
apps/backend/tests/test_auth.py          # new integration tests
apps/frontend/app/(dashboard)/settings/page.tsx  # now functional (profile update)
apps/frontend/components/providers/auth-provider.tsx  # logout hits backend
```

## 3. APIs

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/auth/register` | — | Create account, returns user + tokens (201) |
| POST | `/api/v1/auth/login` | — | Email+password → tokens |
| POST | `/api/v1/auth/refresh` | refresh token | Rotate to a new access+refresh pair |
| POST | `/api/v1/auth/logout` | access | Revoke refresh token(s); body `refresh_token` = single, none = all |
| GET | `/api/v1/auth/me` | access | Current user profile |
| PATCH | `/api/v1/auth/me` | access | Update `full_name` / `password` |
| GET | `/api/v1/auth/users` | admin | Paginated user list (`?limit&offset`) |
| GET | `/api/v1/auth/users/{id}` | admin | Fetch one user |
| PATCH | `/api/v1/auth/users/{id}/role` | admin | Change RBAC role (`admin`/`member`/`viewer`) |

## 4. Database schema

`users` table (migrated in M1 — `001_create_users_table`):
`id uuid PK`, `email varchar unique`, `hashed_password varchar`, `full_name varchar`, `role varchar default 'member'`, `is_active bool`, `created_at`, `updated_at`.

Redis: key `refresh_tokens:{user_id}` — a **set** of valid refresh-token `jti`s, TTL = `REFRESH_TOKEN_EXPIRE_DAYS`. Logout removes one (or all) jti(s); refresh validates membership.

## 5. Environment variables

No new vars beyond M1. `JWT_SECRET_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES=15`, `REFRESH_TOKEN_EXPIRE_DAYS=7` drive token lifetimes.

## 6. Required packages

No new backend deps (python-jose, passlib, redis, asyncpg already present).

## 7. Best practices

- Refresh tokens are **revocable** (stored in Redis by jti) — never trust a refresh token that isn't in the set.
- `PATCH /auth/me` accepts partial updates; only non-null fields are written.
- RBAC enforced via `require_role(...)` dependency factory — keep it on every admin route.
- The global exception handler in `main.py` re-raises `HTTPException` so correct status codes (401/403/409) survive.
- Frontend never stores the password; tokens live in `localStorage`, sent as `Bearer` on every request.

## 8. Common bugs

| Symptom | Cause | Fix |
|---|---|---|
| Refresh always 401 after logout | token revoked but client retries cached one | client clears tokens and redirects to /login on 401 |
| 500 instead of 401/409 | custom `Exception` handler swallowed `HTTPException` | re-raise `HTTPException` inside the handler |
| Duplicate email 500 | repo `create` raises IntegrityError before service check | service checks `exists_by_email` first (returns 409) |
| Logout doesn't invalidate other sessions | per-token key, not a set | use `refresh_tokens:{user_id}` set; logout-without-body deletes the whole set |

## 9. Testing checklist

- [ ] Register returns 201 with tokens; duplicate email returns 409.
- [ ] Login with wrong password returns 401; correct returns 200.
- [ ] `/auth/me` without token → 401; with token → 200.
- [ ] Refresh returns new token pair; using the **old** refresh after refresh still works until logout.
- [ ] Logout (no body) revokes all; subsequent refresh with old token → 401.
- [ ] `PATCH /auth/me` updates name/password; re-login with new password works.
- [ ] Admin can list users and change a role; non-admin hitting `/auth/users` → 403.
- [ ] `pytest` (auth tests) pass against Postgres+Redis; skip gracefully without them.
- [ ] Frontend `/settings` saves profile and reflects changes; logout redirects to /login.
- [ ] `ruff check` and `npm run lint` clean; `next build` succeeds.
