"""Authentication API endpoints with salted scrypt password hashing and JWT tokens."""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.infrastructure.db.models import User
from nexusai.infrastructure.db.session import _get_engine

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)

# In-memory fallback store used when PostgreSQL is unavailable (e.g. unit tests)
_USERS: dict[str, dict[str, Any]] = {}
_TOKENS: dict[str, str] = {}


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, salt_hex, hash_hex = stored_hash.split("$", 2)
        if scheme != "scrypt":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
        return hmac.compare_digest(derived, expected)
    except Exception:
        return False


def _create_token(user_id: str, email: str = "") -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.AUTH_TOKEN_EXPIRE_MINUTES)).timestamp()),
    }
    token = jwt.encode(payload, settings.AUTH_SECRET_KEY, algorithm=settings.AUTH_ALGORITHM)
    _TOKENS[token] = user_id
    return token


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a signed JWT access token."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.AUTH_SECRET_KEY,
            algorithms=[settings.AUTH_ALGORITHM],
        )
        if not payload.get("sub"):
            return None
        return payload
    except JWTError:
        # Support legacy/test tokens stored in _TOKENS if explicitly present
        user_id = _TOKENS.get(token)
        if user_id:
            return {"sub": user_id}
        return None


async def _find_user_by_email(email: str) -> dict[str, Any] | None:
    try:
        _, session_maker = _get_engine()
        async with session_maker() as session:
            row = (
                await session.execute(select(User).where(User.email == email))
            ).scalar_one_or_none()
            if row is not None:
                return {
                    "id": str(row.id),
                    "email": row.email,
                    "name": row.full_name or "",
                    "hashed_password": row.hashed_password,
                }
    except Exception as exc:
        logger.debug("DB user lookup skipped (falling back to memory store): %s", exc)
    return _USERS.get(email)


async def _find_user_by_id(user_id: str) -> dict[str, Any] | None:
    try:
        uid = uuid.UUID(user_id)
        _, session_maker = _get_engine()
        async with session_maker() as session:
            row = (
                await session.execute(select(User).where(User.id == uid))
            ).scalar_one_or_none()
            if row is not None:
                return {
                    "id": str(row.id),
                    "email": row.email,
                    "name": row.full_name or "",
                }
    except Exception as exc:
        logger.debug("DB user lookup by id skipped: %s", exc)
    for user in _USERS.values():
        if user["id"] == user_id:
            return {"id": user["id"], "email": user["email"], "name": user["name"]}
    return None


async def _create_user_record(email: str, hashed_password: str, name: str) -> dict[str, Any]:
    user_uuid = uuid.uuid4()
    user_id = str(user_uuid)
    record = {
        "id": user_id,
        "email": email,
        "name": name,
        "hashed_password": hashed_password,
    }
    try:
        _, session_maker = _get_engine()
        async with session_maker() as session:
            db_user = User(
                id=user_uuid,
                email=email,
                hashed_password=hashed_password,
                full_name=name or None,
            )
            session.add(db_user)
            await session.commit()
    except Exception as exc:
        logger.debug("DB user insert skipped (stored in memory): %s", exc)
    _USERS[email] = record
    return record


async def _get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict[str, Any]:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


@router.post("/register", response_model=dict[str, Any])
async def register(body: dict[str, Any]) -> dict[str, Any]:
    email = str(body.get("email", "")).strip().lower()
    password = str(body.get("password", ""))
    name = str(body.get("name") or body.get("full_name") or "").strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password are required")

    existing = await _find_user_by_email(email)
    if existing is not None:
        raise HTTPException(status_code=409, detail="User already exists")

    hashed = _hash_password(password)
    record = await _create_user_record(email=email, hashed_password=hashed, name=name)
    token = _create_token(record["id"], email=email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": record["id"], "email": email, "name": name},
    }


@router.post("/login", response_model=dict[str, Any])
async def login(body: dict[str, Any]) -> dict[str, Any]:
    email = str(body.get("email", "")).strip().lower()
    password = str(body.get("password", ""))
    user = await _find_user_by_email(email)
    if not user or not _verify_password(password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_token(user["id"], email=user["email"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
        },
    }


@router.get("/me", response_model=dict[str, Any])
async def get_me(payload: dict[str, Any] = Depends(_get_current_user)) -> dict[str, Any]:
    user_id = str(payload["sub"])
    user = await _find_user_by_id(user_id)
    if user is not None:
        return user
    # Fallback when JWT is valid even if ephemeral memory store cleared
    if payload.get("email"):
        return {"id": user_id, "email": payload["email"], "name": ""}
    raise HTTPException(status_code=404, detail="User not found")
