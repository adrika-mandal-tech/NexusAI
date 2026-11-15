"""PostgreSQL and resilient fallback repository implementations."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.entities.knowledge import (
    Conversation,
    ConversationRole,
    IngestionSource,
    KnowledgeGraph,
    Message,
    SourceStatus,
    SourceType,
)
from nexusai.domain.ports.knowledge_repository import KnowledgeRepositoryPort
from nexusai.infrastructure.db.models import (
    Conversation as ConversationModel,
)
from nexusai.infrastructure.db.models import (
    ConversationRole as DBConversationRole,
)
from nexusai.infrastructure.db.models import (
    IngestionSource as SourceModel,
)
from nexusai.infrastructure.db.models import (
    Message as MessageModel,
)
from nexusai.infrastructure.db.models import (
    SourceStatus as DBSourceStatus,
)
from nexusai.infrastructure.db.models import (
    SourceType as DBSourceType,
)
from nexusai.infrastructure.db.models import (
    User as UserModel,
)
from nexusai.infrastructure.db.models import (
    UserRole as DBUserRole,
)
from nexusai.infrastructure.db.session import get_session

logger = get_logger(__name__)


def _utc_now_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _coerce_db_source_type(st: SourceType | str) -> DBSourceType:
    val = st.value if isinstance(st, SourceType) else str(st).lower()
    try:
        return DBSourceType(val)
    except ValueError:
        return DBSourceType.TXT


def _coerce_domain_source_type(st: Any) -> SourceType:
    val = st.value if hasattr(st, "value") else str(st).lower()
    try:
        return SourceType(val)
    except ValueError:
        return SourceType.TXT


def _coerce_db_status(st: SourceStatus | str) -> DBSourceStatus:
    val = st.value if isinstance(st, SourceStatus) else str(st).lower()
    return DBSourceStatus(val)


def _coerce_domain_status(st: Any) -> SourceStatus:
    val = st.value if hasattr(st, "value") else str(st).lower()
    return SourceStatus(val)


class PostgresKnowledgeRepository(KnowledgeRepositoryPort):
    """PostgreSQL implementation of KnowledgeRepositoryPort."""

    async def _ensure_user(self, session: Any, user_id: UUID) -> None:
        """Ensure a corresponding row exists in `users` so FK constraints succeed."""
        existing = await session.execute(select(UserModel.id).where(UserModel.id == user_id))
        if existing.scalar_one_or_none() is not None:
            return
        placeholder_email = (
            "system@nexusai.local"
            if str(user_id) == "00000000-0000-0000-0000-000000000000"
            else f"user-{user_id}@nexusai.local"
        )
        session.add(
            UserModel(
                id=user_id,
                email=placeholder_email,
                hashed_password="!",
                full_name="NexusAI Workspace User",
                role=DBUserRole.ADMIN,
                is_active=True,
            )
        )
        await session.flush()

    async def create_source(self, source: IngestionSource) -> IngestionSource:
        async with get_session() as session:
            await self._ensure_user(session, source.user_id)
            model = SourceModel(
                id=source.id,
                user_id=source.user_id,
                title=source.title,
                source_type=_coerce_db_source_type(source.source_type),
                status=_coerce_db_status(source.status),
                file_path=source.file_path,
                url=source.url,
                meta=dict(source.metadata or {}),
                chunk_count=source.chunk_count,
                error_message=source.error_message,
                created_at=source.created_at,
                updated_at=source.updated_at,
            )
            session.add(model)
            await session.flush()
            await session.refresh(model)
            return self._to_domain(model)

    async def get_source(self, source_id: UUID) -> IngestionSource | None:
        async with get_session() as session:
            result = await session.execute(select(SourceModel).where(SourceModel.id == source_id))
            model = result.scalar_one_or_none()
            if model:
                return self._to_domain(model)
            return None

    async def list_sources(self, user_id: UUID, limit: int, offset: int) -> tuple[list[IngestionSource], int]:
        async with get_session() as session:
            count_result = await session.execute(
                select(func.count(SourceModel.id)).where(SourceModel.user_id == user_id)
            )
            total = count_result.scalar_one()

            result = await session.execute(
                select(SourceModel)
                .where(SourceModel.user_id == user_id)
                .order_by(SourceModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            models = result.scalars().all()
            return [self._to_domain(m) for m in models], total

    async def update_source(self, source: IngestionSource) -> IngestionSource:
        async with get_session() as session:
            await session.execute(
                update(SourceModel)
                .where(SourceModel.id == source.id)
                .values(
                    title=source.title,
                    source_type=_coerce_db_source_type(source.source_type),
                    status=_coerce_db_status(source.status),
                    file_path=source.file_path,
                    url=source.url,
                    meta=dict(source.metadata or {}),
                    chunk_count=source.chunk_count,
                    error_message=source.error_message,
                    updated_at=_utc_now_naive(),
                )
            )
            await session.flush()
            result = await session.execute(select(SourceModel).where(SourceModel.id == source.id))
            model = result.scalar_one()
            return self._to_domain(model)

    async def find_source_by_hash(self, user_id: UUID, content_hash: str) -> IngestionSource | None:
        async with get_session() as session:
            result = await session.execute(
                select(SourceModel)
                .where(SourceModel.user_id == user_id)
                .order_by(SourceModel.created_at.desc())
                .limit(200)
            )
            for model in result.scalars().all():
                meta = model.meta or {}
                if meta.get("content_hash") == content_hash:
                    return self._to_domain(model)
            return None

    async def delete_source(self, source_id: UUID) -> None:
        async with get_session() as session:
            await session.execute(delete(SourceModel).where(SourceModel.id == source_id))

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        async with get_session() as session:
            await self._ensure_user(session, conversation.user_id)
            model = ConversationModel(
                id=conversation.id,
                user_id=conversation.user_id,
                title=conversation.title,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )
            session.add(model)
            await session.flush()
            await session.refresh(model)
            return self._to_domain_conversation(model, messages=[])

    async def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        async with get_session() as session:
            result = await session.execute(
                select(ConversationModel).where(ConversationModel.id == conversation_id)
            )
            model = result.scalar_one_or_none()
            if not model:
                return None
            msg_res = await session.execute(
                select(MessageModel)
                .where(MessageModel.conversation_id == conversation_id)
                .order_by(MessageModel.created_at.asc())
            )
            msg_models = msg_res.scalars().all()
            return self._to_domain_conversation(
                model,
                messages=[self._to_domain_message(m) for m in msg_models],
            )

    async def list_conversations(self, user_id: UUID, limit: int, offset: int) -> tuple[list[Conversation], int]:
        async with get_session() as session:
            count_result = await session.execute(
                select(func.count(ConversationModel.id)).where(ConversationModel.user_id == user_id)
            )
            total = count_result.scalar_one()

            result = await session.execute(
                select(ConversationModel)
                .where(ConversationModel.user_id == user_id)
                .order_by(ConversationModel.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            models = result.scalars().all()
            conversations: list[Conversation] = []
            for m in models:
                msg_res = await session.execute(
                    select(MessageModel)
                    .where(MessageModel.conversation_id == m.id)
                    .order_by(MessageModel.created_at.asc())
                )
                msgs = [self._to_domain_message(mm) for mm in msg_res.scalars().all()]
                conversations.append(self._to_domain_conversation(m, messages=msgs))
            return conversations, total

    async def delete_conversation(self, conversation_id: UUID) -> None:
        async with get_session() as session:
            await session.execute(
                delete(MessageModel).where(MessageModel.conversation_id == conversation_id)
            )
            await session.execute(
                delete(ConversationModel).where(ConversationModel.id == conversation_id)
            )

    async def add_message(self, message: Message) -> Message:
        async with get_session() as session:
            role_val = message.role.value if isinstance(message.role, ConversationRole) else str(message.role)
            model = MessageModel(
                id=message.id,
                conversation_id=message.conversation_id,
                role=DBConversationRole(role_val),
                content=message.content,
                citations=list(message.citations or []),
                meta=dict(message.metadata or {}),
                created_at=message.created_at,
            )
            session.add(model)
            await session.flush()
            await session.refresh(model)

            await session.execute(
                update(ConversationModel)
                .where(ConversationModel.id == message.conversation_id)
                .values(updated_at=_utc_now_naive())
            )

            return self._to_domain_message(model)

    async def get_graph(self, user_id: UUID) -> KnowledgeGraph | None:
        """Return empty graph — graph entities live in Neo4j."""
        return KnowledgeGraph(nodes=[], edges=[])

    def _to_domain(self, model: SourceModel) -> IngestionSource:
        return IngestionSource(
            id=model.id,
            user_id=model.user_id,
            title=model.title,
            source_type=_coerce_domain_source_type(model.source_type),
            status=_coerce_domain_status(model.status),
            file_path=model.file_path,
            url=model.url,
            metadata=dict(model.meta or {}),
            chunk_count=model.chunk_count or 0,
            error_message=model.error_message,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _to_domain_conversation(
        self,
        model: ConversationModel,
        messages: list[Message] | None = None,
    ) -> Conversation:
        return Conversation(
            id=model.id,
            user_id=model.user_id,
            title=model.title,
            messages=messages if messages is not None else [],
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _to_domain_message(self, model: MessageModel) -> Message:
        role_val = model.role.value if hasattr(model.role, "value") else str(model.role)
        return Message(
            id=model.id,
            conversation_id=model.conversation_id,
            role=ConversationRole(role_val),
            content=model.content,
            citations=list(model.citations or []),
            metadata=dict(model.meta or {}),
            created_at=model.created_at,
        )


class LocalJsonKnowledgeRepository(KnowledgeRepositoryPort):
    """Disk-backed JSON repository fallback used only when PostgreSQL is unavailable."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        base_dir = Path(settings.UPLOAD_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)
        self._path = path or (base_dir / "knowledge_store.json")
        self._lock = asyncio.Lock()

    def _load_sync(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"sources": {}, "conversations": {}, "messages": {}}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {"sources": {}, "conversations": {}, "messages": {}}

    def _save_sync(self, data: dict[str, Any]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._path)

    async def create_source(self, source: IngestionSource) -> IngestionSource:
        async with self._lock:
            data = await asyncio.to_thread(self._load_sync)
            data["sources"][str(source.id)] = source.model_dump(mode="json")
            await asyncio.to_thread(self._save_sync, data)
            return source

    async def get_source(self, source_id: UUID) -> IngestionSource | None:
        data = await asyncio.to_thread(self._load_sync)
        raw = data["sources"].get(str(source_id))
        return IngestionSource.model_validate(raw) if raw else None

    async def list_sources(self, user_id: UUID, limit: int, offset: int) -> tuple[list[IngestionSource], int]:
        data = await asyncio.to_thread(self._load_sync)
        items = [
            IngestionSource.model_validate(v)
            for v in data["sources"].values()
            if str(v.get("user_id")) == str(user_id)
        ]
        items.sort(key=lambda s: s.created_at, reverse=True)
        return items[offset : offset + limit], len(items)

    async def update_source(self, source: IngestionSource) -> IngestionSource:
        async with self._lock:
            data = await asyncio.to_thread(self._load_sync)
            updated = source.model_copy(update={"updated_at": _utc_now_naive()})
            data["sources"][str(source.id)] = updated.model_dump(mode="json")
            await asyncio.to_thread(self._save_sync, data)
            return updated

    async def find_source_by_hash(self, user_id: UUID, content_hash: str) -> IngestionSource | None:
        data = await asyncio.to_thread(self._load_sync)
        for raw in data["sources"].values():
            if str(raw.get("user_id")) == str(user_id):
                meta = raw.get("metadata") or {}
                if meta.get("content_hash") == content_hash:
                    return IngestionSource.model_validate(raw)
        return None

    async def delete_source(self, source_id: UUID) -> None:
        async with self._lock:
            data = await asyncio.to_thread(self._load_sync)
            data["sources"].pop(str(source_id), None)
            await asyncio.to_thread(self._save_sync, data)

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        async with self._lock:
            data = await asyncio.to_thread(self._load_sync)
            payload = conversation.model_dump(mode="json")
            payload.pop("messages", None)
            data["conversations"][str(conversation.id)] = payload
            data["messages"].setdefault(str(conversation.id), [])
            await asyncio.to_thread(self._save_sync, data)
            return conversation

    async def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        data = await asyncio.to_thread(self._load_sync)
        raw = data["conversations"].get(str(conversation_id))
        if not raw:
            return None
        msgs = [
            Message.model_validate(m)
            for m in data["messages"].get(str(conversation_id), [])
        ]
        return Conversation.model_validate({**raw, "messages": [m.model_dump(mode="json") for m in msgs]})

    async def list_conversations(self, user_id: UUID, limit: int, offset: int) -> tuple[list[Conversation], int]:
        data = await asyncio.to_thread(self._load_sync)
        items: list[Conversation] = []
        for cid, raw in data["conversations"].items():
            if str(raw.get("user_id")) != str(user_id):
                continue
            msgs = data["messages"].get(cid, [])
            items.append(Conversation.model_validate({**raw, "messages": msgs}))
        items.sort(key=lambda c: c.updated_at, reverse=True)
        return items[offset : offset + limit], len(items)

    async def delete_conversation(self, conversation_id: UUID) -> None:
        async with self._lock:
            data = await asyncio.to_thread(self._load_sync)
            data["conversations"].pop(str(conversation_id), None)
            data["messages"].pop(str(conversation_id), None)
            await asyncio.to_thread(self._save_sync, data)

    async def add_message(self, message: Message) -> Message:
        async with self._lock:
            data = await asyncio.to_thread(self._load_sync)
            cid = str(message.conversation_id)
            data["messages"].setdefault(cid, []).append(message.model_dump(mode="json"))
            if cid in data["conversations"]:
                data["conversations"][cid]["updated_at"] = _utc_now_naive().isoformat()
            await asyncio.to_thread(self._save_sync, data)
            return message

    async def get_graph(self, user_id: UUID) -> KnowledgeGraph | None:
        return KnowledgeGraph(nodes=[], edges=[])


class ResilientKnowledgeRepository(KnowledgeRepositoryPort):
    """Uses PostgreSQL primarily and seamlessly degrades to local JSON store if Postgres is down."""

    def __init__(
        self,
        primary: PostgresKnowledgeRepository | None = None,
        fallback: LocalJsonKnowledgeRepository | None = None,
    ) -> None:
        self._primary = primary or PostgresKnowledgeRepository()
        self._fallback = fallback or LocalJsonKnowledgeRepository()

    async def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            fn = getattr(self._primary, method_name)
            return await fn(*args, **kwargs)
        except Exception as exc:
            logger.warning(
                "PostgreSQL %s failed (%s); falling back to LocalJsonKnowledgeRepository",
                method_name,
                exc,
            )
            fn = getattr(self._fallback, method_name)
            return await fn(*args, **kwargs)

    async def create_source(self, source: IngestionSource) -> IngestionSource:
        return await self._call("create_source", source)

    async def get_source(self, source_id: UUID) -> IngestionSource | None:
        return await self._call("get_source", source_id)

    async def find_source_by_hash(self, user_id: UUID, content_hash: str) -> IngestionSource | None:
        return await self._call("find_source_by_hash", user_id, content_hash)

    async def list_sources(self, user_id: UUID, limit: int, offset: int) -> tuple[list[IngestionSource], int]:
        return await self._call("list_sources", user_id, limit, offset)

    async def update_source(self, source: IngestionSource) -> IngestionSource:
        return await self._call("update_source", source)

    async def delete_source(self, source_id: UUID) -> None:
        await self._call("delete_source", source_id)

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        return await self._call("create_conversation", conversation)

    async def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        return await self._call("get_conversation", conversation_id)

    async def list_conversations(self, user_id: UUID, limit: int, offset: int) -> tuple[list[Conversation], int]:
        return await self._call("list_conversations", user_id, limit, offset)

    async def delete_conversation(self, conversation_id: UUID) -> None:
        await self._call("delete_conversation", conversation_id)

    async def add_message(self, message: Message) -> Message:
        return await self._call("add_message", message)

    async def get_graph(self, user_id: UUID) -> KnowledgeGraph | None:
        return await self._call("get_graph", user_id)
