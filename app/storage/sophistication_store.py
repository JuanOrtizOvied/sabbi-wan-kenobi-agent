from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    Float,
    DateTime,
    Text,
    JSON,
)
from sqlalchemy.ext.asyncio import AsyncEngine

metadata = MetaData()

investor_sophistication_table = Table(
    "investor_sophistication",
    metadata,
    Column("id", String(64), primary_key=True),               # id determinístico o uuid
    Column("session_id", String(256), index=True, nullable=False),
    Column("level", String(32), nullable=False),
    Column("score", Float, nullable=False),
    Column("confidence", String(16), nullable=False),
    Column("signals", JSON, nullable=True),
    Column("evidence", JSON, nullable=True),
    Column("notes", Text, nullable=True),
    Column("raw", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


async def ensure_sophistication_table(engine: AsyncEngine) -> None:
    """
    Crea la tabla si no existe.
    - Úsalo en startup si CREATE_SESSION_TABLES=True, o siempre si te da igual.
    """
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


async def insert_sophistication(
    engine: AsyncEngine,
    *,
    row_id: str,
    session_id: str,
    payload: Dict[str, Any],
    raw: Optional[Dict[str, Any]] = None,
) -> None:
    now = datetime.now(timezone.utc)
    values = {
        "id": row_id,
        "session_id": session_id,
        "level": str(payload.get("level") or "BASICO"),
        "score": float(payload.get("score") or 0.0),
        "confidence": str(payload.get("confidence") or "baja"),
        "signals": payload.get("signals") or [],
        "evidence": payload.get("evidence") or [],
        "notes": payload.get("notes") or "",
        "raw": raw or payload,
        "created_at": now,
    }

    async with engine.begin() as conn:
        await conn.execute(investor_sophistication_table.insert().values(**values))
