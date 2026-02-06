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

# Private table: not part of chat transcript
investor_sophistication_table = Table(
    "investor_sophistication",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("session_id", String(256), index=True, nullable=False),

    # New: helps segment / trace
    Column("event_type", String(64), nullable=True),
    Column("philosophy_hash", String(64), nullable=True),

    Column("level", String(32), nullable=False),
    Column("score", Float, nullable=False),
    Column("confidence", String(16), nullable=False),
    Column("signals", JSON, nullable=True),
    Column("evidence", JSON, nullable=True),
    Column("notes", Text, nullable=True),

    # Keep raw for audit/debug (never shown to user)
    Column("raw", JSON, nullable=True),

    Column("created_at", DateTime(timezone=True), nullable=False),
)


async def ensure_sophistication_table(engine: AsyncEngine) -> None:
    """
    Ensure the private sophistication table exists.
    Also tries best-effort schema upgrades for new columns (event_type, philosophy_hash)
    without failing startup if ALTER is not supported by the DB/dialect/permissions.
    """
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

        # Best-effort schema upgrade for existing deployments
        # (works for Postgres; other DBs may ignore/raise — we swallow errors)
        for col_name, col_sql in [
            ("event_type", "VARCHAR(64)"),
            ("philosophy_hash", "VARCHAR(64)"),
        ]:
            # Try Postgres-style IF NOT EXISTS first
            try:
                await conn.exec_driver_sql(
                    f'ALTER TABLE investor_sophistication ADD COLUMN IF NOT EXISTS {col_name} {col_sql}'
                )
                continue
            except Exception:
                pass
            # Fallback: generic ALTER ADD COLUMN (may fail if already exists)
            try:
                await conn.exec_driver_sql(
                    f'ALTER TABLE investor_sophistication ADD COLUMN {col_name} {col_sql}'
                )
            except Exception:
                pass


async def insert_sophistication(
    engine: AsyncEngine,
    *,
    row_id: str,
    session_id: str,
    payload: Dict[str, Any],
    raw: Optional[Dict[str, Any]] = None,
    event_type: Optional[str] = None,
    philosophy_hash: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc)

    # Always include event_type/philosophy_hash in raw for traceability,
    # even if the DB table doesn't have these columns (fallback insert).
    _raw = dict(raw or {})
    if event_type is not None:
        _raw["event_type"] = event_type
    if philosophy_hash is not None:
        _raw["philosophy_hash"] = philosophy_hash

    values = {
        "id": row_id,
        "session_id": session_id,
        "event_type": event_type,
        "philosophy_hash": philosophy_hash,

        "level": str(payload.get("level") or "BASICO"),
        "score": float(payload.get("score") or 0.0),
        "confidence": str(payload.get("confidence") or "baja"),
        "signals": payload.get("signals") or [],
        "evidence": payload.get("evidence") or [],
        "notes": payload.get("notes") or "",
        "raw": _raw,
        "created_at": now,
    }

    async with engine.begin() as conn:
        try:
            await conn.execute(investor_sophistication_table.insert().values(**values))
        except Exception:
            # Fallback for older schemas that don't have event_type/philosophy_hash columns.
            fallback_values = dict(values)
            fallback_values.pop("event_type", None)
            fallback_values.pop("philosophy_hash", None)
            await conn.execute(investor_sophistication_table.insert().values(**fallback_values))
