from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Optional

from agents import Runner, RunConfig
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from app.agents.sophistication_evaluator import investor_sophistication_evaluator
from app.storage.sophistication_store import insert_sophistication


def _safe_json_loads(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    # If model ever returns fenced code, strip it.
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()
    return json.loads(s)


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


async def evaluate_and_store_sophistication(
    *,
    engine: Any,
    session_id: str,
    session: SQLAlchemySession,
    presence: Optional[Dict[str, bool]] = None,
    event_type: Optional[str] = None,
    philosophy_hash: Optional[str] = None,
) -> None:
    """
    Evaluate sophistication INTERNALLY using gpt-4.1 and store results in a private table.

    IMPORTANT:
    - Call this ONLY when a philosophy is generated/re-generated (builder_initial/builder_edit)
    - Do NOT call on every question turn (to avoid latency/cost)
    - It never returns anything to the user.

    Args:
      event_type: e.g. "builder_initial" | "builder_edit"
      philosophy_hash: SHA-256 hex digest of the generated philosophy text (so you can join later)
    """
    if engine is None:
        return

    # Compact transcript for evaluator (avoid sending huge JSONs)
    history = await session.get_items(limit=250)

    # Build a concise transcript: last N items only
    max_items = 60
    items = list(history or [])[-max_items:]
    lines = []
    for it in items:
        role = (it.get("role") if isinstance(it, dict) else getattr(it, "role", None)) or "unknown"
        content = (it.get("content") if isinstance(it, dict) else getattr(it, "content", None))
        if isinstance(content, list):
            # keep only text parts, truncated
            text_parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            txt = "\n".join(text_parts).strip()
        elif isinstance(content, str):
            txt = content.strip()
        else:
            txt = str(content).strip()

        if not txt:
            continue
        if len(txt) > 1200:
            txt = txt[:1200] + "…(truncado)"
        lines.append(f"{role.upper()}: {txt}")

    transcript = "\n".join(lines)

    prompt = (
        "Evalúa el nivel de sofisticación del inversionista usando este transcript.\n"
        "Devuelve SOLO JSON válido.\n\n"
        f"PRESENCE_FLAGS: {presence or {}}\n\n"
        f"TRANSCRIPT:\n{transcript}\n"
    )

    # NOTE: No session=... here → evaluator output is not written to chat memory
    result = await Runner.run(
        investor_sophistication_evaluator,
        prompt,
        run_config=RunConfig(
            trace_metadata={
                "__trace_source__": "fastapi-service",
                "model_route": "sophistication_evaluator",
                "session_id": session_id,
                "event_type": event_type or "",
                "philosophy_hash": philosophy_hash or "",
            }
        ),
    )

    raw_out = result.final_output_as(str)
    try:
        payload = _safe_json_loads(raw_out)
    except Exception:
        payload = {
            "level": "BASICO",
            "score": 0.0,
            "confidence": "baja",
            "signals": ["parse_error"],
            "evidence": [],
            "notes": "No se pudo parsear JSON del evaluator.",
        }

    row_id = hashlib.sha256(
        (session_id + "|" + str(datetime.utcnow().timestamp())).encode("utf-8")
    ).hexdigest()[:64]

    await insert_sophistication(
        engine,
        row_id=row_id,
        session_id=session_id,
        payload=payload,
        raw={"raw_output": raw_out, "presence": presence or {}},
        event_type=event_type,
        philosophy_hash=philosophy_hash,
    )
