from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents import RunConfig, Runner
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from app.agents.sophistication_evaluator import investor_sophistication_evaluator
from app.storage.sophistication_store import insert_sophistication


def _get_field(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                txt = part.get("text") or part.get("content") or ""
                if isinstance(txt, str) and txt:
                    parts.append(txt)
            else:
                txt = getattr(part, "text", None)
                if isinstance(txt, str) and txt:
                    parts.append(txt)
        return "\n".join(parts)
    txt = getattr(content, "text", None)
    return txt if isinstance(txt, str) else str(content)


def _extract_text_from_item(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    content = _get_field(item, "content")
    if content is not None:
        return _extract_text_from_content(content)
    return str(item)


def _build_eval_transcript(items: List[Any], *, max_lines: int = 60, max_chars_per_line: int = 800) -> str:
    """Compact transcript for evaluator (avoid huge portfolio context dumps)."""
    lines: List[str] = []
    for it in items[-max_lines:]:
        role = (_get_field(it, "role") or "unknown").upper()
        txt = (_extract_text_from_item(it) or "").strip()
        if not txt:
            continue

        # Skip bulky seeded context blocks
        if txt.startswith("CONTEXTO —"):
            continue

        if len(txt) > max_chars_per_line:
            txt = txt[:max_chars_per_line] + "…(truncado)"

        lines.append(f"{role}: {txt}")

    return "\n".join(lines)


def _safe_json_loads(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    # strip accidental fences
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return json.loads(s)


async def evaluate_and_store_sophistication(
    *,
    engine: Any,
    session_id: str,
    session: SQLAlchemySession,
    presence: Optional[Dict[str, bool]] = None,
) -> None:
    """Runs gpt-4.1 evaluator and stores result in a private table.

    - presence: optional flags like:
        {
          "has_portafolio_promedio": True/False,
          "has_portafolio_inversionista": True/False,
          "has_club_deals_concepts": True/False,
          "has_club_deals_opinion": True/False
        }
    """
    if engine is None:
        return

    history = await session.get_items(limit=250)
    transcript = _build_eval_transcript(history)

    presence = presence or {}
    presence_text = "\n".join([f"- {k}: {v}" for k, v in presence.items()]) or "- (no flags provided)"

    prompt = (
        "Evalúa la sofisticación del inversionista.\n\n"
        "INPUTS DISPONIBLES (flags):\n"
        f"{presence_text}\n\n"
        "TRANSCRIPT (reciente):\n"
        f"{transcript}"
    )

    result = await Runner.run(
        investor_sophistication_evaluator,
        prompt,
        # IMPORTANT: do NOT pass `session=` to keep it private
        run_config=RunConfig(
            trace_metadata={
                "__trace_source__": "fastapi-service",
                "model_route": "sophistication_evaluator",
                "session_id": session_id,
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
        raw={
            "raw_output": raw_out,
            "presence": presence,
        },
    )
