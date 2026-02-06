from __future__ import annotations

import hashlib
import json

from app.agents.sophistication_evaluator import investor_sophistication_evaluator
from app.storage.sophistication_store import ensure_sophistication_table, insert_sophistication


from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents import Runner, RunConfig, trace
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from app.agents.filosofia_wow_agent import (
    PHILOSOPHY_HEADER,
    QUESTIONS_TARGET,
    STARTING_QUESTIONS,
    filosofia_builder_agent,
    filosofia_questions_agent,
    filosofia_refine_question_agent,
)
from app.core.config import CREATE_SESSION_TABLES, MESSAGES_TABLE, SESSIONS_TABLE, SESSION_DB_URL
from app.utils.json_utils import pretty_json

try:
    from agents import TResponseInputItem
except Exception:  # pragma: no cover
    from agents.items import TResponseInputItem


class AgentService:
    """
    Model routing (performance + quality):
    - gpt-4.1 → realiza las preguntas (Q1..Q5) y también pregunta de afinado post-filosofía.
    - gpt-5.1 (reasoning high) → genera/actualiza la Filosofía de Inversión (WOW).

    Inputs (opcionales):
    - portafolio_promedio (JSON)
    - portafolio_inversionista (JSON)
    - mi_filosofia (texto)
    - club_deals_concepts (texto)
    - club_deals_opinion (texto)
    - club_deals_information (legacy; se interpreta como concepts si concepts no viene)
    """

    def __init__(self) -> None:
        self._bootstrap: Optional[SQLAlchemySession] = None
        self._engine = None

    async def startup(self) -> None:
        self._bootstrap = SQLAlchemySession.from_url(
            "bootstrap",
            url=SESSION_DB_URL,
            create_tables=CREATE_SESSION_TABLES,
            sessions_table=SESSIONS_TABLE,
            messages_table=MESSAGES_TABLE,
            engine_kwargs={"pool_pre_ping": True},
        )
        self._engine = self._bootstrap.engine

        if CREATE_SESSION_TABLES:
            await self._bootstrap.get_items(limit=1)

        # crea tabla privada para sophistication (opcional pero recomendado)
        try:
            await ensure_sophistication_table(self._engine)
        except Exception:
            # si prefieres, loguea en vez de silenciar
            pass

    def _make_session(self, session_id: str) -> SQLAlchemySession:
        if self._engine is None:
            raise RuntimeError("AgentService not initialized (engine missing). Did startup run?")
        return SQLAlchemySession(
            session_id=session_id,
            engine=self._engine,
            create_tables=False,
            sessions_table=SESSIONS_TABLE,
            messages_table=MESSAGES_TABLE,
        )

    # -------------------------
    # Context seeding / updates
    # -------------------------
    def _context_items(
        self,
        *,
        portafolio_promedio: Optional[Dict[str, Any]],
        portafolio_inversionista: Optional[Dict[str, Any]],
        mi_filosofia: Optional[str],
        club_deals_concepts: Optional[str],
        club_deals_opinion: Optional[str],
    ) -> List[TResponseInputItem]:
        def txt(label: str, value: Optional[str]) -> str:
            return f"{label}:\n" + (value if (value is not None and str(value).strip() != "") else "NO PROVISTA")

        content: List[Dict[str, Any]] = [
            {
                "type": "input_text",
                "text": "CONTEXTO — portafolio_promedio (JSON):\n"
                + (pretty_json(portafolio_promedio) if portafolio_promedio is not None else "NO PROVISTA"),
            },
            {
                "type": "input_text",
                "text": "CONTEXTO — portafolio_inversionista (JSON):\n"
                + (pretty_json(portafolio_inversionista) if portafolio_inversionista is not None else "NO PROVISTA"),
            },
            {"type": "input_text", "text": txt("CONTEXTO — mi_filosofia (texto)", mi_filosofia)},
            {"type": "input_text", "text": txt("CONTEXTO — club_deals_concepts (texto)", club_deals_concepts)},
            {"type": "input_text", "text": txt("CONTEXTO — club_deals_opinion (texto)", club_deals_opinion)},
        ]
        return [{"role": "user", "content": content}]

    async def _seed_context_if_empty(
        self,
        session: SQLAlchemySession,
        *,
        portafolio_promedio: Optional[Dict[str, Any]],
        portafolio_inversionista: Optional[Dict[str, Any]],
        mi_filosofia: Optional[str],
        club_deals_concepts: Optional[str],
        club_deals_opinion: Optional[str],
    ) -> None:
        existing = await session.get_items(limit=1)
        if existing:
            return
        await session.add_items(
            self._context_items(
                portafolio_promedio=portafolio_promedio,
                portafolio_inversionista=portafolio_inversionista,
                mi_filosofia=mi_filosofia,
                club_deals_concepts=club_deals_concepts,
                club_deals_opinion=club_deals_opinion,
            )
        )

    @staticmethod
    def _val_to_fingerprint(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            s = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            s = str(value)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @classmethod
    def _get_field(cls, item: Any, key: str) -> Any:
        if item is None:
            return None
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key, None)

    @staticmethod
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
                    if isinstance(txt, str):
                        parts.append(txt)
                else:
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str):
                        parts.append(txt)
            return "\n".join([p for p in parts if p])
        txt = getattr(content, "text", None)
        if isinstance(txt, str):
            return txt
        return str(content)

    @classmethod
    def _extract_text_from_item(cls, item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item
        content = cls._get_field(item, "content")
        if content is not None:
            return cls._extract_text_from_content(content)
        return str(item)

    @staticmethod
    def _parse_ts(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            v = value.strip()
            if v.isdigit():
                try:
                    return float(v)
                except Exception:
                    return None
            try:
                if v.endswith("Z"):
                    v = v[:-1] + "+00:00"
                return datetime.fromisoformat(v).timestamp()
            except Exception:
                return None
        return None

    def _safe_extract_text(self, items: List[Any], max_items: int = 60, max_chars_per_item: int = 1200) -> str:
        """
        Convierte historial en un 'transcript' compacto para el evaluator.
        - Incluye roles
        - Trunca mensajes largos (portafolios enormes)
        """
        items_sorted = self._sorted_items(items)[-max_items:]
        lines: List[str] = []
        for it in items_sorted:
            role = self._get_field(it, "role") or "unknown"
            txt = (self._extract_text_from_item(it) or "").strip()
            if not txt:
                continue
            if len(txt) > max_chars_per_item:
                txt = txt[:max_chars_per_item] + "…(truncado)"
            lines.append(f"{role.upper()}: {txt}")
        return "\n".join(lines)

    @staticmethod
    def _safe_json_loads(s: str) -> Dict[str, Any]:
        """
        Parse robusto por si el modelo mete whitespace o algo raro.
        (igual el evaluator está instruido JSON-only)
        """
        s = (s or "").strip()
        # si por error viniera con ```json ... ```
        if s.startswith("```"):
            s = s.strip("`")
            s = s.replace("json", "", 1).strip()
        return json.loads(s)

    async def _evaluate_and_store_sophistication(self, session_id: str, session: SQLAlchemySession) -> None:
        """
        1) Lee historial de la sesión
        2) Ejecuta evaluator (gpt-4.1, store=False) SIN session memory
        3) Guarda payload en tabla privada
        """
        if self._engine is None:
            return

        history = await session.get_items(limit=250)
        transcript = self._safe_extract_text(history)

        # Prompt compacto: el evaluator ya tiene instructions, aquí solo le damos data
        prompt = (
            "EVALUA SOFISTICACIÓN del inversionista basado en este transcript.\n\n"
            "TRANSCRIPT:\n"
            f"{transcript}\n"
        )

        result = await Runner.run(
            investor_sophistication_evaluator,
            prompt,
            # 👇 NO pasamos session=... para evitar que se escriba en el chat
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
            payload = self._safe_json_loads(raw_out)
        except Exception:
            # Si falla parsing, no rompas el chat. Guarda algo mínimo:
            payload = {
                "level": "BASICO",
                "score": 0.0,
                "confidence": "baja",
                "signals": ["parse_error"],
                "evidence": [],
                "notes": "No se pudo parsear JSON del evaluator.",
            }

        # ID determinístico simple (evita uuid)
        row_id = hashlib.sha256(
            (session_id + "|" + str(datetime.utcnow().timestamp())).encode("utf-8")
        ).hexdigest()[:64]

        await insert_sophistication(
            self._engine,
            row_id=row_id,
            session_id=session_id,
            payload=payload,
            raw={"raw_output": raw_out, "payload": payload},
        )

    @classmethod
    def _sorted_items(cls, items: List[Any]) -> List[Any]:
        scored: List[Tuple[float, int, Any]] = []
        for idx, it in enumerate(items or []):
            ts = (
                cls._parse_ts(cls._get_field(it, "created_at"))
                or cls._parse_ts(cls._get_field(it, "createdAt"))
                or cls._parse_ts(cls._get_field(it, "timestamp"))
                or cls._parse_ts(cls._get_field(it, "ts"))
                or 0.0
            )
            scored.append((ts, idx, it))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [x[2] for x in scored]

    @classmethod
    def _last_philosophy(cls, items: List[Any]) -> Optional[str]:
        for it in reversed(cls._sorted_items(items)):
            if cls._get_field(it, "role") != "assistant":
                continue
            txt = cls._extract_text_from_item(it)
            if PHILOSOPHY_HEADER in (txt or ""):
                return txt
        return None

    @classmethod
    def _count_assistant_questions(cls, items: List[Any]) -> int:
        count = 0
        for it in cls._sorted_items(items):
            if cls._get_field(it, "role") != "assistant":
                continue
            txt = (cls._extract_text_from_item(it) or "").strip()
            if not txt:
                continue
            if PHILOSOPHY_HEADER in txt:
                continue  # filosofía no cuenta como pregunta
            if txt.endswith("?"):
                count += 1
        return count

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    @classmethod
    def _user_accepts(cls, message: str) -> bool:
        m = cls._norm(message)
        if not m:
            return False
        if m in {"acepto", "aceptado"}:
            return True
        if m.startswith("acepto") and not any(w in m for w in ["pero", "cambia", "ajusta", "modifica", "corrige"]):
            return True
        return False

    @staticmethod
    def _pick_starting_question(session_id: str) -> str:
        # determinístico por sesión para evitar "random" no reproducible
        h = hashlib.sha256(session_id.encode("utf-8")).digest()
        idx = int.from_bytes(h[:2], "big") % len(STARTING_QUESTIONS)
        return STARTING_QUESTIONS[idx]

    async def _append_user_message(self, session: SQLAlchemySession, message: str) -> None:
        await session.add_items([{"role": "user", "content": [{"type": "input_text", "text": message}]}])

    async def _append_context_updates_if_any(
        self,
        session: SQLAlchemySession,
        history: List[Any],
        *,
        portafolio_promedio: Optional[Dict[str, Any]],
        portafolio_inversionista: Optional[Dict[str, Any]],
        mi_filosofia: Optional[str],
        club_deals_concepts: Optional[str],
        club_deals_opinion: Optional[str],
    ) -> None:
        # Dedupe simple por fingerprint guardado en el texto del update
        latest = self._last_context_fingerprints(history)

        updates: List[Dict[str, Any]] = []

        def maybe_add(key: str, label: str, value: Any, renderer) -> None:
            if value is None:
                return
            fp = self._val_to_fingerprint(value)
            if latest.get(key) == fp:
                return
            updates.append({"type": "input_text", "text": f"UPDATE — {label}:\n{renderer(value)}\n__fp__={fp}"})

        maybe_add("portafolio_promedio", "portafolio_promedio (JSON)", portafolio_promedio, lambda v: pretty_json(v))
        maybe_add(
            "portafolio_inversionista",
            "portafolio_inversionista (JSON)",
            portafolio_inversionista,
            lambda v: pretty_json(v),
        )
        maybe_add("mi_filosofia", "mi_filosofia (texto)", mi_filosofia, lambda v: str(v))
        maybe_add("club_deals_concepts", "club_deals_concepts (texto)", club_deals_concepts, lambda v: str(v))
        maybe_add("club_deals_opinion", "club_deals_opinion (texto)", club_deals_opinion, lambda v: str(v))

        if updates:
            await session.add_items([{"role": "user", "content": updates}])

    @classmethod
    def _last_context_fingerprints(cls, items: List[Any]) -> Dict[str, str]:
        # Busca fingerprints en los UPDATE — ...
        fps: Dict[str, str] = {}
        for it in reversed(cls._sorted_items(items)):
            if cls._get_field(it, "role") != "user":
                continue
            txt = cls._extract_text_from_item(it)
            if "UPDATE —" not in txt or "__fp__=" not in txt:
                continue

            # Puede haber varios updates en un mismo content; extraemos todos
            for block in txt.split("UPDATE —"):
                if "__fp__=" not in block:
                    continue
                # detecta key por label
                label_line = block.strip().splitlines()[0] if block.strip() else ""
                fp = block.split("__fp__=")[-1].strip().splitlines()[0]
                if "portafolio_promedio" in label_line and "portafolio_promedio" not in fps:
                    fps["portafolio_promedio"] = fp
                elif "portafolio_inversionista" in label_line and "portafolio_inversionista" not in fps:
                    fps["portafolio_inversionista"] = fp
                elif "mi_filosofia" in label_line and "mi_filosofia" not in fps:
                    fps["mi_filosofia"] = fp
                elif "club_deals_concepts" in label_line and "club_deals_concepts" not in fps:
                    fps["club_deals_concepts"] = fp
                elif "club_deals_opinion" in label_line and "club_deals_opinion" not in fps:
                    fps["club_deals_opinion"] = fp

            if len(fps) >= 5:
                break
        return fps

    async def _run_refine_question(self, session: SQLAlchemySession) -> str:
        with trace("Filosofia WOW (Refine Question)"):
            rq = await Runner.run(
                filosofia_refine_question_agent,
                "Haz la pregunta de afinado ahora.",
                session=session,
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "fastapi-service",
                        "model_route": "refine_question",
                    }
                ),
            )
        return rq.final_output_as(str)

    async def chat(
        self,
        *,
        session_id: str,
        message: str,
        portafolio_promedio: Optional[Dict[str, Any]] = None,
        portafolio_inversionista: Optional[Dict[str, Any]] = None,
        mi_filosofia: Optional[str] = None,
        club_deals_concepts: Optional[str] = None,
        club_deals_opinion: Optional[str] = None,
        club_deals_information: Optional[str] = None,  # legacy
    ) -> str:
        # legacy mapping
        if club_deals_concepts is None and club_deals_information:
            club_deals_concepts = club_deals_information

        session = self._make_session(session_id)

        # Seed inicial con lo que haya (todo opcional)
        await self._seed_context_if_empty(
            session,
            portafolio_promedio=portafolio_promedio,
            portafolio_inversionista=portafolio_inversionista,
            mi_filosofia=mi_filosofia,
            club_deals_concepts=club_deals_concepts,
            club_deals_opinion=club_deals_opinion,
        )

        history = await session.get_items(limit=250)

        # Si el cliente manda inputs en mensajes posteriores, guárdalos como updates (dedupe)
        await self._append_context_updates_if_any(
            session,
            history,
            portafolio_promedio=portafolio_promedio,
            portafolio_inversionista=portafolio_inversionista,
            mi_filosofia=mi_filosofia,
            club_deals_concepts=club_deals_concepts,
            club_deals_opinion=club_deals_opinion,
        )

        # Re-fetch por si se agregaron updates
        history = await session.get_items(limit=300)
        last_philosophy = self._last_philosophy(history)

        # -------------------------
        # Fase: refinamiento (ya hay filosofía)
        # -------------------------
        if last_philosophy:
            # Guarda el mensaje del usuario aunque no usemos Runner (observabilidad)
            await self._append_user_message(session, message)

            if self._user_accepts(message):
                return "Perfecto. Queda como versión final:\n\n" + last_philosophy

            run_message = (
                "Actualiza la filosofía anterior según los cambios solicitados por el usuario. "
                "Mantén la estructura obligatoria y los requisitos WOW.\n\n"
                f"CAMBIOS SOLICITADOS:\n{message}"
            )

            with trace("Filosofia WOW (Builder Edit)"):
                result = await Runner.run(
                    filosofia_builder_agent,
                    run_message,
                    session=session,
                    run_config=RunConfig(
                        trace_metadata={
                            "__trace_source__": "fastapi-service",
                            "model_route": "builder_edit",
                        }
                    ),
                )

            philosophy = result.final_output_as(str)
            refine_q = await self._run_refine_question(session)
            return philosophy + "\n\n" + refine_q

        # -------------------------
        # Fase: entrevista (Q1..Q5)
        # -------------------------
        question_count = self._count_assistant_questions(history)

        # Q1: hardcode (de tu lista) — y SOLO UNA pregunta
        if question_count == 0:
            await self._append_user_message(session, message)
            q1 = self._pick_starting_question(session_id)
            await session.add_items([{"role": "assistant", "content": q1}])
            return q1

        # Si ya se hicieron 5 preguntas, el usuario está respondiendo a Q5 → generar filosofía
        if question_count >= QUESTIONS_TARGET:
            await self._append_user_message(session, message)
            run_message = (
                "Genera mi Filosofía de Inversión WOW final ahora, usando todo el contexto y las respuestas previas. "
                "Entrega SOLO la filosofía completa con la estructura obligatoria."
            )

            with trace("Filosofia WOW (Builder Initial)"):
                result = await Runner.run(
                    filosofia_builder_agent,
                    run_message,
                    session=session,
                    run_config=RunConfig(
                        trace_metadata={
                            "__trace_source__": "fastapi-service",
                            "model_route": "builder_initial",
                        }
                    ),
                )

            philosophy = result.final_output_as(str)
            refine_q = await self._run_refine_question(session)
            return philosophy + "\n\n" + refine_q

        # Si faltan preguntas: gpt-4.1 pregunta #2..#5 considerando inputs disponibles
        with trace("Filosofia WOW (Questions)"):
            result = await Runner.run(
                filosofia_questions_agent,
                message,
                session=session,
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "fastapi-service",
                        "model_route": "questions",
                    }
                ),
            )

        # 👇 evaluar y guardar (interno)
        await self._evaluate_and_store_sophistication(session_id, session)

        return result.final_output_as(str)
