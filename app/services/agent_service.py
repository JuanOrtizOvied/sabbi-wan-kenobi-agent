from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import hashlib

from agents import Runner, RunConfig, trace
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from app.agents.filosofia_wow_agent import (
    PHILOSOPHY_HEADER,
    QUESTIONS_TARGET,
    REFINE_QUESTION,
    STARTING_QUESTIONS,
    filosofia_de_inversion_builder,
    filosofia_de_inversion_questions,
)
from app.core.config import CREATE_SESSION_TABLES, MESSAGES_TABLE, SESSIONS_TABLE, SESSION_DB_URL
from app.utils.json_utils import pretty_json

try:
    from agents import TResponseInputItem
except Exception:  # pragma: no cover
    from agents.items import TResponseInputItem


class AgentService:
    """
    Inputs (todos opcionales):
    - portafolio_inversionista (JSON)
    - portafolio_promedio (JSON)
    - mi_filosofia (texto)
    - club_deals_concepts (definición/racional)
    - club_deals_opinion (qué piensa el inversionista)

    Backward-compat:
    - club_deals_information (legacy) se interpreta como club_deals_concepts si no viene concepts.

    Lógica:
    1) Primera respuesta del sistema (inicio de chat): 1 pregunta hardcodeada (de lista) → cuenta como Q1/5
    2) gpt-4.1 hace las siguientes 4 preguntas (una por mensaje) hasta completar 5
    3) gpt-5.1 (reasoning high) genera la filosofía
    4) Loop de refinamiento (gpt-5.1) hasta que el usuario responda “acepto”
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

    def _context_items(
        self,
        *,
        portafolio_promedio: Optional[Dict[str, Any]],
        portafolio_inversionista: Optional[Dict[str, Any]],
        mi_filosofia: Optional[str],
        club_deals_concepts: Optional[str],
        club_deals_opinion: Optional[str],
    ) -> List[TResponseInputItem]:
        content: List[Dict[str, Any]] = [
            {
                "type": "input_text",
                "text": "CONTEXTO — portafolio_promedio (JSON):\n"
                + (pretty_json(portafolio_promedio) if portafolio_promedio else "NO PROVISTA"),
            },
            {
                "type": "input_text",
                "text": "CONTEXTO — portafolio_inversionista (JSON):\n"
                + (pretty_json(portafolio_inversionista) if portafolio_inversionista else "NO PROVISTA"),
            },
            {
                "type": "input_text",
                "text": "CONTEXTO — mi_filosofia (texto):\n" + (mi_filosofia if mi_filosofia else "NO PROVISTA"),
            },
            {
                "type": "input_text",
                "text": "CONTEXTO — club_deals_concepts:\n"
                + (club_deals_concepts if club_deals_concepts else "NO PROVISTA"),
            },
            {
                "type": "input_text",
                "text": "CONTEXTO — club_deals_opinion:\n"
                + (club_deals_opinion if club_deals_opinion else "NO PROVISTA"),
            },
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

    async def _append_context_updates(
        self,
        session: SQLAlchemySession,
        *,
        portafolio_promedio: Optional[Dict[str, Any]],
        portafolio_inversionista: Optional[Dict[str, Any]],
        mi_filosofia: Optional[str],
        club_deals_concepts: Optional[str],
        club_deals_opinion: Optional[str],
    ) -> None:
        """
        Si en requests posteriores llegan valores no-None, los guardamos como UPDATE en memoria.
        Esto permite que el usuario comparta portafolios/filosofía a mitad de flujo sin romper el contexto.
        """
        updates: List[str] = []
        if portafolio_promedio is not None:
            updates.append("UPDATE — portafolio_promedio (JSON):\n" + pretty_json(portafolio_promedio))
        if portafolio_inversionista is not None:
            updates.append("UPDATE — portafolio_inversionista (JSON):\n" + pretty_json(portafolio_inversionista))
        if mi_filosofia is not None:
            updates.append("UPDATE — mi_filosofia (texto):\n" + (mi_filosofia or "VACÍO"))
        if club_deals_concepts is not None:
            updates.append("UPDATE — club_deals_concepts:\n" + (club_deals_concepts or "VACÍO"))
        if club_deals_opinion is not None:
            updates.append("UPDATE — club_deals_opinion:\n" + (club_deals_opinion or "VACÍO"))

        if not updates:
            return

        await session.add_items([{"role": "user", "content": "\n\n".join(updates)}])

    # -------------------------
    # Helpers de parsing (robustos a dict/obj + content como string/lista)
    # -------------------------
    @staticmethod
    def _get_field(item: Any, key: str) -> Any:
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
    def _count_interview_questions(cls, items: List[Any]) -> int:
        count = 0
        for it in cls._sorted_items(items):
            if cls._get_field(it, "role") != "assistant":
                continue
            txt = (cls._extract_text_from_item(it) or "").strip()
            if not txt or PHILOSOPHY_HEADER in txt:
                continue
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
        if m in {"acepto", "aceptado", "ok", "okey", "listo", "perfecto", "de acuerdo", "conforme"}:
            return True
        if "acepto" in m and not any(w in m for w in ["pero", "cambia", "ajusta", "modifica", "corrige"]):
            return True
        return False

    @staticmethod
    def _pick_starting_question(session_id: str) -> str:
        # determinístico por session_id para no "saltar" entre servidores
        h = hashlib.sha256(session_id.encode("utf-8")).digest()
        idx = int.from_bytes(h[:2], "big") % len(STARTING_QUESTIONS)
        return STARTING_QUESTIONS[idx]

    async def _ensure_first_question(self, session: SQLAlchemySession, session_id: str) -> Optional[str]:
        history = await session.get_items(limit=80)
        if self._count_interview_questions(history) > 0:
            return None

        q = self._pick_starting_question(session_id)
        # Guardamos la pregunta como mensaje assistant para que cuente como Q1/5.
        await session.add_items([{"role": "assistant", "content": q}])
        return q

    async def chat(
        self,
        *,
        session_id: str,
        message: str,
        portafolio_promedio: Optional[Dict[str, Any]],
        portafolio_inversionista: Optional[Dict[str, Any]],
        mi_filosofia: Optional[str],
        club_deals_concepts: Optional[str],
        club_deals_opinion: Optional[str],
        # Backward-compat:
        club_deals_information: Optional[str] = None,
    ) -> str:
        # Backward-compat mapping
        if club_deals_concepts is None and club_deals_information:
            club_deals_concepts = club_deals_information

        session = self._make_session(session_id)
        await self._seed_context_if_empty(
            session,
            portafolio_promedio=portafolio_promedio,
            portafolio_inversionista=portafolio_inversionista,
            mi_filosofia=mi_filosofia,
            club_deals_concepts=club_deals_concepts,
            club_deals_opinion=club_deals_opinion,
        )

        # Si llegan updates en mensajes posteriores, los guardamos.
        await self._append_context_updates(
            session,
            portafolio_promedio=portafolio_promedio,
            portafolio_inversionista=portafolio_inversionista,
            mi_filosofia=mi_filosofia,
            club_deals_concepts=club_deals_concepts,
            club_deals_opinion=club_deals_opinion,
        )

        # Primera pregunta hardcodeada (solo al inicio real del chat)
        first_q = await self._ensure_first_question(session, session_id)
        if first_q:
            return first_q

        history = await session.get_items(limit=250)
        last_philosophy = self._last_philosophy(history)

        # -------------------------
        # Fase 2: Refinamiento (ya existe filosofía)
        # -------------------------
        if last_philosophy:
            if self._user_accepts(message):
                return "Perfecto. Queda como versión final:\n\n" + last_philosophy

            run_message = (
                "Actualiza la filosofía anterior según estos cambios solicitados por el usuario. "
                "Mantén la estructura obligatoria y los requisitos WOW.\n\n"
                f"CAMBIOS SOLICITADOS:\n{message}"
            )
            agent = filosofia_de_inversion_builder

            with trace("Filosofia WOW (FastAPI)"):
                result = await Runner.run(
                    agent,
                    run_message,
                    session=session,
                    run_config=RunConfig(
                        trace_metadata={
                            "__trace_source__": "fastapi-service",
                            "workflow_id": "wf_693a02d72190819097a8a7b5234510f70851015287e3b178",
                            "model_route": "builder_edit",
                        }
                    ),
                )

            return result.final_output_as(str) + "\n\n" + REFINE_QUESTION

        # -------------------------
        # Fase 1: Entrevista (5 preguntas en total, Q1 hardcode + Q2-Q5 con gpt-4.1)
        # -------------------------
        question_count = self._count_interview_questions(history)

        # Respuesta del usuario a la 5ta → generar filosofía con gpt-5.1 high reasoning
        if question_count >= QUESTIONS_TARGET:
            run_message = (
                "Genera mi Filosofía de Inversión WOW final ahora, usando todo el contexto y las respuestas previas. "
                "Entrega SOLO la filosofía completa con la estructura obligatoria."
            )
            agent = filosofia_de_inversion_builder

            with trace("Filosofia WOW (FastAPI)"):
                result = await Runner.run(
                    agent,
                    run_message,
                    session=session,
                    run_config=RunConfig(
                        trace_metadata={
                            "__trace_source__": "fastapi-service",
                            "workflow_id": "wf_693a02d72190819097a8a7b5234510f70851015287e3b178",
                            "model_route": "builder_initial",
                        }
                    ),
                )

            return result.final_output_as(str) + "\n\n" + REFINE_QUESTION

        # Aún faltan preguntas → seguir entrevistando con gpt-4.1
        agent = filosofia_de_inversion_questions

        with trace("Filosofia WOW (FastAPI)"):
            result = await Runner.run(
                agent,
                message,
                session=session,
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "fastapi-service",
                        "workflow_id": "wf_693a02d72190819097a8a7b5234510f70851015287e3b178",
                        "model_route": "questions",
                    }
                ),
            )

        return result.final_output_as(str)
