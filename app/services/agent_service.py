# app/services/agent_service.py
from __future__ import annotations

import hashlib
import hashlib as _hashlib
from typing import Any, Dict, List, Optional, Tuple

from agents import Runner, RunConfig, trace
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from app.agents.filosofia_wow_agent import (
    PHILOSOPHY_HEADER,
    SOCIAL_VERSION_HEADER,
    QUESTIONS_TARGET,
    STARTING_QUESTIONS,
    filosofia_de_inversion_builder,
    filosofia_questions_agent,
    filosofia_refine_question_agent,
    filosofia_social_version_agent,
)
from app.core.config import CREATE_SESSION_TABLES, MESSAGES_TABLE, SESSIONS_TABLE, SESSION_DB_URL
from app.utils.json_utils import pretty_json
from app.storage.sophistication_store import ensure_sophistication_table
from app.services.sophistication_eval import evaluate_and_store_sophistication

try:
    from agents import TResponseInputItem
except Exception:  # pragma: no cover
    from agents.items import TResponseInputItem


class AgentService:
    """
    Flujo:
    - Primera respuesta del asistente (inicio de sesión): pregunta #1 (hardcode desde STARTING_QUESTIONS)
    - Luego: preguntas #2..#5 con gpt-4.1
    - Tras la 5ta respuesta del usuario: generar filosofía con gpt-5.1 (reasoning high)
      (incluye versión para redes)
    - Después de generar: entrar en "modo refinamiento":
        - gpt-4.1 pregunta iterativamente qué tema pulir
        - SOLO cuando el usuario lo pida ("genera nueva versión", "actualiza", etc.), se regenera con gpt-5.1
        - Termina cuando el usuario responde "acepto"

    Retro-compat:
    - Si ya existe una filosofía anterior SIN versión para redes, se puede crear un bloque social
      a pedido del usuario ("versión para redes", "copy para redes", etc.) sin hacer preguntas extra.
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

        await ensure_sophistication_table(self._engine)

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

    # -------------------------
    # Context helpers
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
        content: List[Dict[str, Any]] = []

        if portafolio_promedio is not None:
            content.append(
                {"type": "input_text", "text": "CONTEXTO — portafolio_promedio (JSON):\n" + pretty_json(portafolio_promedio)}
            )
        if portafolio_inversionista is not None:
            content.append(
                {"type": "input_text", "text": "CONTEXTO — portafolio_inversionista (JSON):\n" + pretty_json(portafolio_inversionista)}
            )
        if mi_filosofia:
            content.append({"type": "input_text", "text": "CONTEXTO — mi_filosofia (texto):\n" + mi_filosofia})
        if club_deals_concepts:
            content.append({"type": "input_text", "text": "CONTEXTO — club_deals_concepts:\n" + club_deals_concepts})
        if club_deals_opinion:
            content.append({"type": "input_text", "text": "CONTEXTO — club_deals_opinion:\n" + club_deals_opinion})

        if not content:
            return []

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

        items = self._context_items(
            portafolio_promedio=portafolio_promedio,
            portafolio_inversionista=portafolio_inversionista,
            mi_filosofia=mi_filosofia,
            club_deals_concepts=club_deals_concepts,
            club_deals_opinion=club_deals_opinion,
        )
        if items:
            await session.add_items(items)

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
        content: List[Dict[str, Any]] = []
        if portafolio_promedio is not None:
            content.append({"type": "input_text", "text": "UPDATE — portafolio_promedio (JSON):\n" + pretty_json(portafolio_promedio)})
        if portafolio_inversionista is not None:
            content.append({"type": "input_text", "text": "UPDATE — portafolio_inversionista (JSON):\n" + pretty_json(portafolio_inversionista)})
        if mi_filosofia is not None:
            content.append({"type": "input_text", "text": "UPDATE — mi_filosofia:\n" + (mi_filosofia or "VACÍO")})
        if club_deals_concepts is not None:
            content.append({"type": "input_text", "text": "UPDATE — club_deals_concepts:\n" + (club_deals_concepts or "VACÍO")})
        if club_deals_opinion is not None:
            content.append({"type": "input_text", "text": "UPDATE — club_deals_opinion:\n" + (club_deals_opinion or "VACÍO")})

        if content:
            await session.add_items([{"role": "user", "content": content}])

    # -------------------------
    # History parsing
    # -------------------------
    @staticmethod
    def _get_field(item: Any, key: str) -> Any:
        if item is None:
            return None
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key, None)

    @classmethod
    def _extract_text_from_item(cls, item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item
        content = cls._get_field(item, "content")
        if content is None:
            return str(item)

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

    @classmethod
    def _last_philosophy(cls, items: List[Any]) -> Optional[str]:
        for it in reversed(items or []):
            if cls._get_field(it, "role") != "assistant":
                continue
            txt = cls._extract_text_from_item(it)
            if PHILOSOPHY_HEADER in (txt or ""):
                return txt
        return None

    @classmethod
    def _count_questions_before_philosophy(cls, items: List[Any]) -> int:
        count = 0
        for it in items or []:
            if cls._get_field(it, "role") != "assistant":
                continue
            txt = (cls._extract_text_from_item(it) or "").strip()
            if not txt:
                continue
            if PHILOSOPHY_HEADER in txt:
                break
            if txt.endswith("?"):
                count += 1
        return count

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    @classmethod
    def _user_accepts(cls, message: str) -> bool:
        m = cls._norm(message)
        return m in {"acepto", "aceptado", "ok", "listo", "perfecto", "de acuerdo", "conforme"}

    @classmethod
    def _user_requests_regenerate(cls, message: str) -> bool:
        m = cls._norm(message)
        triggers = [
            "genera nueva version",
            "genera nueva versión",
            "genera una nueva version",
            "genera una nueva versión",
            "regenera",
            "actualiza la filosofia",
            "actualiza la filosofía",
            "aplica cambios",
            "crea una nueva",
            "reformula",
            "haz una nueva",
            "nueva version",
            "nueva versión",
        ]
        return any(t in m for t in triggers)

    @classmethod
    def _user_requests_social_version(cls, message: str) -> bool:
        m = cls._norm(message)
        triggers = [
            "version para redes",
            "versión para redes",
            "copy para redes",
            "texto para redes",
            "post para redes",
            "caption",
            "instagram",
            "linkedin",
            "para compartir",
            "para publicar",
            "version corta",
            "versión corta",
        ]
        return any(t in m for t in triggers)

    # -------------------------
    # Starting question selection
    # -------------------------
    @staticmethod
    def _pick_starting_question(session_id: str) -> str:
        if not STARTING_QUESTIONS:
            return "¿Cómo describirías tu filosofía de inversión, en general?"
        idx = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest(), 16) % len(STARTING_QUESTIONS)
        return STARTING_QUESTIONS[idx]

    async def _append_user_message(self, session: SQLAlchemySession, message: str) -> None:
        await session.add_items([{"role": "user", "content": [{"type": "input_text", "text": message}]}])

    async def _append_assistant_message(self, session: SQLAlchemySession, message: str) -> None:
        await session.add_items([{"role": "assistant", "content": message}])

    @classmethod
    def _infer_presence_from_history(
        cls,
        history: List[Any],
        *,
        portafolio_promedio: Optional[Dict[str, Any]],
        portafolio_inversionista: Optional[Dict[str, Any]],
        club_deals_concepts: Optional[str],
        club_deals_opinion: Optional[str],
    ) -> Dict[str, bool]:
        text = "\n".join(cls._extract_text_from_item(it) for it in (history or []))
        return {
            "has_portafolio_promedio": (portafolio_promedio is not None) or ("portafolio_promedio" in text),
            "has_portafolio_inversionista": (portafolio_inversionista is not None) or ("portafolio_inversionista" in text),
            "has_club_deals_concepts": bool(club_deals_concepts) or ("club_deals_concepts" in text),
            "has_club_deals_opinion": bool(club_deals_opinion) or ("club_deals_opinion" in text),
        }

    @staticmethod
    def _extract_social_block(philosophy_text: str) -> str:
        txt = philosophy_text or ""
        idx = txt.find(SOCIAL_VERSION_HEADER)
        if idx == -1:
            return ""
        return txt[idx:].strip()

    async def _ensure_social_version_for_existing_philosophy(
        self,
        *,
        session: SQLAlchemySession,
        last_philosophy: str,
    ) -> str:
        if SOCIAL_VERSION_HEADER in (last_philosophy or ""):
            return last_philosophy

        social_prompt = (
            "Genera SOLO el bloque de versión para redes sociales en primera persona, "
            "basado en la filosofía existente.\n\n"
            f"FILOSOFÍA EXISTENTE:\n{last_philosophy}\n"
        )
        with trace("Filosofia WOW (Social Only)"):
            social_res = await Runner.run(
                filosofia_social_version_agent,
                social_prompt,
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "fastapi-service",
                        "model_route": "social_only",
                    }
                ),
            )
        social_block = (social_res.final_output_as(str) or "").strip()

        combined = (last_philosophy or "").rstrip() + "\n\n" + social_block
        await self._append_assistant_message(session, combined)
        return combined

    # -------------------------
    # Main chat
    # -------------------------
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
        club_deals_information: Optional[str] = None,
    ) -> str:
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

        await self._append_context_updates(
            session,
            portafolio_promedio=portafolio_promedio,
            portafolio_inversionista=portafolio_inversionista,
            mi_filosofia=mi_filosofia,
            club_deals_concepts=club_deals_concepts,
            club_deals_opinion=club_deals_opinion,
        )

        history = await session.get_items(limit=250)
        last_philosophy = self._last_philosophy(history)

        assistant_exists = any(self._get_field(it, "role") == "assistant" for it in history or [])
        if not assistant_exists:
            await self._append_user_message(session, message)
            q1 = self._pick_starting_question(session_id)
            await self._append_assistant_message(session, q1)
            return q1

        # 1) Ya existe filosofía
        if last_philosophy:
            # ✅ SOCIAL-ONLY: si el usuario pide versión para redes, devolver SOLO ese bloque (sin preguntas)
            if self._user_requests_social_version(message):
                if SOCIAL_VERSION_HEADER not in last_philosophy:
                    last_philosophy = await self._ensure_social_version_for_existing_philosophy(
                        session=session,
                        last_philosophy=last_philosophy,
                    )
                social_block = self._extract_social_block(last_philosophy)
                return social_block or "No pude encontrar/generar la versión para redes. ¿Quieres que la regenere?"

            if self._user_accepts(message):
                return "Perfecto. Queda como versión final:\n\n" + last_philosophy

            if self._user_requests_regenerate(message):
                run_message = (
                    "Actualiza la filosofía anterior (incluida la versión para redes) según los cambios "
                    "y aclaraciones del usuario en la conversación. Devuelve los DOS BLOQUES obligatorios."
                )
                with trace("Filosofia WOW (Builder Edit)"):
                    result = await Runner.run(
                        filosofia_de_inversion_builder,
                        run_message,
                        session=session,
                        run_config=RunConfig(
                            trace_metadata={
                                "__trace_source__": "fastapi-service",
                                "model_route": "builder_edit",
                            }
                        ),
                    )
                philosophy_text = result.final_output_as(str)

                presence = self._infer_presence_from_history(
                    history=await session.get_items(limit=250),
                    portafolio_promedio=portafolio_promedio,
                    portafolio_inversionista=portafolio_inversionista,
                    club_deals_concepts=club_deals_concepts,
                    club_deals_opinion=club_deals_opinion,
                )
                philosophy_hash = _hashlib.sha256((philosophy_text or "").encode("utf-8")).hexdigest()
                await evaluate_and_store_sophistication(
                    engine=self._engine,
                    session_id=session_id,
                    session=session,
                    presence=presence,
                    event_type="builder_edit",
                    philosophy_hash=philosophy_hash,
                )

                refine_prompt = (
                    "Esta es la filosofía actual (incluye versión para redes). "
                    "Haz una sola pregunta para precisar qué tema pulir a continuación.\n\n"
                    f"FILOSOFÍA:\n{philosophy_text}\n"
                )
                refine_result = await Runner.run(
                    filosofia_refine_question_agent,
                    refine_prompt,
                    session=session,
                    run_config=RunConfig(
                        trace_metadata={
                            "__trace_source__": "fastapi-service",
                            "model_route": "refine_question",
                        }
                    ),
                )
                refine_q = refine_result.final_output_as(str)

                return philosophy_text + "\n\n" + refine_q

            # Refinamiento normal (1 pregunta)
            refine_prompt = (
                "Tenemos una filosofía ya generada. "
                "El usuario respondió esto:\n"
                f"{message}\n\n"
                "Haz UNA pregunta para precisar el tema específico a pulir y qué cambio desea. "
                "Si el usuario está conforme, puede responder 'acepto'. "
                "Si quiere una nueva versión, que diga 'genera nueva versión'.\n\n"
                f"FILOSOFÍA ACTUAL:\n{last_philosophy}\n"
            )
            with trace("Filosofia WOW (Refine Question)"):
                refine_result = await Runner.run(
                    filosofia_refine_question_agent,
                    refine_prompt,
                    session=session,
                    run_config=RunConfig(
                        trace_metadata={
                            "__trace_source__": "fastapi-service",
                            "model_route": "refine_question",
                        }
                    ),
                )
            return refine_result.final_output_as(str)

        # 2) No hay filosofía aún → entrevista
        question_count = self._count_questions_before_philosophy(history)

        if question_count >= QUESTIONS_TARGET:
            run_message = (
                "Genera mi Filosofía de Inversión WOW final ahora, usando todo el contexto y respuestas previas. "
                "Devuelve los DOS BLOQUES obligatorios: filosofía completa + versión para redes."
            )
            with trace("Filosofia WOW (Builder Initial)"):
                result = await Runner.run(
                    filosofia_de_inversion_builder,
                    run_message,
                    session=session,
                    run_config=RunConfig(
                        trace_metadata={
                            "__trace_source__": "fastapi-service",
                            "model_route": "builder_initial",
                        }
                    ),
                )
            philosophy_text = result.final_output_as(str)

            presence = self._infer_presence_from_history(
                history=await session.get_items(limit=250),
                portafolio_promedio=portafolio_promedio,
                portafolio_inversionista=portafolio_inversionista,
                club_deals_concepts=club_deals_concepts,
                club_deals_opinion=club_deals_opinion,
            )
            philosophy_hash = _hashlib.sha256((philosophy_text or "").encode("utf-8")).hexdigest()
            await evaluate_and_store_sophistication(
                engine=self._engine,
                session_id=session_id,
                session=session,
                presence=presence,
                event_type="builder_initial",
                philosophy_hash=philosophy_hash,
            )

            refine_prompt = (
                "Esta es la filosofía recién generada (incluye versión para redes). "
                "Haz una sola pregunta para que el usuario indique qué desea afinar y en qué tema específico. "
                "Indica que puede responder 'acepto' o 'genera nueva versión'.\n\n"
                f"FILOSOFÍA:\n{philosophy_text}\n"
            )
            refine_result = await Runner.run(
                filosofia_refine_question_agent,
                refine_prompt,
                session=session,
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "fastapi-service",
                        "model_route": "refine_question",
                    }
                ),
            )
            refine_q = refine_result.final_output_as(str)
            return philosophy_text + "\n\n" + refine_q

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

        return result.final_output_as(str)
