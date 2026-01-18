from typing import Any, Dict, List, Optional

from agents import Runner, RunConfig, trace
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from app.agents.filosofia_wow_agent import filosofia_de_inversion
from app.core.config import CREATE_SESSION_TABLES, MESSAGES_TABLE, SESSIONS_TABLE, SESSION_DB_URL
from app.utils.json_utils import pretty_json

try:
    from agents import TResponseInputItem
except Exception:  # pragma: no cover
    from agents.items import TResponseInputItem


class AgentService:
    """
    - Crea 1 engine compartido
    - Crea SQLAlchemySession por request (session_id)
    - "Siembra" contexto SOLO si la sesión está vacía:
        - portafolio_promedio (req)
        - portafolio_inversionista (req)
        - mi_filosofia (opcional)
        - club_deals_information (opcional)
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
        portafolio_promedio: Dict[str, Any],
        portafolio_inversionista: Dict[str, Any],
        mi_filosofia: Optional[str],
        club_deals_information: Optional[str],
    ) -> List[TResponseInputItem]:
        # Siempre incluimos los 2 JSON (son el core)
        content: List[Dict[str, Any]] = [
            {
                "type": "input_text",
                "text": "CONTEXTO — portafolio_promedio (JSON):\n" + pretty_json(portafolio_promedio),
            },
            {
                "type": "input_text",
                "text": "CONTEXTO — portafolio_inversionista (JSON):\n" + pretty_json(portafolio_inversionista),
            },
        ]

        # Opcionales: igual los “marcamos” explícitamente si no llegan
        content.append(
            {
                "type": "input_text",
                "text": "CONTEXTO — mi_filosofia (texto):\n" + (mi_filosofia if mi_filosofia else "NO PROVISTA"),
            }
        )
        content.append(
            {
                "type": "input_text",
                "text": "CONTEXTO — club_deals_information:\n" + (club_deals_information if club_deals_information else "NO PROVISTA"),
            }
        )

        return [{"role": "user", "content": content}]

    async def _seed_context_if_empty(
        self,
        session: SQLAlchemySession,
        *,
        portafolio_promedio: Optional[Dict[str, Any]],
        portafolio_inversionista: Optional[Dict[str, Any]],
        mi_filosofia: Optional[str],
        club_deals_information: Optional[str],
    ) -> None:
        existing = await session.get_items(limit=1)
        if existing:
            return  # ya hay contexto (y conversación), no volvemos a sembrar

        # Si es la primera interacción de la sesión, exigimos los JSON
        if portafolio_promedio is None or portafolio_inversionista is None:
            raise ValueError(
                "Falta contexto inicial: 'portafolio_promedio' y 'portafolio_inversionista' "
                "son requeridos al menos en el primer request de la sesión."
            )

        await session.add_items(
            self._context_items(
                portafolio_promedio=portafolio_promedio,
                portafolio_inversionista=portafolio_inversionista,
                mi_filosofia=mi_filosofia,
                club_deals_information=club_deals_information,
            )
        )

    async def chat(
        self,
        *,
        session_id: str,
        message: str,
        portafolio_promedio: Optional[Dict[str, Any]],
        portafolio_inversionista: Optional[Dict[str, Any]],
        mi_filosofia: Optional[str],
        club_deals_information: Optional[str],
    ) -> str:
        session = self._make_session(session_id)
        await self._seed_context_if_empty(
            session,
            portafolio_promedio=portafolio_promedio,
            portafolio_inversionista=portafolio_inversionista,
            mi_filosofia=mi_filosofia,
            club_deals_information=club_deals_information,
        )

        with trace("Filosofia WOW (FastAPI)"):
            result = await Runner.run(
                filosofia_de_inversion,
                message,              # ✅ string input
                session=session,      # ✅ session memory enabled
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "fastapi-service",
                        "workflow_id": "wf_693a02d72190819097a8a7b5234510f70851015287e3b178",
                    }
                ),
            )

        return result.final_output_as(str)
