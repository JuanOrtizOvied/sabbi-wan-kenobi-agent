from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional

from agents import Agent, ModelSettings, Runner
from openai import OpenAI
from openai.types.shared.reasoning import Reasoning

from app.core.config import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "gpt-5.2"
AGENT_NAME: Final[str] = "RiesgoEstructuralAgent"
UPLOAD_FILENAME: Final[str] = "score_data.json"
UPLOAD_MIMETYPE: Final[str] = "application/json"
UPLOAD_PURPOSE: Final[str] = "assistants"

USER_INSTRUCTION: Final[str] = (
    "Aquí están los datos del portafolio del cliente. "
    "Redacta la sección 'Calidad de Portafolio' siguiendo "
    "exactamente el formato y tono indicados en las instrucciones."
)

PERSONALITY_PROMPT: Final[str] = """\
Actúa como analista senior de riesgo estructural en Sabbi, comunicando a clientes con conocimiento medio/bajo.
Vas a redactar ÚNICAMENTE la sección “Riesgo estructural del portafolio”.
No calcules nada. No inventes datos. Usa solo el input.
TONO
- Claro, profesional, no vendedor.
- Urgencia estratégica sin alarmismo: “postergar aumenta vulnerabilidad / reduce resiliencia”.
- Evita tecnicismos. Si aparece “correlación”, explícalo simple (“se mueven juntos”).
ESTRUCTURA (obligatoria, similar al informe)
1) Intro (2–4 líneas)
Explica que comparar rentabilidad es fácil, pero entender riesgo es clave; por eso se revisan varias dimensiones.
2) “Riesgos estructurales del portafolio”
Tabla con columnas EXACTAS:
Dimensión de riesgo | Score (1–10) | Explicación
Filas a incluir y cómo llenarlas:
- Concentración / Diversificación → concentracion.score + concentracion.interpretacion (explicación simple)
- Correlación del portafolio → correlacion.score + correlacion.interpretacion (explicación simple)
- Riesgo del gestor → gestor.score (redondear a 1 decimal) + lectura simple (“calidad promedio de quién toma decisiones”)
- Riesgo del administrador → administrador.score (1 decimal) + lectura simple (“solidez operativa/regulatoria”)
- Riesgo de moneda → moneda.score + lectura simple (“exposición relevante a PEN” si pen_pct es alto)
4) “Conclusión del riesgo estructural”
1–2 párrafos, priorizando:
- cuál es el riesgo dominante (usa los scores más bajos)
- cómo se manifiesta en el portafolio (sin listar todos los productos)
- recomendación estructural general (ej. diversificar drivers, reducir dependencia país/moneda con flujos futuros)
- urgencia racional sin pánico
REGLAS
- No listar la matriz de correlación ni números internos de la matriz.
- No mencionar nombres de productos salvo que sea imprescindible (preferir “bloques” o “exposiciones”).
- No proponer ventas forzadas.
INPUT
Recibirás un JSON con:
- global_score
- concentracion{score, interpretacion, hhi, inversiones_totales}
- correlacion{score, interpretacion, total_correlation}
- gestor{score}
- administrador{score}
- moneda{score, pen_pct}
SALIDA
Solo el texto final de la sección + la tabla en Markdown.
No expliques el proceso.
"""


@dataclass(frozen=True, slots=True)
class AgentReply:
    text: str
    response_id: str


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


class AgentService:
    """
    Service wrapper around an Agents SDK Agent that:
      - uploads JSON input as a file attachment
      - runs the agent
      - optionally deletes the uploaded file
    """

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        agent: Optional[Agent] = None,
        *,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._openai = openai_client or self._build_openai_client()
        self._agent = agent or self._build_agent(model=model)

    @staticmethod
    def _build_openai_client() -> OpenAI:
        if not settings.OPENAI_API_KEY:
            raise ConfigError("OPENAI_API_KEY is missing")
        return OpenAI(api_key=settings.OPENAI_API_KEY)

    @staticmethod
    def _build_agent(*, model: str) -> Agent:
        return Agent(
            name=AGENT_NAME,
            instructions=PERSONALITY_PROMPT,
            model=model,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="high", summary="auto"),
            ),
        )

    @staticmethod
    def _json_bytes(json_data: Mapping[str, Any]) -> bytes:
        return json.dumps(json_data, ensure_ascii=False, indent=2).encode("utf-8")

    def _upload_json_file(self, json_data: Mapping[str, Any]) -> str:
        """Upload JSON input as a file and return its OpenAI file_id."""
        file_obj = io.BytesIO(self._json_bytes(json_data))
        uploaded = self._openai.files.create(
            file=(UPLOAD_FILENAME, file_obj, UPLOAD_MIMETYPE),
            purpose=UPLOAD_PURPOSE,
        )
        return uploaded.id

    def _delete_file_safely(self, file_id: str) -> None:
        """Best-effort cleanup; never fail the request due to cleanup errors."""
        try:
            self._openai.files.delete(file_id)
        except Exception:
            log.exception("Failed to delete uploaded file_id=%s", file_id)

    @staticmethod
    def _build_user_message(*, file_id: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "input_text", "text": USER_INSTRUCTION},
                {"type": "input_file", "file_id": file_id},
            ],
        }

    def reply(
        self,
        json_data: Mapping[str, Any],
        previous_response_id: Optional[str] = None,
        *,
        cleanup_uploaded_file: bool = True,
    ) -> AgentReply:
        """
        Runs the agent and returns the generated 'Calidad de Portafolio' section.

        previous_response_id:
          - None for the first message
          - The last response.id for follow-up turns
        """
        if not isinstance(json_data, Mapping):
            raise TypeError("json_data must be a mapping (dict-like)")

        file_id = self._upload_json_file(json_data)
        try:
            result = Runner.run_sync(
                self._agent,
                input=[self._build_user_message(file_id=file_id)],
                previous_response_id=previous_response_id,
            )

            final_text = getattr(result, "final_output", None)
            last_id = getattr(result, "last_response_id", None)
            if not isinstance(final_text, str) or not isinstance(last_id, str):
                raise RuntimeError("Runner returned an unexpected result shape")

            return AgentReply(text=final_text, response_id=last_id)
        finally:
            if cleanup_uploaded_file:
                self._delete_file_safely(file_id)