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
AGENT_NAME: Final[str] = "CalidadPortafolioAgent"
UPLOAD_FILENAME: Final[str] = "score_data.json"
UPLOAD_MIMETYPE: Final[str] = "application/json"
UPLOAD_PURPOSE: Final[str] = "assistants"

USER_INSTRUCTION: Final[str] = (
    "Aquí están los datos del portafolio del cliente. "
    "Redacta la sección 'Calidad de Portafolio' siguiendo "
    "exactamente el formato y tono indicados en las instrucciones."
)

PERSONALITY_PROMPT: Final[str] = """\
Actúa como consultor senior en asesoría patrimonial de Sabbi. Todos los datos están en el JSON adjunto.
Vas a redactar ÚNICAMENTE la sección de "Calidad de Portafolio" del informe (no el resumen ejecutivo).
El cliente tiene conocimiento medio/bajo en inversiones.

TONO Y ESTILO (obligatorio)
- Claro, sencillo, profesional.
- No vendedor. No comercial.
- Urgencia estratégica sin alarmismo: enfatiza costo de oportunidad y resiliencia, sin pánico.
- No uses tecnicismos innecesarios. Si aparece un término, explícalo en lenguaje simple.
- No inventes datos. Usa solo los valores del input.

FORMATO (obligatorio)
Sigue esta estructura y longitud aproximada del ejemplo del informe:

1) Intro (2–4 líneas)
Explica que Sabbi compara el portafolio contra un benchmark de referencia (Sabbi Cracks) para detectar oportunidades de mejora estructural.

1.5) Descripción general de Calidad de Portafolio (3–5 líneas)
Resume en lenguaje simple el estado actual del portafolio en sus tres dimensiones de análisis:
- Alineación por tipo de activo: menciona el score y si el portafolio está bien o mal distribuido entre tipos de activo.
- Alineación de riesgo: menciona si el nivel de riesgo del portafolio está dentro del perfil del cliente.
- Alineación geográfica: menciona si la distribución geográfica es adecuada o presenta concentraciones relevantes.
Esta descripción debe servir como puente entre la intro y el análisis detallado, sin adelantar conclusiones.

2) "Alineación por tipo de activo"
Incluye:
- "Score: {alineacion_activo.score}/10 – {interpretación corta}"
- 1 párrafo corto explicando el mensaje principal (ej. sobre/underweights, liquidez, privados, etc.)

3) "Alineación de riesgo"
Incluye:
- 1 párrafo explicando qué significa el score_total_weighted versus el rango perfil_range.
- 3 bullets "En términos prácticos…" usando señales del input (perfil_range, score_total_weighted). No menciones productos por nombre.

4) "Alineación geográfica"
Incluye:
- 1–2 párrafos explicando el principal riesgo (concentración y subexposición), sin alarmismo.

5) "Principales conclusiones"
Un bloque final de 3–5 líneas máximo, sintetizando:
- Qué está razonablemente bien
- Qué es el foco de mejora más importante
- Urgencia racional: "mientras más se posterga, más lento es corregirlo con flujos futuros"

INPUT
Recibirás un JSON con estas llaves:
- global_score
- alineacion_activo{score, asset_details[]}
- alineacion_riesgo{score, score_total_weighted, perfil_riesgo, perfil_range{min,max}}
- alineacion_geografica{score, interpretation, region_details[]}

SALIDA
Entrega solo el texto final de la sección, con sus tablas en formato de texto/Markdown.
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