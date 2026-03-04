from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional

from agents import Agent, ModelSettings, Runner
from openai import OpenAI
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel

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

REGLAS ESTRICTAS DE SALIDA (obligatorio)
- PROHIBIDO usar tablas (Markdown tables o cualquier formato de tabla).
- PROHIBIDO mostrar scores o puntajes de cualquier tipo.
  (No escribas "Score", "x/10", ni valores de campos como score_total_weighted.)
- No inventes datos. Usa solo señales del input.
- Si incluyes números, que sea solo para pesos/porcentajes u otros datos descriptivos, siempre en texto (nunca en tabla).

TONO Y ESTILO (obligatorio)
- Claro, sencillo, profesional.
- No vendedor. No comercial.
- Urgencia estratégica sin alarmismo: enfatiza costo de oportunidad y resiliencia, sin pánico.
- No uses tecnicismos innecesarios. Si aparece un término, explícalo en lenguaje simple.

FORMATO (obligatorio)
Sigue esta estructura y longitud aproximada del ejemplo del informe:

1) calidad_portafolio_description — Descripción general (2–3 líneas, UN SOLO PÁRRAFO)
- No uses viñetas.
- Describe de forma fluida: mezcla por tipo de activo, alineación de riesgo al perfil, concentraciones geográficas.
- Sirve como puente entre la intro y el análisis, sin adelantar conclusiones.

2) alineacion_activo_description — Alineación por tipo de activo
- Máximo 5–7 líneas.
- 1 párrafo (o 2 muy cortos) con el mensaje principal (sobre/subpeso, diversificación, estabilidad).
- Puedes mencionar 2–4 desbalances clave en texto, SIN tablas y SIN scores.

3) alineacion_riesgo_description — Alineación de riesgo
- 1 párrafo explicando el contraste entre el riesgo agregado del portafolio y el rango objetivo del perfil.
- 3 bullets "En términos prácticos…" (permitidos aquí), sin mencionar scores ni valores numéricos de scoring.

4) alineacion_geografica_description — Alineación geográfica
- 1–2 párrafos explicando el principal riesgo (concentración y subexposición), sin alarmismo.
- Si mencionas porcentajes, hazlo dentro del texto (sin tablas).

5) conclusions — Principales conclusiones
- 3 a 5 bullet points máximo.
- Sintetiza: qué está razonablemente bien, cuál es el foco de mejora más importante,
  y urgencia racional: "mientras más se posterga, más lento es corregirlo con flujos futuros".

INPUT
Recibirás un JSON con estas llaves:
- global_score
- alineacion_activo{score, asset_details[]}
- alineacion_riesgo{score, score_total_weighted, perfil_riesgo, perfil_range{min,max}}
- alineacion_geografica{score, interpretation, region_details[]}
"""


class CalidadPortafolioOutput(BaseModel):
    calidad_portafolio_description: str
    alineacion_activo_description: str
    alineacion_riesgo_description: str
    alineacion_geografica_description: str
    conclusions: str


@dataclass(frozen=True, slots=True)
class AgentReply:
    response_id: str
    parsed: dict[str, str]


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


class AgentService:
    """
    Service wrapper around an Agents SDK Agent that:
      - uploads JSON input as a file attachment
      - runs the agent with a structured output_type
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
            output_type=CalidadPortafolioOutput,
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

            parsed = result.final_output
            last_id = getattr(result, "last_response_id", None)
            if not isinstance(parsed, CalidadPortafolioOutput) or not isinstance(last_id, str):
                raise RuntimeError("Runner returned an unexpected result shape")

            return AgentReply(response_id=last_id, parsed=parsed.model_dump())
        finally:
            if cleanup_uploaded_file:
                self._delete_file_safely(file_id)
