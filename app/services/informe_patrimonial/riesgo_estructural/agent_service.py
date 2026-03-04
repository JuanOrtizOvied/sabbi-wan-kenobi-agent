from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional

from agents import Agent, ModelSettings, Runner
from openai import OpenAI
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel, Field

from app.core.config import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "gpt-5.2"
AGENT_NAME: Final[str] = "RiesgoEstructuralAgent"
UPLOAD_FILENAME: Final[str] = "score_data.json"
UPLOAD_MIMETYPE: Final[str] = "application/json"
UPLOAD_PURPOSE: Final[str] = "assistants"

# ---------------------------------------------------------------------------
# ReportLab rich-text formatting reference (for the model)
# ---------------------------------------------------------------------------
# Use these inline tags/conventions when producing each text field:
#
#   <b>text</b>          → Bold
#   <i>text</i>          → Italic / cursive
#   <u>text</u>          → Underline
#   <br/>                → Line break
#   • item               → Bullet point  (use literal "•" character)
#   <title>text</title>  → Section title  (rendered as bold + larger font)
#   <strike>text</strike>→ Strikethrough  (use sparingly)
#
# Rules:
#   - Wrap risk-level labels in <b>…</b>  (e.g. <b>Riesgo Medio</b>)
#   - Wrap dimension names in <i>…</i> the first time they appear
#   - Use "• " bullets for enumerated recommendations or risk factors
#   - Separate logical paragraphs with <br/><br/>
#   - conclusions field: open with a <title> line summarising the section
# ---------------------------------------------------------------------------

USER_INSTRUCTION: Final[str] = (
    "Aquí están los datos del portafolio del cliente adjuntos en el archivo JSON.\n\n"
    "Tu tarea es redactar ÚNICAMENTE la sección 'Riesgo estructural del portafolio' "
    "siguiendo exactamente el formato, tono y estructura indicados en las instrucciones del sistema.\n\n"
    "FORMATO DE SALIDA OBLIGATORIO:\n"
    "Devuelve un objeto JSON con exactamente estos seis campos:\n"
    "  • explicacion_concentracion   — explicación de la dimensión Concentración/Diversificación\n"
    "  • explicacion_correlacion      — explicación de la dimensión Correlación del portafolio\n"
    "  • explicacion_riesgo_gestor    — explicación de la dimensión Riesgo del gestor\n"
    "  • explicacion_riesgo_administrador — explicación de la dimensión Riesgo del administrador\n"
    "  • explicacion_moneda           — explicación de la dimensión Riesgo de moneda\n"
    "  • conclusions                  — conclusión integrada del riesgo estructural (1–2 párrafos)\n\n"
    "REGLAS PARA LOS CAMPOS DE EXPLICACIÓN (explicacion_*):\n"
    "  • PROHIBIDO mencionar o mostrar el score numérico (ej. '7', '6.5', '8/10') dentro del texto.\n"
    "    El score se presenta por separado en la tabla; la explicación solo describe la situación en lenguaje natural.\n"
    "  • Máximo 200 caracteres de texto plano por campo (sin contar etiquetas HTML).\n\n"
    "REGLAS DE FORMATO PARA EL TEXTO (ReportLab rich-text):\n"
    "  • Usa <b>…</b> para negritas (ej. nivel de riesgo, términos clave).\n"
    "  • Usa <i>…</i> para itálicas/cursiva (ej. nombres de dimensiones la primera vez).\n"
    "  • Usa <u>…</u> para subrayado (ej. alertas o recomendaciones clave).\n"
    "  • Usa el carácter '• ' para bullets cuando enumeres factores o recomendaciones.\n"
    "  • Usa <br/><br/> para separar párrafos dentro de un mismo campo.\n"
    "  • En el campo conclusions, abre con una línea de título en <b><u>…</u></b>.\n"
    "  • No inventes datos; usa solo los valores del JSON adjunto.\n"
    "  • No expliques el proceso ni agregues campos adicionales."
)

PERSONALITY_PROMPT: Final[str] = """\
Actúa como analista senior de riesgo estructural en Sabbi, comunicando a clientes con conocimiento medio/bajo.
Vas a redactar ÚNICAMENTE la sección "Riesgo estructural del portafolio".
No calcules nada. No inventes datos. Usa solo el input.

TONO
- Claro, profesional, no vendedor.
- Urgencia estratégica sin alarmismo: "postergar aumenta vulnerabilidad / reduce resiliencia".
- Evita tecnicismos. Si aparece "correlación", explícalo simple ("se mueven juntos").

CONTENIDO POR CAMPO
Cada campo del JSON de salida corresponde a una dimensión de la tabla de riesgos.
REGLA CRÍTICA: ningún campo de explicación (explicacion_*) debe mencionar ni mostrar
el valor numérico del score. Describe la situación únicamente en lenguaje natural.

explicacion_concentracion
  Basada en concentracion.interpretacion (NO menciones el score numérico).
  Explica de forma simple qué tan diversificado está el portafolio.

explicacion_correlacion
  Basada en correlacion.interpretacion (NO menciones el score numérico).
  Explica si los activos "se mueven juntos" y qué implica eso.

explicacion_riesgo_gestor
  Basada en gestor.score solo para inferir el nivel cualitativo (NO lo menciones en el texto).
  Lectura simple sobre la calidad promedio de quién toma las decisiones de inversión.

explicacion_riesgo_administrador
  Basada en administrador.score solo para inferir el nivel cualitativo (NO lo menciones en el texto).
  Lectura simple sobre solidez operativa y regulatoria de quienes custodian el portafolio.

explicacion_moneda
  Basada en moneda.pen_pct para determinar la exposición (NO menciones el score numérico).
  Si pen_pct es alto, indicar exposición relevante a PEN y sus implicancias.

conclusions
  1–2 párrafos cortos (máx. 2–4 líneas c/u) que integren:
  - el riesgo dominante (scores más bajos = mayor riesgo)
  - cómo se manifiesta en el portafolio (sin listar productos)
  - recomendación estructural general (diversificar drivers, reducir dependencia moneda/país con flujos futuros)
  - urgencia racional sin pánico
  Regla del nivel global usando global_score (redondeado): <=4 → <b>Alto</b>, 5–7 → <b>Medio</b>, >=8 → <b>Bajo</b>.
  Abre con una línea resumen del nivel: ej. "Nivel de riesgo estructural: <b>Medio</b>".

REGLAS GENERALES
- No listar la matriz de correlación ni números internos de la matriz.
- No mencionar nombres de productos salvo que sea imprescindible (preferir "bloques" o "exposiciones").
- No proponer ventas forzadas.
- Respetar siempre el formato ReportLab indicado en USER_INSTRUCTION.

INPUT
Recibirás un JSON con:
- global_score
- concentracion{score, interpretacion, hhi, inversiones_totales}
- correlacion{score, interpretacion, total_correlation}
- gestor{score}
- administrador{score}
- moneda{score, pen_pct}

SALIDA
Devuelve ÚNICAMENTE el objeto JSON con los seis campos definidos. Sin texto adicional.
"""


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class RiesgoEstructuralOutput(BaseModel):
    """Structured output for the 'Riesgo estructural del portafolio' section."""

    explicacion_concentracion: str = Field(
        description=(
            "Explicación de la dimensión Concentración/Diversificación. "
            "Máx. 200 caracteres de texto plano. Puede contener etiquetas ReportLab."
        )
    )
    explicacion_correlacion: str = Field(
        description=(
            "Explicación de la dimensión Correlación del portafolio. "
            "Máx. 200 caracteres de texto plano. Puede contener etiquetas ReportLab."
        )
    )
    explicacion_riesgo_gestor: str = Field(
        description=(
            "Explicación de la dimensión Riesgo del gestor. "
            "Máx. 200 caracteres de texto plano. Puede contener etiquetas ReportLab."
        )
    )
    explicacion_riesgo_administrador: str = Field(
        description=(
            "Explicación de la dimensión Riesgo del administrador. "
            "Máx. 200 caracteres de texto plano. Puede contener etiquetas ReportLab."
        )
    )
    explicacion_moneda: str = Field(
        description=(
            "Explicación de la dimensión Riesgo de moneda. "
            "Máx. 200 caracteres de texto plano. Puede contener etiquetas ReportLab."
        )
    )
    conclusions: str = Field(
        description=(
            "Conclusión integrada del riesgo estructural: 1–2 párrafos con nivel global, "
            "riesgo dominante, cómo se manifiesta y recomendación estructural. "
            "Formato ReportLab; abrir con línea de título en <b><u>…</u></b>."
        )
    )


# ---------------------------------------------------------------------------
# Reply container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AgentReply:
    output: RiesgoEstructuralOutput
    response_id: str


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AgentService:
    """
    Service wrapper around an Agents SDK Agent that:
      - uploads JSON input as a file attachment
      - runs the agent with a structured output type (RiesgoEstructuralOutput)
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
            output_type=RiesgoEstructuralOutput,
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
        Runs the agent and returns the structured 'Riesgo estructural' section.

        previous_response_id:
          - None for the first message
          - The last response.id for follow-up turns

        Returns an AgentReply whose `.output` is a validated RiesgoEstructuralOutput
        instance with ReportLab-formatted rich text in every field.
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

            structured_output = getattr(result, "final_output", None)
            last_id = getattr(result, "last_response_id", None)

            if not isinstance(structured_output, RiesgoEstructuralOutput):
                raise RuntimeError(
                    f"Runner returned unexpected final_output type: {type(structured_output)}"
                )
            if not isinstance(last_id, str):
                raise RuntimeError("Runner returned an unexpected last_response_id type")

            return AgentReply(output=structured_output, response_id=last_id)
        finally:
            if cleanup_uploaded_file:
                self._delete_file_safely(file_id)
