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
# Global-score interpretive ranges for Riesgo Estructural
# ---------------------------------------------------------------------------
SCORE_RANGES: Final[list[tuple[float, float, str]]] = [
    (1.0,  2.99999, "Muy Alto (crítico)"),
    (3.0,  4.99999, "Alto (deficiente)"),
    (5.0,  6.99999, "Medio (aceptable)"),
    (7.0,  8.99999, "Bajo (sólido)"),
    (9.0, 10.0,     "Muy Bajo (excelente)"),
]

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
    "METODOLOGÍA DE CÁLCULO DEL SCORE DE RIESGO ESTRUCTURAL\n"
    "El score global se calcula como promedio ponderado de cinco dimensiones:\n"
    "  • Concentración / Diversificación (25%) — distribución del patrimonio entre inversiones; "
    "identifica posiciones con peso relevante que amplifican el impacto de eventos específicos.\n"
    "  • Correlación del portafolio (25%) — qué tan relacionados están los activos entre sí, "
    "agrupados por motor económico; cuando se mueven juntos, la diversificación real se reduce.\n"
    "  • Riesgo del gestor (15%) — calidad y experiencia de los gestores que toman decisiones de "
    "inversión, según ranking interno del equipo de inversiones de Sabbi.\n"
    "  • Riesgo del administrador (15%) — riesgo operativo y regulatorio de las entidades que "
    "estructuran y administran los fondos (solidez, regulación, trayectoria).\n"
    "  • Riesgo de moneda (20%) — exposición a variaciones del tipo de cambio según la proporción "
    "del patrimonio invertida en monedas distintas a la moneda de referencia.\n\n"
    "RANGOS DE INTERPRETACIÓN DEL SCORE GLOBAL (campo global_score)\n"
    "  • 1.0 – 2.99  → Muy Alto (crítico): el portafolio presenta vulnerabilidades severas.\n"
    "  • 3.0 – 4.99  → Alto (deficiente): existen riesgos estructurales que requieren acción prioritaria.\n"
    "  • 5.0 – 6.99  → Medio (aceptable): la estructura es funcional pero con espacios de mejora relevantes.\n"
    "  • 7.0 – 8.99  → Bajo (sólido): el portafolio está bien estructurado con ajustes menores posibles.\n"
    "  • 9.0 – 10.0  → Muy Bajo (excelente): estructura altamente resiliente y diversificada.\n"
    "Usa estos rangos para calibrar el tono (más urgente o más positivo) "
    "pero NUNCA menciones los valores numéricos del score ni estos rangos en el texto de salida.\n\n"
    "FORMATO DE SALIDA OBLIGATORIO:\n"
    "Devuelve un objeto JSON con exactamente estos siete campos:\n"
    "  1. explicacion_concentracion   — explicación de la dimensión Concentración/Diversificación\n"
    "  2. explicacion_correlacion      — explicación de la dimensión Correlación del portafolio\n"
    "  3. explicacion_riesgo_gestor    — explicación de la dimensión Riesgo del gestor\n"
    "  4. explicacion_riesgo_administrador — explicación de la dimensión Riesgo del administrador\n"
    "  5. explicacion_moneda           — explicación de la dimensión Riesgo de moneda\n"
    "  6. conclusions                  — conclusión integrada del riesgo estructural (1–2 párrafos). "
    "Debe usar el campo nivel_de_riesgo_estructural del JSON para establecer el nivel global de riesgo.\n"
    "  7. titulo_riesgo_estructural    — título corto que resuma el diagnóstico de conclusions\n\n"
    "IMPORTANTE: Redacta PRIMERO las explicaciones y conclusions, y DESPUÉS genera "
    "titulo_riesgo_estructural como un titular corto que sintetice lo que ya escribiste.\n\n"
    "REGLAS PARA LOS CAMPOS DE EXPLICACIÓN (explicacion_*):\n"
    "  • PROHIBIDO mencionar o mostrar el score numérico (ej. '7', '6.5', '8/10') dentro del texto.\n"
    "    El score se presenta por separado en la tabla; la explicación solo describe la situación en lenguaje natural.\n"
    "  • PROHIBIDO mencionar los pesos porcentuales de cada dimensión (25%, 15%, 20%).\n"
    "  • PROHIBIDO iniciar el texto con el nombre de la dimensión como prefijo (ej. NO escribir "
    "\"Concentración:\", \"Correlación:\", \"Riesgo de administrador:\", etc.). "
    "El nombre de la dimensión ya aparece en la tabla; la explicación debe ir directo al contenido.\n"
    "  • Máximo 200 caracteres de texto plano por campo (sin contar etiquetas HTML).\n\n"
    "REGLAS DE FORMATO PARA EL TEXTO (ReportLab rich-text):\n"
    "  • Usa <b>…</b> para negritas (ej. nivel de riesgo, términos clave).\n"
    "  • Usa <i>…</i> para itálicas/cursiva (ej. nombres de dimensiones la primera vez).\n"
    "  • Usa <u>…</u> para subrayado (ej. alertas o recomendaciones clave).\n"
    "  • Usa el carácter '• ' para bullets cuando enumeres factores o recomendaciones.\n"
    "  • Usa <br/><br/> para separar párrafos dentro de un mismo campo.\n"
    "  • En el campo conclusions, abre con una línea de título en <b><u>…</u></b>.\n"
    "  • El campo titulo_riesgo_estructural es texto plano corto, SIN marcado XML.\n"
    "  • No inventes datos; usa solo los valores del JSON adjunto.\n"
    "  • No expliques el proceso ni agregues campos adicionales."
)

PERSONALITY_PROMPT: Final[str] = """\
Actúa como analista senior de riesgo estructural en Sabbi, comunicando a clientes con conocimiento medio/bajo.
Vas a redactar ÚNICAMENTE la sección "Riesgo estructural del portafolio".
No calcules nada. No inventes datos. Usa solo el input.

METODOLOGÍA DE CÁLCULO (contexto interno — NO incluir en la salida)
El score global de riesgo estructural se obtiene como promedio ponderado de cinco dimensiones:
  Concentración/Diversificación (25%) + Correlación (25%) + Riesgo gestor (15%) \
+ Riesgo administrador (15%) + Riesgo moneda (20%).

Descripción de cada dimensión (para tu comprensión, NO reproducir textualmente):
  • Concentración / Diversificación — distribución del patrimonio entre inversiones; \
identifica posiciones con peso relevante que amplifican el impacto de eventos específicos.
  • Correlación del portafolio — qué tan relacionados están los activos, agrupados por motor \
económico; cuando se mueven juntos, la diversificación real se reduce.
  • Riesgo del gestor — calidad y experiencia de los gestores que toman decisiones de inversión, \
según ranking interno del equipo de inversiones de Sabbi.
  • Riesgo del administrador — riesgo operativo y regulatorio de las entidades que estructuran \
y administran los fondos (solidez, regulación, trayectoria).
  • Riesgo de moneda — exposición a variaciones del tipo de cambio según la proporción del \
patrimonio invertida en monedas distintas a la moneda de referencia.

Rangos de interpretación del score global:
  1.0–2.99 → Muy Alto (crítico) | 3.0–4.99 → Alto (deficiente) | 5.0–6.99 → Medio (aceptable) \
| 7.0–8.99 → Bajo (sólido) | 9.0–10.0 → Muy Bajo (excelente).
Usa estos rangos para calibrar el tono (más urgente o más positivo), \
pero NUNCA menciones valores numéricos del score, ponderaciones ni rangos en el texto de salida.

TONO
- Claro, profesional, no vendedor.
- Urgencia estratégica sin alarmismo: "postergar aumenta vulnerabilidad / reduce resiliencia".
- Evita tecnicismos. Si aparece "correlación", explícalo simple ("se mueven juntos").

CONTENIDO POR CAMPO
Cada campo del JSON de salida corresponde a una dimensión de la tabla de riesgos.
REGLA CRÍTICA: ningún campo de explicación (explicacion_*) debe mencionar ni mostrar
el valor numérico del score. Describe la situación únicamente en lenguaje natural.
PROHIBIDO iniciar el texto con el nombre de la dimensión como prefijo
(ej. NO escribir "Concentración:", "Correlación:", "Riesgo del gestor:", etc.).
El nombre ya aparece en la tabla; ve directo al contenido.

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
  DEBE usar el campo nivel_de_riesgo_estructural del JSON para establecer y comunicar
  el nivel global de riesgo en la conclusión. Este campo ya viene precalculado con el rango
  correcto — NO intentes clasificar el score tú mismo, usa directamente el valor de nivel_de_riesgo_estructural.
  1–2 párrafos cortos (máx. 2–4 líneas c/u) que integren:
  - el nivel de riesgo global tomado de nivel_de_riesgo_estructural (ej. "Medio (aceptable)")
  - el riesgo dominante (scores más bajos = mayor riesgo)
  - cómo se manifiesta en el portafolio (sin listar productos)
  - recomendación estructural general (diversificar drivers, reducir dependencia moneda/país con flujos futuros)
  - urgencia racional sin pánico
  Abre con una línea resumen del nivel usando el valor de nivel_de_riesgo_estructural:
  ej. "Nivel de riesgo estructural: <b>Medio</b>".

titulo_riesgo_estructural
  ORDEN: redacta PRIMERO todas las explicaciones y conclusions, DESPUÉS genera este título.
  Titular corto (máx. 8 palabras) que sintetice el diagnóstico que ya escribiste en conclusions.
  Lee tu propia conclusions y extrae la idea central como titular.
  Calibra el tono según lo que redactaste:
    - Si conclusions señala vulnerabilidades severas → tono de alerta fuerte.
    - Si señala riesgos relevantes → tono de advertencia.
    - Si describe estructura funcional con mejoras → tono neutro-constructivo.
    - Si describe solidez → tono positivo.
    - Si describe excelencia → tono muy positivo.
  Texto plano, SIN marcado XML. PROHIBIDO incluir valores numéricos o scores.

REGLAS GENERALES
- No listar la matriz de correlación ni números internos de la matriz.
- No mencionar nombres de productos salvo que sea imprescindible (preferir "bloques" o "exposiciones").
- No proponer ventas forzadas.
- No mencionar los pesos porcentuales de cada dimensión (25%, 15%, 20%).
- Respetar siempre el formato ReportLab indicado en USER_INSTRUCTION.

INPUT
Recibirás un JSON con:
- global_score (calculado como promedio ponderado de las 5 dimensiones)
- nivel_de_riesgo_estructural (string con el rango interpretativo, e.g. "Medio (aceptable)")
- concentracion{score, interpretacion, hhi, inversiones_totales}
- correlacion{score, interpretacion, total_correlation}
- gestor{score}
- administrador{score}
- moneda{score, pen_pct}

SALIDA
Devuelve ÚNICAMENTE el objeto JSON con los siete campos definidos. Sin texto adicional.
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
            "Conclusión integrada del riesgo estructural: 1–2 párrafos con nivel global "
            "(tomado del campo nivel_de_riesgo_estructural del JSON), "
            "riesgo dominante, cómo se manifiesta y recomendación estructural. "
            "Formato ReportLab; abrir con línea de título en <b><u>…</u></b>."
        )
    )
    titulo_riesgo_estructural: str = Field(
        description=(
            "Título corto (máx. 8 palabras) que sintetice el diagnóstico redactado en conclusions. "
            "Texto plano, sin marcado XML. Sin valores numéricos ni scores."
        )
    )


# ---------------------------------------------------------------------------
# Reply container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AgentReply:
    output: dict[str, str]
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

    @staticmethod
    def _classify_global_score(score: float) -> str:
        """Return the interpretive range label for a given global_score.

        Ranges use continuous boundaries (e.g. 5.0–6.99999 → Medio).
        The raw score is compared directly without rounding.
        """
        for lo, hi, label in SCORE_RANGES:
            if lo <= score <= hi:
                return label
        return "fuera_de_rango"

    @classmethod
    def _enrich_json(cls, json_data: Mapping[str, Any]) -> dict[str, Any]:
        """Add derived fields (e.g. nivel_de_riesgo_estructural) to the input data."""
        enriched = dict(json_data)
        global_score = enriched.get("global_score")
        if isinstance(global_score, (int, float)):
            enriched["nivel_de_riesgo_estructural"] = cls._classify_global_score(float(global_score))
        return enriched

    def _upload_json_file(self, json_data: Mapping[str, Any]) -> str:
        """Upload JSON input as a file and return its OpenAI file_id."""
        enriched = self._enrich_json(json_data)
        file_obj = io.BytesIO(self._json_bytes(enriched))
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

            return AgentReply(output=structured_output.model_dump(), response_id=last_id)
        finally:
            if cleanup_uploaded_file:
                self._delete_file_safely(file_id)
