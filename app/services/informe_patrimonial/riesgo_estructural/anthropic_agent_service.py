from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional

import anthropic
from pydantic import BaseModel, Field

from app.core.config import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "claude-sonnet-4-5-20250514"
MAX_TOKENS: Final[int] = 16_384

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
    "Devuelve un objeto JSON con exactamente estos ocho campos:\n"
    "  1. introduccion_riesgo         — párrafo introductorio personalizado (ver instrucción abajo)\n"
    "  2. explicacion_concentracion   — explicación de la dimensión Concentración/Diversificación\n"
    "  3. explicacion_correlacion      — explicación de la dimensión Correlación del portafolio\n"
    "  4. explicacion_riesgo_gestor    — explicación de la dimensión Riesgo del gestor\n"
    "  5. explicacion_riesgo_administrador — explicación de la dimensión Riesgo del administrador\n"
    "  6. explicacion_moneda           — explicación de la dimensión Riesgo de moneda\n"
    "  7. conclusions                  — conclusión integrada del riesgo estructural (1–2 párrafos)\n"
    "  8. titulo_riesgo_estructural    — título corto que resuma el diagnóstico de conclusions\n\n"
    "IMPORTANTE: Redacta PRIMERO introduccion_riesgo, luego las explicaciones y conclusions, "
    "y DESPUÉS genera titulo_riesgo_estructural como un titular corto que sintetice lo que ya escribiste.\n\n"
    "INSTRUCCIÓN PARA introduccion_riesgo:\n"
    "  • Es un párrafo corto (2–3 líneas máximo) que sintetiza el nivel de riesgo global\n"
    "    y los principales focos de atención ESPECÍFICOS de este portafolio.\n"
    "  • DEBE variar entre clientes — no usar texto genérico ni plantilla fija.\n"
    "  • Debe nombrar explícitamente las 2–3 dimensiones con mayor riesgo (scores más bajos)\n"
    "    para este cliente concreto, en lenguaje simple.\n"
    "  • Formato: usar el campo nivel_de_riesgo_estructural del JSON para el nivel global.\n"
    "  • Ejemplo correcto (Héctor, riesgo moneda bajo y concentración alta):\n"
    "    'El riesgo global del portafolio es Medio (aceptable). Los principales focos de atención\n"
    "    son la alta dependencia del sol peruano y la concentración en pocas posiciones,\n"
    "    mientras que la calidad de gestores y la diversificación entre activos mitigan el riesgo.'\n"
    "  • Ejemplo correcto (Ursula, sin riesgo de moneda pero concentración crítica):\n"
    "    'El riesgo global del portafolio es Medio (aceptable). El principal foco de atención\n"
    "    es la concentración en pocas posiciones con calidad de gestores heterogénea;\n"
    "    la ausencia de exposición al sol peruano es un punto a favor.'\n"
    "  • PROHIBIDO usar el texto genérico actual: 'Los principales focos de atención se concentran\n"
    "    en la correlación entre los activos, la concentración del portafolio y la exposición a moneda'\n"
    "    — ese texto no puede aparecer en ningún informe.\n\n"
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
    "  • En el campo conclusions, NO incluir línea de título (la sección ya tiene encabezado propio).\n"
    "  • El campo titulo_riesgo_estructural es texto plano corto, SIN marcado XML.\n"
    "  • No inventes datos; usa solo los valores del JSON adjunto.\n"
    "  • No expliques el proceso ni agregues campos adicionales.\n\n"
    "REGLA DE FORMATO PARA conclusions:\n"
    "  • Usar párrafo corrido (NO bullets).\n"
    "  • 1–2 párrafos cortos.\n"
    "  • Integrar: nivel global de riesgo, riesgo dominante, manifestación en el portafolio,\n"
    "    recomendación estructural general y urgencia racional sin pánico.\n"
    "  • Comenzar directamente con el contenido — NO incluir título ni encabezado."
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
- Lenguaje simple: evitar "iliquidez", "rebalanceo", "drivers", "benchmark", "drawdown".

CONTENIDO POR CAMPO

introduccion_riesgo
  Párrafo personalizado que resume el nivel de riesgo global y los focos específicos
  de ESTE portafolio. Ver instrucción detallada en USER_INSTRUCTION.
  PROHIBIDO usar texto genérico. Debe nombrar las dimensiones con mayor riesgo
  para este cliente concreto.

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
  Si pen_pct es bajo o cero, indicar que el riesgo cambiario está acotado.
  Adaptar siempre a la situación real del cliente — no usar texto genérico.

conclusions
  DEBE usar el campo nivel_de_riesgo_estructural del JSON para establecer y comunicar
  el nivel global de riesgo en la conclusión.
  Formato: párrafo corrido (NO bullets). 1–2 párrafos cortos.
  Integrar:
  - el nivel de riesgo global tomado de nivel_de_riesgo_estructural
  - el riesgo dominante (dimensiones con scores más bajos)
  - cómo se manifiesta en el portafolio (sin listar productos)
  - recomendación estructural general
  - urgencia racional sin pánico
  NO incluir línea de título.
  Comenzar directamente con el contenido.

titulo_riesgo_estructural
  ORDEN: redacta PRIMERO todas las explicaciones y conclusions, DESPUÉS genera este título.
  Titular corto (máx. 8 palabras) que sintetice el diagnóstico que ya escribiste en conclusions.
  Calibra el tono según lo que redactaste:
    - vulnerabilidades severas → tono de alerta fuerte
    - riesgos relevantes → tono de advertencia
    - estructura funcional con mejoras → tono neutro-constructivo
    - solidez → tono positivo
  Texto plano, SIN marcado XML. PROHIBIDO incluir valores numéricos o scores.

REGLAS GENERALES
- No listar la matriz de correlación ni números internos de la matriz.
- No mencionar nombres de productos salvo que sea imprescindible.
- No proponer ventas forzadas.
- No mencionar los pesos porcentuales de cada dimensión.
- Respetar siempre el formato ReportLab indicado en USER_INSTRUCTION.

INPUT
Recibirás un JSON con:
- global_score
- nivel_de_riesgo_estructural
- concentracion{score, interpretacion, hhi, inversiones_totales}
- correlacion{score, interpretacion, total_correlation}
- gestor{score}
- administrador{score}
- moneda{score, pen_pct}

SALIDA
Devuelve ÚNICAMENTE el objeto JSON con los ocho campos definidos. Sin texto adicional.
"""


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class RiesgoEstructuralOutput(BaseModel):
    """Structured output for the 'Riesgo estructural del portafolio' section."""

    introduccion_riesgo: str = Field(
        description=(
            "Párrafo introductorio personalizado (2–3 líneas) que sintetiza el nivel "
            "de riesgo global y los focos de atención ESPECÍFICOS de este portafolio. "
            "Debe nombrar las dimensiones con mayor riesgo para este cliente. "
            "Puede contener etiquetas ReportLab. PROHIBIDO usar texto genérico."
        )
    )
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
            "Formato ReportLab; NO incluir línea de título (la sección ya tiene encabezado propio)."
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
    Service that uses the Anthropic Messages API to:
      - enrich the input JSON with derived fields (nivel_de_riesgo_estructural)
      - send the data inline in the user message
      - return a schema-guaranteed structured output via messages.parse()

    Each string value in AgentReply.output contains ReportLab XML markup
    ready to be consumed by reportlab_utils.field_to_flowables().
    """

    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self._client = client or self._build_client()
        self._model = model
        self._max_tokens = max_tokens

    @staticmethod
    def _build_client() -> anthropic.Anthropic:
        if not settings.ANTHROPIC_API_KEY:
            raise ConfigError("ANTHROPIC_API_KEY is missing")
        return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

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

    @classmethod
    def _build_user_content(cls, json_data: Mapping[str, Any]) -> str:
        """
        Build the full user message content.

        Enriches the JSON with derived fields and embeds it inline.
        """
        enriched = cls._enrich_json(json_data)
        portfolio_json = json.dumps(enriched, ensure_ascii=False, indent=2)
        return (
            f"{USER_INSTRUCTION}\n\n"
            f"--- INICIO DATOS DEL PORTAFOLIO ---\n"
            f"{portfolio_json}\n"
            f"--- FIN DATOS DEL PORTAFOLIO ---"
        )

    def reply(
        self,
        json_data: Mapping[str, Any],
    ) -> AgentReply:
        """
        Runs the model and returns the structured 'Riesgo estructural' section.

        AgentReply.output is a dict[str, str] where each value contains
        ReportLab-formatted rich text. Pass each value through
        reportlab_utils.field_to_flowables() to obtain PDF-ready flowables.
        """
        if not isinstance(json_data, Mapping):
            raise TypeError("json_data must be a mapping (dict-like)")

        user_content = self._build_user_content(json_data)

        response = self._client.messages.parse(
            model=self._model,
            max_tokens=self._max_tokens,
            system=PERSONALITY_PROMPT,
            thinking={"type": "adaptive"},
            messages=[
                {"role": "user", "content": user_content},
            ],
            output_format=RiesgoEstructuralOutput,
        )

        parsed = response.parsed_output
        if not isinstance(parsed, RiesgoEstructuralOutput):
            raise RuntimeError(
                f"API returned unexpected output type: {type(parsed)}"
            )

        return AgentReply(
            output=parsed.model_dump(),
            response_id=response.id,
        )
