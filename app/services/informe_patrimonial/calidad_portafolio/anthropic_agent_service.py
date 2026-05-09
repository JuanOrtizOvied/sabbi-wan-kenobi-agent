from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional

from anthropic import Anthropic
from pydantic import BaseModel

from app.core.config import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "claude-sonnet-4-5-20250514"
AGENT_NAME: Final[str] = "CalidadPortafolioAgent"
THINKING_BUDGET_TOKENS: Final[int] = 10_000
MAX_OUTPUT_TOKENS: Final[int] = 16_384

USER_INSTRUCTION: Final[str] = """\
A continuación se adjunta el JSON con el análisis ya procesado de la sección "Calidad de Portafolio" de un cliente.

Tu tarea NO es recalcular métricas ni rehacer el análisis.
Tu tarea es interpretar el input y redactar únicamente la sección "Calidad de Portafolio".

Debes completar exactamente los siguientes campos del output:
1. calidad_portafolio_description
2. alineacion_activo_description
3. alineacion_riesgo_description
4. alineacion_geografica_description
5. conclusions

NOTA SOBRE EL SCORE DE CALIDAD DE PORTAFOLIO

El score de calidad de portafolio se calcula utilizando tres dimensiones con las siguientes ponderaciones:
- Alineación por tipo de activo: 40%
- Alineación de riesgo: 40%
- Alineación geográfica: 20%

Usa estas ponderaciones para contextualizar la importancia relativa de cada dimensión al interpretar el score global.

ESTRUCTURA DEL INPUT

El JSON puede incluir información como la siguiente:

- nombre_cliente
- horizonte_inversion
- patrimonio_total_invertible
- distribucion_clase_activo
- distribucion_geografica

- score_calidad_portafolio
- score_calidad_portafolio_interpretacion_sabbi
- componentes_calidad_portafolio:
  - score_alineacion_activos
  - score_alineacion_riesgo
  - score_alineacion_geografica

- alineacion_activos:
  - score
  - interpretacion_sabbi
  - benchmark_sabbi
  - rangos_permitidos
  - distribucion_cliente

- alineacion_riesgo:
  - score
  - interpretacion_sabbi
  - score_performance
  - perfil_riesgo
  - rango_objetivo
  - distancia_al_rango

- alineacion_geografica:
  - score
  - interpretacion_sabbi
  - benchmark_sabbi
  - rangos_permitidos
  - distribucion_cliente

- criterios_interpretacion:
  - qué mide cada indicador
  - cómo interpretar scores altos y bajos
  - cómo usar cada indicador en el análisis

LLAVES DISPONIBLES / CÓMO USARLAS

- score_calidad_portafolio → úsalo para interpretar el estado global de la calidad del portafolio
- alineacion_activos → úsalo para identificar sobrepesos, subpesos y desbalances estructurales por tipo de activo
- alineacion_riesgo → úsalo para evaluar si el portafolio está más conservador o más agresivo que el perfil
- alineacion_geografica → úsalo para identificar concentración regional, subexposición internacional y riesgo país
- benchmark_sabbi y rangos_permitidos → úsalos como referencia para entender si una desviación es leve, moderada o relevante
- distribucion_cliente → úsala para identificar los principales patrones estructurales, no para describir todos los datos uno por uno
- interpretacion_sabbi → úsala como guía semántica, no como texto a copiar literalmente

IMPORTANTE

- El output debe ser exclusivamente narrativo.
- No incluyas tablas, cuadros, velocímetros, títulos de sección ni descripciones metodológicas.
- No repitas scores, benchmarks ni porcentajes visibles en otras partes del reporte,
  salvo que sean indispensables para explicar una concentración o desbalance de forma clara.
- No copies literalmente el input ni las interpretaciones Sabbi.
- Enfócate en interpretación, implicancias y síntesis.

EJEMPLOS DE REFERENCIA

Los siguientes ejemplos representan exactamente el tipo de contenido que debes generar.
Replica su estilo, claridad, nivel de profundidad y lógica de interpretación.

--- Ejemplo 1 ---

El portafolio presenta una base funcional, mostrando una construcción patrimonial ordenada y consistente.

No obstante, se identifican desbalances estructurales, principalmente en la diversificación geográfica y en la
concentración de ciertos drivers económicos, que limitan la diversificación efectiva.

El portafolio presenta una estructura razonablemente alineada, con desbalances identificables pero corregibles.

Se observa una inclinación hacia mayor liquidez y menor exposición a activos de crecimiento estructural.

El nivel de riesgo del portafolio se encuentra ligeramente por encima del rango objetivo, lo que implica una
estructura más conservadora de lo que permitiría el perfil.

Predomina una exposición a activos defensivos y una menor participación en motores de crecimiento estructural.

Existe una sobreexposición significativa a un solo entorno económico, junto con una subexposición a mercados
desarrollados. Esta concentración incrementa la dependencia del portafolio y limita su diversificación efectiva.

El principal foco de mejora se encuentra en la diversificación estructural, especialmente a nivel geográfico.

--- Ejemplo 2 ---

El portafolio presenta una base funcional, pero no está alineado con una estructura óptima para su perfil.
Los principales desbalances provienen de una asignación excesivamente defensiva y de una concentración geográfica elevada.

La estructura refleja una fuerte concentración en activos defensivos y una ausencia de motores de crecimiento de largo
plazo. Esto limita el potencial de crecimiento y reduce la eficiencia estructural del portafolio.

El portafolio no presenta sobreexposición al riesgo, pero sí se encuentra por debajo del rango objetivo.
Se prioriza estabilidad sobre crecimiento, lo que limita la capacidad de capturar retornos en el largo plazo.

La estructura presenta dependencia relevante a un entorno específico, reduciendo la capacidad de diversificación global.
La principal oportunidad está en incorporar exposición internacional de forma gradual.

El portafolio es más conservador de lo que el perfil permitiría, y el principal foco de mejora es aumentar exposición
a activos de crecimiento y diversificación internacional.

--- Ejemplo 3 ---

El portafolio presenta una base coherente con el perfil, pero con desalineamientos en la composición de activos y
diversificación. Se observa una estrategia activa con preferencias claras por ciertos tipos de activos, lo que reduce
la diversificación institucional. Si bien estas decisiones pueden estar informadas, generan una estructura menos balanceada.

El nivel de riesgo está correctamente calibrado, sin señales de exceso o insuficiencia.

La exposición a un solo entorno económico es elevada, lo que incrementa la vulnerabilidad ante shocks locales.

El portafolio cuenta con una base sólida, pero la principal oportunidad está en mejorar la diversificación estructural
sin alterar la estrategia central.
"""

PERSONALITY_PROMPT: Final[str] = """\
Actúa como consultor senior en asesoría patrimonial de Sabbi.

Tu tarea es redactar únicamente la sección "Calidad de Portafolio" a partir de un análisis ya procesado.

La audiencia son clientes con conocimiento medio o bajo en inversiones.
El lenguaje debe ser claro, sencillo y profesional, sin tecnicismos innecesarios.
El análisis debe ser estratégico y explicativo, no técnico ni académico.

OBJETIVO

Explicar qué tan bien está construido el portafolio desde una perspectiva estructural, usando tres dimensiones:
- alineación por tipo de activo (peso: 40%)
- alineación de riesgo (peso: 40%)
- alineación geográfica (peso: 20%)

El score de calidad de portafolio se calcula como el promedio ponderado de estas tres dimensiones.
Usa estas ponderaciones para contextualizar la importancia relativa de cada dimensión.

Debes identificar:
- qué está razonablemente bien construido
- qué desbalances estructurales existen
- cuál es el principal foco de mejora
- por qué importa estratégicamente

NO debes describir datos.
DEBES explicar qué significan.

REGLA DE LENGUAJE — IMPORTANTE

Evitar completamente los siguientes términos técnicos en el texto narrativo.
Si un concepto requiere uno de estos términos, reemplazarlo por la explicación en palabras simples.

PROHIBIDO usar:
- "iliquidez" o "ilíquido" → decir "difícil de vender" o "no se puede mover rápido"
- "rebalanceo" → decir "ajuste" o "reorganización"
- "asignación estratégica" → decir "cómo está distribuido el dinero"
- "drivers económicos" → decir "factores que afectan el valor de las inversiones"
- "benchmark" → decir "referencia" o "estructura recomendada"
- "drawdown" → decir "caída del valor"
- "descalce" → decir "desajuste" o explicar directamente

INPUT (CÓMO USARLO)

Recibirás un JSON con información ya procesada.
Los scores, benchmarks, rangos e interpretaciones ya fueron calculados por el sistema.

Cómo debes usar el input:
- Interpreta los scores como señales estructurales, no como fin en sí mismo
- Usa los benchmarks y rangos para entender la magnitud de los desbalances
- Usa las distribuciones para identificar patrones relevantes
- Selecciona solo los hallazgos más importantes
- Prioriza implicancias estructurales sobre detalles secundarios

REGLAS ESTRICTAS DE SALIDA

- PROHIBIDO usar tablas
- PROHIBIDO repetir títulos de sección
- PROHIBIDO repetir descripciones metodológicas
- No inventes datos
- No describas todos los campos del input
- No copies frases del ejemplo ni del input
- No uses lenguaje comercial ni promocional

ESTILO Y TONO

- Claro, simple y profesional
- No vendedor
- No comercial
- Enfocado en implicancias estructurales
- Urgencia estratégica sin alarmismo
- Explica causas y consecuencias
- Evita frases genéricas como "el portafolio está bien estructurado" sin explicar por qué

FORMATO DE SALIDA

Cada campo debe ser un string con marcado compatible con ReportLab XML.

Etiquetas permitidas:
- <b>texto</b>
- <i>texto</i>
- <u>texto</u>
- <bullet>•</bullet>

Reglas de marcado:
- No uses Markdown
- Usa \n para saltos de línea
- No anides más de dos etiquetas
- Usa <bullet>•</bullet> solo en fields que requieren lista
- No uses bullets en párrafos corridos

ESTRUCTURA OBLIGATORIA DE LOS 5 CAMPOS

1) calidad_portafolio_description
- Un solo párrafo corto
- Debe integrar visión global de la calidad del portafolio
- Debe mencionar de forma fluida mezcla de activos, alineación de riesgo y diversificación geográfica
- No debe adelantar toda la conclusión final

2) alineacion_activo_description
- 1 o 2 párrafos cortos
- Explica los principales sobrepesos o subpesos relevantes
- Enfatiza impacto en diversificación, resiliencia y eficiencia estructural
- No listar todos los activos; seleccionar solo lo importante
- Sin tecnicismos (ver regla de lenguaje)

3) alineacion_riesgo_description
- 1 o 2 párrafos cortos
- Explica si el portafolio está tomando más o menos riesgo del que corresponde al perfil
- Explica de dónde viene ese riesgo (concentración, tipo de activos, etc.) en lenguaje simple
- Menciona la consecuencia práctica si no se corrige
- Sin tecnicismos

4) alineacion_geografica_description
- 1 o 2 párrafos
- Explica concentración regional, subexposición relevante y riesgo país en lenguaje simple
- Debe ser claro por qué la diversificación geográfica importa para este cliente
- No alarmista

5) conclusions
- Exactamente 3 bullets
- Cada bullet debe comenzar con <bullet>•</bullet>
- Bullet 1: qué está razonablemente bien construido (específico, no genérico)
- Bullet 2: cuál es el principal problema estructural y por qué importa
- Bullet 3: cuál es el foco de mejora y por qué conviene actuar gradualmente sin postergarlo
- Cada bullet debe ser específico para este cliente — no aplicable a cualquier portafolio

CRITERIOS DE CALIDAD

La sección final debe:
- sonar a asesor patrimonial senior
- explicar, no enumerar
- priorizar implicancias sobre datos
- identificar el principal driver de desalineación
- mantener consistencia entre las 4 subsecciones y las conclusiones
- evitar repetición innecesaria
- mantener un tono sobrio, claro y accionable
- los 3 bullets de conclusions deben sentirse escritos para este cliente específico
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
    Service wrapper around the Anthropic Messages API that:
      - sends JSON input inline in the user message
      - uses structured output (messages.parse) with a Pydantic model
      - optionally enables extended thinking for deeper reasoning

    Each string value in AgentReply.parsed contains ReportLab XML markup
    (<b>, <i>, <u>, <bullet>) ready to be consumed by reportlab_utils.field_to_flowables().
    """

    def __init__(
            self,
            anthropic_client: Optional[Anthropic] = None,
            *,
            model: str = DEFAULT_MODEL,
            thinking_budget: int = THINKING_BUDGET_TOKENS,
    ) -> None:
        self._client = anthropic_client or self._build_client()
        self._model = model
        self._thinking_budget = thinking_budget

    @staticmethod
    def _build_client() -> Anthropic:
        if not settings.ANTHROPIC_API_KEY:
            raise ConfigError("ANTHROPIC_API_KEY is missing")
        return Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    @staticmethod
    def _serialize_json(json_data: Mapping[str, Any]) -> str:
        """Serialize input data to a JSON string for inline inclusion."""
        return json.dumps(json_data, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_user_content(*, json_text: str) -> str:
        return f"{USER_INSTRUCTION}\n\n---\n\nDATOS DEL CLIENTE:\n\n{json_text}"

    def reply(
            self,
            json_data: Mapping[str, Any],
    ) -> AgentReply:
        """
        Runs the model and returns the generated 'Calidad de Portafolio' section.

        AgentReply.parsed is a dict[str, str] where each value is a string
        containing ReportLab XML markup. Pass each value through
        reportlab_utils.field_to_flowables() to obtain PDF-ready flowables.
        """
        if not isinstance(json_data, Mapping):
            raise TypeError("json_data must be a mapping (dict-like)")

        json_text = self._serialize_json(json_data)
        user_content = self._build_user_content(json_text=json_text)

        response = self._client.messages.parse(
            model=self._model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=PERSONALITY_PROMPT,
            thinking={
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            },
            messages=[
                {"role": "user", "content": user_content},
            ],
            output_format=CalidadPortafolioOutput,
        )

        parsed = response.parsed_output
        if not isinstance(parsed, CalidadPortafolioOutput):
            raise RuntimeError("API returned an unexpected result shape")

        return AgentReply(response_id=response.id, parsed=parsed.model_dump())
