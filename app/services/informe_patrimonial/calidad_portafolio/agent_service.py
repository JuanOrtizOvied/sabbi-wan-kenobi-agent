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

USER_INSTRUCTION: Final[str] = """\
A continuación se adjunta el archivo JSON con el análisis ya procesado de la sección "Calidad de Portafolio" de un cliente.

Tu tarea NO es recalcular métricas ni rehacer el análisis.
Tu tarea es interpretar el input y redactar únicamente la sección "Calidad de Portafolio".

Debes completar exactamente los siguientes campos del output:
1. calidad_portafolio_description
2. alineacion_activo_title
3. alineacion_activo_description
4. alineacion_riesgo_title
5. alineacion_riesgo_description
6. alineacion_geografica_title
7. alineacion_geografica_description
8. conclusions

NOTA SOBRE EL SCORE DE CALIDAD DE PORTAFOLIO

El score de calidad de portafolio se calcula utilizando tres dimensiones con las siguientes ponderaciones:
- Alineación por tipo de activo: 40%
- Alineación de riesgo: 40%
- Alineación geográfica: 20%

Usa estas ponderaciones para contextualizar la importancia relativa de cada dimensión al interpretar el score global.

TÍTULOS FIJOS POR SCORE — REGLA IMPORTANTE

Los campos alineacion_activo_title y alineacion_geografica_title deben usar
EXACTAMENTE los títulos fijos definidos abajo según el score correspondiente.
No los modifiques ni los reescribas.

El campo alineacion_riesgo_title debe generarse según la instrucción específica
para ese campo (ver sección ESTRUCTURA OBLIGATORIA).

TÍTULOS FIJOS — Alineación por tipo de activo:
  Score 1–2.99  → "Estructura crítica y altamente desbalanceada, con impacto en la resiliencia"
  Score 3–4.99  → "Estructura desalineada, con riesgos relevantes en la distribución de activos"
  Score 5–6.99  → "Estructura funcional, pero con desbalances que reducen eficiencia y diversificación"
  Score 7–8.99  → "Estructura sólida y bien balanceada, con desbalances menores no críticos"
  Score 9–10    → "Estructura altamente optimizada, con asignación eficiente entre clases de activo"

TÍTULOS FIJOS — Alineación geográfica:
  Score 1–2.99  → "Dependencia extrema de un solo país o región"
  Score 3–4.99  → "Alta concentración geográfica"
  Score 5–6.99  → "Alta concentración geográfica"
  Score 7–8.99  → "Concentración relevante en una región"
  Score 9–10    → "Diversificación geográfica adecuada"

TEXTOS FIJOS — Alineación de riesgo (para el campo alineacion_riesgo_title):
  Score 10      → "Riesgo ideal: calibrado exactamente para tu perfil"
  Score 9       → "Riesgo bien calibrado y coherente con tus objetivos"
  Score 8       → "Riesgo en rango, con margen de mejora en sus fuentes"
  Score 7       → "Riesgo controlado, aunque con fuentes concentradas"
  Score 6       → "Riesgo ligeramente fuera de rango: hay ajuste posible"
  Score 5       → "Riesgo con desajuste claro respecto a tu perfil"
  Score 4       → "El riesgo no encaja con lo que declaraste tolerar"
  Score 3       → "El portafolio asume mucho más (o menos) riesgo del adecuado"
  Score 2       → "El riesgo está fuera de control para tu perfil"
  Score 1       → "El riesgo supera ampliamente tu tolerancia declarada"

  Regla: usar el score entero (floor) de alineacion_riesgo.score para seleccionar el texto.
  Copiar el texto exactamente como aparece arriba — no modificar.

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
- Los campos de título (*_title) deben ser texto plano sin etiquetas de marcado

ESTRUCTURA OBLIGATORIA DE LOS 8 CAMPOS

1) calidad_portafolio_description
- Un solo párrafo corto
- Debe integrar visión global de la calidad del portafolio
- Debe mencionar de forma fluida mezcla de activos, alineación de riesgo y diversificación geográfica
- No debe adelantar toda la conclusión final

2) alineacion_activo_title
- USAR EXACTAMENTE el título fijo definido en USER_INSTRUCTION según el score de alineacion_activos
- No modificar ni reescribir

3) alineacion_activo_description
- 1 o 2 párrafos cortos
- Explica los principales sobrepesos o subpesos relevantes
- Enfatiza impacto en diversificación, resiliencia y eficiencia estructural
- No listar todos los activos; seleccionar solo lo importante
- Sin tecnicismos (ver regla de lenguaje)

4) alineacion_riesgo_title
- USAR EXACTAMENTE el texto fijo definido en USER_INSTRUCTION según el score de alineacion_riesgo
  (usar floor del score para seleccionar el texto)
- No modificar ni reescribir

5) alineacion_riesgo_description
- 1 o 2 párrafos cortos
- Explica si el portafolio está tomando más o menos riesgo del que corresponde al perfil
- Explica de dónde viene ese riesgo (concentración, tipo de activos, etc.) en lenguaje simple
- Menciona la consecuencia práctica si no se corrige
- Sin tecnicismos

6) alineacion_geografica_title
- USAR EXACTAMENTE el título fijo definido en USER_INSTRUCTION según el score de alineacion_geografica
- No modificar ni reescribir

7) alineacion_geografica_description
- 1 o 2 párrafos
- Explica concentración regional, subexposición relevante y riesgo país en lenguaje simple
- Debe ser claro por qué la diversificación geográfica importa para este cliente
- No alarmista

8) conclusions
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
    alineacion_activo_title: str
    alineacion_riesgo_description: str
    alineacion_riesgo_title: str
    alineacion_geografica_description: str
    alineacion_geografica_title: str
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

    Each string value in AgentReply.parsed contains ReportLab XML markup
    (<b>, <i>, <u>, <bullet>) ready to be consumed by reportlab_utils.field_to_flowables().
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

        AgentReply.parsed is a dict[str, str] where each value is a string
        containing ReportLab XML markup. Pass each value through
        reportlab_utils.field_to_flowables() to obtain PDF-ready flowables.

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
