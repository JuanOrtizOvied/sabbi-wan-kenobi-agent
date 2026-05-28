from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional

from anthropic import Anthropic
from pydantic import BaseModel

from app.core.config import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "claude-sonnet-4-6"
AGENT_NAME: Final[str] = "CalidadPortafolioAgent"
THINKING_BUDGET_TOKENS: Final[int] = 10_000
MAX_OUTPUT_TOKENS: Final[int] = 16_384

USER_INSTRUCTION: Final[str] = """\
A continuación se adjunta el archivo JSON con el análisis ya procesado de la sección 
"Calidad de Portafolio" de un cliente.

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

El score de calidad de portafolio se calcula utilizando tres dimensiones con las 
siguientes ponderaciones:
- Alineación por tipo de activo: 40%
- Alineación de riesgo: 40%
- Alineación geográfica: 20%

Usa estas ponderaciones para contextualizar la importancia relativa de cada dimensión 
al interpretar el score global.

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

  ESCALA DE PERFORMANCE — CÓMO LEERLA ANTES DE SELECCIONAR EL TÍTULO:

  La escala de performance va de 1 a 10, donde:
  - Score de performance ALTO (cercano a 10) = activos muy conservadores (cash, renta fija 
    corta) → el portafolio es MÁS CONSERVADOR de lo que el perfil permite.
  - Score de performance BAJO (cercano a 1) = activos muy agresivos (PE, VC, cripto) → el 
    portafolio es MÁS ARRIESGADO de lo que el perfil tolera.

  El rango objetivo del perfil define en qué zona de la escala debería estar el portafolio.
  Si el score de performance está POR ENCIMA del rango → portafolio demasiado conservador.
  Si el score de performance está POR DEBAJO del rango → portafolio demasiado agresivo.

  Los títulos fijos tienen DOS variantes por score según la dirección del desajuste.
  Usa el campo "distancia_al_rango" o compara score_performance vs rango_objetivo para 
  determinar la dirección ANTES de seleccionar el título.

  Score 10:
    - Performance sobre rango → "Portafolio muy conservador para tu perfil: capital sin trabajar"
    - Performance bajo rango  → [no aplica en score 10]
  Score 9:
    - Performance sobre rango → "El portafolio está siendo mucho más cauto de lo que tu perfil necesita"
    - Performance bajo rango  → "El riesgo supera ampliamente lo que declaraste tolerar"
  Score 8:
    - Performance sobre rango → "El portafolio está siendo más cauto de lo que tu perfil permite"
    - Performance bajo rango  → "El riesgo está por encima de lo que declaraste tolerar"
  Score 7:
    - Performance sobre rango → "Portafolio conservador: dentro del rango, pero con margen para más"
    - Performance bajo rango  → "Riesgo ligeramente elevado respecto a tu perfil"
  Score 6:
    - Performance sobre rango → "Portafolio más conservador de lo óptimo: hay ajuste posible"
    - Performance bajo rango  → "Riesgo ligeramente fuera de rango: hay ajuste posible"
  Score 5:
    - Performance sobre rango → "El portafolio asume menos riesgo del que tu perfil podría aprovechar"
    - Performance bajo rango  → "Riesgo con desajuste claro respecto a tu perfil"
  Score 4:
    - Performance sobre rango → "El portafolio es demasiado conservador para tu perfil declarado"
    - Performance bajo rango  → "El riesgo no encaja con lo que declaraste tolerar"
  Score 3:
    - Performance sobre rango → "El portafolio está muy por debajo del riesgo que tu perfil permite"
    - Performance bajo rango  → "El portafolio asume mucho más riesgo del adecuado"
  Score 2:
    - Performance sobre rango → "Exceso de cautela: el capital no está trabajando para tu perfil"
    - Performance bajo rango  → "El riesgo está fuera de control para tu perfil"
  Score 1:
    - Performance sobre rango → "Portafolio casi inactivo: la estructura no está diseñada para crecer"
    - Performance bajo rango  → "El riesgo supera ampliamente tu tolerancia declarada"

  Regla: usar el score entero (floor) de alineacion_riesgo.score para seleccionar el texto.
  Determinar la dirección comparando score_performance con rango_objetivo antes de seleccionar.
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
- alineacion_activos → úsalo para identificar sobrepesos, subpesos y desbalances estructurales 
  por tipo de activo
- alineacion_riesgo → úsalo para evaluar si el portafolio está más conservador o más agresivo 
  que el perfil. SIEMPRE verificar la dirección del desajuste comparando score_performance 
  con rango_objetivo antes de redactar.
- alineacion_geografica → úsalo para identificar concentración regional, subexposición 
  internacional y riesgo país
- benchmark_sabbi y rangos_permitidos → úsalos como referencia para entender si una 
  desviación es leve, moderada o relevante
- distribucion_cliente → úsala para identificar los principales patrones estructurales, no para 
  describir todos los datos uno por uno
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

El portafolio presenta una base funcional, mostrando una construcción patrimonial ordenada 
y consistente.

No obstante, se identifican desbalances estructurales, principalmente en la diversificación 
geográfica y en la concentración de ciertos factores que afectan el valor de las inversiones, 
que limitan la diversificación efectiva.

El portafolio presenta una estructura razonablemente alineada, con desbalances identificables 
pero corregibles.

Se observa una inclinación hacia mayor liquidez y menor exposición a activos de crecimiento 
estructural.

El nivel de riesgo del portafolio se encuentra ligeramente por encima del rango objetivo, lo 
que implica una estructura más conservadora de lo que permitiría el perfil.

Predomina una exposición a activos defensivos y una menor participación en motores de 
crecimiento estructural.

Existe una sobreexposición significativa a un solo entorno económico, junto con una 
subexposición a mercados desarrollados. Esta concentración incrementa la dependencia 
del portafolio y limita su diversificación efectiva.

El principal foco de mejora se encuentra en la diversificación estructural, especialmente a 
nivel geográfico.

--- Ejemplo 2 ---

El portafolio presenta una base funcional, pero no está alineado con una estructura óptima 
para su perfil. Los principales desbalances provienen de una asignación excesivamente 
defensiva y de una concentración geográfica elevada.

La estructura refleja una fuerte concentración en activos defensivos y una ausencia de 
motores de crecimiento de largo plazo. Esto limita el potencial de crecimiento y reduce la 
eficiencia estructural del portafolio.

El portafolio no presenta sobreexposición al riesgo, pero sí se encuentra por debajo del 
rango objetivo. Se prioriza estabilidad sobre crecimiento, lo que limita la capacidad de 
capturar retornos en el largo plazo.

La estructura presenta dependencia relevante a un entorno específico, reduciendo la 
capacidad de diversificación global. La principal oportunidad está en incorporar exposición 
internacional de forma gradual.

El portafolio es más conservador de lo que el perfil permitiría, y el principal foco de mejora 
es aumentar exposición a activos de crecimiento y diversificación internacional.

--- Ejemplo 3 ---

El portafolio presenta una base coherente con el perfil, pero con desalineamientos en la 
composición de activos y diversificación. Se observa una estrategia activa con preferencias 
claras por ciertos tipos de activos, lo que reduce la diversificación institucional. Si bien estas 
decisiones pueden estar informadas, generan una estructura menos balanceada.

El nivel de riesgo está correctamente calibrado, sin señales de exceso o insuficiencia.

La exposición a un solo entorno económico es elevada, lo que incrementa la vulnerabilidad 
ante shocks locales.

El portafolio cuenta con una base sólida, pero la principal oportunidad está en mejorar la 
diversificación estructural sin alterar la estrategia central.

"""

PERSONALITY_PROMPT: Final[str] = """\
Actúa como consultor senior en asesoría patrimonial de Sabbi.

Tu tarea es redactar únicamente la sección "Calidad de Portafolio" a partir de un análisis 
ya procesado.

La audiencia son clientes con conocimiento medio o bajo en inversiones.
El lenguaje debe ser claro, sencillo y profesional, sin tecnicismos innecesarios.
El análisis debe ser estratégico y explicativo, no técnico ni académico.

OBJETIVO

Explicar qué tan bien está construido el portafolio desde una perspectiva estructural, 
usando tres dimensiones:
- alineación por tipo de activo (peso: 40%)
- alineación de riesgo (peso: 40%)
- alineación geográfica (peso: 20%)

El score de calidad de portafolio se calcula como el promedio ponderado de estas tres 
dimensiones. Usa estas ponderaciones para contextualizar la importancia relativa de 
cada dimensión.

Debes identificar:
- qué está razonablemente bien construido
- qué desbalances estructurales existen
- cuál es el principal foco de mejora
- por qué importa estratégicamente

NO debes describir datos.
DEBES explicar qué significan.

REGLA CRÍTICA — DIRECCIÓN DEL DESAJUSTE DE RIESGO

Antes de redactar cualquier texto sobre alineación de riesgo, debes determinar 
obligatoriamente la dirección del desajuste. Este paso no es opcional.

La escala de performance funciona de forma INVERSA a lo que intuitivamente parece:
- Score de performance ALTO (cercano a 10) → activos conservadores (cash, renta fija 
  corta plazo). Si está POR ENCIMA del rango objetivo → el portafolio es DEMASIADO 
  CONSERVADOR para el perfil. El problema es exceso de cautela.
- Score de performance BAJO (cercano a 1) → activos agresivos (PE, VC, cripto). Si está 
  POR DEBAJO del rango objetivo → el portafolio es DEMASIADO AGRESIVO para el 
  perfil. El problema es exceso de riesgo.

PASO OBLIGATORIO antes de redactar alineacion_riesgo_description:
  1. Leer score_performance del input
  2. Leer rango_objetivo del perfil
  3. Determinar: ¿score_performance está por encima o por debajo del rango?
  4. Si está POR ENCIMA → redactar en clave de "portafolio demasiado conservador"
  5. Si está POR DEBAJO → redactar en clave de "portafolio demasiado agresivo"
  6. Si está DENTRO del rango → redactar en clave de "riesgo bien calibrado"

EJEMPLOS DE DIRECCIÓN CORRECTA:

  Caso A — Performance 8.5, rango objetivo 4–5 (perfil Moderado & Arriesgado):
    Score performance (8.5) > rango superior (5) → portafolio DEMASIADO CONSERVADOR
    CORRECTO: "El portafolio está siendo mucho más cauto de lo que tu perfil necesita. 
    Casi todo el capital está en efectivo e instrumentos de muy bajo riesgo, cuando tu 
    perfil te permitiría —y necesita— activos con mayor potencial de crecimiento."
    INCORRECTO: "El portafolio está asumiendo un nivel de riesgo que no calza con lo 
    que declaraste tolerar." ← esta frase implica exceso de riesgo, dirección opuesta.

  Caso B — Performance 2.5, rango objetivo 5–6 (perfil Moderado):
    Score performance (2.5) < rango inferior (5) → portafolio DEMASIADO AGRESIVO
    CORRECTO: "El portafolio está tomando más riesgo del que tu perfil declara tolerar. 
    La concentración en activos de alta volatilidad supera lo que corresponde a un 
    inversionista Moderado."
    INCORRECTO: "El portafolio es más conservador de lo que el perfil permitiría."

  Caso C — Performance 5.5, rango objetivo 5–6 (perfil Moderado):
    Score performance (5.5) dentro del rango → riesgo BIEN CALIBRADO
    CORRECTO: "El nivel de riesgo del portafolio está bien alineado con tu perfil, 
    reflejando una combinación equilibrada entre activos defensivos y de crecimiento."

PROHIBIDO en cualquier caso:
- Usar lenguaje que implique exceso de riesgo cuando el desajuste es por exceso de 
  cautela, y viceversa.
- Redactar sobre alineación de riesgo sin haber determinado la dirección primero.

REGLA DE LENGUAJE — IMPORTANTE

Evitar completamente los siguientes términos técnicos en el texto narrativo.
Si un concepto requiere uno de estos términos, reemplazarlo por la explicación en 
palabras simples.

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
- Para alineación de riesgo: SIEMPRE verificar dirección del desajuste antes de redactar

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
- Debe mencionar de forma fluida mezcla de activos, alineación de riesgo y diversificación 
  geográfica
- No debe adelantar toda la conclusión final

2) alineacion_activo_title
- USAR EXACTAMENTE el título fijo definido en USER_INSTRUCTION según el score de 
  alineacion_activos
- No modificar ni reescribir

3) alineacion_activo_description
- 1 o 2 párrafos cortos
- Explica los principales sobrepesos o subpesos relevantes
- Enfatiza impacto en diversificación, resiliencia y eficiencia estructural
- No listar todos los activos; seleccionar solo lo importante
- Sin tecnicismos (ver regla de lenguaje)

4) alineacion_riesgo_title
- USAR EXACTAMENTE el texto fijo definido en USER_INSTRUCTION según el score de 
  alineacion_riesgo Y la dirección del desajuste (sobre o bajo rango)
- Determinar dirección ANTES de seleccionar
- No modificar ni reescribir

5) alineacion_riesgo_description
- 1 o 2 párrafos cortos
- PASO 1 OBLIGATORIO: determinar dirección del desajuste (score_performance vs 
  rango_objetivo) antes de escribir una sola frase
- Si score_performance > rango_superior → explicar que el portafolio es demasiado 
  conservador: qué tipo de activos generan esa cautela excesiva y qué oportunidad se está 
  perdiendo
- Si score_performance < rango_inferior → explicar que el portafolio asume más riesgo del 
  adecuado: de dónde viene ese riesgo y cuál es la consecuencia práctica
- Si score_performance está dentro del rango → explicar el buen calibrado y de dónde vienen 
  las fuentes de riesgo
- Mencionar la consecuencia práctica si no se corrige el desajuste
- Sin tecnicismos

6) alineacion_geografica_title
- USAR EXACTAMENTE el título fijo definido en USER_INSTRUCTION según el score de 
  alineacion_geografica
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
- Bullet 3: cuál es el foco de mejora y por qué conviene actuar gradualmente sin 
  postergarlo
- Cada bullet debe ser específico para este cliente — no aplicable a cualquier portafolio
- En Bullet 2: si el problema de riesgo es exceso de cautela, formularlo como oportunidad 
  perdida, no como riesgo de pérdida

CRITERIOS DE CALIDAD

La sección final debe:
- sonar a asesor patrimonial senior
- explicar, no enumerar
- priorizar implicancias sobre datos
- identificar el principal driver de desalineación
- mantener consistencia entre las 4 subsecciones y las conclusiones
- la dirección del desajuste de riesgo debe ser consistente en título, descripción y 
  conclusiones — nunca contradictoria
- evitar repetición innecesaria
- mantener un tono sobrio, claro y accionable
- los 3 bullets de conclusions deben sentirse escritos para este cliente específico
"""


class CalidadPortafolioOutput(BaseModel):
    calidad_portafolio_description: str
    alineacion_activo_title: str
    alineacion_activo_description: str
    alineacion_riesgo_title: str
    alineacion_riesgo_description: str
    alineacion_geografica_title: str
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
    