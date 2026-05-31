from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional

import anthropic
from pydantic import BaseModel

from app.core.config import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "claude-opus-4-6"
MAX_TOKENS: Final[int] = 16_384

PERSONALITY_PROMPT: Final[str] = """\
Actúa como consultor senior en asesoría patrimonial de Sabbi.

Tu tarea es analizar las respuestas de un cliente al Typeform de perfilamiento
y redactar la sección "Lo que Sabbi detectó" de su Radiografía Patrimonial.

La audiencia son clientes peruanos con conocimiento medio o bajo en inversiones,
con patrimonio típico entre USD 100k y USD 1M.
El lenguaje debe ser claro, cercano y profesional — como si un asesor senior
le hablara directamente al cliente en una reunión.

CONTEXTO DE USO

Esta sección aparece dentro de la Radiografía Patrimonial, un documento que:
- Se presenta al cliente durante una primera reunión con el equipo Sabbi
- Se entrega después por correo para que el cliente lo relea solo
- Precede la explicación del servicio de Asesoría Patrimonial Inicial

El objetivo de esta sección NO es vender ni alarmar.
Es nombrar 2 a 4 observaciones genuinamente útiles que el cliente
probablemente no se había planteado de forma explícita — y que justifican
una conversación más profunda sobre su patrimonio.

TONO

- Directo y específico para este cliente — nunca genérico
- Cercano pero profesional
- Sin alarmismo ni dramatismo
- Sin lenguaje comercial ni promocional
- Urgencia racional: hay conversaciones pendientes que tienen un costo concreto,
  pero se comunican sin catastrofismo

REGLA DE LENGUAJE — IMPORTANTE

Evitar completamente los siguientes términos en el texto narrativo.
Si un concepto requiere uno de estos términos, reemplazarlo por lenguaje simple.

PROHIBIDO usar:
- "iliquidez" o "ilíquido" → decir "dinero que no puedes mover fácilmente"
- "rebalanceo" → decir "ajuste" o "reorganización"
- "benchmark" → no mencionarlo
- "asignación estratégica" → decir "cómo está distribuido el dinero"
- "drivers económicos" → decir "factores que afectan el valor de las inversiones"
- "descalce" → decir "desajuste" o explicar directamente
- "drawdown" → decir "caída del valor"
- "diversificación efectiva" → decir "que realmente protejan entre sí"
- "score" o "scoring" → no mencionarlo

REGLAS GENERALES

- No inventes datos que no estén en el input
- No copies frases del input literalmente
- No describas los datos — interprétarlos
- No menciones a Sabbi en tono promocional
- No recomiendes productos específicos
- No repitas información que ya aparece en otras páginas de la Radiografía
  (arquetipo, tolerancia al riesgo, objetivo, horizonte)
- Cada observación debe sentirse escrita para este cliente,
  no aplicable a cualquier persona
"""

USER_INSTRUCTION: Final[str] = """\
A continuación se adjuntan las respuestas del cliente al Typeform de perfilamiento.

Tu tarea es generar la sección "Lo que Sabbi detectó" de la Radiografía Patrimonial.

Esta sección contiene entre 2 y 4 observaciones personalizadas.
Cada observación nombra algo concreto sobre la situación del cliente
que merece una conversación más profunda.

════════════════════════════════════════
PASO 1 — DETECTAR QUÉ APLICA
════════════════════════════════════════

Revisa las respuestas del cliente y evalúa cada regla de detección de la siguiente
biblioteca. Una regla se activa cuando su condición se cumple con los datos disponibles.
No fuerces reglas que no apliquen claramente.

──────────────────────────────────────
BIBLIOTECA DE REGLAS DE DETECCIÓN
──────────────────────────────────────

GRUPO A — BRECHA PATRIMONIO / EXPERIENCIA

A1 — Patrimonio alto con instrumentos básicos
  Condición: patrimonio disponible para invertir > USD 300k
              AND nivel de experiencia = "Básica" o "Novel"
  Insight: el tamaño del patrimonio y la sofisticación de los instrumentos
           que usa no van de la mano. Hay opciones que probablemente nunca
           ha explorado y que podrían ser más adecuadas para lo que busca.

A2 — Experiencia declarada alta pero instrumentos simples
  Condición: experiencia = "Experto" o "Amplia"
              AND solo marca 1 o 2 tipos de instrumento en el Typeform
  Insight: hay una brecha entre la experiencia que declara y la variedad
           de instrumentos que usa. La experiencia también puede tener
           sus propios puntos ciegos.

──────────────────────────────────────
GRUPO B — CONTRADICCIÓN OBJETIVO / SITUACIÓN

B1 — Quiere ingresos pasivos pero está en etapa de acumulación
  Condición: objetivo = "generar ingresos pasivos"
              AND no tiene fuente de ingresos recurrente adicional (responde "No")
              AND edad < 50
  Insight: hay una decisión importante que no se ha tomado conscientemente:
           ¿priorizo crecer el capital ahora para tener más flujo después,
           o busco flujo inmediato aunque sea menor?
           Esa respuesta cambia completamente cómo debería invertir hoy.

B2 — Necesita flujo mensual concreto pero sus instrumentos no lo generan
  Condición: declara monto de ingresos mensuales necesarios (campo > 0)
              AND los únicos instrumentos que marca son Fondos Mutuos,
              Bonos o Depósitos — ninguno orientado a distribución de flujo
  Insight: el monto que necesita mensualmente es alcanzable con su patrimonio,
           pero requiere que sus inversiones estén estructuradas específicamente
           para generarlo. No todos los instrumentos lo hacen.

B3 — Objetivo desactualizado respecto a su situación
  Condición: situacion laboral = "Jubilado o retirado"
              AND objetivo = "planificar tus ahorros para jubilarte"
  Insight: el objetivo declarado no refleja la etapa en que realmente está.
           Si ya se retiró, la estrategia cambia completamente:
           ya no se trata de acumular, sino de proteger y distribuir lo que tiene.

──────────────────────────────────────
GRUPO C — CONCENTRACIÓN / VISIBILIDAD

C1 — Preferencia por evitar Perú pero instrumentos mayormente locales
  Condición: responde "Prefiero evitarlo" a la pregunta sobre invertir en Perú
              AND los instrumentos que marca son principalmente Fondos Mutuos
              locales, Bonos locales o Compra de inmuebles
  Insight: hay una contradicción entre la preferencia declarada y
           el tipo de instrumentos que usa. La pregunta pendiente es:
           ¿su portafolio actual refleja esa preferencia o es algo
           que todavía no ha podido implementar?
  Nota: no afirmar que está concentrado en Perú — solo señalar la pregunta.

C2 — Visibilidad limitada por pocos tipos de activo
  Condición: marca 1 o 2 tipos de instrumento AND patrimonio > USD 200k
  Insight: con el nivel de patrimonio que tiene, la pregunta no es si tiene
           diversificación en papel — es si esa diversificación funciona
           en la práctica. Con pocos tipos de activo declarados, no es
           posible saberlo sin ver el detalle completo.
  Nota importante: NO asumir que está mal diversificado. Puede estar muy
           bien diversificado dentro de cada categoría. El insight es
           sobre visibilidad, no sobre un problema confirmado.

──────────────────────────────────────
GRUPO D — COMPORTAMIENTO / DELEGACIÓN

D1 — Mucho tiempo gestionando sin estrategia unificada
  Condición: dedica "Más de 4 horas" al mes a gestionar inversiones
              AND lo que más valora = "tener una estrategia clara para
              todo tu patrimonio, no solo un producto puntual"
  Insight: dedica tiempo real a sus inversiones pero lo que más valora
           es tener una estrategia que lo organice todo. Eso señala algo:
           hay actividad, pero quizás sin un norte claro. Mucha gestión
           sin estrategia puede dar sensación de control sin los
           resultados esperados.

D2 — Ha delegado antes con mala experiencia
  Condición: responde "Sí, pero la experiencia no fue buena y ahora tengo dudas"
  Insight: ya intentó delegar la gestión de su dinero y no terminó bien.
           La pregunta que vale la pena hacerse es qué falló exactamente:
           ¿el producto, la persona, la falta de comunicación,
           o la ausencia de un plan claro desde el inicio?

D3 — Quiere delegar completamente pero nunca lo ha hecho con éxito
  Condición: nivel de comodidad para delegar = "Muy cómodo. Prefiero
              delegar completamente a un equipo profesional"
              AND (nunca ha delegado OR ha delegado con mala experiencia)
  Insight: la disposición a delegar ya está. Lo que probablemente
           no ha encontrado todavía es a quién hacerlo con confianza real.

──────────────────────────────────────
GRUPO E — RIESGO / COMPORTAMIENTO

E1 — Intolerancia total a caídas con horizonte largo
  Condición: responde "Cualquier caída me haría sentir incómodo"
              AND horizonte de inversión = "Largo Plazo" o > 5 años
  Insight: en horizontes largos las caídas temporales son inevitables.
           El reto no es evitarlas — es construir un portafolio que
           pueda sostenerse emocionalmente cuando ocurran, sin tomar
           decisiones que dañen el patrimonio en el peor momento.

E2 — Contradicción entre tolerancia declarada y cartera elegida
  Condición: se autocalifica con tolerancia "alta" o "media"
              AND la distribución de cartera que elige tiene más del
              60% en activos "bajo" (conservador)
  Insight: hay una pequeña contradicción entre cómo se define como
           inversionista y la cartera que elegiría en la práctica.
           Puede significar que la tolerancia real es más baja de lo
           que cree, o que no está del todo familiarizado con lo que
           implica cada nivel de riesgo.

E3 — Deuda significativa mientras busca invertir
  Condición: deudas declaradas > USD 50,000
              AND tiene capital disponible para invertir
  Insight: hay una decisión pendiente sobre el orden de prioridades:
           ¿el retorno esperado de las inversiones supera el costo
           de la deuda que mantiene? Si no, cada sol invertido
           puede estar trabajando menos de lo que parece.

════════════════════════════════════════
PASO 2 — SELECCIONAR Y PRIORIZAR
════════════════════════════════════════

De todas las reglas que se activaron, selecciona EXACTAMENTE 2 observaciones
siguiendo este orden de prioridad:

  1. Primero: reglas de comportamiento (Grupo D)
  2. Segundo: reglas de contradicción objetivo/situación (Grupo B)
  3. Tercero: reglas de brecha patrimonio/experiencia (Grupo A)
  4. Cuarto: reglas de concentración/visibilidad (Grupo C)
  5. Último: reglas de riesgo/comportamiento (Grupo E)

Reglas adicionales de selección:
- Selecciona siempre las 2 con mayor tensión genuina para este cliente
- No incluyas dos observaciones del mismo grupo
- Si solo se activa 1 regla claramente, redacta 2 observaciones:
  la regla activada + la más relevante del Grupo E o C
- Si ninguna regla se activa con claridad, devuelve
  observaciones_count = 0 y observaciones = [] sin inventar insights

════════════════════════════════════════
FORMATO DE SALIDA
════════════════════════════════════════

Devuelve ÚNICAMENTE un objeto JSON con esta estructura exacta.
Sin texto adicional fuera del JSON.

{
  "observaciones_count": 2,
  "observaciones": [
    {
      "grupo": "<letra del grupo: A, B, C, D o E>",
      "regla": "<código de regla: A1, B2, D1, etc.>",
      "titulo_observacion": "<string, texto plano sin etiquetas>",
      "descripcion_observacion": "<string, máximo 3 líneas, \\n para saltos>"
    },
    {
      "grupo": "<letra del grupo: A, B, C, D o E>",
      "regla": "<código de regla>",
      "titulo_observacion": "<string, texto plano sin etiquetas>",
      "descripcion_observacion": "<string, máximo 3 líneas, \\n para saltos>"
    }
  ]
}

════════════════════════════════════════
PASO 3 — REDACTAR
════════════════════════════════════════

Para cada observación seleccionada, redacta usando esta estructura:

titulo_observacion
  4 a 7 palabras. Directo y específico.
  Debe nombrar la tensión o pregunta central — no describirla genéricamente.
  Ejemplos de buenos títulos:
    ✓ "Tu patrimonio supera lo que tus instrumentos pueden manejar"
    ✓ "Quieres flujo, pero tu portafolio no está armado para darlo"
    ✓ "Mucha gestión, todavía sin estrategia clara"
    ✓ "Ya delegaste antes — y algo salió mal"
  Ejemplos de títulos prohibidos:
    ✗ "Observación sobre tu experiencia"
    ✗ "Consideración importante"
    ✗ "Punto a revisar"

descripcion_observacion
  MÁXIMO 3 líneas. Párrafo corrido, sin bullets.
  Debe:
  - Ir directo al punto — sin introducción ni contexto innecesario
  - Usar los datos concretos del cliente cuando estén disponibles
    (edad, monto declarado, objetivo, instrumentos, años invirtiendo)
  - Terminar con una pregunta corta o frase que abra la conversación
  No debe:
  - Explicar ni desarrollar — el asesor lo hace en voz alta en la reunión
  - Afirmar problemas que no se pueden confirmar con el Typeform
  - Sonar como diagnóstico definitivo ni como crítica
  - Usar lenguaje técnico prohibido
  - Repetir información ya visible en otras páginas de la Radiografía

  EJEMPLO DE TONO Y EXTENSIÓN CORRECTOS:
  "Tienes 33 años, sin ingresos recurrentes, y quieres vivir de tus inversiones.
  Con menos de $200k disponibles, el flujo que generarías hoy sería mínimo.
  ¿La prioridad es crecer primero o generar algo ahora?"

════════════════════════════════════════
FORMATO DE SALIDA
════════════════════════════════════════

Devuelve ÚNICAMENTE un objeto JSON con esta estructura exacta.
Sin texto adicional fuera del JSON.

{
  "observaciones_count": <número entero entre 0 y 4>,
  "observaciones": [
    {
      "grupo": "<letra del grupo: A, B, C, D o E>",
      "regla": "<código de regla: A1, B2, D1, etc.>",
      "titulo_observacion": "<string, texto plano sin etiquetas>",
      "descripcion_observacion": "<string, máximo 3 líneas, \\n para saltos>"
    }
  ]
}

Reglas de formato del texto:
- Usa <b>dato concreto</b> para destacar números, montos o datos
  específicos del cliente dentro de descripcion_observacion
- No uses Markdown
- Usa \\n para saltos de línea dentro de descripcion_observacion
- titulo_observacion es texto plano sin etiquetas

════════════════════════════════════════
CRITERIOS DE CALIDAD
════════════════════════════════════════

Antes de devolver el output, verifica:

✓ ¿Cada descripción tiene máximo 3 líneas?
✓ ¿Va directo al punto sin introducción innecesaria?
✓ ¿Termina con una pregunta o frase que abre conversación?
✓ ¿Alguna observación podría aplicarse a cualquier cliente sin cambiar
  una sola palabra? Si sí → reescribir con datos específicos o eliminar
✓ ¿Alguna descripción afirma un problema que el Typeform no confirma?
  Si sí → suavizar a pregunta o señal, no conclusión
✓ ¿Se usó alguna palabra de la lista prohibida? Si sí → reemplazar
✓ ¿El tono es de asesor que abre una conversación,
  no de sistema que emite un diagnóstico? Si no → reescribir
"""


# ── Pydantic output models ───────────────────────────────────────────


class Observacion(BaseModel):
    """A single profiling observation detected from Typeform responses."""
    grupo: str
    regla: str
    titulo_observacion: str
    descripcion_observacion: str


class ObservacionesOutput(BaseModel):
    """
    Top-level structured output returned by the profiling agent.

    Maps 1-to-1 with the JSON schema defined in USER_INSTRUCTION.
    Uses Pydantic so it can be passed directly to messages.parse()
    for guaranteed schema compliance via constrained decoding.
    """
    observaciones_count: int
    observaciones: list[Observacion]


# ── Reply wrapper ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProfilingAgentReply:
    observaciones: ObservacionesOutput
    raw: dict[str, Any]
    message_id: str


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


class AgentService:
    """
    Service that uses the Anthropic Messages API to:
      - receive Typeform profiling responses as inline JSON
      - leverage extended thinking for the internal rule-evaluation phase
      - return a schema-validated set of personalized observations

    Uses streaming (messages.stream) to handle long-running requests
    that exceed the 10-minute HTTP timeout. The output_format parameter
    enables constrained decoding, and the response is validated against
    the ObservacionesOutput Pydantic schema after streaming completes.
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

    # ── Construction helpers ──────────────────────────────────────

    @staticmethod
    def _build_client() -> anthropic.Anthropic:
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ConfigError("ANTHROPIC_API_KEY is missing")
        return anthropic.Anthropic(api_key=api_key)

    # ── Message building ─────────────────────────────────────────

    @staticmethod
    def _serialize_typeform(json_data: Mapping[str, Any]) -> str:
        """Serialize Typeform response data to a JSON string for inline embedding."""
        return json.dumps(json_data, ensure_ascii=False, indent=2)

    @classmethod
    def _build_user_content(cls, json_data: Mapping[str, Any]) -> str:
        """
        Build the full user message content.

        The Typeform JSON is embedded inline in the user message,
        separated by clear delimiters for the model to parse.
        """
        typeform_json = cls._serialize_typeform(json_data)
        return (
            f"{USER_INSTRUCTION}\n\n"
            f"--- INICIO RESPUESTAS TYPEFORM ---\n"
            f"{typeform_json}\n"
            f"--- FIN RESPUESTAS TYPEFORM ---"
        )

    # ── Main entry point ─────────────────────────────────────────

    def analyze(
            self,
            json_data: Mapping[str, Any],
    ) -> ProfilingAgentReply:
        """
        Run the Typeform profiling analysis via Anthropic's Messages API
        and return the structured observations.

        Uses extended thinking (adaptive) so the model can perform its
        internal rule-evaluation phase (PASO 1) before producing the output.

        Uses messages.stream() to avoid the 10-minute HTTP timeout on
        long-running requests. The output_format parameter still enables
        constrained decoding, and the JSON is validated against the
        ObservacionesOutput Pydantic schema after streaming completes.

        Returns a ProfilingAgentReply with:
          - observaciones: parsed ObservacionesOutput (Pydantic model)
          - raw: dict serialisation of the observations for logging/storage
          - message_id: Anthropic message ID for tracing
        """
        if not isinstance(json_data, Mapping):
            raise TypeError("json_data must be a mapping (dict-like)")

        user_content = self._build_user_content(json_data)

        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=PERSONALITY_PROMPT,
            thinking={"type": "adaptive", "effort": "high"},
            messages=[
                {"role": "user", "content": user_content},
            ],
            output_format=ObservacionesOutput,
        ) as stream:
            response = stream.get_final_message()

        # Extract JSON text from the response content blocks
        json_text = next(
            (block.text for block in response.content if block.type == "text"),
            None,
        )
        if not json_text:
            raise RuntimeError(
                "Anthropic response contained no text content block"
            )

        observaciones = ObservacionesOutput.model_validate_json(json_text)

        return ProfilingAgentReply(
            observaciones=observaciones,
            raw=observaciones.model_dump(),
            message_id=response.id,
        )
