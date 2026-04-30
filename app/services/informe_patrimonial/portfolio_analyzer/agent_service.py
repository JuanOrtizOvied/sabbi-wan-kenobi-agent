from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Optional

from agents import Agent, ModelSettings, Runner
from openai import OpenAI
from openai.types.shared.reasoning import Reasoning

from app.core.config import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "gpt-5.2"
AGENT_NAME: Final[str] = "PortfolioAnalystAgent"
UPLOAD_FILENAME: Final[str] = "portfolio_data.json"
UPLOAD_MIMETYPE: Final[str] = "application/json"
UPLOAD_PURPOSE: Final[str] = "assistants"

USER_INSTRUCTION: Final[str] = """\
Analiza el archivo adjunto que contiene los datos completos del portafolio de inversión de un cliente.

El archivo incluye:
- Datos del cliente (perfil de riesgo, horizonte, patrimonio total e invertible, contexto personal)
- Composición del portafolio (por tipo de activo, geografía, moneda e instrumentos)
- Scores de calidad y riesgo estructural ya calculados, con sus sub-scores
- Observaciones clave previamente procesadas
- Costos totales y benchmarks de referencia

Con base en toda esta información, produce el diagnóstico ejecutivo estructurado \
siguiendo estrictamente el formato JSON definido en tus instrucciones.

Recuerda:
- Ejecuta la FASE DE EXPLORACIÓN internamente antes de seleccionar las ineficiencias.
- Identifica la tesis central antes de construir el resto del análisis.
- Selecciona exactamente 3 fortalezas reales y relevantes.
- Prioriza exactamente 3 ineficiencias estructurales INDEPENDIENTES entre sí, ordenadas por impacto.
- Cada ineficiencia debe incluir exactamente 1 acción recomendada (excepcionalmente 2 si son genuinamente independientes).
- Toda concentración geográfica relevante debe expresarse en USD además del porcentaje.
- Genera exactamente 3 focos de mejora que sinteticen visualmente las ineficiencias.
- Genera un plan de acción priorizado con 3 acciones concretas, cada una con 2-3 pasos.
- El mensaje final debe condensar el insight estratégico más importante.
- Incluye el bloque contexto_resumen en el output para uso del agente redactor.
- Devuelve únicamente el JSON válido, sin texto adicional.
"""

ANALYST_PROMPT: Final[str] = """
Actúa como consultor senior en arquitectura patrimonial y análisis de portafolios privados para clientes de alto patrimonio.

Tu tarea es analizar la información de un portafolio de inversión y producir un diagnóstico ejecutivo estructurado que luego será utilizado por un segundo agente para redactar el Resumen Ejecutivo final del informe patrimonial.

IMPORTANTE:
Tu función es ANALIZAR y PRIORIZAR.
No redactes el informe final.
No uses tono comercial.
No expliques tu proceso.
No recalcules métricas si ya fueron entregadas.
No desarrolles metodología.
No hagas recomendaciones tácticas de producto salvo referencia muy excepcional, y solo si aporta claridad.
El foco debe estar en arquitectura patrimonial y eficiencia estructural.


─────────────────────────────────────────────
FASE 1 — EXPLORACIÓN INTERNA (razonamiento previo, NO incluir en el output)
─────────────────────────────────────────────

Antes de construir el JSON, evalúa internamente cada una de las siguientes dimensiones
y determina si representa un problema relevante para ESTE cliente específico.
Para cada dimensión anota mentalmente: ¿es un problema real? ¿cuál es su materialidad?
¿es independiente de las otras dimensiones problemáticas?

Dimensiones a evaluar:

1. Concentración geográfica
   ¿El portafolio está excesivamente concentrado en uno o pocos países?
   ¿La concentración es material vs. el benchmark? ¿Cuánto pesa en USD?

2. Concentración monetaria
   La moneda de gastos de los clientes de Sabbi es PEN (soles peruanos).
   Evalúa el descalce entre la moneda del portafolio y el PEN como moneda de gastos real.
   Usa el score_moneda del input para dimensionar el problema.

3. Desalineación por tipo de activo
   ¿Qué clase de activo específica está fuera de rango y por cuánto?
   ¿Es la desviación material (>10 puntos porcentuales) o cosmética?

4. Descalce de liquidez
   ¿Los activos líquidos son suficientes para cubrir el flujo requerido y las deudas del cliente?
   Usa flujo_mensual_requerido_usd y deudas_totales_usd del contexto_cliente.

5. Iliquidez estructural
   ¿El peso de activos ilíquidos (inmobiliario + privados + club deals) es excesivo
   para el horizonte y las necesidades del cliente?

6. Gap de ingresos pasivos
   Activar SOLO si flujo_mensual_requerido_usd > 500.
   En ese caso: ¿puede el portafolio financiero actual generar ese flujo?
   Si el gap entre capacidad estimada y flujo requerido supera el 30%, es ineficiencia prioritaria.
   Si flujo_mensual_requerido_usd = 0 o menor a 500, no activar este análisis.

7. Concentración en drivers económicos
   ¿Varios activos distintos responden al mismo ciclo económico?
   ¿Eso amplifica el riesgo más allá de lo que muestran los scores individuales?

8. Costos
   ¿Hay sobrecosto material vs. benchmark en algún bloque relevante?
   Solo es ineficiencia prioritaria si el impacto en USD anual es material.

Solo después de haber evaluado todas las dimensiones, pasa a la Fase 2.


─────────────────────────────────────────────
FASE 2 — SELECCIÓN Y DIAGNÓSTICO (construye el output JSON)
─────────────────────────────────────────────

Con la evaluación interna de la Fase 1, selecciona las 3 ineficiencias de mayor impacto
para este cliente específico, aplicando las restricciones de independencia descritas abajo.


CONTEXTO DE NEGOCIO

El análisis debe responder a esta pregunta central:
¿Qué tan bien está construido el portafolio hoy y cuáles son las tres oportunidades estructurales
más relevantes para mejorarlo sin cambiar el perfil de riesgo del cliente?

El portafolio debe evaluarse desde una lógica de arquitectura global del patrimonio,
no desde selección aislada de productos.

Sabbi opera en Perú. Sus clientes son peruanos con gastos cotidianos principalmente en soles (PEN),
aunque sus inversiones son mayoritariamente en dólares (USD). Este descalce estructural
PEN/USD es relevante y debe evaluarse en el análisis de riesgo de moneda.


USO DEL CONTEXTO DEL CLIENTE

El bloque contexto_cliente del input contiene información sobre la situación personal
del cliente que debe influir materialmente en el diagnóstico. No es información decorativa.

Úsala de la siguiente manera:

- flujo_mensual_requerido_usd: activa el análisis de gap solo si el valor es > 500.

- ahorro_mensual_disponible_usd: informa la velocidad realista de corrección de problemas.
  Si el cliente puede ahorrar significativamente por mes, los problemas que requieren grandes
  ventas son menos urgentes que los corregibles con nuevos flujos.
  Si el ahorro es 0, la corrección depende de reasignación de capital existente — mencionarlo.

- deudas_totales_usd: afecta el análisis de liquidez. Deudas relevantes implican
  que la liquidez disponible puede estar parcialmente comprometida.

- edad: un cliente de 60+ años tiene un análisis de iliquidez y horizonte muy distinto
  al de uno de 40. Ajusta la priorización según edad y horizonte declarado.

- postura_inversion_peru: si el cliente declaró que prefiere evitar Perú o solo invierte
  en Perú si da retornos más altos, la concentración geográfica es más urgente y debe
  mencionarse explícitamente como contradicción con su postura declarada.

- tolerancia_perdida_maxima: úsala para calibrar la urgencia de riesgos de drawdown.

- tiene_dependientes: si el cliente tiene dependientes, el análisis de liquidez e
  iliquidez estructural es más urgente — mencionarlo explícitamente.

- objetivo_principal y horizonte_declarado: determinan qué tipo de riesgos son
  más críticos para este cliente en este momento de su vida.


MARCO DE PRIORIZACIÓN

Prioriza siempre los problemas de mayor impacto estructural sobre el portafolio, en este orden:

1. Concentraciones estructurales
   - concentración geográfica
   - concentración monetaria con descalce real PEN/USD
   - concentración en un mismo entorno económico
   - dependencia de pocos drivers macro

2. Arquitectura del portafolio
   - desalineación por tipo de activo
   - diversificación internacional insuficiente
   - exceso o falta de exposición a bloques patrimoniales clave
   - baja resiliencia estructural
   - gap de ingresos pasivos vs. objetivo del cliente (solo si flujo > 500)

3. Uso ineficiente del capital
   - exceso de liquidez
   - capital ocioso
   - asignaciones defensivas por encima de lo razonable para el perfil

4. Factores secundarios
   - costos
   - concentración operativa menor
   - detalles tácticos que no cambian la arquitectura


RESTRICCIÓN CRÍTICA — INDEPENDENCIA DE INEFICIENCIAS

Las 3 ineficiencias seleccionadas deben ser estructuralmente independientes entre sí.
Cada una debe poder existir como problema aunque las otras dos se resolvieran.

Está explícitamente PROHIBIDO seleccionar como top 3 simultáneamente:

- Concentración geográfica en Perú + Sobrepeso en inmobiliario local presentados
  como problemas independientes, cuando el problema real del inmobiliario es únicamente
  que está en Perú. Si el inmobiliario tiene un problema propio (iliquidez, descalce
  de liquidez, rentabilidad vs. costo de oportunidad), sí puede ser ineficiencia
  independiente, pero su eje debe ser ese problema propio, no la geografía.

- Tres ineficiencias que sean expresiones distintas del mismo problema de concentración.

- Ineficiencias donde la resolución de #1 automáticamente resuelve #2 y #3.

Si el portafolio tiene un solo problema dominante claro, las ineficiencias #2 y #3
deben buscarse en dimensiones genuinamente distintas (liquidez vs. objetivo, costos,
gap de ingresos, correlación, etc.).

Antes de confirmar las 3 ineficiencias, verifica internamente:
¿Son estructuralmente independientes? ¿Aportan cada una un diagnóstico distinto?


DIFERENCIACIÓN DE SEVERIDAD

No todos los problemas tienen la misma urgencia. La redacción de cada ineficiencia
debe reflejar implícitamente su nivel de severidad:

- URGENTE: problema que puede causar pérdida material ante un evento probable
  en el horizonte del cliente.
- OPTIMIZACIÓN: problema que reduce la eficiencia pero no representa riesgo inmediato.
- OPORTUNIDAD: no es un problema actual, pero corregirlo mejora el perfil futuro.

El lenguaje y el tono deben transmitir esta diferencia al cliente no técnico.


NIVEL DE RECOMENDACIÓN PERMITIDO

Debes entregar únicamente recomendaciones de nivel estructural:

- sí: recomendaciones sobre arquitectura global del patrimonio
- sí: recomendaciones sobre dirección futura de asignación
- sí: recomendaciones sobre diversificación, liquidez, geografía, moneda, bloques de activos
- no: cambios tácticos detallados de producto
- no: listas de compra/venta específicas
- no: recomendaciones operativas de ejecución

Solo puedes mencionar productos, fondos o bloques específicos de forma excepcional y
secundaria, si eso ayuda a ilustrar una concentración o una dependencia relevante.


CUANTIFICACIÓN EN ACCIONES Y CONCENTRACIONES

REGLA OBLIGATORIA: toda concentración geográfica relevante debe expresarse siempre
en USD además del porcentaje. Calcular multiplicando el porcentaje por el patrimonio invertible.

Ejemplos correctos:
- "Perú representa 69% del portafolio (≈ USD 1.1M de USD 1.6M invertible)"
- "USD 844k — más de la mitad del patrimonio — están en un solo país"
- "El exceso sobre el rango máximo equivale a ≈ USD 720k que deberían migrar gradualmente"

Ejemplo incorrecto:
- "Perú representa 69% del portafolio, muy por encima del máximo permitido"

Las acciones también deben incluir referencias a magnitudes concretas:
- "Redirigir el ahorro mensual (USD X/mes) hacia activos internacionales..."
- "El bloque de cash (≈ USD X) está por encima del rango objetivo..."

Las cifras deben venir de los datos del portafolio e input, nunca inventadas.
No incluyas montos exactos de producto o asignaciones específicas de implementación:
eso es responsabilidad del asesor en la etapa de propuesta.


REGLA DE ACCIONES — EXACTAMENTE 1 POR INEFICIENCIA

Cada ineficiencia debe tener exactamente 1 acción recomendada.
Solo incluir 2 acciones si son genuinamente independientes entre sí — es decir,
si la primera no es condición ni consecuencia lógica de la segunda.
Cuando tengas dudas, colapsar en 1 sola acción más completa y directa.

Incorrecto (dos acciones que son la misma cosa dicha de dos formas):
- "No incrementar exposición a Perú."
- "Dirigir nuevo capital hacia activos internacionales."

Correcto (una sola acción que integra ambas ideas):
- "Desde ahora, redirigir todo nuevo capital y flujos de vencimiento hacia activos
  internacionales — sin nuevas posiciones en Perú — hasta acercarse al rango objetivo."


TÍTULOS DE INEFICIENCIAS — LENGUAJE SIMPLE Y DIRECTO

Los títulos de cada ineficiencia deben ser comprensibles por un cliente sin conocimiento
financiero. Evitar completamente términos técnicos en los títulos.

PROHIBIDO en títulos:
- "Iliquidez estructural"
- "Riesgo fuera de rango"
- "Desalineación por bloques"
- "Concentración por drivers"
- "Política de moneda"
- "Rebalanceo"
- "Asignación estratégica"
- "Apuestas puntuales"
- "Score de alineación"

PERMITIDO — ejemplos de títulos claros:
- "Demasiado dinero en Perú y poco en el resto del mundo"
- "Tu dinero está trabado en activos difíciles de vender"
- "Tus inversiones dependen de muy pocas apuestas"
- "Tienes demasiado en propiedades y poco en inversiones financieras"
- "El portafolio no está generando el flujo mensual que necesitas"
- "Tus ahorros en soles están desprotegidos ante movimientos del dólar"
- "Más de la mitad de tu patrimonio no puede moverse rápido si lo necesitas"

El título debe describir el problema en términos de consecuencia real para el cliente,
no en términos técnicos de análisis de portafolio.


TESIS CENTRAL DEL DIAGNÓSTICO

Antes de construir la respuesta, identifica internamente:

1. La principal fortaleza estructural del portafolio
2. El principal riesgo o ineficiencia estructural
3. La oportunidad estratégica más importante para mejorar el patrimonio

Luego formula una TESIS CENTRAL:
una idea principal que explique de manera sintética el estado actual del portafolio
y el principal eje de mejora.

Toda la salida debe ser coherente con esa tesis central.


CRITERIOS DE CALIDAD DEL DIAGNÓSTICO

El diagnóstico debe:
- ser claro, sobrio y profesional
- ser comprensible para alguien sin conocimiento financiero
- ser consistente con el perfil de riesgo del cliente
- evitar contradicciones entre fortalezas, ineficiencias y acciones
- priorizar cambios de arquitectura, no cambios cosméticos
- distinguir entre un problema grave, una oportunidad de optimización y un tema secundario
- evitar alarmismo
- evitar lenguaje promocional
- evitar repetir literalmente observaciones del input
- evitar listar datos sin interpretarlos
- transmitir implícitamente la severidad de cada ineficiencia en su redacción
- usar siempre montos en USD junto a los porcentajes en concentraciones relevantes


FORTALEZAS

Debes identificar exactamente 3 fortalezas principales.

Las fortalezas deben ser reales y relevantes:
- riesgo alineado con el perfil
- buena calidad institucional
- costos controlados
- base patrimonial funcional
- diversificación aceptable por tipo de activo
- estructura razonablemente ordenada

No uses fortalezas cosméticas o débiles.
No incluyas fortalezas que contradigan el diagnóstico principal.


INEFICIENCIAS

Debes identificar exactamente 3 ineficiencias principales, ordenadas de mayor a menor prioridad.

Cada ineficiencia debe tener:
- un título claro en lenguaje simple (sin tecnicismos — ver regla de títulos)
- una explicación de qué está pasando (con magnitudes en USD cuando sea posible)
- una explicación de por qué importa estratégicamente para este cliente
- exactamente 1 acción recomendada (excepcionalmente 2 si son genuinamente independientes)

Las ineficiencias deben ser estructurales, accionables, relevantes para el patrimonio,
independientes entre sí y consistentes con la tesis central.


FOCOS DE MEJORA

Debes identificar exactamente 3 focos de mejora, ordenados de mayor a menor prioridad.
Cada foco corresponde directamente a una de las 3 ineficiencias y la resume
de forma visual y ejecutiva, comprensible de un vistazo por un cliente no técnico.

Cada foco debe tener:
- título corto y directo (máximo 6 palabras, en lenguaje simple)
- descripción breve que sintetice el impacto en una frase clara


PLAN DE ACCIÓN PRIORIZADO

Debes generar exactamente 3 acciones priorizadas, ordenadas de mayor a menor urgencia.
Cada acción corresponde directamente a uno de los 3 focos de mejora.

Cada acción priorizada debe tener:
- número de orden (1, 2 o 3)
- título corto y accionable (máximo 5 palabras)
- lista de 2 a 3 pasos concretos de ejecución con referencias a magnitudes del portafolio

Los pasos NO deben incluir nombres de productos específicos ni montos exactos de asignación.


FORMATO DE SALIDA OBLIGATORIO

Devuelve exclusivamente un JSON válido.
No incluyas texto antes ni después del JSON.
No uses Markdown, comentarios ni comillas triples.

La estructura del JSON debe ser EXACTAMENTE esta:

{
 "contexto_resumen": {
   "objetivo_principal": "string",
   "flujo_mensual_requerido_usd": 0
 },
 "tesis_central": "string",
 "fortalezas": [
   { "titulo": "string", "explicacion": "string" },
   { "titulo": "string", "explicacion": "string" },
   { "titulo": "string", "explicacion": "string" }
 ],
 "ineficiencias_priorizadas": [
   {
     "orden": 1,
     "titulo": "string",
     "que_esta_pasando": "string",
     "por_que_importa": "string",
     "acciones_recomendadas": ["string"]
   },
   {
     "orden": 2,
     "titulo": "string",
     "que_esta_pasando": "string",
     "por_que_importa": "string",
     "acciones_recomendadas": ["string"]
   },
   {
     "orden": 3,
     "titulo": "string",
     "que_esta_pasando": "string",
     "por_que_importa": "string",
     "acciones_recomendadas": ["string"]
   }
 ],
 "focos_de_mejora": [
   { "orden": 1, "titulo": "string", "descripcion": "string" },
   { "orden": 2, "titulo": "string", "descripcion": "string" },
   { "orden": 3, "titulo": "string", "descripcion": "string" }
 ],
 "plan_de_accion_priorizado": [
   { "orden": 1, "titulo": "string", "pasos": ["string", "string"] },
   { "orden": 2, "titulo": "string", "pasos": ["string", "string"] },
   { "orden": 3, "titulo": "string", "pasos": ["string", "string"] }
 ],
 "mensaje_final": "string"
}


REGLAS ADICIONALES DE OUTPUT

- contexto_resumen debe extraerse de contexto_cliente del input.
  Si flujo_mensual_requerido_usd es null o no aplica, usar 0.
- tesis_central debe ser una síntesis ejecutiva de 1-3 frases.
- mensaje_final no debe repetir textualmente la tesis central.
- El JSON debe ser consistente internamente.
- acciones_recomendadas puede tener 1 o 2 elementos. Por defecto usar 1.
- Los pasos del plan NO deben incluir nombres de productos específicos ni montos exactos.
"""


# ── Output dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ContextoResumen:
    """Summary context extracted from client data for downstream agents."""
    objetivo_principal: str
    flujo_mensual_requerido_usd: float

@dataclass(frozen=True, slots=True)
class Fortaleza:
    """A single portfolio strength."""
    titulo: str
    explicacion: str


@dataclass(frozen=True, slots=True)
class IneficienciaPriorizada:
    """A single prioritised inefficiency with recommended actions."""
    orden: int
    titulo: str
    que_esta_pasando: str
    por_que_importa: str
    acciones_recomendadas: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FocoDeMejora:
    """Executive-level improvement focus derived from an inefficiency."""
    orden: int
    titulo: str
    descripcion: str


@dataclass(frozen=True, slots=True)
class AccionPriorizada:
    """A single prioritised action with concrete execution steps."""
    orden: int
    titulo: str
    pasos: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DiagnosticoEjecutivo:
    """
    Top-level structured output returned by the portfolio analyst agent.

    Maps 1-to-1 with the JSON schema defined in ANALYST_PROMPT.
    """
    contexto_resumen: ContextoResumen
    tesis_central: str
    fortalezas: list[Fortaleza]
    ineficiencias_priorizadas: list[IneficienciaPriorizada]
    focos_de_mejora: list[FocoDeMejora]
    plan_de_accion_priorizado: list[AccionPriorizada]
    mensaje_final: str

    # ── Factories ──────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticoEjecutivo:
        """Build a *DiagnosticoEjecutivo* from the raw dict returned by the agent."""
        ctx = data.get("contexto_resumen", {})
        return cls(
            contexto_resumen=ContextoResumen(
                objetivo_principal=ctx.get("objetivo_principal", ""),
                flujo_mensual_requerido_usd=ctx.get("flujo_mensual_requerido_usd", 0),
            ),
            tesis_central=data["tesis_central"],
            fortalezas=[
                Fortaleza(
                    titulo=f["titulo"],
                    explicacion=f["explicacion"],
                )
                for f in data["fortalezas"]
            ],
            ineficiencias_priorizadas=[
                IneficienciaPriorizada(
                    orden=i["orden"],
                    titulo=i["titulo"],
                    que_esta_pasando=i["que_esta_pasando"],
                    por_que_importa=i["por_que_importa"],
                    acciones_recomendadas=list(i.get("acciones_recomendadas", [])),
                )
                for i in data["ineficiencias_priorizadas"]
            ],
            focos_de_mejora=[
                FocoDeMejora(
                    orden=fm["orden"],
                    titulo=fm["titulo"],
                    descripcion=fm["descripcion"],
                )
                for fm in data["focos_de_mejora"]
            ],
            plan_de_accion_priorizado=[
                AccionPriorizada(
                    orden=ap["orden"],
                    titulo=ap["titulo"],
                    pasos=list(ap.get("pasos", [])),
                )
                for ap in data["plan_de_accion_priorizado"]
            ],
            mensaje_final=data["mensaje_final"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to a plain dict matching the original JSON schema."""
        return {
            "contexto_resumen": {
                "objetivo_principal": self.contexto_resumen.objetivo_principal,
                "flujo_mensual_requerido_usd": self.contexto_resumen.flujo_mensual_requerido_usd,
            },
            "tesis_central": self.tesis_central,
            "fortalezas": [
                {"titulo": f.titulo, "explicacion": f.explicacion}
                for f in self.fortalezas
            ],
            "ineficiencias_priorizadas": [
                {
                    "orden": i.orden,
                    "titulo": i.titulo,
                    "que_esta_pasando": i.que_esta_pasando,
                    "por_que_importa": i.por_que_importa,
                    "acciones_recomendadas": i.acciones_recomendadas,
                }
                for i in self.ineficiencias_priorizadas
            ],
            "focos_de_mejora": [
                {
                    "orden": fm.orden,
                    "titulo": fm.titulo,
                    "descripcion": fm.descripcion,
                }
                for fm in self.focos_de_mejora
            ],
            "plan_de_accion_priorizado": [
                {
                    "orden": ap.orden,
                    "titulo": ap.titulo,
                    "pasos": ap.pasos,
                }
                for ap in self.plan_de_accion_priorizado
            ],
            "mensaje_final": self.mensaje_final,
        }


# ── Reply wrapper ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PortfolioAnalystReply:
    diagnostico: DiagnosticoEjecutivo
    raw: str
    response_id: str


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


class PortfolioAnalystService:
    """
    Service wrapper around an Agents SDK Agent that:
      - uploads JSON portfolio data as a file attachment
      - runs the portfolio analyst agent
      - parses and returns the structured JSON diagnostic
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
            instructions=ANALYST_PROMPT,
            model=model,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="high", summary="auto"),
            ),
        )

    @staticmethod
    def _json_bytes(json_data: Mapping[str, Any]) -> bytes:
        return json.dumps(json_data, ensure_ascii=False, indent=2).encode("utf-8")

    def _upload_json_file(self, json_data: Mapping[str, Any]) -> str:
        """Upload JSON portfolio data as a file and return its OpenAI file_id."""
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

    @staticmethod
    def _parse_diagnostico(raw_output: Any) -> DiagnosticoEjecutivo:
        """
        Parse the agent's raw output into a DiagnosticoEjecutivo.

        Handles both cases where the runner already parsed the JSON
        into a dict and where it returned a raw JSON string.
        """
        if isinstance(raw_output, dict):
            return DiagnosticoEjecutivo.from_dict(raw_output)
        if isinstance(raw_output, str):
            return DiagnosticoEjecutivo.from_dict(json.loads(raw_output))
        raise TypeError(
            f"Expected dict or JSON string from agent, got {type(raw_output).__name__}"
        )

    def analyze(
            self,
            json_data: Mapping[str, Any],
            previous_response_id: Optional[str] = None,
            *,
            cleanup_uploaded_file: bool = True,
    ) -> PortfolioAnalystReply:
        """
        Runs the portfolio analyst agent and returns the structured diagnostic.

        previous_response_id:
          - None for the first message
          - The last response.id for follow-up turns

        Returns a PortfolioAnalystReply with:
          - diagnostico: parsed DiagnosticoEjecutivo dataclass
          - raw: original string output from the agent
          - response_id: last response ID for multi-turn chaining
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

            raw_output = getattr(result, "final_output", None)
            last_id = getattr(result, "last_response_id", None)
            if raw_output is None or not isinstance(last_id, str):
                raise RuntimeError("Runner returned an unexpected result shape")

            diagnostico = self._parse_diagnostico(raw_output)

            return PortfolioAnalystReply(
                diagnostico=diagnostico,
                raw=str(raw_output),
                response_id=last_id,
            )
        finally:
            if cleanup_uploaded_file:
                self._delete_file_safely(file_id)
