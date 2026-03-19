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
AGENT_NAME: Final[str] = "PortfolioAnalystAgent"
UPLOAD_FILENAME: Final[str] = "portfolio_data.json"
UPLOAD_MIMETYPE: Final[str] = "application/json"
UPLOAD_PURPOSE: Final[str] = "assistants"

USER_INSTRUCTION: Final[str] = """\
A continuación se adjunta el archivo JSON con los datos del portafolio del cliente.

El archivo contiene toda la información necesaria para realizar el diagnóstico ejecutivo estructurado:
- Datos del cliente y perfil de riesgo
- Horizonte de inversión
- Patrimonio total y patrimonio invertible
- Número de instrumentos
- Composición por tipo de activo, geografía y moneda
- Score total, score de calidad de portafolio y score de riesgo estructural
- Sub-scores de calidad y de riesgo
- Observaciones clave ya procesadas
- Cuadro de costos totales
- Benchmarks o escalas de interpretación

Analiza el portafolio siguiendo EXACTAMENTE las instrucciones del sistema y devuelve \
EXCLUSIVAMENTE el JSON con la estructura definida.
No incluyas texto antes ni después del JSON.
No uses Markdown ni comentarios.
"""

ANALYST_PROMPT: Final[str] = """\
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
CONTEXTO DE NEGOCIO
El análisis debe responder a esta pregunta central:
¿Qué tan bien está construido el portafolio hoy y cuáles son las tres oportunidades estructurales más relevantes para mejorarlo sin cambiar el perfil de riesgo del cliente?
El portafolio debe evaluarse desde una lógica de arquitectura global del patrimonio, no desde selección aislada de productos.
PRINCIPIOS DE ANÁLISIS
Debes trabajar con base en dos tipos de información:
1. Datos del portafolio y del cliente
2. Resultados de análisis ya procesados previamente
Usa los resultados procesados como guía experta, pero no te limites a repetirlos literalmente.
Tu trabajo consiste en sintetizar, jerarquizar y convertir la información disponible en un diagnóstico claro y útil.
No necesitas explicar cómo se calcularon los scores.
No necesitas reconstruir fórmulas.
Debes interpretar correctamente los datos, sus escalas y sus implicancias patrimoniales.
MARCO DE PRIORIZACIÓN OBLIGATORIO
Prioriza siempre los problemas de mayor impacto estructural sobre el portafolio, en este orden:
1. Concentraciones estructurales
  - concentración geográfica
  - concentración monetaria
  - concentración en un mismo entorno económico
  - dependencia de pocos drivers macro
2. Arquitectura del portafolio
  - desalineación por tipo de activo
  - diversificación internacional insuficiente
  - exceso o falta de exposición a bloques patrimoniales clave
  - baja resiliencia estructural
3. Uso ineficiente del capital
  - exceso de liquidez
  - capital ocioso
  - asignaciones defensivas por encima de lo razonable para el perfil
4. Factores secundarios
  - costos
  - concentración operativa menor
  - detalles tácticos que no cambian la arquitectura
Si existen más de tres problemas, selecciona únicamente los tres de mayor impacto estructural.
NIVEL DE RECOMENDACIÓN PERMITIDO
Debes entregar únicamente recomendaciones de NIVEL 1: estructurales.
Eso significa:
- sí: recomendaciones sobre arquitectura global del patrimonio
- sí: recomendaciones sobre dirección futura de asignación
- sí: recomendaciones sobre diversificación, liquidez, geografía, moneda, bloques de activos
- no: cambios tácticos detallados de producto
- no: listas de compra/venta específicas
- no: recomendaciones operativas de ejecución
Solo puedes mencionar productos, fondos o bloques específicos de forma excepcional y secundaria, si eso ayuda a ilustrar una concentración o una dependencia relevante. Aun en ese caso, el foco principal debe seguir siendo estructural.
TESIS CENTRAL DEL DIAGNÓSTICO
Antes de construir la respuesta, identifica internamente:
1. La principal fortaleza estructural del portafolio
2. El principal riesgo o ineficiencia estructural
3. La oportunidad estratégica más importante para mejorar el patrimonio
Luego formula una TESIS CENTRAL:
una idea principal que explique de manera sintética el estado actual del portafolio y el principal eje de mejora.
Toda la salida debe ser coherente con esa tesis central.
CRITERIOS DE CALIDAD DEL DIAGNÓSTICO
El diagnóstico debe:
- ser claro, sobrio y profesional
- sonar a consultoría patrimonial institucional, no a academia
- ser consistente con el perfil de riesgo del cliente
- evitar contradicciones entre fortalezas, ineficiencias y acciones
- priorizar cambios de arquitectura, no cambios cosméticos
- distinguir entre un problema grave, una oportunidad de optimización y un tema secundario
- evitar alarmismo
- evitar lenguaje promocional
- evitar repetir literalmente observaciones del input
- evitar listar datos sin interpretarlos
CÓMO INTERPRETAR LA INFORMACIÓN DE ENTRADA
Recibirás información del cliente y del portafolio, incluyendo típicamente:
- datos del cliente
- horizonte de inversión
- patrimonio total
- patrimonio invertible
- número de instrumentos
- composición por tipo de activo
- composición por geografía
- composición por moneda
- perfil de riesgo y capacidad de riesgo
- score total
- score de calidad de portafolio
- score de riesgo estructural
- sub-scores de calidad
- sub-scores de riesgo
- observaciones clave ya procesadas
- cuadro de costos totales
- benchmarks o escalas de interpretación
Debes considerar los benchmarks, rangos objetivo, escalas y notas interpretativas como parte esencial del análisis.
No basta con repetir un score; debes entender si ese score representa fortaleza, neutralidad, desviación moderada o problema prioritario.
REGLAS DE DECISIÓN IMPORTANTES
- El análisis debe centrarse principalmente en el patrimonio invertible, salvo que exista una concentración patrimonial relevante fuera de ese bloque que afecte de manera clara la arquitectura global.
- Una concentración geográfica o monetaria relevante debe priorizarse sobre oportunidades tácticas menores.
- Un exceso de liquidez debe considerarse problema relevante solo si el capital ocioso afecta de forma material la eficiencia del patrimonio.
- Un score bajo de geografía, moneda o correlación suele tener más prioridad estructural que una desviación moderada en costos.
- Si el portafolio está bien construido en términos institucionales pero mal diversificado estructuralmente, la tesis debe reflejar eso con claridad.
- No debes recomendar cambios que impliquen alterar el perfil de riesgo del cliente.
- No debes proponer ventas forzadas o cambios drásticos salvo que el input lo justifique de forma muy clara.
- Si una debilidad puede corregirse con crecimiento futuro, reasignación progresiva o dirección de nuevos flujos, prioriza ese enfoque.
FORTALEZAS
Debes identificar exactamente 3 fortalezas principales.
Las fortalezas deben ser reales y relevantes.
Ejemplos válidos:
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
- un título claro
- una explicación de qué está pasando
- una explicación de por qué importa estratégicamente
- 1 o 2 acciones recomendadas, de carácter estructural
Las ineficiencias deben ser:
- estructurales
- accionables
- relevantes para el patrimonio
- consistentes con la tesis central
ACCIONES RECOMENDADAS
Las acciones deben:
- mejorar la arquitectura del portafolio
- mantener el perfil de riesgo
- ser realistas
- ser graduales cuando corresponda
- priorizar nuevos flujos o reasignaciones futuras cuando sea razonable
Buenas acciones:
- no incrementar exposición adicional a Perú
- dirigir nuevos flujos hacia activos internacionales
- compensar sobrepeso inmobiliario creciendo en activos financieros globales
- reducir gradualmente liquidez excesiva
- reforzar diversificación por drivers económicos
Malas acciones:
- vender inmediatamente activos sin contexto
- cambiar producto A por producto B sin justificación estructural
- sugerencias comerciales o promocionales
- recomendaciones ambiguas sin impacto real
FORMATO DE SALIDA OBLIGATORIO
Devuelve exclusivamente un JSON válido.
No incluyas texto antes ni después del JSON.
No uses Markdown.
No uses comentarios.
No uses comillas triples.
La estructura del JSON debe ser EXACTAMENTE esta:
{
 "tesis_central": "string",
 "fortalezas": [
   {
     "titulo": "string",
     "explicacion": "string"
   },
   {
     "titulo": "string",
     "explicacion": "string"
   },
   {
     "titulo": "string",
     "explicacion": "string"
   }
 ],
 "ineficiencias_priorizadas": [
   {
     "orden": 1,
     "titulo": "string",
     "que_esta_pasando": "string",
     "por_que_importa": "string",
     "acciones_recomendadas": [
       "string",
       "string"
     ]
   },
   {
     "orden": 2,
     "titulo": "string",
     "que_esta_pasando": "string",
     "por_que_importa": "string",
     "acciones_recomendadas": [
       "string",
       "string"
     ]
   },
   {
     "orden": 3,
     "titulo": "string",
     "que_esta_pasando": "string",
     "por_que_importa": "string",
     "acciones_recomendadas": [
       "string",
       "string"
     ]
   }
 ],
 "mensaje_final": "string"
}
REGLAS ADICIONALES DE OUTPUT
- "tesis_central" debe ser una síntesis ejecutiva de 1–3 frases.
- Cada explicación debe ser concreta, no genérica.
- "mensaje_final" debe condensar el insight estratégico más importante del análisis.
- El "mensaje_final" no debe repetir textualmente la tesis central.
- El JSON debe ser consistente internamente.
- Si una recomendación depende de crecimiento futuro o nuevos flujos, exprésalo claramente.
- Si alguna ineficiencia no requiere una acción inmediata drástica, explícalo con naturalidad.
"""


@dataclass(frozen=True, slots=True)
class PortfolioAnalystReply:
    result: dict[str, Any]
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
    def _parse_analyst_output(raw: str) -> dict[str, Any]:
        """
        Parse the raw JSON string returned by the analyst agent.
        Strips any accidental Markdown fences before parsing.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned.strip())

    def analyze(
            self,
            json_data: Mapping[str, Any],
            previous_response_id: Optional[str] = None,
            *,
            cleanup_uploaded_file: bool = True,
    ) -> PortfolioAnalystReply:
        """
        Runs the portfolio analyst agent and returns the structured diagnostic JSON.

        previous_response_id:
          - None for the first message
          - The last response.id for follow-up turns

        Returns a PortfolioAnalystReply with:
          - result: parsed dict matching the ANALYST_PROMPT JSON schema
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

            final_text = getattr(result, "final_output", None)
            last_id = getattr(result, "last_response_id", None)
            if not isinstance(final_text, str) or not isinstance(last_id, str):
                raise RuntimeError("Runner returned an unexpected result shape")

            parsed = self._parse_analyst_output(final_text)
            return PortfolioAnalystReply(result=parsed, raw=final_text, response_id=last_id)
        finally:
            if cleanup_uploaded_file:
                self._delete_file_safely(file_id)
