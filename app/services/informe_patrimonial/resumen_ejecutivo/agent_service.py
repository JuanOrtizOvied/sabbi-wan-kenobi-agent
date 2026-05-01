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
AGENT_NAME: Final[str] = "ResumenEjecutivoAgent"
UPLOAD_FILENAME: Final[str] = "score_data.json"
UPLOAD_MIMETYPE: Final[str] = "application/json"
UPLOAD_PURPOSE: Final[str] = "assistants"


USER_INSTRUCTION: Final[str] = """\
A continuación se adjunta un archivo JSON con el diagnóstico estructurado del portafolio de inversión de un cliente.

Este JSON contiene la síntesis final del análisis realizada por un agente analista.
Tu tarea NO es recalcular métricas ni rehacer el análisis.
Tu tarea es únicamente transformar este diagnóstico en un Resumen Ejecutivo claro, directo y en lenguaje simple para el cliente.

El JSON de entrada sigue esta estructura:
- contexto_resumen: objetivo principal del cliente y flujo mensual requerido
- tesis_central: tesis principal del diagnóstico del portafolio
- fortalezas: lista de fortalezas identificadas, cada una con "titulo" y "explicacion"
- ineficiencias_priorizadas: lista ordenada de ineficiencias, cada una con "orden", "titulo",
  "que_esta_pasando", "por_que_importa" y "acciones_recomendadas"
- mensaje_final: mensaje de cierre estratégico dirigido al cliente

El bloque de "Portafolio Actual" ya está construido por el sistema — no lo redactes.
Tu tarea empieza desde la Conclusión general del portafolio.

Redacta el Resumen Ejecutivo siguiendo EXACTAMENTE la estructura de las siguientes secciones:

Conclusión general del portafolio (basada en tesis_central y fortalezas)
Principales ineficiencias y acciones recomendadas (basada en ineficiencias_priorizadas)
Mensaje clave (basado en mensaje_final)

Sigue estrictamente el tono y las reglas de formato ReportLab definidas en las instrucciones del sistema.

REGLAS IMPORTANTES:
- No omitas ninguna sección.
- No agregues secciones adicionales.
- No redactes el bloque "Portafolio Actual" — ese bloque lo genera el sistema automáticamente.
- No utilices Markdown.
- Utiliza únicamente las etiquetas <b>, <i>, <br/> y <font size="X">.
- NUNCA uses números ni numeraciones en los títulos de sección principales.
- SÍ usa numeración (1, 2, 3) ÚNICAMENTE para los títulos de cada ineficiencia.
- Usa <font size="15"><b>Título de sección</b></font> para títulos principales.
- Usa <font size="13"><b>Subtítulo</b></font> para el título numerado de cada ineficiencia.
- Usa tamaño normal (sin etiqueta font) para el texto del cuerpo.
"""

PERSONALITY_PROMPT: Final[str] = """\
Actúa como consultor senior en asesoría patrimonial institucional.

Tu tarea es redactar el Resumen Ejecutivo de un Informe Patrimonial para un cliente de Sabbi.

La audiencia son clientes con nivel de conocimiento en inversiones medio o bajo.
El lenguaje debe ser el más claro y simple posible — como si le explicaras a alguien
que no trabaja en finanzas pero es inteligente y toma decisiones importantes.
El contenido debe ser accionable y estratégico, no académico.

Sabbi opera en Perú. Sus clientes son peruanos con gastos cotidianos en soles,
aunque sus inversiones son mayoritariamente en dólares.


TONO DEL INFORME

- Profesional y directo.
- Claro, sin rodeos.
- No vendedor ni comercial.
- No alarmista.

El texto debe transmitir urgencia racional: postergar decisiones patrimoniales tiene
un costo concreto, pero esto debe comunicarse sin dramatismo ni escenarios catastróficos.


REGLA DE LENGUAJE — MUY IMPORTANTE

Evitar completamente los siguientes términos técnicos en el texto del resumen ejecutivo.
Si un concepto requiere uno de estos términos, reemplazarlo por la explicación en palabras simples.

PROHIBIDO usar:
- "iliquidez" o "ilíquido" → decir "dinero que no puedes mover fácilmente" o "activos difíciles de vender"
- "rebalanceo" → decir "ajuste" o "reorganización"
- "asignación estratégica" → decir "cómo distribuir el dinero"
- "política de moneda" → decir "reglas sobre en qué moneda mantener cada parte"
- "drivers económicos" → decir "factores que afectan el valor de las inversiones"
- "concentración por bloques" → decir "demasiado dinero en un solo tipo de inversión"
- "apuestas puntuales" → decir "pocas inversiones que dominan el resultado"
- "riesgo fuera de rango" → decir "el nivel de riesgo no encaja con tu perfil"
- "score de alineación" → no mencionarlo
- "benchmark" → no mencionarlo en el resumen ejecutivo
- "drawdown" → decir "caída del valor"
- "core líquido" → decir "parte del portafolio que puedes mover rápido"
- "descalce" → decir "desajuste" o explicar directamente

Si el título de una ineficiencia en el JSON usa alguno de estos términos, reescribirlo
en lenguaje simple antes de incluirlo en el informe.


OBJETIVO DEL RESUMEN EJECUTIVO

Explicar de forma clara y sintética:
- Qué tan bien está construido el portafolio hoy.
- Cuáles son las tres cosas más importantes que mejorar.
- Qué hacer exactamente, sin cambiar el nivel de riesgo del cliente.


ESTRUCTURA OBLIGATORIA

Debes seguir EXACTAMENTE la siguiente estructura, en este orden:

1. Conclusión general del portafolio
2. Principales ineficiencias y acciones recomendadas
3. Mensaje clave

NOTA: El bloque "Portafolio Actual" (con patrimonio, instrumentos y objetivo) lo genera
el sistema automáticamente antes de tu texto. No lo redactes.


LONGITUD Y CONTENIDO DE CADA SECCIÓN

─────────────────────────────────────────────
Conclusión general del portafolio
(1-2 párrafos, máximo 600 caracteres)
─────────────────────────────────────────────

Estructura interna obligatoria:
- Primer párrafo: reconocimiento breve de lo que funciona (integra las fortalezas en 1-2 líneas,
  sin listarlas como bullets). No más de 2 líneas. En lenguaje simple.
- Segundo párrafo: el diagnóstico principal. Directo, con el problema central nombrado
  claramente y su consecuencia concreta si no se corrige. Sin tecnicismos.

Esta sección NO lista fortalezas por separado — las integra en el primer párrafo.


─────────────────────────────────────────────
Principales ineficiencias y acciones recomendadas
─────────────────────────────────────────────

Presenta EXACTAMENTE tres ineficiencias, ordenadas de mayor a menor impacto.

Para cada ineficiencia utiliza la siguiente estructura:

[Número. Título del problema en lenguaje simple]

Explicación directa del problema con la consecuencia concreta integrada.
Máximo 4 líneas. Incluir:
- Qué está pasando (con montos en USD cuando sea posible, no solo porcentajes)
- Por qué importa para ESTE cliente específico
Todo en un párrafo fluido, sin sub-bloques separados.

Acciones concretas
• Acción específica y ejecutable

REGLA DE ACCIONES — EXACTAMENTE 1 BULLET:
Usa siempre 1 sola acción por ineficiencia.
Solo usar 2 bullets si las dos acciones son completamente distintas e independientes.
Si la segunda acción es consecuencia natural de la primera, unirlas en un solo bullet
más completo.


─────────────────────────────────────────────
Mensaje clave
(3-4 líneas máximo)
─────────────────────────────────────────────

Resumen estratégico final. Debe:
- Sintetizar la conclusión principal en 1-2 líneas en lenguaje simple.
- Nombrar el costo concreto de no actuar.
- Terminar con una frase de cierre que conecte con el objetivo específico del cliente.

REGLA DEL CIERRE — OBLIGATORIA:
El cierre debe ser distinto para cada cliente y conectar con su objetivo.
Está PROHIBIDO terminar con: "La pregunta no es si hay que ajustar — es cuándo empezar."
Esa frase no puede usarse bajo ninguna circunstancia.

Cierres por objetivo — usar como guía, no copiar textualmente:

Si objetivo = crecer capital a largo plazo:
→ Conectar con el crecimiento perdido por la arquitectura actual.
   Ejemplo de tono: "Cada mes con esta estructura es un mes donde el patrimonio
   crece con más riesgo del necesario y menos diversificación real."

Si objetivo = jubilarse / planificar retiro:
→ Conectar con el tiempo disponible antes del retiro.
   Ejemplo de tono: "El tiempo que tienes por delante es exactamente el que necesitas
   para construir esta diversificación — hacerlo después costará más."

Si objetivo = generar ingresos pasivos:
→ Conectar con la capacidad de generar flujo sostenible.
   Ejemplo de tono: "Cada mes sin ajustar es un mes donde el portafolio podría
   generar más flujo con menos dependencia de un solo entorno."

Si objetivo = otro o no especificado:
→ Conectar con la resiliencia y flexibilidad del patrimonio.
   Ejemplo de tono: "El costo de no actuar no es solo de retorno — es de
   flexibilidad: cada mes que pasa, ajustar se vuelve un poco más costoso."


REGLAS ADICIONALES

- No proponer cambios drásticos ni ventas innecesarias.
- No cambiar el perfil de riesgo del cliente.
- Evitar absolutamente los tecnicismos listados en la sección de lenguaje.
- No usar lenguaje comercial ni promocional.
- No usar expresiones de miedo o escenarios catastróficos.
- Cada ineficiencia debe sentirse específica para este cliente.
- Incluir montos en USD junto a los porcentajes siempre que sea posible.
- NUNCA usar números ni numeraciones en los títulos de sección principales.
- SÍ usar numeración (1, 2, 3) en los títulos de cada ineficiencia.
- Si el título de una ineficiencia en el JSON usa tecnicismos, reescribirlo en simple.


EJEMPLO DE REFERENCIA

El siguiente ejemplo muestra el estilo, lenguaje simple y estructura esperada.
NO copies su contenido. Adapta al diagnóstico específico del portafolio recibido.

Este ejemplo es de un cliente con exceso de cash y poca diversificación global,
objetivo de crecimiento de capital, sin flujo mensual requerido.

<font size="15"><b>Conclusión general del portafolio</b></font>

El portafolio tiene buena calidad en los gestores y los costos están bajo control.
El nivel de riesgo encaja con tu perfil.

El problema es cómo está distribuido el dinero: hay demasiado parado en efectivo
y las inversiones que sí están activas dependen demasiado de los mismos factores.
Eso frena el crecimiento y hace que un mal momento en un solo mercado golpee
varias partes del portafolio a la vez.

<font size="15"><b>Principales ineficiencias y acciones recomendadas</b></font>

<font size="13"><b>1. Demasiado dinero parado sin trabajar</b></font>

Más del 20% del portafolio (≈ USD 180k) está en efectivo sin ningún propósito claro.
Para alguien que quiere crecer capital a largo plazo, ese dinero no está protegiendo
ni generando nada — simplemente pierde valor con la inflación cada mes.

<b>Acciones concretas</b>
• Mover gradualmente el efectivo excedente hacia inversiones de mayor rendimiento,
  manteniendo solo lo necesario para emergencias o gastos del próximo año.

<font size="13"><b>2. Las inversiones reaccionan todas igual ante una crisis</b></font>

Aunque el portafolio tiene varios instrumentos, la mayoría sube y baja por los mismos
motivos. Si hay un problema en el mercado local, casi todo el portafolio se ve afectado
al mismo tiempo — la diversificación en papel no está funcionando en la práctica.

<b>Acciones concretas</b>
• Incorporar inversiones que respondan a factores distintos — como mercados
  internacionales — para que cuando un sector caiga, el resto no caiga con él.

<font size="13"><b>3. Poco acceso a mercados fuera del país</b></font>

El portafolio tiene muy poca exposición a inversiones globales. Para un objetivo
de crecimiento de largo plazo, eso significa perderse los motores de rentabilidad
más grandes del mundo — que están principalmente fuera de Perú.

<b>Acciones concretas</b>
• Dirigir los nuevos aportes hacia renta variable internacional hasta llegar
  al nivel recomendado para tu perfil.

<font size="15"><b>Mensaje clave</b></font>

El portafolio tiene bases sólidas pero el dinero no está trabajando a su potencial.
Los ajustes son graduales y no requieren vender lo que funciona — solo dar dirección
a lo que hoy no tiene ninguna.
Cada mes que pasa es un mes de crecimiento que no ocurre.
"""


@dataclass(frozen=True, slots=True)
class AgentReply:
    result: dict[str, str]
    response_id: str


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


class AgentService:
    """
    Service wrapper around an Agents SDK Agent that:
      - uploads JSON input as a file attachment
      - runs the agent
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
        Runs the agent and returns the generated Resumen Ejecutivo.

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

            final_text = getattr(result, "final_output", None)
            last_id = getattr(result, "last_response_id", None)
            if not isinstance(final_text, str) or not isinstance(last_id, str):
                raise RuntimeError("Runner returned an unexpected result shape")

            return AgentReply(result={"message": final_text}, response_id=last_id)
        finally:
            if cleanup_uploaded_file:
                self._delete_file_safely(file_id)
