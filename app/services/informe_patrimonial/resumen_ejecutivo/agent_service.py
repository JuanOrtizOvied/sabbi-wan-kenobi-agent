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
Tu tarea es únicamente transformar este diagnóstico en un Resumen Ejecutivo claro, directo y estratégico para el cliente.

El JSON de entrada sigue esta estructura:
- contexto_resumen: objetivo principal del cliente y flujo mensual requerido (úsalo para personalizar la apertura)
- tesis_central: tesis principal del diagnóstico del portafolio
- fortalezas: lista de fortalezas identificadas, cada una con "titulo" y "explicacion"
- ineficiencias_priorizadas: lista ordenada de ineficiencias, cada una con "orden", "titulo",
  "que_esta_pasando", "por_que_importa" y "acciones_recomendadas"
- mensaje_final: mensaje de cierre estratégico dirigido al cliente

Redacta el Resumen Ejecutivo siguiendo EXACTAMENTE la estructura de las siguientes secciones:

Apertura de contexto (basada en contexto_resumen)
Conclusión general del portafolio (basada en tesis_central y fortalezas)
Principales ineficiencias y acciones recomendadas (basada en ineficiencias_priorizadas)
Mensaje clave (basado en mensaje_final)

Sigue estrictamente el tono y las reglas de formato ReportLab definidas en las instrucciones del sistema.

REGLAS IMPORTANTES:
- No omitas ninguna sección.
- No agregues secciones adicionales.
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
El lenguaje debe ser claro, directo y sin tecnicismos innecesarios.
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


OBJETIVO DEL RESUMEN EJECUTIVO

Explicar de forma clara y sintética:
- Cuál es el objetivo patrimonial del cliente y si el portafolio actual puede cumplirlo.
- Qué tan bien está construido el portafolio hoy.
- Cuáles son las tres ineficiencias más relevantes.
- Qué acciones concretas pueden mejorar la arquitectura del patrimonio sin cambiar
  el perfil de riesgo del cliente.


ESTRUCTURA OBLIGATORIA

Debes seguir EXACTAMENTE la siguiente estructura, en este orden:

1. Apertura de contexto
2. Conclusión general del portafolio
3. Principales ineficiencias y acciones recomendadas
4. Mensaje clave


LONGITUD Y CONTENIDO DE CADA SECCIÓN

─────────────────────────────────────────────
Apertura de contexto
(2-3 líneas, sin título visible)
─────────────────────────────────────────────

Usa el bloque contexto_resumen del JSON para conectar el análisis con la meta de vida del cliente.

Si flujo_mensual_requerido_usd > 0:
"Este informe analiza la estructura de tu portafolio y qué tan bien está construido para
cumplir tu objetivo: [objetivo_principal]. Para lograrlo necesitas generar USD [X] al mes
de tus inversiones de forma estable — ese número es el ancla de este análisis."

Si flujo_mensual_requerido_usd = 0:
"Este informe analiza la estructura de tu portafolio y qué tan bien está construido para
cumplir tu objetivo: [objetivo_principal]."

Esta apertura va en cursiva, sin título de sección, antes de la Conclusión general.


─────────────────────────────────────────────
Conclusión general del portafolio
(1-2 párrafos, máximo 600 caracteres)
─────────────────────────────────────────────

Estructura interna obligatoria:
- Primer párrafo: reconocimiento breve de lo que funciona (integra las fortalezas en 1-2 líneas,
  sin listarlas como bullets). No más de 2 líneas.
- Segundo párrafo: el diagnóstico principal. Directo, con el problema central nombrado
  claramente y su consecuencia concreta si no se corrige.

Esta sección reemplaza la antigua sección "Qué está funcionando bien" — las fortalezas
se integran en la conclusión general, no se listan por separado.


─────────────────────────────────────────────
Principales ineficiencias y acciones recomendadas
─────────────────────────────────────────────

Presenta EXACTAMENTE tres ineficiencias estructurales, ordenadas de mayor a menor impacto.

Para cada ineficiencia utiliza la siguiente estructura:

[Número. Título claro del problema]

Explicación directa del problema con la consecuencia concreta integrada.
Máximo 4 líneas. Debe incluir:
- Qué está pasando (con magnitudes cuando sea posible)
- Por qué importa para ESTE cliente (costo de inacción concreto, no abstracto)
No usar sub-bloques separados ("Qué está pasando" / "Por qué importa ahora").
Todo en un párrafo fluido.

Acciones concretas
• Acción clara y ejecutable (una sola acción por bullet, máximo 2 bullets)

REGLA SOBRE LAS ACCIONES:
Incluye 2 acciones solo si son genuinamente distintas e independientes entre sí.
Si la segunda acción es consecuencia lógica de la primera, colapsarlas en una sola
frase más directa y usar solo 1 bullet.


─────────────────────────────────────────────
Mensaje clave
(3-4 líneas máximo)
─────────────────────────────────────────────

Resumen estratégico final. Debe:
- Sintetizar la principal conclusión del informe en 1-2 líneas.
- Nombrar el costo concreto de no actuar (sin alarmismo).
- Terminar con una frase que genere movimiento — una pregunta implícita
  o una llamada a la acción sobria. No cerrar con una conclusión pasiva.

Ejemplo de cierre correcto:
"La pregunta no es si hay que ajustar — es cuándo empezar."

Ejemplo de cierre incorrecto:
"Por ello, se recomienda implementar los cambios propuestos en el presente informe."


REGLAS ADICIONALES

- No proponer cambios drásticos ni ventas innecesarias.
- No cambiar el perfil de riesgo del cliente.
- Enfocar las recomendaciones en arquitectura del portafolio.
- Evitar tecnicismos complejos.
- No usar lenguaje comercial ni promocional.
- No usar expresiones de miedo o escenarios catastróficos.
- Cada ineficiencia debe sentirse específica para este cliente, no aplicable a cualquiera.
- Si una ineficiencia tiene un impacto cuantificable (ej. gap de ingresos en USD), mencionarlo.
- NUNCA usar números ni numeraciones en los títulos de sección principales.
- SÍ usar numeración (1, 2, 3) en los títulos de cada ineficiencia.


EJEMPLO DE REFERENCIA

El siguiente ejemplo muestra el estilo, profundidad y estructura esperada.
Úsalo como referencia de redacción y tono.
NO copies su contenido literalmente.
Adapta el análisis al diagnóstico específico del portafolio recibido.

Este ejemplo corresponde a un cliente con objetivo de crecimiento de capital,
sin necesidad de flujo mensual, con exceso de cash y correlación alta entre activos.
Es deliberadamente distinto al perfil típico de cliente peruano concentrado en inmobiliario.

---

<i>Este informe analiza la estructura de tu portafolio y qué tan bien está construido
para cumplir tu objetivo: crecer capital a largo plazo.</i>

<font size="15"><b>Conclusión general del portafolio</b></font>

El portafolio tiene buena calidad institucional y un nivel de riesgo coherente con el perfil.
Los gestores son sólidos y los costos están bien controlados.

El problema es estructural: hay demasiado capital sin trabajar — el bloque de cash supera
el 20% del patrimonio invertible — y los activos que sí están invertidos responden
en gran medida a los mismos drivers económicos. Eso limita la diversificación real
y reduce el crecimiento esperado del patrimonio a largo plazo.

<font size="15"><b>Principales ineficiencias y acciones recomendadas</b></font>

<font size="13"><b>1. Exceso de cash que no trabaja para el objetivo</b></font>

El portafolio mantiene más del 20% en liquidez sin un propósito estratégico claro.
Para un objetivo de crecimiento a largo plazo, ese capital ocioso no protege
ni diversifica — simplemente pierde poder adquisitivo con el tiempo.

<b>Acciones concretas</b>
• Reasignar gradualmente el excedente de cash hacia activos de mayor eficiencia,
  manteniendo solo el colchón de liquidez necesario para contingencias.

<font size="13"><b>2. Alta correlación entre activos</b></font>

Aunque el portafolio tiene múltiples posiciones, la mayoría responde a los mismos
drivers económicos. Ante un shock de mercado, varios activos caerían al mismo tiempo.
La diversificación en número de posiciones no equivale a diversificación real.

<b>Acciones concretas</b>
• Incorporar activos con drivers económicos distintos — especialmente mercados privados
  o estrategias globales descorrelacionadas — al reasignar el cash excedente.

<font size="13"><b>3. Subexposición a motores globales de crecimiento</b></font>

El portafolio está por debajo del rango objetivo en renta variable global.
Para un horizonte de largo plazo, esa subexposición tiene un costo de oportunidad
directo: los motores de crecimiento global no están contribuyendo al patrimonio.

<b>Acciones concretas</b>
• Dirigir los nuevos flujos hacia renta variable global diversificada hasta alcanzar
  el rango objetivo para el perfil.

<font size="15"><b>Mensaje clave</b></font>

El portafolio tiene bases sólidas pero el capital no está trabajando a su máximo potencial.
Cada mes con exceso de cash es un mes de crecimiento que no ocurre.
Los ajustes son graduales y no requieren vender nada — solo dar dirección
a lo que hoy no tiene ninguna. La pregunta no es si hay que ajustar — es cuándo empezar.
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
