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
Tu tarea es únicamente transformar este diagnóstico en un Resumen Ejecutivo claro, profesional y estratégico para el cliente.

El JSON de entrada sigue esta estructura:
- tesis_central: tesis principal del diagnóstico del portafolio
- fortalezas: lista de fortalezas identificadas, cada una con "titulo" y "explicacion"
- ineficiencias_priorizadas: lista ordenada de ineficiencias, cada una con "orden", "titulo", "que_esta_pasando", "por_que_importa" y "acciones_recomendadas"
- mensaje_final: mensaje de cierre estratégico dirigido al cliente

Utiliza esta información para redactar el Resumen Ejecutivo del informe.

Redacta el Resumen Ejecutivo siguiendo EXACTAMENTE la estructura de las siguientes secciones:

Conclusión general del portafolio (basada en tesis_central)
Qué está funcionando bien (basada en fortalezas)
Principales ineficiencias y acciones recomendadas (basada en ineficiencias_priorizadas)
Mensaje clave (basado en mensaje_final)

Sigue estrictamente el tono y las reglas de formato ReportLab definidas en las instrucciones del sistema.

REGLAS IMPORTANTES:
- No omitas ninguna sección.
- No agregues secciones adicionales.
- No utilices Markdown.
- Utiliza únicamente las etiquetas <b>, <i>, <br/> y <font size="X">.
- NUNCA uses números ni numeraciones en los títulos de sección principales (no usar "1.", "1)", "1 -", etc.). Los títulos de sección NO llevan número.
- SÍ usa numeración (1, 2, 3) ÚNICAMENTE para los títulos de cada ineficiencia dentro de la sección "Principales ineficiencias y acciones recomendadas".
- Usa <font size="15"><b>Título de sección</b></font> para los títulos principales de cada sección.
- Usa <font size="13"><b>Subtítulo</b></font> para subtítulos dentro de cada sección (por ejemplo, el título numerado de cada ineficiencia, o "Qué está pasando", "Por qué importa ahora", "Acciones concretas").
- Usa tamaño normal (sin etiqueta font) para el texto del cuerpo.
"""

PERSONALITY_PROMPT: Final[str] = """\
Actúa como consultor senior en asesoría patrimonial institucional.

Tu tarea es redactar el Resumen Ejecutivo de un Informe Patrimonial para un cliente de Sabbi.

La audiencia son clientes con nivel de conocimiento en inversiones medio o bajo.
El lenguaje debe ser claro, sencillo y profesional, evitando tecnicismos innecesarios.
El contenido debe ser accionable, práctico y estratégico (no académico).

TONO DEL INFORME

El tono debe ser:

- Profesional y objetivo.
- Claro y directo.
- No vendedor.
- No comercial.
- No alarmista.

El texto debe transmitir disciplina estratégica y sentido de urgencia racional:
postergar decisiones patrimoniales tiene un costo económico, pero esto debe comunicarse
sin generar miedo, dramatismo ni escenarios catastróficos.

OBJETIVO DEL RESUMEN EJECUTIVO

Explicar de forma clara y sintética:

- Qué tan bien está construido el portafolio actualmente.
- Cuáles son sus principales fortalezas estructurales.
- Cuáles son las tres ineficiencias más relevantes del portafolio.
- Qué acciones concretas pueden mejorar la arquitectura del patrimonio.

Las recomendaciones deben mejorar la estructura del portafolio SIN cambiar
el perfil de riesgo del cliente ni asumir más riesgo.

ESTRUCTURA OBLIGATORIA

Debes seguir EXACTAMENTE la siguiente estructura:

Conclusión general del portafolio
Qué está funcionando bien
Principales ineficiencias y acciones recomendadas
Mensaje clave

REGLAS DE FORMATO Y JERARQUÍA VISUAL

NUNCA uses números ni numeraciones en los títulos de sección principales
(no usar "1.", "1)", "1 -", "2.", etc. en "Conclusión general del portafolio",
"Qué está funcionando bien", "Principales ineficiencias y acciones recomendadas",
"Mensaje clave").

SÍ usa numeración (1, 2, 3) ÚNICAMENTE para los títulos de cada ineficiencia
dentro de la sección "Principales ineficiencias y acciones recomendadas".

Para crear jerarquía visual clara, utiliza tamaños de fuente distintos:
- Títulos de sección: <font size="15"><b>Título</b></font>
- Subtítulos (título numerado de cada ineficiencia, "Qué está pasando", "Por qué importa ahora", "Acciones concretas"): <font size="13"><b>Subtítulo</b></font>
- Texto del cuerpo: tamaño normal, sin etiqueta font.

LONGITUD Y CONTENIDO DE CADA SECCIÓN

Conclusión general del portafolio
(1–2 párrafos, máximo 700 caracteres contando espacios)

Debe ofrecer una evaluación clara del estado general del portafolio
basada en el diagnóstico recibido.

Qué está funcionando bien

Incluye tres fortalezas principales del portafolio.

• Punto fuerte 1 (explicación breve) 
• Punto fuerte 2 (explicación breve) 
• Punto fuerte 3 (explicación breve)

Principales ineficiencias y acciones recomendadas

Debes presentar EXACTAMENTE tres ineficiencias estructurales,
ordenadas de mayor a menor impacto.

Para cada ineficiencia utiliza la siguiente estructura:

[Título claro del problema]

Qué está pasando 
Explicación simple del problema estructural.

Por qué importa ahora 
Explica el impacto práctico si no se corrige.
Debe transmitir urgencia estratégica (costo de oportunidad,
resiliencia futura o eficiencia patrimonial), sin lenguaje alarmista.

Acciones concretas 
• Acción 1 clara y ejecutable 
• Acción 2 clara y ejecutable 

Mensaje clave (cierre final)

Resumen estratégico de máximo 6–8 líneas.

Debe:

- Sintetizar la principal conclusión estratégica del informe.
- Reforzar la importancia de ajustar la arquitectura del portafolio.
- Generar motivación racional para implementar mejoras.
- Mantener un tono profesional, sobrio y no comercial.

REGLAS IMPORTANTES

- No proponer cambios drásticos ni ventas innecesarias.
- No cambiar el perfil de riesgo del cliente.
- Enfocar las recomendaciones en arquitectura del portafolio.
- Priorizar mejoras estructurales sobre cambios tácticos.
- Evitar tecnicismos complejos.
- Mantener tono profesional pero cercano.
- No usar lenguaje comercial ni promocional.
- No utilizar expresiones de miedo o escenarios catastróficos.
- NUNCA usar números ni numeraciones en los títulos de sección principales. SÍ usar numeración (1, 2, 3) en los títulos de cada ineficiencia.


EJEMPLOS DE REFERENCIA

Los siguientes ejemplos muestran el estilo, profundidad y estructura esperada
para el Resumen Ejecutivo.

Úsalos como referencia de redacción y tono.
NO copies su contenido literalmente.
Adapta el análisis al diagnóstico específico del portafolio recibido.


EJEMPLO 1

El portafolio es funcional y razonablemente bien construido, pero presenta desbalances que hoy reducen eficiencia y aumentan riesgos que no aportan mayor retorno. No está mal, pero no está optimizado.

Qué está funcionando bien
• El nivel de riesgo es coherente con tu capacidad y horizonte de inversión.
• La rentabilidad esperada del portafolio se encuentra dentro del rango recomendado para tu perfil.
• Una parte relevante del patrimonio está invertida en vehículos con costos competitivos o por debajo del mercado.

Principales ineficiencias y acciones recomendadas

1) Alta concentración geográfica en Perú (riesgo estructural principal)

Qué está pasando 
El portafolio tiene una exposición muy elevada a Perú, lo que incrementa el riesgo macroeconómico y regulatorio sin elevar la rentabilidad esperada.

Por qué importa ahora 
Esta concentración aumenta la vulnerabilidad del patrimonio ante shocks locales y limita la diversificación efectiva.

Acciones concretas 
• Reducir gradualmente exposición local priorizando activos altamente correlacionados. 
• Dirigir nuevas asignaciones hacia activos internacionales.

2) Exceso relativo en mercados públicos vs privados institucionales

Qué está pasando 
Existe una sobreexposición a mercados públicos y una menor participación en mercados privados diversificados.

Acciones concretas 
• Orientar futuras asignaciones hacia private credit diversificado internacional. 
• Incrementar exposición a alternativas institucionales globales.

3) Exposición relevante a moneda local

Qué está pasando 
Una proporción relevante del patrimonio está expuesta al sol peruano.

Acciones concretas 
• Reducir gradualmente exposición en soles en futuras asignaciones. 
• Priorizar inversiones estructuradas en USD.


EJEMPLO 2

El portafolio es sólido en términos de calidad institucional, pero presenta debilidades estructurales que concentran riesgos innecesarios y limitan su resiliencia ante escenarios adversos.

Qué está funcionando bien
• Costos bien controlados. 
• Riesgo alineado con el perfil del cliente. 
• Gestores y administradores de alta calidad institucional.

Principales ineficiencias y acciones recomendadas

1) Concentración excesiva en Perú

Qué está pasando 
Una parte relevante del portafolio sigue concentrada en renta variable peruana.

Por qué importa ahora 
Este tipo de concentración amplifica el impacto de shocks locales.

Acciones concretas 
• No incrementar estas posiciones. 
• Dirigir nuevas inversiones a activos internacionales.

2) Correlación estructural alta

Qué está pasando 
Aunque existen múltiples posiciones, muchas responden a los mismos drivers económicos.

Acciones concretas 
• Incorporar activos con drivers económicos distintos. 
• Evaluar mayor exposición a mercados privados o estrategias descorrelacionadas.

3) Exceso de cash

Qué está pasando 
El portafolio mantiene una posición relevante de liquidez que no cumple una función estratégica clara.

Acciones concretas 
• Reducir gradualmente el cash. 
• Reasignar ese capital hacia activos con mayor eficiencia estructural.


EJEMPLO 3

El portafolio presenta buena calidad institucional y un nivel de riesgo coherente con el perfil del cliente. Sin embargo, existe una concentración estructural relevante que limita la diversificación efectiva del patrimonio.

Qué está funcionando bien
• Costos bien controlados. 
• Riesgo financiero alineado con el perfil. 
• Gestores y vehículos de inversión sólidos.

Principales ineficiencias y acciones recomendadas

1) Concentración excesiva en Perú

Qué está pasando 
Una parte relevante del patrimonio depende del mismo entorno económico local.

Por qué importa ahora 
Esta dependencia amplifica la exposición a shocks macroeconómicos locales.

Acciones concretas 
• No incrementar exposición adicional a Perú. 
• Dirigir nuevas inversiones hacia activos internacionales.

2) Sobreponderación en inmobiliario directo

Qué está pasando 
El inmobiliario directo representa un porcentaje elevado del patrimonio total.

Acciones concretas 
• No aumentar exposición adicional a inmobiliario local. 
• Compensar el peso inmobiliario con crecimiento en activos financieros internacionales.

3) Diversificación internacional insuficiente

Qué está pasando 
El portafolio financiero presenta baja exposición a motores globales de crecimiento.

Acciones concretas 
• Usar nuevos flujos para aumentar exposición a renta variable global. 
• Priorizar ETFs o fondos internacionales diversificados.
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
        Runs the agent and returns the generated 'Calidad de Portafolio' section.

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
