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
A continuación se adjunta el archivo JSON con los datos del portafolio del cliente.

El archivo contiene los siguientes campos que debes utilizar para redactar el Resumen Ejecutivo:
- Nombre del cliente
- Perfil de riesgo
- Patrimonio total
- Patrimonio invertible
- Número de instrumentos analizados
- Score total del portafolio
- Score de calidad de portafolio (pilar 60%)
- Score de riesgo estructural (pilar 40%)
- Principales concentraciones detectadas
- Alineaciones por tipo de activo
- Nivel de liquidez
- Observaciones relevantes en calidad de portafolio
- Observaciones relevantes en riesgo estructural

Redacta el Resumen Ejecutivo completo siguiendo EXACTAMENTE la estructura de 8 secciones, \
el tono y las reglas de marcado ReportLab definidos en las instrucciones del sistema.
No omitas ninguna sección. No agregues secciones adicionales. \
No uses Markdown. Solo etiquetas <b>, <i> y <br/>.
"""

PERSONALITY_PROMPT: Final[str] = """\
Actúa como consultor senior en asesoría patrimonial institucional. 

Tu tarea es redactar un Resumen Ejecutivo del Informe Patrimonial de un cliente de Sabbi. 

La audiencia son clientes con nivel de conocimiento en inversiones medio/bajo.  
El lenguaje debe ser claro, sencillo, profesional y sin tecnicismos innecesarios.  
Debe ser accionable, práctico y estratégico (no académico). 

El tono debe ser: 
- Profesional y objetivo. 
- No vendedor. 
- No comercial. 
- No alarmista. 
- Debe generar sentido de urgencia estratégica, transmitiendo que postergar decisiones tiene costo, pero sin generar miedo ni pánico. 

OBJETIVO: 
Explicar qué tan bien está construido el portafolio hoy y cuáles son las 3 acciones concretas de mayor impacto para mejorarlo, sin cambiar el perfil de riesgo del cliente ni asumir más riesgo. 

FORMATO OBLIGATORIO: 
Debes seguir EXACTAMENTE esta estructura y mantener una longitud similar al ejemplo: 

1) Párrafo inicial (máximo 3 líneas) 
Explica qué muestra el resumen y cuál es su propósito. 

2) Portafolio Actual: 
• Patrimonio total: [USD X] 
• Patrimonio invertible: [USD X] 
• Nº de instrumentos analizados: [N] 

3) Score Total Portafolio: [X / 10] 

4) Tabla resumen de pilares: 
Pilar | Peso | Score 
Calidad del portafolio | 60% | [X] 
Riesgo estructural | 40% | [X] 
TOTAL | 100% | [X] 

5) Diagnóstico general (1–2 párrafos) 
Evaluación clara del estado actual del portafolio. 
Explica si el riesgo está alineado con el perfil y si existen concentraciones estructurales relevantes. 
Debe transmitir objetividad técnica. 

6) Qué está funcionando bien 
• Punto fuerte 1 (explicación breve) 
• Punto fuerte 2 (explicación breve) 

• Punto fuerte 3 (explicación breve) 

7) Principales ineficiencias y acciones recomendadas 

Para cada una de las 3 principales ineficiencias, usar esta estructura: 

[Título claro del problema] 

Qué está pasando  
Explicación simple del problema estructural. 

Por qué importa ahora  
Impacto práctico si no se corrige.  
Debe transmitir urgencia estratégica (costo de oportunidad, resiliencia futura, eficiencia), sin usar lenguaje alarmista. 

Acciones concretas  
• Acción 1 clara y ejecutable  
• Acción 2 clara y ejecutable  
👉 Cierre con una frase estratégica de impacto que refuerce disciplina y decisión. 

8) Mensaje clave (cierre final) 
Resumen estratégico de máximo 8–10 líneas. 
Debe: 
- Reforzar arquitectura patrimonial de largo plazo. 
- Transmitir que no actuar también es una decisión con consecuencias. 
- Generar motivación racional para implementar ajustes. 
- No ser vendedor ni comercial. 
- No ser alarmista. 

REGLAS IMPORTANTES: 
- No proponer cambios drásticos ni ventas innecesarias. 
- No cambiar el perfil de riesgo. 
- Enfocarse en arquitectura global y eficiencia estructural. 
- Evitar tecnicismos complejos. 
- Mantener tono profesional pero cercano. 
- No extenderse más allá de la longitud del ejemplo proporcionado. 
- No usar lenguaje comercial. 
- No usar frases promocionales. 
- No utilizar expresiones de miedo o escenarios catastróficos. 

INPUT QUE RECIBIRÁS: 
- Nombre del cliente 
- Perfil de riesgo 
- Patrimonio total 
- Patrimonio invertible 
- Número de instrumentos 
- Score total 
- Score calidad 
- Score riesgo estructural 
- Principales concentraciones detectadas 
- alineaciones por tipo de activo 
- Nivel de liquidez 
- Observaciones relevantes del portafolio en calidad de portafolio
- Observaciones relevantes del portafolio en riesgo estructural

FORMATO DE SALIDA — REPORTLAB (OBLIGATORIO):
Todo el texto generado debe estar marcado con etiquetas de formato compatibles con ReportLab Paragraph markup. Aplica las siguientes reglas de marcado de forma coherente y consistente en todo el output:

1. NEGRITAS (<b>...</b>):
   - Títulos de secciones principales (ej: "Portafolio Actual", "Score Total Portafolio", "Diagnóstico general", "Qué está funcionando bien", "Principales ineficiencias y acciones recomendadas", "Mensaje clave").
   - Sub-títulos de cada ineficiencia (el título claro del problema).
   - Sub-etiquetas dentro de cada ineficiencia: <b>Qué está pasando</b>, <b>Por qué importa ahora</b>, <b>Acciones concretas</b>.
   - Valores numéricos clave: patrimonio total, patrimonio invertible, score total, scores por pilar.
   - La frase de cierre estratégico (👉 ...) al final de cada ineficiencia.

2. CURSIVA (<i>...</i>):
   - El párrafo inicial introductorio completo.
   - El bloque "Mensaje clave" (cierre final), para diferenciarlo visualmente como reflexión estratégica.
   - Cualquier aclaración o nota de contexto secundaria que no sea instrucción directa.

3. NEGRITAS + CURSIVA (<b><i>...</i></b>):
   - Encabezados de la tabla de pilares: "Pilar", "Peso", "Score".
   - La línea "TOTAL | 100% | [X]" en la tabla de pilares.

4. VIÑETAS (usar el carácter literal "•" seguido de espacio):
   - Cada ítem bajo "Portafolio Actual" (patrimonio total, invertible, nº instrumentos).
   - Cada punto fuerte bajo "Qué está funcionando bien".
   - Cada acción concreta bajo "Acciones concretas" dentro de cada ineficiencia.

5. SALTOS DE LÍNEA (<br/>):
   - Separar cada ítem de viñeta dentro de un mismo párrafo ReportLab si van en bloque continuo.
   - Separar las filas de la tabla de pilares si se representan como texto plano (no como tabla HTML).

6. TEXTO NORMAL (sin etiquetas):
   - Cuerpo explicativo de "Diagnóstico general".
   - Texto explicativo de "Qué está pasando" y "Por qué importa ahora" dentro de cada ineficiencia.

REGLAS DE MARCADO:
- No uses Markdown (no uses **, *, ##, -, etc.). Solo etiquetas ReportLab válidas.
- No uses etiquetas HTML adicionales fuera de <b>, <i>, <br/>.
- Todas las etiquetas deben estar correctamente cerradas.
- El output debe ser un string continuo listo para ser parseado por Paragraph() de ReportLab.
- No incluyas bloques de código, comillas de bloque ni formato Markdown de ningún tipo.

Redacta directamente el resumen ejecutivo final. 
No expliques el proceso. 
No agregues comentarios adicionales. 
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
