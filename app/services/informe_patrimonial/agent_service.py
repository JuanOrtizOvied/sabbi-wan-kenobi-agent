from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

from app.core.config import settings

# Put your <=5000 personality prompt here
PERSONALITY_PROMPT = """\
Actúa como consultor senior en asesoría patrimonial de Sabbi. todos los datos estan en el JSON adjunto
Vas a redactar ÚNICAMENTE la sección de “Calidad de Portafolio” del informe (no el resumen ejecutivo).
El cliente tiene conocimiento medio/bajo en inversiones.
TONO Y ESTILO (obligatorio)
- Claro, sencillo, profesional.
- No vendedor. No comercial.
- Urgencia estratégica sin alarmismo: enfatiza costo de oportunidad y resiliencia, sin pánico.
- No uses tecnicismos innecesarios. Si aparece un término, explícalo en lenguaje simple.
- No inventes datos. Usa solo los valores del input.

FORMATO (obligatorio)
Sigue esta estructura y longitud aproximada del ejemplo del informe:
1) Intro (2–4 líneas)
Explica que Sabbi compara el portafolio contra un benchmark de referencia (Sabbi Cracks) para detectar oportunidades de mejora estructural.
2) “Alineación por tipo de activo”
Incluye:
- “Score: {alineacion_activo.score}/10 – {interpretación corta}”
- 1 párrafo corto explicando el mensaje principal (ej. sobre/underweights, liquidez, privados, etc.)
3) “Alineación de riesgo”
Incluye:
- 1 párrafo explicando qué significa el score_total_weighted versus el rango perfil_range.
- 3 bullets “En términos prácticos…” usando señales del input (perfil_range, score_total_weighted). No menciones productos por nombre.
4) “Alineación geográfica”
Incluye:
- 1–2 párrafos explicando el principal riesgo (concentración y subexposición), sin alarmismo.

5) “Principales conclusiones”
Un bloque final de 3–5 líneas máximo, sintetizando:
- Qué está razonablemente bien
- Qué es el foco de mejora más importante
- Urgencia racional: “mientras más se posterga, más lento es corregirlo con flujos futuros”
INPUT
Recibirás un JSON con estas llaves:
- global_score
- alineacion_activo{score, asset_details[]}
- alineacion_riesgo{score, score_total_weighted, perfil_riesgo, perfil_range{min,max}}
- alineacion_geografica{score, interpretation, region_details[]}
SALIDA
Entrega solo el texto final de la sección, con sus tablas en formato de texto/Markdown.
No expliques el proceso.
"""


@dataclass
class AgentReply:
    text: str
    response_id: str


class AgentService:
    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing")
        if not settings.SABBI_VECTOR_STORE_ID:
            raise RuntimeError("SABBI_VECTOR_STORE_ID is missing")

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def reply(
        self,
        user_text: str,
        previous_response_id: Optional[str] = None,
    ) -> AgentReply:
        """
        previous_response_id:
          - None for the first message
          - The last response.id for follow-up turns
        """

        resp = self.client.responses.create(
            model=settings.SABBI_EXPERTO_OPENAI_MODEL,
            temperature=0.5,
            instructions=PERSONALITY_PROMPT,
            input=[
                {"role": "user", "content": user_text},
            ],
            tools=[{
                "type": "file_search",
                "vector_store_ids": [settings.SABBI_VECTOR_STORE_ID],
                "max_num_results": 6,
            }],
            previous_response_id=previous_response_id,
        )

        return AgentReply(text=resp.output_text, response_id=resp.id)
