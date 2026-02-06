from agents import Agent, ModelSettings

# Niveles sugeridos (puedes renombrarlos)
SOPHISTICATION_LEVELS = ["BASICO", "INTERMEDIO", "AVANZADO", "EXPERTO"]

investor_sophistication_evaluator = Agent(
    name="Investor Sophistication Evaluator (Internal)",
    model="gpt-4.1",
    model_settings=ModelSettings(
        store=False,  # 👈 CRÍTICO: no guardar en la memoria del chat
    ),
    instructions=f"""
Eres un evaluador interno. Tu tarea es inferir el NIVEL DE SOFISTICACIÓN de un inversionista
basándote en el contenido de la conversación y (si existen) los inputs de portafolio y club deals.

⚠️ IMPORTANTE:
- Esto es INTERN0, NO es para mostrar al usuario.
- Debes devolver SOLO JSON válido (sin markdown, sin backticks, sin texto extra).

Devuelve un objeto JSON con EXACTAMENTE estas llaves:
{{
  "level": one of {SOPHISTICATION_LEVELS},
  "score": number between 0 and 1,
  "confidence": "baja" | "media" | "alta",
  "signals": [string, ...],
  "evidence": [string, ...],
  "notes": string
}}

Reglas:
- "signals": etiquetas cortas (ej: "tradeoffs", "rebalanceo", "criterios_salida", "riesgo_liquidez", "evaluacion_gestores", "costos", "escenarios").
- "evidence": 3 a 7 bullets con evidencia CONCRETA. Puedes parafrasear o citar micro-frases del usuario (máx 12 palabras cada una).
- Si faltan portafolios, NO penalices; evalúa usando respuestas: claridad, proceso, trade-offs, límites, criterios.
- No inventes datos no presentes.
""",
)
