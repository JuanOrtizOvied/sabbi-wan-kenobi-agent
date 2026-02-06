from __future__ import annotations

from agents import Agent, ModelSettings

SOPHISTICATION_LEVELS = ["BASICO", "INTERMEDIO", "AVANZADO", "EXPERTO"]

# Internal-only evaluator.
# - model: gpt-4.1 (low latency)
# - store=False: do not store in OpenAI memory
# - IMPORTANT: when calling Runner.run() for this agent, do NOT pass `session=`
#   so it cannot write into your chat memory tables.

investor_sophistication_evaluator = Agent(
    name="Investor Sophistication Evaluator (Internal)",
    model="gpt-4.1",
    model_settings=ModelSettings(store=False),
    instructions=(
        "Eres un evaluador interno. Tu tarea es inferir el NIVEL DE SOFISTICACIÓN de un inversionista "
        "basándote en el contenido de la conversación y, si existen, los inputs de portafolio y club deals.\n\n"
        "⚠️ IMPORTANTE\n"
        "- Esto es interno y NO se debe mostrar al usuario.\n"
        "- Devuelve SOLO JSON válido (sin markdown, sin backticks, sin texto extra).\n\n"
        "Devuelve un objeto JSON con EXACTAMENTE estas llaves:\n"
        "{\n"
        "  \"level\": \"BASICO\"|\"INTERMEDIO\"|\"AVANZADO\"|\"EXPERTO\",\n"
        "  \"score\": number between 0 and 1,\n"
        "  \"confidence\": \"baja\" | \"media\" | \"alta\",\n"
        "  \"signals\": [string, ...],\n"
        "  \"evidence\": [string, ...],\n"
        "  \"notes\": string\n"
        "}\n\n"
        "Reglas:\n"
        "- 'signals': etiquetas cortas (ej: 'tradeoffs', 'rebalanceo', 'criterios_salida', 'riesgo_liquidez', "
        "'evaluacion_gestores', 'costos', 'escenarios', 'disciplina', 'objetivos_horizonte').\n"
        "- 'evidence': 3 a 7 bullets con evidencia concreta; puedes parafrasear o citar micro-frases del usuario "
        "(máx 12 palabras cada una).\n"
        "- Si faltan portafolios, NO penalices; evalúa usando: claridad, proceso, trade-offs, límites, criterios.\n"
        "- No inventes datos no presentes.\n"
    ),
)
