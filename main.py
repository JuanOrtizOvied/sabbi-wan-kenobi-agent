import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai.types.shared.reasoning import Reasoning

from agents import Agent, ModelSettings, Runner, RunConfig, trace
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

# Depending on your Agents SDK version, this import can vary:
try:
    from agents import TResponseInputItem
except Exception:
    from agents.items import TResponseInputItem

load_dotenv()

# -----------------------------
# ENV
# -----------------------------
SESSION_DB_URL = os.getenv("SESSION_DB_URL", "postgresql+asyncpg://admin:pass@localhost:5432/ai-experiments")
CREATE_SESSION_TABLES = os.getenv("CREATE_SESSION_TABLES", "1").strip().lower() in ("1", "true", "yes")

SESSIONS_TABLE = os.getenv("AGENT_SESSIONS_TABLE", "agent_sessions")
MESSAGES_TABLE = os.getenv("AGENT_MESSAGES_TABLE", "agent_messages")


# -----------------------------
# Agent (all behavior in prompt)
# -----------------------------
filosofia_de_inversion = Agent(
    name="Filosofía de Inversión — WOW",
    model="gpt-5.1",
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="high", summary="auto"),
    ),
    instructions=(
        "Eres un experto creando una **FILOSOFÍA DE INVERSIÓN personalizada con efecto “WOW”**.\n"
        "Tu rol es cuestionar, interpretar y destilar el pensamiento real del inversionista a partir de su portafolio y sus respuestas, "
        "y luego transformarlo en una filosofía clara, profunda y accionable.\n\n"
        "## 📥 INSUMOS DISPONIBLES (YA EN CONTEXTO)\n"
        "1) portafolio_inversionista (JSON)\n"
        "2) portafolio_promedio (JSON)\n"
        "3) mi_filosofia (texto libre del inversionista)\n"
        "4) club_deals_information (definición y racional)\n\n"
        "## 🎯 OBJETIVO\n"
        "Crear una **Filosofía de Inversión WOW**, coherente y justificable, alineada a los insumos, que refleje:\n"
        "- cómo piensa realmente el inversionista\n"
        "- su nivel de sofisticación (inferido, no declarado)\n"
        "- sus convicciones, sesgos y criterios de decisión\n"
        "- el rol estratégico de cada tipo de activo\n\n"
        "## 🧠 REGLAS DE INTERACCIÓN (OBLIGATORIAS)\n"
        "### 1) PREGUNTAS OBLIGATORIAS (POR RONDA)\n"
        "Antes de generar la filosofía final, debes hacer **exactamente 4 preguntas** en una primera ronda.\n"
        "- Las preguntas deben **cuestionar directamente el portafolio del inversionista**.\n"
        "- No puedes generar la filosofía sin haber hecho y recibido respuesta a estas 4 preguntas.\n\n"
        "### 2) FORMATO DE PREGUNTAS\n"
        "- Las preguntas se hacen **DE UNA EN UNA**.\n"
        "- En cada respuesta tuya, entrega **SOLO 1 pregunta y nada más**.\n"
        "- Prohibido: listas, dobles preguntas, preámbulos, explicaciones, diagnósticos o resúmenes antes de terminar la ronda.\n\n"
        "### 3) ORDEN Y COBERTURA DE LAS 4 PREGUNTAS\n"
        "Las 4 preguntas, en conjunto, deben cubrir:\n"
        "1. **Convicción central**: qué principio guía su asignación actual.\n"
        "2. **Decisión reveladora**: qué parte del portafolio refleja mayor convicción y cuál le genera duda/tensión.\n"
        "3. **Diagnóstico de sofisticación** (sin preguntarlo explícitamente): cómo evalúa riesgo, managers, rebalanceos o salidas.\n"
        "4. **Gestión de tensiones / incoherencias**: si hay contradicciones entre discurso y portafolio, prioriza resolverlas; si no, profundiza disciplina y reglas.\n\n"
        "### 4) CONTRADICCIONES\n"
        "- Si detectas contradicciones entre mi_filosofia y portafolio_inversionista, o frente al portafolio_promedio, "
        "debes priorizarlas en la **siguiente pregunta disponible**.\n"
        "- No inventes contradicciones.\n\n"
        "### 5) SOFISTICACIÓN (INFERIDA, NO DECLARADA)\n"
        "- NO preguntes escalas ni 'qué tan avanzado eres'.\n"
        "- Debes **inferir** el nivel de sofisticación por la calidad de sus respuestas: claridad, profundidad, trade-offs, "
        "lenguaje, entendimiento de riesgo/liquidez/ciclos, criterios de manager selection, etc.\n"
        "- Ajusta el nivel de tecnicismo y profundidad del output final según lo inferido.\n\n"
        "### 6) GATE PARA CONTINUAR AFINANDO (OBLIGATORIO)\n"
        "Después de la 4ta pregunta y su respuesta, NO generes aún la filosofía. "
        "Debes hacer una pregunta de gate (y solo esa pregunta) para decidir si se afina más o se genera:\n"
        "'¿Quieres que haga 4 preguntas más para afinar tu filosofía de inversión o prefieres que ya la genere?'\n"
        "Si el usuario responde que quiere seguir afinando (ej. 'continuemos', 'más', 'afinar'), inicias otra ronda de **exactamente 4 preguntas** "
        "(una por mensaje) siguiendo las mismas reglas.\n"
        "Si el usuario responde que ya está listo (ej. 'genera', 'listo', 'ya'), entonces generas la filosofía final.\n\n"
        "## ✨ CUANDO GENERES LA FILOSOFÍA FINAL (WOW)\n"
        "Entrega la filosofía con esta estructura obligatoria:\n"
        "## PRINCIPIOS FUNDAMENTALES\n"
        "## OBJETIVOS DE INVERSIÓN\n"
        "## ESTRATEGIA / METODOLOGÍA\n"
        "## GESTIÓN DEL RIESGO\n"
        "## DISCIPLINA Y SESGOS\n"
        "## REFLEXIÓN FINAL\n\n"
        "### 🌟 REQUISITOS WOW\n"
        "- Debe sentirse **profundamente personalizada**: usa detalles concretos de portafolio_inversionista y mi_filosofia.\n"
        "- Compara brevemente vs portafolio_promedio y explica qué diferencias son intencionales vs qué ajustes conceptuales se justifican.\n"
        "- Define y justifica el rol de **Club Deals** usando club_deals_information.\n"
        "- Incluye **3–5 reglas accionables** (criterio + gatillo + límite + qué monitorear).\n"
        "- Si hubo contradicciones, explícitalas y muestra cómo se resolvieron o qué supuesto se tomó.\n"
        "- Mantén un tono sofisticado, claro y narrativo; evita sonar académico o genérico.\n\n"
        "## 🚫 RESTRICCIONES\n"
        "- No recomendar productos.\n"
        "- No sugerir compra/venta.\n"
        "- No usar jerga innecesaria.\n"
        "- No mencionar porcentajes en el texto final (puedes razonar internamente con ellos, pero no mostrarlos).\n\n"
        "## 🟢 FLUJO\n"
        "1) Haz la pregunta 1 (solo la pregunta).\n"
        "2) Con la respuesta, haz la pregunta 2 (solo la pregunta).\n"
        "3) Con la respuesta, haz la pregunta 3 (solo la pregunta).\n"
        "4) Con la respuesta, haz la pregunta 4 (solo la pregunta).\n"
        "5) Haz el gate: '¿Quieres que haga 4 preguntas más para afinar tu filosofía de inversión o prefieres que ya la genere?'\n"
        "6) Si 'afinar' → repite ronda de 4 preguntas y vuelve al gate.\n"
        "7) Si 'generar' → entrega Filosofía WOW completa.\n"
    ),
)


# -----------------------------
# API Models
# -----------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique id per user/conversation thread")
    message: str = Field(..., min_length=1, description="User message (one turn)")


class ChatResponse(BaseModel):
    session_id: str
    output_text: str


# -----------------------------
# Service
# -----------------------------
def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


class AgentService:
    """
    Owns:
    - one SQLAlchemy engine (shared)
    - session creation per request (by session_id)
    - seeding the initial context ONCE per session
    """

    def __init__(self) -> None:
        self._bootstrap: Optional[SQLAlchemySession] = None
        self._engine = None

        # TODO: Replace placeholders with your real objects/strings
        self.state: Dict[str, Any] = {
            "custom_input": {
                "mi_filosofia": "En mi filosofía de inversión, por ejemplo La primera división que yo hago de asset allocation es en propiedades, mercados públicos, mercados privados y esta especie de club deals, que son inversiones en las cuales yo tengo cierto conocimiento peculiar y específico que teóricamente me permiten tomar una decisión de mejor riesgo/retorno.       Después, evidentemente, dentro de cada categoría, el criterio de selección es… o sea, los motivos para tenerlos en el portafolio son distintos. Por ejemplo, en el caso de las propiedades creo que son una buena protección contra la inflación. Yo particularmente, como trabajo digamos en el mundo de Real Estate, tengo cierto conocimiento sobre algún tipo de Real Estate específico para poder tomar decisiones directas con mayor conocimiento que me deberían dar una buena relación de riesgo/retorno.       Si, por ejemplo, en mi portafolio considero que hoy por hoy está sobreexpuesto a este asset class y es un asset class que me gustaría disminuir.       Después, la segunda categoría que son mercados públicos. Evidentemente, los mercados públicos tienen un componente de volatilidad alta pero son una parte importantísima de cualquier cartera de inversión.       Mi approach a invertir en mercados públicos es que no hay digamos mucho valor generalmente en el management activo, entonces quiero que la mayoría del portafolio se componga de ETFs. Yo no me considero un experto en mercados públicos como para realmente poder conseguir un alfa, de manera que prefiero esta exposición pasiva.       Mi forma de decidir en mercados públicos es básicamente decir qué tan expuesto o no estoy al asset class como un todo. Es decir, concretamente ahora, digamos estoy con una exposición bastante baja en mercados públicos porque siento que sí estamos en una digamos en la fase final tal vez de un ciclo de crecimiento bastante importante.       Entonces, mi idea sí sería aumentar mi exposición en mercados públicos una vez que esta fase, una vez que el mercado tenga una corrección. O sea, yo sí a pesar de que creo digamos en todo el potencial de la inteligencia artificial, sí creo que hay ciertos factores sistémicos que hacen que las valuaciones hoy por hoy que puedan demandar las empresas en mercados públicos sean un poco mayores a las históricas.       Sí creo que igual estamos un poco demasiado altos, por lo que por lo que estoy siendo relativamente conservador con mi exposición a mercados públicos. Ya entro en mercados públicos y tengo una location que es prioritariamente renta variable y menos renta fija. Obviamente, porque mi horizonte de inversión es de largo plazo y no necesito este capital en el corto plazo.       ¿Qué más? Ah, sí, evidentemente también trato de hoy por hoy estar un poco más enfocado en mercados ex, o sea, fuera de Estados Unidos, porque siento que este efecto de la valorización exagerada también se da principalmente en Estados Unidos.      Ya después, sobre cómo hacer el asset allocation y posteriormente el security selection ya en mercados privados.       Yo básicamente ahí trato de decidir un asset allocation que me haga sentido, nuevamente priorizando crecimiento a largo plazo. Me enfoco más en asset classes como private equity (tal vez un poco de venture capital). Bueno, algo de deuda privada porque, en realidad, digamos en los últimos años las rentas han estado o sea los retornos de deuda privada han estado súper interesantes.       De manera que la prima, digamos de retorno que te da el private equity, desde mi punto de vista, no se justificaba tanto con retornos de deuda de alrededor del 10%. Yo creo que eso está cambiando y creo que también en el corto o mediano plazo voy a ir disminuyendo un poco mi exposición de renta y deuda versus equity, para llegar igual a un 80-20, que es parecido a lo que tengo en mercados públicos.       Después creo que real estate e infraestructura te dan este componente de estabilidad y flujos y protección contra la inflación. Inversión en activos reales no, que me parece también súper valioso como parte de darle estabilidad a un portafolio y con un efecto parecido de hedge funds que te otorgan baja correlación con en general con el componente de mercados públicos.       Entonces, tal vez digamos mi componente de hedge funds aumente un poco también si aumento mi exposición a mercados públicos. Si evidentemente dentro de mercados privados es importantísimo el manager selection, el asset, la selección del manager.       Entonces, ahí dado que tampoco me considero un súper experto en mangers, trato principalmente de elegir a los mejores managers y managers ya probados dentro del mercado. No estoy buscando realmente el alfa extraordinario, escoger al manager chico que es un crack.       Creo que hay un montón de valor en eso, pero creo que yo no tengo la capacidad actualmente. Me gusta ir con los managers grandes y probados que probablemente estén en el segundo cuartil superior o tal vez algunos en el cuartil superior, pero claramente no en el decíl superior.       Otra cosa, o el punto final que me gustaría aclarar es sobre los Club Deals.       Yo creo que los Club Deals tienen un lugar importante en el portafolio, pero un Club Deal es inversión, digamos, en una inversión específica o un manager pequeño, en donde tienes calificación para tomar una buena decisión riesgo-retorno, que no está disponible, digamos, en los mercados públicos, ni siquiera en los grandes managers de mercados privados.       En mi caso, digamos, yo al gestionar fondos de Real Estate y conocer bastante sobre un nicho específico de Real Estate Perú, no me siento capacitado para tomar ciertas de esas decisiones en ese nicho específico. Pero, digamos, también me puedo apalancar de personas en las que confío que son expertas en algún otro mercado o un nicho específico que tienen ese mismo conocimiento.       Lo veo como no lo veo como mercados privados per se, lo veo en otra categoría porque acá, teóricamente, la inversión se basa más en lo que conoces de esta situación específica. Lo veo descorrosionado incluso con mercados privados, y ese es el factor que me gusta. Si le meto creo que cuánto de tu portafolio le metes a este tipo de inversiones depende justamente de qué tan confiado te sientes de que esa inversión va a tener un retorno extraordinario.       Evidentemente, este tipo de inversiones para mí tienen que comandar un retorno más alto. O sea, tienen que comandar un retorno más alto porque, si hacer un manager pequeño generalmente implica mucha concentración, digamos, un cierto riesgo riesgo oculto mayor; no, entonces, definitivamente tiene que comandar un retorno más alto.       Pero sí creo que ocupan una parte importante dentro de mi portafolio de inversión. ",
                "club_deals_information": "# 📌 **¿Qué son los Club Deals? (Definición Integrada)**  Los **Club Deals** son inversiones en las que **un grupo reducido de inversionistas participa directamente en una oportunidad privada específica**, en lugar de invertir en un fondo grande y diversificado como los de Blackstone u otros gestores globales.  A diferencia de los fondos tradicionales, donde existen muchas capas de intermediación y gestión, en un Club Deal los inversionistas suelen estar **más cerca del gestor** y de la operación en sí, lo que reduce comisiones y permite mayor visibilidad del proyecto.  Los Club Deals se encuentran principalmente en **mercados privados**, y pueden pertenecer a distintas categorías:  * **Real Estate – Club Deals:** proyectos inmobiliarios específicos (como desarrollos tipo Edifica). * **Deuda Privada – Club Deals:** financiamiento directo a empresas u operaciones estructuradas. * **Otros – Club Deals:** oportunidades privadas en sectores como energía, agricultura, infraestructura, venture capital, o estrategias especiales.  ---  # 🎯 **¿Por qué se usan en los portafolios? (visión del CEO + técnico)**  Los Club Deals ofrecen:  ### **1. Diversificación y descorrelación**  No se mueven igual que los mercados públicos (bolsa), ni siquiera igual que algunos fondos privados tradicionales. Esto ayuda a mejorar la estabilidad del portafolio.  ### **2. Mayor cercanía al gestor**  Al ser vehículos más pequeños, el inversionista está más cerca de quien ejecuta la estrategia. Esto implica:  * mayor visibilidad del proyecto * menos capas de comisiones * alineación más directa entre gestor e inversionista  ### **3. Mejor potencial riesgo–retorno**  Al tener acceso directo a una transacción puntual —y no a un fondo enorme y genérico— el inversionista puede:  * entender mejor la operación * evaluar riesgos con más claridad * capturar retornos más altos por asumir riesgos específicos  ### **4. Oportunidades que no existen en fondos grandes**  Algunos proyectos pequeños o medianos (como los inmobiliarios locales tipo Edifica) **no califican para fondos globales masivos**. Los Club Deals permiten entrar en ese tipo de oportunidades que grandes gestores no consideran, pero que pueden ser atractivas y rentables.  ---  # 🧾 **Resumen integrado**  | Tema                              | Descripción                                                                                                   | | --------------------------------- | ------------------------------------------------------------------------------------------------------------- | | **Qué son**                       | Inversiones privadas en las que pocos inversionistas participan directamente en un proyecto o transacción.    | | **Por qué no son fondos grandes** | Son vehículos pequeños, con gestores más cercanos, menos capas y costos más bajos.                            | | **Ventajas**                      | Descorrelación, acceso directo, comisiones más bajas, mejor entendimiento del riesgo y potencial de retornos. | | **Ejemplos**                      | Edifica, proyectos inmobiliarios específicos, préstamos privados, energía, infraestructura, VC, etc.          |  ---",
                "portafolio_promedio": {
                    "PROPIEDADES_DIRECTAS": {
                        "data": [
                            {
                                "name": "Empresas",
                                "percentage": 0.00
                            },
                            {
                                "name": "Prop Peru Residencial",
                                "percentage": 12.05
                            },
                            {
                                "name": "Prop Peru Oficinas",
                                "percentage": 4.74
                            },
                            {
                                "name": "Prop Peru Comercial/Indus.",
                                "percentage": 7.51
                            },
                            {
                                "name": "Prop Extranjero",
                                "percentage": 4.71
                            }
                        ],
                        "subtotal": 29.02
                    },
                    "ALTERNATIVES": {
                        "data": [
                            {
                                "name": "Private Credit",
                                "percentage": 11.17
                            },
                            {
                                "name": "Private Equity",
                                "percentage": 5.51
                            },
                            {
                                "name": "Venture Capital",
                                "percentage": 0.56
                            },
                            {
                                "name": "Real Estate",
                                "percentage": 1.97
                            },
                            {
                                "name": "Hedge Funds",
                                "percentage": 2.51
                            },
                            {
                                "name": "Infrastructure",
                                "percentage": 1.16
                            }
                        ],
                        "subtotal": 22.89
                    },
                    "CLUB_DEALS": {
                        "data": [
                            {
                                "name": "Real Estate - Club Deals",
                                "percentage": 4.22
                            },
                            {
                                "name": "Deuda Privada - Club Deals",
                                "percentage": 5.60
                            },
                            {
                                "name": "Otros - Club Deals",
                                "percentage": 0.40
                            }
                        ],
                        "subtotal": 10.23
                    },
                    "MERCADOS_PUBLICOS": {
                        "RENTA_VARIABLE": {
                            "data": [
                                {
                                    "name": "US Large Cap",
                                    "percentage": 13.20
                                },
                                {
                                    "name": "US Mid and Small Cap",
                                    "percentage": 2.13
                                },
                                {
                                    "name": "Mercados Desarrollados (ex US)",
                                    "percentage": 3.58
                                },
                                {
                                    "name": "Mercados Emergentes (ex Peru)",
                                    "percentage": 0.75
                                },
                                {
                                    "name": "Perú",
                                    "percentage": 1.97
                                }
                            ],
                            "subtotal": 21.62
                        },
                        "RENTA_FIJA": {
                            "data": [
                                {
                                    "name": "US Treasuries (Bonos del Tesoro de US)",
                                    "percentage": 1.00
                                },
                                {
                                    "name": "Bonos Corporativos Investment Grade (AAA–BBB)",
                                    "percentage": 2.38
                                },
                                {
                                    "name": "Bonos High Yield (BB o menor)",
                                    "percentage": 2.10
                                },
                                {
                                    "name": "Bonos de Mercados Emergentes",
                                    "percentage": 0.65
                                },
                                {
                                    "name": "Bonos Latinoamérica",
                                    "percentage": 1.48
                                },
                                {
                                    "name": "Bonos Perú",
                                    "percentage": 1.38
                                }
                            ],
                            "subtotal": 8.98
                        },
                        "subtotal_general": 30.60
                    },
                    "OTROS_ACTIVOS": {
                        "data": [
                            {
                                "name": "Cripto",
                                "percentage": 0.46
                            },
                            {
                                "name": "Commodities",
                                "percentage": 0.84
                            }
                        ],
                        "subtotal": 1.30
                    },
                    "CASH": {
                        "data": [
                            {
                                "name": "Cash",
                                "percentage": 5.97
                            }
                        ],
                        "subtotal": 5.97
                    },
                    "TOTAL": 100.00
                },
                "portafolio_inversionista": {
                    "cliente": "MARTIN BEDOYA",
                    "portfolio": [
                        {
                            "asset_class": "PROPIEDADES DIRECTAS",
                            "percentage": 19.85,
                            "data": [
                                {
                                    "name": "Empresas",
                                    "percentage": 0
                                },
                                {
                                    "name": "Propiedades Peru Residencial",
                                    "percentage": 12.2
                                },
                                {
                                    "name": "Propiedades Peru Oficinas",
                                    "percentage": 7.65
                                },
                                {
                                    "name": "Propiedades Peru Comercial/Indus.",
                                    "percentage": 0
                                },
                                {
                                    "name": "Propiedades Extranjero",
                                    "percentage": 0
                                }
                            ]
                        },
                        {
                            "asset_class": "ALTERNATIVES",
                            "percentage": 40.34,
                            "data": [
                                {
                                    "name": "Private Credit",
                                    "percentage": 11.41
                                },
                                {
                                    "name": "Private Equity",
                                    "percentage": 13.28
                                },
                                {
                                    "name": "Venture Capital",
                                    "percentage": 4.58
                                },
                                {
                                    "name": "Real Estate",
                                    "percentage": 4.07
                                },
                                {
                                    "name": "Hedge Funds",
                                    "percentage": 3.52
                                },
                                {
                                    "name": "Infrastructure",
                                    "percentage": 3.48
                                }
                            ]
                        },
                        {
                            "asset_class": "CLUB DEALS",
                            "percentage": 12.78,
                            "data": [
                                {
                                    "name": "Real Estate - Club Deals",
                                    "percentage": 6.99
                                },
                                {
                                    "name": "Deuda Privada - Club Deals",
                                    "percentage": 2.91
                                },
                                {
                                    "name": "Otros - Club Deals",
                                    "percentage": 2.88
                                }
                            ]
                        },
                        {
                            "asset_class": "MERCADOS PUBLICOS",
                            "percentage": 17.81,
                            "subcategories": [
                                {
                                    "asset_class": "RENTA VARIABLE - Mercados Públicos",
                                    "percentage": 14.73,
                                    "data": [
                                        {
                                            "name": "US Large Cap",
                                            "percentage": 5.79
                                        },
                                        {
                                            "name": "US Mid and Small Cap",
                                            "percentage": 0.33
                                        },
                                        {
                                            "name": "Mercados Desarrollados (ex US)",
                                            "percentage": 4.14
                                        },
                                        {
                                            "name": "Mercados Emergentes (ex Perú)",
                                            "percentage": 4.48
                                        },
                                        {
                                            "name": "Perú",
                                            "percentage": 0
                                        }
                                    ]
                                },
                                {
                                    "asset_class": "RENTA FIJA - Mercados Públicos",
                                    "percentage": 3.08,
                                    "data": [
                                        {
                                            "name": "US Treasuries (Bonos del Tesoro de US)",
                                            "percentage": 0.94
                                        },
                                        {
                                            "name": "Bonos Corporativos Investment Grade (AAA-BBB)",
                                            "percentage": 1.49
                                        },
                                        {
                                            "name": "Bonos High Yield (BB o menor)",
                                            "percentage": 0.49
                                        },
                                        {
                                            "name": "Bonos de Mercados Emergentes",
                                            "percentage": 0.16
                                        },
                                        {
                                            "name": "Bonos Latinoamérica",
                                            "percentage": 0
                                        },
                                        {
                                            "name": "Bonos Perú",
                                            "percentage": 0
                                        }
                                    ]
                                }
                            ]
                        },
                        {
                            "asset_class": "OTROS",
                            "percentage": 2.97,
                            "data": [
                                {
                                    "name": "Cripto",
                                    "percentage": 2.79
                                },
                                {
                                    "name": "Commodities",
                                    "percentage": 0.18
                                }
                            ]
                        },
                        {
                            "asset_class": "CASH",
                            "percentage": 6.25,
                            "data": [
                                {
                                    "name": "Cash",
                                    "percentage": 6.25
                                }
                            ]
                        },
                        {
                            "asset_class": "TOTAL GENERAL",
                            "percentage": 100,
                            "data": [
                                {
                                    "name": "Total",
                                    "percentage": 100
                                }
                            ]
                        }
                    ]
                }
            }
        }

    async def startup(self) -> None:
        # Create a single engine for the whole process
        self._bootstrap = SQLAlchemySession.from_url(
            "bootstrap",
            url=SESSION_DB_URL,
            create_tables=CREATE_SESSION_TABLES,
            sessions_table=SESSIONS_TABLE,
            messages_table=MESSAGES_TABLE,
            engine_kwargs={"pool_pre_ping": True},
        )
        self._engine = self._bootstrap.engine

        # Force table creation if enabled
        if CREATE_SESSION_TABLES:
            await self._bootstrap.get_items(limit=1)

    def _make_session(self, session_id: str) -> SQLAlchemySession:
        if self._engine is None:
            raise RuntimeError("AgentService not initialized (engine missing). Did startup run?")
        return SQLAlchemySession(
            session_id=session_id,
            engine=self._engine,
            create_tables=False,
            sessions_table=SESSIONS_TABLE,
            messages_table=MESSAGES_TABLE,
        )

    def _context_items(self) -> List[TResponseInputItem]:
        ci = self.state["custom_input"]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "CONTEXTO — portafolio_promedio (JSON):\n" + _pretty(ci["portafolio_promedio"])},
                    {"type": "input_text", "text": "CONTEXTO — portafolio_inversionista (JSON):\n" + _pretty(ci["portafolio_inversionista"])},
                    {"type": "input_text", "text": "CONTEXTO — mi_filosofia (texto):\n" + ci["mi_filosofia"]},
                    {"type": "input_text", "text": "CONTEXTO — club_deals_information:\n" + ci["club_deals_information"]},
                ],
            }
        ]

    async def _seed_context_if_empty(self, session: SQLAlchemySession) -> None:
        existing = await session.get_items(limit=1)
        if not existing:
            await session.add_items(self._context_items())  # OK

    async def chat(self, session_id: str, message: str) -> str:
        session = self._make_session(session_id)
        await self._seed_context_if_empty(session)

        with trace("Filosofia WOW (FastAPI)"):
            result = await Runner.run(
                filosofia_de_inversion,
                message,  # ✅ string input (NOT a list)
                session=session,  # ✅ session memory enabled
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "fastapi-service",
                        "workflow_id": "wf_693a02d72190819097a8a7b5234510f70851015287e3b178",
                    }
                ),
            )

        return result.final_output_as(str)


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Sabbi WOW Philosophy Agent")
service = AgentService()


@app.on_event("startup")
async def on_startup():
    await service.startup()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        output_text = await service.chat(session_id=req.session_id, message=req.message)
        return ChatResponse(session_id=req.session_id, output_text=output_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
