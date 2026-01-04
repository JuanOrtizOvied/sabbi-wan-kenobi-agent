import json
import asyncio
from agents import Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

filosofia_de_inversion = Agent(
  name="filosofia de inversion",
  instructions=(
    "Genera una filosofía de inversión personalizada y coherente usando CUATRO insumos:\n"
    "1) portafolio_inversionista\n"
    "2) portafolio_promedio\n"
    "3) mi_filosofia (texto del inversionista sobre su enfoque)\n"
    "4) club_deals_information (definición y racional de Club Deals)\n\n"
    "Requisitos:\n"
    "- Integra los 4 insumos: explica similitudes y diferencias entre el portafolio del inversionista y el promedio.\n"
    "- Aterriza el texto de mi_filosofia en principios accionables (no solo resumen).\n"
    "- Usa club_deals_information para definir Club Deals y justificar su rol en el portafolio.\n"
    "- Si detectas incoherencias (por ejemplo, lo que dice mi_filosofia vs el portafolio actual), señálalas claramente "
    "y propone una forma de alineación.\n"
    "- Entrega un resultado claro, justificable y accionable.\n\n"
    "Formato sugerido:\n"
    "## PRINCIPIOS FUNDAMENTALES\n"
    "## OBJETIVOS DE INVERSIÓN\n"
    "## ESTRATEGIA / METODOLOGÍA\n"
    "## GESTIÓN DEL RIESGO\n"
    "## DISCIPLINA Y SESGOS\n"
    "## REFLEXIÓN FINAL\n"
  ),
  model="gpt-5.1",
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="high",
      summary="auto"
    )
  )
)


class WorkflowInput(BaseModel):
  input_as_text: str


def _pretty(obj) -> str:
  """Readable JSON-ish dump for the model."""
  return json.dumps(obj, ensure_ascii=False, indent=2)


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  with trace("New agent"):
    state = {
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
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": workflow["input_as_text"]
          }
        ]
      }
    ]
    filosofia_de_inversion_result_temp = await Runner.run(
      filosofia_de_inversion,
      input=[
        *conversation_history,
        {
          "role": "user",
          "content": [
            {
              "type": "input_text",
              "text": "CONTEXTO — portafolio_promedio (JSON):\n" + _pretty(
                state["custom_input"]["portafolio_promedio"]),
            },
            {
              "type": "input_text",
              "text": "CONTEXTO — portafolio_inversionista (JSON):\n" + _pretty(
                state["custom_input"]["portafolio_inversionista"]),
            },
            {
              "type": "input_text",
              "text": "CONTEXTO — mi_filosofia (texto):\n" + state["custom_input"]["mi_filosofia"],
            },
            {
              "type": "input_text",
              "text": "CONTEXTO — club_deals_information (definición):\n" + state["custom_input"][
                "club_deals_information"],
            },
          ]
        }
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_693a02d72190819097a8a7b5234510f70851015287e3b178"
      })
    )

    conversation_history.extend([item.to_input_item() for item in filosofia_de_inversion_result_temp.new_items])

    filosofia_de_inversion_result = {
      "output_text": filosofia_de_inversion_result_temp.final_output_as(str)
    }

a    return filosofia_de_inversion_result

if __name__ == "__main__":
    # ✅ Correct Pydantic init + ✅ run async
    result = asyncio.run(run_workflow(WorkflowInput(input_as_text="Crea mi fiilosofía de inversión")))
    print(result["output_text"])
