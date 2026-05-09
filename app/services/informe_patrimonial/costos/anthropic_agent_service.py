from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional

import anthropic
from pydantic import BaseModel, Field

from app.core.config import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "claude-sonnet-4-6"
MAX_TOKENS: Final[int] = 16_384

USER_INSTRUCTION: Final[str] = """Analiza el archivo JSON del portafolio adjunto y genera un reporte estructurado de costos y comisiones.

El reporte debe incluir la **Agrupación inteligente** (`grouped_output`):
   - Agrupa productos por `comision_sin_igv`
   - Para cada grupo, crea un nombre descriptivo basado en los nombres de los productos
   - El nombre debe reflejar las características comunes de los productos del grupo
   - NO uses nombres genéricos como "comision 0.0065" o "grupo 1"
   - Calcula el monto total por grupo
   - Calcula el costo como: `costo = total_amount * fee` (solo si la comisión es numérica)
   - Lista cada producto con su nombre y monto dentro del grupo

**Reglas para nombres de grupo**:
- Analiza los nombres de productos en cada grupo (misma comisión)
- Identifica patrones comunes: proveedores, tipos, características compartidas
- Crea nombres descriptivos basados en esos patrones
- Si es un solo producto, usa su nombre directamente
- Si los productos comparten proveedor o características, refleja eso en el nombre

**Casos especiales**:
- Si `comision_sin_igv` es `"0"` Y los productos son ahorro/cash/liquidez → nombre: `"Cash / ahorro"`
- Si `comision_sin_igv` es `"0"` Y los productos NO son ahorro → nombre: `"Activos sin comisión explícita"`
- Si `comision_sin_igv` contiene múltiples valores (ej: "Clase A 1.75% - Clase B 1.05%"), extrae todos los porcentajes y usa el **más alto** como fee numérico (ej: 0.0175)

**Regla de salida para `fee`**:
- El campo `fee` en el output SIEMPRE debe ser un número (float), nunca un string
- Convierte porcentajes a decimal si es necesario (ej: 1.75% → 0.0175)
- Si hay múltiples valores, usa el más alto convertido a decimal

BENCHMARKS DE REFERENCIA POR TIPO DE PRODUCTO

Usa esta tabla para identificar si algún fee está en el rango alto para su categoría.
Esta información debe usarse ÚNICAMENTE para el campo lectura_ejecutiva, no para calcular costos.

  Tipo de producto                            Rango típico de fee anual
  Acciones en bolsa (via SAB/broker)          0.50% – 0.75%
  Fondos mutuos renta fija                    0.50% – 1.00%
  Fondos mutuos renta variable                0.75% – 1.50%
  Notas estructuradas                         1.00% – 1.75%
  Fondos de mercados privados (PE/crédito)    1.50% – 2.50%
  Club deals / fondos inmobiliarios           1.50% – 3.00%
  Depósitos a plazo / cash                    0%

REGLA PARA lectura_ejecutiva

La lectura ejecutiva debe seguir esta lógica condicional:

Párrafo base (siempre incluir):
"Con la información disponible, no se observan señales evidentes de sobrecostos estructurales
frente a benchmarks comparables. Este análisis debe leerse como una evaluación de visibilidad
y control de costos: no todos los costos 'all-in' pueden ser trazados con precisión hoy,
especialmente en productos estructurados y alternativos."

Línea adicional condicional 1 — incluir SI más del 30% del patrimonio invertible
está en grupos con comisión cero (excluyendo cash):
"Una parte relevante del patrimonio ([X]%) está en activos sin comisión explícita,
lo que limita la visibilidad del costo total del portafolio."

Línea adicional condicional 2 — incluir SI algún instrumento tiene un fee
que supera el límite superior de su rango típico según la tabla de benchmarks:
"El [nombre del instrumento] tiene un fee de [X]%, en el rango alto para este tipo de producto."

Si ninguna condición aplica, usar solo el párrafo base.

**Importante**: No inventes datos que no estén en el archivo JSON."""

PERSONALITY_PROMPT: Final[str] = """Eres un analista financiero experto en optimización de portafolios y estructura de costos.

Tu misión es procesar datos de portafolio y generar agrupaciones inteligentes de costos y comisiones,
junto con una lectura ejecutiva que sea útil y honesta para el cliente.

**Responsabilidades**:
1. Analizar la estructura de comisiones del portafolio
2. Agrupar productos de manera inteligente
3. Crear nombres de grupo descriptivos basados en la información real de los productos
4. Redactar una lectura ejecutiva personalizada según las condiciones del portafolio

**Lógica de agrupación y nomenclatura**:

1. **Agrupa por comisión**: Todos los productos con el mismo `comision_sin_igv` van juntos

2. **Analiza cada grupo**:
   - Examina los nombres de los productos en el grupo
   - Identifica patrones comunes: proveedores recurrentes, características similares, categorías
   - Busca términos compartidos en los nombres

3. **Crea nombres descriptivos**:
   - Si todos los productos comparten un proveedor (ej: "Credicorp Capital", "Sabadell"), inclúyelo en el nombre
   - Si hay características comunes visibles en los nombres (ej: tickers bursátiles, instrumentos similares), descríbelas
   - Si es un solo producto, usa su nombre completo o una versión resumida
   - Si los productos son diversos pero tienen algo en común, encuentra el denominador común

   **Ejemplos de buenos nombres**:
   - Si todos tienen "- Credicorp Capital" o son tickers → "Acciones en bolsa / Credicorp Capital"
   - Si todos son de "Sabadell" y tienen "FUND" → "Bonos Sabadell Investment Grade"
   - Si hay un solo producto "Sabadell - JPMORGAN SOXX PPN" → "Nota estructurada Sabadell - JPMORGAN SOXX PPN"
   - Si es solo "Sabbi Oportunidad" → "Sabbi Oportunidad"

   **Evita nombres genéricos**:
   - ❌ "comision 0.0065"
   - ❌ "grupo 1"
   - ❌ "productos varios"

4. **Casos especiales para comisión "0"**:
   - Si los productos tienen "Ahorro", "Cash", o "Liquidez" en el nombre → `"Cash / ahorro"`
   - Si NO tienen esas palabras → `"Activos sin comisión explícita"`

5. **Reglas de cálculo**:
   - `total_amount` = suma de todos los amounts de productos en el grupo
   - `fee` = el valor de comision_sin_igv convertido a número decimal (float)
   - Si `comision_sin_igv` contiene múltiples valores, toma el más alto y conviértelo a decimal
   - `fee` en el output SIEMPRE debe ser un número (float), nunca un string
   - `costo` = total_amount * fee
   - Para fee = 0: costo = 0.0

6. **Lectura ejecutiva**:
   - Seguir estrictamente la lógica condicional definida en USER_INSTRUCTION
   - Incluir siempre el párrafo base
   - Añadir líneas adicionales solo si se cumplen las condiciones definidas
   - No inventar sobrecostos ni hacer juicios de valor más allá de lo que los datos permiten
   - Si un fee está en el rango alto de su categoría según la tabla de benchmarks, mencionarlo
     de forma factual y neutral — no alarmista

**Formato de salida**:
- Usa SIEMPRE el esquema estructurado `PortfolioReport`

**Principios**:
- Precisión: Solo reporta datos que estén en el archivo
- Honestidad: No concluir sobrecostos cuando no hay información suficiente para garantizarlo
- Inteligencia: Deriva nombres descriptivos de la información real de los productos
- Claridad: Los nombres deben ser inmediatamente comprensibles

**Restricciones**:
- NO uses nombres genéricos como "comision X.XX" o "grupo N"
- NO inventes datos faltantes
- NO agregues campos adicionales al esquema
- NO proporciones explicaciones fuera de los campos definidos
- SIEMPRE valida que los cálculos sean correctos antes de responder
- SIEMPRE deriva nombres de grupo de la información real de los productos"""


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
# Example output with intelligent group names derived from product information:
# [
#   {
#     "group_name": "Acciones en bolsa / Credicorp Capital",
#     "total_amount": 259891.0,
#     "fee": 0.0065,
#     "costo": 1689.29,
#     "products": [
#       {"name": "SNJUANC1", "amount": 91469.0},
#       {"name": "BAP - Credicorp Capital", "amount": 58545.0},
#       ...
#     ]
#   },
#   {
#     "group_name": "Cash / ahorro",
#     "total_amount": 199922.41,
#     "fee": 0,
#     "costo": 0.0,
#     "products": [...]
#   }
# ]
# ---------------------------------------------------------------------------

class ProductAmount(BaseModel):
    """Individual product with its amount within a fee group."""
    name: str = Field(description="Product name")
    amount: float = Field(description="Product amount in the portfolio")


class FeeGroup(BaseModel):
    """Portfolio products grouped by fee with names derived from product information."""
    group_name: str = Field(
        description=(
            "Descriptive name derived from analyzing product names in the group. "
            "Should reflect common characteristics like shared providers, product types, "
            "or individual product names. NOT generic names like 'comision 0.0065'."
        )
    )
    total_amount: float = Field(description="Sum of all product amounts in this group")
    fee: float = Field(
        description=(
            "Fee/commission rate as a decimal number (e.g., 0.0065, 0.0175). "
            "When the original value contains multiple rates, use the highest one."
        )
    )
    costo: float = Field(description="Calculated cost (total_amount * fee).")
    products: list[ProductAmount] = Field(
        description="List of products in this group with individual amounts"
    )


class PortfolioReport(BaseModel):
    """Complete portfolio cost analysis report with intelligent grouping."""
    grouped_output: list[FeeGroup] = Field(
        description=(
            "Portfolio grouped by fee with descriptive names derived from product information. "
            "Group names reflect common characteristics found in product names (providers, types, etc.) "
            "rather than generic labels like 'comision X.XX'."
        )
    )
    lectura_ejecutiva: str = Field(
        description=(
            "Executive reading that follows the conditional logic defined in USER_INSTRUCTION. "
            "Always includes the base paragraph. Adds conditional lines only when the data "
            "meets the specified thresholds (>30% in zero-fee non-cash groups, or fees above "
            "benchmark upper bounds). Factual and neutral tone."
        )
    )


# ---------------------------------------------------------------------------
# Reply container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AgentReply:
    """Container for agent response with structured output."""
    output: dict[str, Any]
    response_id: str


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AgentService:
    """
    Service that uses the Anthropic Messages API for portfolio cost analysis.

    Workflow:
      1. Sends portfolio JSON data inline in the user message
      2. Uses structured output (messages.parse) to guarantee schema compliance
      3. Agent groups products by fee and derives descriptive names
      4. Returns validated PortfolioReport with grouped data and executive reading
    """

    def __init__(
            self,
            client: Optional[anthropic.Anthropic] = None,
            *,
            model: str = DEFAULT_MODEL,
            max_tokens: int = MAX_TOKENS,
    ) -> None:
        self._client = client or self._build_client()
        self._model = model
        self._max_tokens = max_tokens

    @staticmethod
    def _build_client() -> anthropic.Anthropic:
        if not settings.ANTHROPIC_API_KEY:
            raise ConfigError("ANTHROPIC_API_KEY is missing from configuration")
        return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    @staticmethod
    def _build_user_content(json_data: Mapping[str, Any]) -> str:
        """
        Build the full user message content with portfolio JSON inline.
        """
        portfolio_json = json.dumps(json_data, ensure_ascii=False, indent=2)
        return (
            f"{USER_INSTRUCTION}\n\n"
            f"--- INICIO DATOS DEL PORTAFOLIO ---\n"
            f"{portfolio_json}\n"
            f"--- FIN DATOS DEL PORTAFOLIO ---"
        )

    def reply(
            self,
            json_data: Mapping[str, Any],
    ) -> AgentReply:
        """
        Run the agent and return structured portfolio cost analysis.

        Returns:
            AgentReply with:
                - output: Validated PortfolioReport as dict (grouped data + executive reading)
                - response_id: Anthropic message ID for tracing
        """
        if not isinstance(json_data, Mapping):
            raise TypeError("json_data must be a mapping (dict-like)")

        user_content = self._build_user_content(json_data)

        response = self._client.messages.parse(
            model=self._model,
            max_tokens=self._max_tokens,
            system=PERSONALITY_PROMPT,
            thinking={"type": "adaptive"},
            messages=[
                {"role": "user", "content": user_content},
            ],
            output_format=PortfolioReport,
        )

        parsed = response.parsed_output
        if not isinstance(parsed, PortfolioReport):
            raise RuntimeError(
                f"API returned unexpected output type: {type(parsed)}. "
                f"Expected PortfolioReport."
            )

        log.info(
            "Agent completed successfully. message_id=%s, groups=%d",
            response.id,
            len(parsed.grouped_output),
        )

        return AgentReply(
            output=parsed.model_dump(),
            response_id=response.id,
        )
