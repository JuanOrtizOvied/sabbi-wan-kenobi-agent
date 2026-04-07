from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional

from agents import Agent, ModelSettings, Runner
from openai import OpenAI
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel, Field

from app.core.config import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "gpt-5.2"
AGENT_NAME: Final[str] = "CostosAgent"
UPLOAD_FILENAME: Final[str] = "score_data.json"
UPLOAD_MIMETYPE: Final[str] = "application/json"
UPLOAD_PURPOSE: Final[str] = "assistants"

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
- Si `comision_sin_igv` es `"0"` Y los productos NO son ahorro → nombre: `"Inversiones sin comisión (no cash)"`
- Si `comision_sin_igv` contiene múltiples valores (ej: "Clase A 1.75% - Clase B 1.05%"), extrae todos los porcentajes y usa el **más alto** como fee numérico (ej: 0.0175)

**Regla de salida para `fee`**:
- El campo `fee` en el output SIEMPRE debe ser un número (float), nunca un string
- Convierte porcentajes a decimal si es necesario (ej: 1.75% → 0.0175)
- Si hay múltiples valores, usa el más alto convertido a decimal

**Importante**: No inventes datos que no estén en el archivo JSON."""

PERSONALITY_PROMPT: Final[str] = """Eres un analista financiero experto en optimización de portafolios y estructura de costos.

Tu misión es procesar datos de portafolio y generar agrupaciones inteligentes de costos y comisiones.

**Responsabilidades**:
1. Analizar la estructura de comisiones del portafolio
2. Agrupar productos de manera inteligente
3. Crear nombres de grupo descriptivos basados en la información de los productos

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
   - Si NO tienen esas palabras → `"Inversiones sin comisión (no cash)"`

5. **Reglas de cálculo**:
   - `total_amount` = suma de todos los amounts de productos en el grupo
   - `fee` = el valor de comision_sin_igv convertido a número decimal (float)
   - Si `comision_sin_igv` contiene múltiples valores (ej: "Clase A 1.75% - Clase B 1.05%"), extrae todos los porcentajes, toma el **más alto** y conviértelo a decimal (ej: 1.75% → 0.0175)
   - Si `comision_sin_igv` es un valor simple como "0.0065", úsalo directamente como float
   - `fee` en el output SIEMPRE debe ser un número (float), nunca un string
   - `costo` = total_amount * fee
   - Para fee = 0: costo = 0.0

**Formato de salida**:
- Usa SIEMPRE el esquema estructurado `PortfolioReport`

**Principios**:
- Precisión: Solo reporta datos que estén en el archivo
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
#     "group_name": "Acciones en bolsa / Credicorp Capital",  # Derived from provider in product names
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
#     "group_name": "Bonos Sabadell Investment Grade",  # Derived from common characteristics
#     "total_amount": 118686.85,
#     "fee": 0.007,
#     "costo": 830.81,
#     "products": [...]
#   },
#   {
#     "group_name": "Nota estructurada Sabadell - JPMORGAN SOXX PPN",
#     "total_amount": 100000.0,
#     "fee": 0.0175,
#     "costo": 1750.0,
#     "products": [...]
#   },
#   {
#     "group_name": "Cash / ahorro",  # Special case: fee=0 with savings products
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
    fee: float = Field(description="Fee/commission rate as a decimal number (e.g., 0.0065, 0.0175). When the original value contains multiple rates, use the highest one.")
    costo: float = Field(
        description="Calculated cost (total_amount * fee)."
    )
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


# ---------------------------------------------------------------------------
# Reply container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AgentReply:
    """Container for agent response with structured output and continuation token."""
    output: dict[str, Any]
    response_id: str


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AgentService:
    """
    Service wrapper around an Agents SDK Agent for portfolio cost analysis.

    Workflow:
      1. Uploads JSON portfolio data as a file attachment
      2. Runs the agent with structured output (PortfolioReport)
      3. Agent groups products by fee and derives descriptive names from product information
      4. Returns validated output with grouped portfolio data
      5. Optionally cleans up the uploaded file

    The agent analyzes product names to create descriptive group labels
    (e.g., "Acciones en bolsa / Credicorp Capital" instead of "comision 0.0065")
    and calculates costs per group.
    """

    def __init__(
            self,
            openai_client: Optional[OpenAI] = None,
            agent: Optional[Agent] = None,
            *,
            model: str = DEFAULT_MODEL,
    ) -> None:
        """
        Initialize the agent service.

        Args:
            openai_client: OpenAI client instance (creates one if None)
            agent: Pre-configured Agent instance (creates one if None)
            model: Model name to use (default: gpt-5.2)
        """
        self._openai = openai_client or self._build_openai_client()
        self._agent = agent or self._build_agent(model=model)

    @staticmethod
    def _build_openai_client() -> OpenAI:
        """Create OpenAI client from settings."""
        if not settings.OPENAI_API_KEY:
            raise ConfigError("OPENAI_API_KEY is missing from configuration")
        return OpenAI(api_key=settings.OPENAI_API_KEY)

    @staticmethod
    def _build_agent(*, model: str) -> Agent:
        """Create configured Agent with structured output."""
        return Agent(
            name=AGENT_NAME,
            instructions=PERSONALITY_PROMPT,
            model=model,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="high", summary="auto"),
            ),
            output_type=PortfolioReport,
        )

    @staticmethod
    def _json_bytes(json_data: Mapping[str, Any]) -> bytes:
        """Convert JSON data to UTF-8 encoded bytes."""
        return json.dumps(json_data, ensure_ascii=False, indent=2).encode("utf-8")

    def _upload_json_file(self, json_data: Mapping[str, Any]) -> str:
        """
        Upload JSON portfolio data as a file attachment.

        Args:
            json_data: Portfolio data to upload

        Returns:
            OpenAI file_id for the uploaded file
        """
        file_obj = io.BytesIO(self._json_bytes(json_data))
        uploaded = self._openai.files.create(
            file=(UPLOAD_FILENAME, file_obj, UPLOAD_MIMETYPE),
            purpose=UPLOAD_PURPOSE,
        )
        log.info("Uploaded file with id=%s", uploaded.id)
        return uploaded.id

    def _delete_file_safely(self, file_id: str) -> None:
        """
        Best-effort cleanup of uploaded file.

        Logs exceptions but never fails the request due to cleanup errors.
        """
        try:
            self._openai.files.delete(file_id)
            log.info("Deleted file id=%s", file_id)
        except Exception:
            log.exception("Failed to delete uploaded file_id=%s", file_id)

    @staticmethod
    def _build_user_message(*, file_id: str) -> dict[str, Any]:
        """
        Construct user message with instruction text and file attachment.

        Args:
            file_id: OpenAI file_id of the uploaded portfolio JSON

        Returns:
            Message dict in Agents SDK format
        """
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
        Run the agent and return structured portfolio cost analysis.

        Args:
            json_data: Portfolio data as a mapping (will be uploaded as JSON)
            previous_response_id: None for first message, or last response.id for follow-ups
            cleanup_uploaded_file: Whether to delete the uploaded file after processing

        Returns:
            AgentReply with:
                - output: Validated PortfolioReport as dict (grouped portfolio data)
                - response_id: ID for continuing the conversation

        Raises:
            TypeError: If json_data is not a mapping
            RuntimeError: If agent returns unexpected output type
            ConfigError: If OpenAI API key is missing
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

            structured_output = getattr(result, "final_output", None)
            last_id = getattr(result, "last_response_id", None)

            if not isinstance(structured_output, PortfolioReport):
                raise RuntimeError(
                    f"Agent returned unexpected final_output type: {type(structured_output)}. "
                    f"Expected PortfolioReport."
                )
            if not isinstance(last_id, str):
                raise RuntimeError(
                    f"Agent returned unexpected last_response_id type: {type(last_id)}. "
                    f"Expected str."
                )

            log.info(
                "Agent completed successfully. response_id=%s, groups=%d",
                last_id,
                len(structured_output.grouped_output),
            )

            return AgentReply(
                output=structured_output.model_dump(),
                response_id=last_id
            )
        finally:
            if cleanup_uploaded_file:
                self._delete_file_safely(file_id)
