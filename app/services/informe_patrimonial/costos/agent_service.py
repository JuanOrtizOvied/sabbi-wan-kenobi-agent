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

El reporte debe incluir:

1. **Agrupación por comisiones** (`grouped_output`):
   - Agrupa los productos por `comision_sin_igv`
   - Calcula el monto total por grupo
   - Calcula el costo como: `costo = total_amount * fee` (solo si la comisión es numérica)
   - Lista cada producto con su nombre y monto dentro del grupo

2. **Lectura ejecutiva** (`lectura_ejecutiva`):
   - Máximo 3 párrafos cortos en español
   - Cada párrafo no debe exceder 3 líneas
   - Resalta: concentración de costos, diferencias entre productos con/sin comisión, hallazgos clave
   - **Formato**: Usa etiquetas ReportLab para dar formato al texto:
     * `<b>texto</b>` para negritas
     * `<i>texto</i>` para cursivas
     * `<br/>` para saltos de línea entre párrafos

3. **Oportunidades de mejora** (`oportunidades_mejora`):
   - Un solo párrafo en español, máximo 5 líneas
   - Enfócate en recomendaciones accionables para optimizar costos
   - **Formato**: Usa etiquetas ReportLab (`<b>`, `<i>`, viñetas si es necesario)

**Reglas de agrupación**:
- Si `comision_sin_igv` es `"0"` Y el producto es cash/ahorro/liquidez → Grupo: `Cash / ahorro`
- Si `comision_sin_igv` es `"0"` Y el producto NO es cash/ahorro/liquidez → Grupo: `Inversiones sin comisión (no cash)`
- Si `comision_sin_igv` no es numérica → `costo = null` (no calcular)
- Agrupa el resto por el valor exacto de `comision_sin_igv`

**Importante**: No inventes datos que no estén en el archivo JSON."""

PERSONALITY_PROMPT: Final[str] = """Eres un analista financiero experto en optimización de portafolios y estructura de costos.

Tu misión es procesar datos de portafolio y generar reportes ejecutivos claros, precisos y accionables.

**Responsabilidades**:
1. Analizar la estructura de comisiones del portafolio
2. Identificar concentraciones de costos y oportunidades de ahorro
3. Generar insights ejecutivos con formato profesional
4. Proporcionar recomendaciones basadas en datos

**Formato de salida**:
- Usa SIEMPRE el esquema estructurado `PortfolioReport`
- Los campos de texto (`lectura_ejecutiva`, `oportunidades_mejora`) DEBEN usar formato ReportLab:
  * Negritas: `<b>texto</b>` para conceptos clave
  * Cursivas: `<i>texto</i>` para énfasis o términos técnicos
  * Saltos de párrafo: `<br/><br/>` entre párrafos
  * Ejemplo: `<b>Concentración alta:</b> El <i>80% de los costos</i> proviene de productos con comisión del 2.5%<br/><br/>La diversificación...`

**Principios**:
- Precisión: Solo reporta datos que estén en el archivo
- Claridad: Usa lenguaje ejecutivo directo
- Acción: Enfócate en insights que generen valor
- Formato: Respeta estrictamente el esquema y las etiquetas ReportLab

**Restricciones**:
- NO inventes datos faltantes
- NO agregues campos adicionales al esquema
- NO proporciones explicaciones fuera de los campos definidos
- SIEMPRE valida que los cálculos sean correctos antes de responder"""


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class ProductAmount(BaseModel):
    """Individual product with its amount within a fee group."""
    name: str = Field(description="Product name")
    amount: float = Field(description="Product amount in the portfolio")


class FeeGroup(BaseModel):
    """Portfolio products grouped by commission/fee rate."""
    group_name: str = Field(description="Descriptive name for this fee group")
    total_amount: float = Field(description="Sum of all product amounts in this group")
    fee: str = Field(description="Fee/commission rate as string (e.g., '2.5', '0', 'N/A')")
    costo: float | None = Field(
        description="Calculated cost (total_amount * fee). Null if fee is not numeric."
    )
    products: list[ProductAmount] = Field(
        description="List of products in this group with individual amounts"
    )


class PortfolioReport(BaseModel):
    """Complete portfolio cost analysis report."""
    grouped_output: list[FeeGroup] = Field(
        description="Portfolio grouped by comision_sin_igv with aggregated amounts and costs."
    )
    lectura_ejecutiva: str = Field(
        description=(
            "Executive summary in Spanish. Maximum 3 short paragraphs (3 lines each). "
            "Must use ReportLab formatting tags: <b>bold</b>, <i>italic</i>, <br/> for line breaks. "
            "Highlights cost concentration, fee structure insights, and key findings."
        )
    )
    oportunidades_mejora: str = Field(
        description=(
            "Improvement opportunities in Spanish. One paragraph, maximum 5 lines. "
            "Must use ReportLab formatting tags: <b>bold</b>, <i>italic</i>. "
            "Focused on actionable recommendations to optimize portfolio costs."
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
      3. Returns validated output with ReportLab-formatted text fields
      4. Optionally cleans up the uploaded file

    The agent groups products by fee, calculates costs, and generates
    executive insights with proper ReportLab markup for PDF rendering.
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
                - output: Validated PortfolioReport as dict (with ReportLab-formatted text)
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
