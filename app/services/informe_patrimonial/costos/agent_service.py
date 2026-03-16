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

USER_INSTRUCTION: Final[str] = (
    """Procesa el archivo JSON del portafolio adjunto y devuelve la salida en formato `PortfolioReport`.

Necesito:
1. `grouped_output` con:
   - `group_name`
   - `total_amount`
   - `fee`
   - `costo`
   - `products` como lista de objetos con:
     - `name`
     - `amount`

2. `lectura_ejecutiva` en español, máximo 3 párrafos, con máximo 3 líneas por párrafo.

3. `oportunidades_mejora` en un solo párrafo, máximo 5 líneas.

Reglas:
- Agrupar por `comision_sin_igv`.
- Si la comisión es `"0"` y no pertenece a la categoría de cash, ahorro o liquidez, agruparlo aparte.
- Si la comisión no es numérica, no calcular el costo y devolver `null`.
- No inventar datos faltantes."""
)

PERSONALITY_PROMPT: Final[str] = """\
Eres un analista de portafolios especializado en costos y comisiones. Siempre debes usar la herramienta `group_portfolio_by_fee` para construir la salida final y nunca debes inventar datos.

Tu respuesta debe seguir exactamente el esquema `PortfolioReport`, con estos tres campos:
1. `grouped_output`: lista de grupos con `group_name`, `total_amount`, `fee`, `costo` y `products` con objetos `{name, amount}`.
2. `lectura_ejecutiva`: en español, máximo 3 párrafos. Cada párrafo debe ser breve y no exceder 3 líneas.
3. `oportunidades_mejora`: un solo párrafo en español, máximo 5 líneas.

Reglas obligatorias:
- Agrupa por `comision_sin_igv`.
- Suma el monto total por grupo.
- Calcula `costo = total_amount * fee` solo si la comisión es numérica.
- Si la comisión no es numérica, devuelve `costo = null`.
- Si la comisión es `0` y el producto corresponde a cash, ahorro o liquidez, ubícalo en `Cash / ahorro`.
- Si la comisión es `0` pero no corresponde a cash, ubícalo en `Inversiones sin comisión (no cash)`.
- Dentro de `products`, agrega cada producto con su `amount` agregado.
- En la lectura ejecutiva, resalta concentración de costos, diferencia entre cash y no cash con fee 0, y productos con fee por clase cuando aplique.
- No agregues explicaciones fuera del esquema requerido.
"""


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class ProductAmount(BaseModel):
    name: str
    amount: float


class FeeGroup(BaseModel):
    group_name: str
    total_amount: float
    fee: str
    costo: float | None
    products: list[ProductAmount]


class PortfolioReport(BaseModel):
    grouped_output: list[FeeGroup] = Field(
        description="Portfolio grouped by comision_sin_igv with per-product amounts."
    )
    lectura_ejecutiva: str = Field(
        description=(
            "Maximum 3 short paragraphs. Each paragraph must be at most 3 lines. "
            "Executive reading in Spanish."
        )
    )
    oportunidades_mejora: str = Field(
        description=(
            "One paragraph in Spanish, maximum 5 lines, focused on improvement opportunities."
        )
    )


# ---------------------------------------------------------------------------
# Reply container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AgentReply:
    output: dict[str, str]
    response_id: str


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AgentService:
    """
    Service wrapper around an Agents SDK Agent that:
      - uploads JSON input as a file attachment
      - runs the agent with a structured output type (CostosOutput)
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
            output_type=PortfolioReport,
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
        Runs the agent and returns the structured 'Riesgo estructural' section.

        previous_response_id:
          - None for the first message
          - The last response.id for follow-up turns

        Returns an AgentReply whose `.output` is a validated CostosOutput
        instance with ReportLab-formatted rich text in every field.
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
                    f"Runner returned unexpected final_output type: {type(structured_output)}"
                )
            if not isinstance(last_id, str):
                raise RuntimeError("Runner returned an unexpected last_response_id type")

            return AgentReply(output=structured_output.model_dump(), response_id=last_id)
        finally:
            if cleanup_uploaded_file:
                self._delete_file_safely(file_id)
