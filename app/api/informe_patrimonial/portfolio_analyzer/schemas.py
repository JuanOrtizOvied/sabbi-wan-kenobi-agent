# app/api/schemas_analyzer.py
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────


class PortfolioAnalyzerRequest(BaseModel):
    """
    Request body for POST /portfolio-analyzer.

    The `portfolio_data` field must contain the full portfolio JSON
    payload (client context, composition, scores, observations, costs).
    """
    portfolio_data: dict[str, Any] = Field(
        ...,
        description=(
            "Complete portfolio JSON containing client data, "
            "composition, scores, observations, and costs."
        ),
    )


# ── Response sub-models ──────────────────────────────────────────────
class DiagnosticoGeneral(BaseModel):
    """High-level portfolio diagnostic headline (titulo ≤ 200 chars, subtitulo ≤ 200 chars)."""
    titulo: str
    subtitulo: str


class ContextoResumenOut(BaseModel):
    objetivo_principal: str
    flujo_mensual_requerido_usd: float


class FortalezaOut(BaseModel):
    titulo: str
    explicacion: str


class IneficienciaPriorizadaOut(BaseModel):
    orden: int
    titulo: str
    que_esta_pasando: str
    por_que_importa: str
    acciones_recomendadas: list[str]


class FocoDeMejoraOut(BaseModel):
    orden: int
    titulo: str
    descripcion: str


class AccionPriorizadaOut(BaseModel):
    orden: int
    titulo: str
    pasos: list[str]


class DiagnosticoEjecutivoOut(BaseModel):
    diagnostico_general: DiagnosticoGeneral
    contexto_resumen: ContextoResumenOut
    tesis_central: str
    fortalezas: list[FortalezaOut]
    ineficiencias_priorizadas: list[IneficienciaPriorizadaOut]
    focos_de_mejora: list[FocoDeMejoraOut]
    plan_de_accion_priorizado: list[AccionPriorizadaOut]
    mensaje_final: str


# ── Response ─────────────────────────────────────────────────────────


class PortfolioAnalyzerResponse(BaseModel):
    """
    Response body for POST /portfolio-analyzer.
    """
    reply: DiagnosticoEjecutivoOut
    message_id: str = Field(
        ...,
        description="Anthropic message ID for tracing / auditing.",
    )
