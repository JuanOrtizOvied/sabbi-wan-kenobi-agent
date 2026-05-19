import io

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.api.informe_patrimonial.schemas import ChatRequest, ChatResponse
from app.services.informe_patrimonial.calidad_portafolio.agent_service import AgentService as AnthropicAgentServiceCalidad
from app.services.informe_patrimonial.calidad_portafolio.anthropic_agent_service import AgentService
from app.services.informe_patrimonial.riesgo_estructural.agent_service import AgentService as StructuralRiskAgentService
from app.services.informe_patrimonial.riesgo_estructural.anthropic_agent_service import AgentService as StructuralRiskAgentServiceAnthropic
from app.services.informe_patrimonial.resumen_ejecutivo.agent_service import \
    AgentService as ResumenEjecutivoAgentService
from app.services.informe_patrimonial.costos.agent_service import AgentService as CostosAgentService
from app.services.informe_patrimonial.costos.anthropic_agent_service import AgentService as CostosAnthropicAgentService
from app.services.informe_patrimonial.portfolio_analyzer.agent_service import \
    PortfolioAnalystService as PortfolioAnalyzerAgentService
from app.services.informe_patrimonial.portfolio_analyzer.anthropic_agent_service import (
    PortfolioAnalyzerService as PortfolioAnalyzerAnthropicAgentService)
from app.services.informe_patrimonial.resumen_ejecutivo.anthropic_agent_service import AgentService as ResumenEjecutivoAnthropicAgent
from app.services.informe_patrimonial.radiografia_patrimonial.anthropic_agent_service import AgentService as RadiografiaPatrimonialAnthropicAgent

from openpyxl import Workbook

from .calidad_portafolio.schemas import ReportRequest
from .riesgo_estructural.schemas import StructuralRiskData
from .portfolio.schemas import PortafolioItem
from app.services.informe_patrimonial.calidad_portafolio.excel import build_asset_sheet, build_risk_sheet, \
    build_geo_sheet
from app.services.informe_patrimonial.riesgo_estructural.excel import build_structural_risk_excel
from app.services.informe_patrimonial.portfolio.excel import generate_portfolio_excel
from app.api.informe_patrimonial.portfolio_analyzer.schemas import PortfolioAnalyzerResponse

informe_patrimonial_router = APIRouter()
agent = AgentService()
anthropic_agent_service_calidad_portafolio = AnthropicAgentServiceCalidad()
structural_risk_agent = StructuralRiskAgentService()
structural_risk_anthropic_agent = StructuralRiskAgentServiceAnthropic()
resumen_ejecutivo_agent = ResumenEjecutivoAgentService()
costos_agent = CostosAgentService()
costos_anthropic_agent = CostosAnthropicAgentService()
portfolio_analyzer_agent = PortfolioAnalyzerAgentService()
portfolio_analyzer_anthropic_agent = PortfolioAnalyzerAnthropicAgentService()
resumen_ejecutivo_anthropic_agent = ResumenEjecutivoAnthropicAgent()
radiografia_patrimonial_anthropic_agent = RadiografiaPatrimonialAnthropicAgent()


@informe_patrimonial_router.post("/calidad-portafolio", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = agent.reply(
            json_data=req.json_data,
            previous_response_id=req.previous_response_id,
        )
        return ChatResponse(reply=out.parsed, response_id=out.response_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@informe_patrimonial_router.post("/anthropic/calidad-portafolio", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = anthropic_agent_service_calidad_portafolio.reply(
            json_data=req.json_data,
            previous_response_id=req.previous_response_id,
        )
        return ChatResponse(reply=out.parsed, response_id=out.response_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@informe_patrimonial_router.post("/riesgo-estructural", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = structural_risk_agent.reply(
            json_data=req.json_data,
            previous_response_id=req.previous_response_id,
        )
        return ChatResponse(reply=out.output, response_id=out.response_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@informe_patrimonial_router.post("/anthropic/riesgo-estructural", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = structural_risk_anthropic_agent.reply(
            json_data=req.json_data,
        )
        return ChatResponse(reply=out.output, response_id=out.response_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@informe_patrimonial_router.post("/resumen-ejecutivo", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = resumen_ejecutivo_agent.reply(
            json_data=req.json_data,
            previous_response_id=req.previous_response_id,
        )
        return ChatResponse(reply=out.result, response_id=out.response_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@informe_patrimonial_router.post("/anthropic/resumen-ejecutivo", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = resumen_ejecutivo_anthropic_agent.reply(
            json_data=req.json_data,
        )
        return ChatResponse(reply=out.result, response_id=out.message_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@informe_patrimonial_router.post("/portfolio-analyzer", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = portfolio_analyzer_agent.analyze(
            json_data=req.json_data,
            previous_response_id=req.previous_response_id,
        )
        return ChatResponse(reply=out.diagnostico.to_dict(), response_id=out.response_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ANTHROPIC
@informe_patrimonial_router.post(
    "/anthropic/portfolio-analyzer",
    response_model=PortfolioAnalyzerResponse,
)
async def portfolio_analyzer(req: ChatRequest):
    """
    Analyze a client's investment portfolio and return a structured
    executive diagnostic (DiagnosticoEjecutivo) using the Anthropic API.
    """
    try:
        reply = portfolio_analyzer_anthropic_agent.analyze(json_data=req.json_data)

        return PortfolioAnalyzerResponse(
            reply=reply.diagnostico.model_dump(),
            message_id=reply.message_id,
        )

    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@informe_patrimonial_router.post(
    "/anthropic/radigrafia-patrimonial",
    response_model=PortfolioAnalyzerResponse,
)
async def radiografia_patrimonial(req: ChatRequest) -> ChatResponse:
    try:
        reply = radiografia_patrimonial_anthropic_agent.analyze(json_data=req.json_data)

        return ChatResponse(reply=reply.observaciones.model_dump(), response_id=reply.message_id)

    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@informe_patrimonial_router.post("/costos", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = costos_agent.reply(
            json_data=req.json_data,
            previous_response_id=req.previous_response_id,
        )
        return ChatResponse(reply=out.output, response_id=out.response_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@informe_patrimonial_router.post("/anthropic/costos", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = costos_anthropic_agent.reply(
            json_data=req.json_data,
        )
        return ChatResponse(reply=out.output, response_id=out.response_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@informe_patrimonial_router.post("/calidad-portafolio-excel")
async def generate_report(req: ReportRequest):
    """Generate the alignment Excel report directly from structured data."""
    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    build_asset_sheet(wb, req)
    build_risk_sheet(wb, req)
    build_geo_sheet(wb, req)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"calidad_portafolio.xlsx"
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@informe_patrimonial_router.post(
    "/structural-risk/excel",
    response_class=Response,
    summary="Generate structural risk Excel report",
    responses={
        200: {
            "content": {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {}
            },
            "description": "Excel workbook with 4 sheets: "
                           "Concentración, Correlación, Gestor y Administrador, Moneda.",
        }
    },
)
async def generate_structural_risk_excel(payload: StructuralRiskData):
    """
    Receive structural risk scoring data and return an Excel report.

    The workbook contains four sheets showing step-by-step calculations:
    - **Concentración**: HHI + max weight → blended score
    - **Correlación**: weighted correlation matrix → score
    - **Gestor y Administrador**: entity-weighted scores
    - **Moneda**: PEN exposure → score + global summary
    """
    try:
        wb = build_structural_risk_excel(payload.model_dump())

        buffer = io.BytesIO()
        wb.save(buffer)
        content = buffer.getvalue()

        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": "attachment; filename=riesgo_estructural.xlsx"
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@informe_patrimonial_router.post("/portfolio/excel")
async def generate_excel(portafolio: List[PortafolioItem]):
    xlsx_bytes = generate_portfolio_excel(portafolio)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"Portfolio-{timestamp}.xlsx"

    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
