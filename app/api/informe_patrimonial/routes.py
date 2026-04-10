import io
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.api.informe_patrimonial.schemas import ChatRequest, ChatResponse
from app.services.informe_patrimonial.calidad_portafolio.agent_service import AgentService
from app.services.informe_patrimonial.riesgo_estructural.agent_service import AgentService as StructuralRiskAgentService
from app.services.informe_patrimonial.resumen_ejecutivo.agent_service import AgentService as ResumenEjecutivoAgentService
from app.services.informe_patrimonial.costos.agent_service import AgentService as CostosAgentService
from app.services.informe_patrimonial.portfolio_analyzer.agent_service import PortfolioAnalystService as PortfolioAnalyzerAgentService

from openpyxl import Workbook

from .calidad_portafolio.schemas import ReportRequest
from app.services.informe_patrimonial.calidad_portafolio.excel import build_asset_sheet, build_risk_sheet, build_geo_sheet

informe_patrimonial_router = APIRouter()
agent = AgentService()
structural_risk_agent = StructuralRiskAgentService()
resumen_ejecutivo_agent = ResumenEjecutivoAgentService()
costos_agent = CostosAgentService()
portfolio_analyzer_agent = PortfolioAnalyzerAgentService()


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

    filename = f"alineacion_{req.inversionista.replace(' ', '_').lower()}.xlsx"
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
