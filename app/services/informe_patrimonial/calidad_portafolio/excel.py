"""
FastAPI service that uses Anthropic SDK to generate alignment analysis
and produces an Excel file with three sheets.
"""
import json
import io
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

app = FastAPI(title="Alignment Report Generator")

# ── Pydantic models ──────────────────────────────────────────────

class AssetDetail(BaseModel):
    asset_name: str
    benchmark_percentage: float
    investor_percentage: float
    min_limit: float
    max_limit: float
    penalty: float
    is_within_limits: bool
    deviation: float

class AssetSummary(BaseModel):
    total_assets_evaluated: int
    assets_within_limits: int
    assets_outside_limits: int
    portfolio_assets_not_in_benchmark: int
    total_positive_deviation: float
    total_negative_deviation: float

class AssetAlignmentData(BaseModel):
    score: int
    distancia_estructural: float
    penalizacion_total: float
    asset_percentage: float
    asset_details: list[AssetDetail]
    summary: AssetSummary

class InvestmentContribution(BaseModel):
    producto_nombre: str
    tipo_activo: list[str]
    monto: float
    peso_inversion: float
    risk_score: float
    aporte: float

class TipoActivoContribution(BaseModel):
    tipo_activo: str
    weight: float
    portfolio_percentage: float
    monto: float
    score: float

class DCalculation(BaseModel):
    score_total: float
    perfil_min: float
    perfil_max: float
    first_quarter: float
    midpoint: float
    third_quarter: float
    zone: str
    reference_point: float
    d_value: float

class PerfilRange(BaseModel):
    min: float
    max: float

class RiskAlignmentData(BaseModel):
    score: int
    score_total_weighted: float
    perfil_riesgo: str
    perfil_range: PerfilRange
    d_value: float
    d_calculation: DCalculation
    tipo_activo_contributions: list[TipoActivoContribution]
    total_portfolio_percentage_by_activo: float
    investment_contributions: list[InvestmentContribution]

class RegionDetail(BaseModel):
    region: str
    benchmark_percentage: float
    portfolio_percentage: float
    tolerance: str
    min_limit: float
    max_limit: float
    deviation: float
    penalty: float
    is_within_limits: bool

class GeoSummary(BaseModel):
    total_regions_in_benchmark: int
    regions_within_limits: int
    regions_outside_limits: int
    portfolio_regions_not_in_benchmark: int

class GeoAlignmentData(BaseModel):
    score: int
    interpretation: str
    total_deviation: float
    region_details: list[RegionDetail]
    unmapped_regions: list
    summary: GeoSummary

class ReportRequest(BaseModel):
    inversionista: str
    total_patrimonio: float
    perfil_riesgo: str
    asset_alignment: AssetAlignmentData
    risk_alignment: RiskAlignmentData
    geo_alignment: GeoAlignmentData


# ── Styling helpers ──────────────────────────────────────────────

HEADER_FILL = PatternFill("solid", fgColor="2F5496")
HEADER_FONT = Font(bold=True, color="FFFFFF", name="Arial", size=10)
TITLE_FONT = Font(bold=True, name="Arial", size=12, color="2F5496")
SECTION_FONT = Font(bold=True, name="Arial", size=10, color="2F5496")
NORMAL_FONT = Font(name="Arial", size=10)
SCORE_FONT = Font(bold=True, name="Arial", size=14, color="2F5496")
GREEN_FILL = PatternFill("solid", fgColor="C6EFCE")
RED_FILL = PatternFill("solid", fgColor="FFC7CE")
YELLOW_FILL = PatternFill("solid", fgColor="FFEB9C")
LIGHT_BLUE_FILL = PatternFill("solid", fgColor="D6E4F0")
THIN_BORDER = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin"),
)

def style_header_row(ws, row, cols):
    for c in range(1, cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = THIN_BORDER

def style_data_cell(ws, row, col, fmt=None):
    cell = ws.cell(row=row, column=col)
    cell.font = NORMAL_FONT
    cell.border = THIN_BORDER
    if fmt:
        cell.number_format = fmt
    return cell

def auto_width(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 4, 45)


# ── Sheet builders ───────────────────────────────────────────────

def build_asset_sheet(wb, req: ReportRequest):
    ws = wb.create_sheet("Alineacion Activo")
    d = req.asset_alignment
    r = 1

    # Title
    ws.cell(row=r, column=1, value="ALINEACIÓN POR TIPO DE ACTIVO").font = TITLE_FONT
    r += 2

    # Section 1: Universe
    ws.cell(row=r, column=1, value="1. Universo del análisis").font = SECTION_FONT
    r += 1
    for label, val in [
        ("Inversionista", req.inversionista.title()),
        ("Patrimonio invertible", f"USD {req.total_patrimonio:,.2f}"),
        ("Perfil de riesgo", req.perfil_riesgo),
        ("Benchmark aplicable", f"{'≥500K' if req.total_patrimonio >= 500000 else '<500K'} – Perfil {req.perfil_riesgo}"),
    ]:
        ws.cell(row=r, column=1, value=label).font = NORMAL_FONT
        ws.cell(row=r, column=2, value=val).font = NORMAL_FONT
        r += 1
    r += 1

    # Section 2: Benchmark
    ws.cell(row=r, column=1, value="2. Benchmark utilizado").font = SECTION_FONT
    r += 1
    headers = ["Clase de activo", "Benchmark %"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=r, column=i, value=h)
    style_header_row(ws, r, len(headers))
    r += 1
    for ad in d.asset_details:
        style_data_cell(ws, r, 1).value = ad.asset_name
        style_data_cell(ws, r, 2, "0.00%").value = ad.benchmark_percentage / 100
        r += 1
    ws.cell(row=r, column=1, value="Total").font = Font(bold=True, name="Arial", size=10)
    ws.cell(row=r, column=2, value=1).number_format = "0.00%"
    ws.cell(row=r, column=2).font = Font(bold=True, name="Arial", size=10)
    r += 2

    # Section 3: Ranges
    ws.cell(row=r, column=1, value="3. Rangos permitidos por clase de activo").font = SECTION_FONT
    r += 1
    headers = ["Clase de activo", "Límite inferior", "Límite superior"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=r, column=i, value=h)
    style_header_row(ws, r, len(headers))
    r += 1
    for ad in d.asset_details:
        style_data_cell(ws, r, 1).value = ad.asset_name
        style_data_cell(ws, r, 2, "0.00%").value = ad.min_limit / 100
        style_data_cell(ws, r, 3, "0.00%").value = ad.max_limit / 100
        r += 1
    r += 1

    # Section 4: Comparison & penalties
    ws.cell(row=r, column=1, value="4. Comparación vs benchmark y penalizaciones").font = SECTION_FONT
    r += 1
    headers = ["Clase de activo", "Benchmark", "Rango Min", "Rango Max", req.inversionista.title(), "Desviación", "Penalización", "¿Dentro?"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=r, column=i, value=h)
    style_header_row(ws, r, len(headers))
    r += 1
    for ad in d.asset_details:
        style_data_cell(ws, r, 1).value = ad.asset_name
        style_data_cell(ws, r, 2, "0.00%").value = ad.benchmark_percentage / 100
        style_data_cell(ws, r, 3, "0.00%").value = ad.min_limit / 100
        style_data_cell(ws, r, 4, "0.00%").value = ad.max_limit / 100
        style_data_cell(ws, r, 5, "0.00%").value = ad.investor_percentage / 100
        style_data_cell(ws, r, 6, "0.00%").value = ad.deviation / 100
        style_data_cell(ws, r, 7, "0.00%").value = ad.penalty / 100
        cell = style_data_cell(ws, r, 8)
        cell.value = "Sí" if ad.is_within_limits else "No"
        cell.fill = GREEN_FILL if ad.is_within_limits else RED_FILL
        r += 1
    # Totals row
    ws.cell(row=r, column=1, value="TOTAL PENALIZACIÓN").font = Font(bold=True, name="Arial", size=10)
    ws.cell(row=r, column=7, value=d.penalizacion_total / 100).number_format = "0.00%"
    ws.cell(row=r, column=7).font = Font(bold=True, name="Arial", size=10)
    r += 2

    # Section 5: Structural distance & score
    ws.cell(row=r, column=1, value="5. Distancia estructural y Score").font = SECTION_FONT
    r += 1
    ws.cell(row=r, column=1, value="Distancia estructural").font = NORMAL_FONT
    ws.cell(row=r, column=2, value=d.distancia_estructural / 100).number_format = "0.00%"
    r += 1
    ws.cell(row=r, column=1, value="Score de Alineación por Tipo de Activo").font = Font(bold=True, name="Arial", size=11)
    cell = ws.cell(row=r, column=2, value=f"{d.score} / 10")
    cell.font = SCORE_FONT
    cell.fill = YELLOW_FILL

    auto_width(ws)


def build_risk_sheet(wb, req: ReportRequest):
    ws = wb.create_sheet("Alineacion Riesgo")
    d = req.risk_alignment
    r = 1

    ws.cell(row=r, column=1, value="ALINEACIÓN POR RIESGO").font = TITLE_FONT
    r += 2

    # Investment contributions table
    ws.cell(row=r, column=1, value="1. Score de performance ponderado por inversión").font = SECTION_FONT
    r += 1
    headers = ["Producto", "Tipo de activo", "Monto (USD)", "Peso (%)", "Risk Score", "Aporte"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=r, column=i, value=h)
    style_header_row(ws, r, len(headers))
    r += 1
    for ic in d.investment_contributions:
        style_data_cell(ws, r, 1).value = ic.producto_nombre
        style_data_cell(ws, r, 2).value = ", ".join(ic.tipo_activo)
        style_data_cell(ws, r, 3, "#,##0.00").value = ic.monto
        style_data_cell(ws, r, 4, "0.00%").value = ic.peso_inversion / 100
        style_data_cell(ws, r, 5, "0.00").value = ic.risk_score
        style_data_cell(ws, r, 6, "0.0000").value = ic.aporte
        r += 1
    # Total
    ws.cell(row=r, column=1, value="TOTAL PONDERADO").font = Font(bold=True, name="Arial", size=10)
    ws.cell(row=r, column=6, value=d.score_total_weighted).font = Font(bold=True, name="Arial", size=10)
    ws.cell(row=r, column=6).number_format = "0.0000"
    r += 2

    # Profile range
    ws.cell(row=r, column=1, value="2. Rango objetivo según perfil").font = SECTION_FONT
    r += 1
    profiles = [
        ("Conservador", "7 – 8"), ("Conservador & Moderado", "6 – 7"),
        ("Moderado", "5 – 6"), ("Moderado & Agresivo", "4 – 5"), ("Agresivo", "3 – 4"),
    ]
    headers = ["Perfil", "Rango objetivo"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=r, column=i, value=h)
    style_header_row(ws, r, len(headers))
    r += 1
    for prof, rng in profiles:
        c = style_data_cell(ws, r, 1)
        c.value = prof
        if prof == d.perfil_riesgo:
            c.fill = YELLOW_FILL
            style_data_cell(ws, r, 2).fill = YELLOW_FILL
        style_data_cell(ws, r, 2).value = rng
        r += 1
    r += 1

    # D calculation explanation
    ws.cell(row=r, column=1, value="3. Cálculo de distancia al rango (d)").font = SECTION_FONT
    r += 1
    dc = d.d_calculation
    for label, val in [
        ("Score total ponderado", f"{dc.score_total:.4f}"),
        ("Rango perfil", f"[{dc.perfil_min}, {dc.perfil_max}]"),
        ("Zona central", f"[{dc.first_quarter}, {dc.third_quarter}]"),
        ("Zona borde inferior", f"[{dc.perfil_min}, {dc.first_quarter})"),
        ("Zona borde superior", f"({dc.third_quarter}, {dc.perfil_max}]"),
        ("Zona del score", dc.zone.replace("_", " ").title()),
        ("Punto de referencia", f"{dc.reference_point}"),
        ("d value", f"{dc.d_value}"),
    ]:
        ws.cell(row=r, column=1, value=label).font = NORMAL_FONT
        ws.cell(row=r, column=2, value=val).font = NORMAL_FONT
        r += 1
    r += 1

    # Score translation table
    ws.cell(row=r, column=1, value="4. Traducción a Score (1-10)").font = SECTION_FONT
    r += 1
    headers = ["Condición", "Score"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=r, column=i, value=h)
    style_header_row(ws, r, len(headers))
    r += 1
    score_table = [
        ("Zona central [L+0.25, U-0.25]", 10),
        ("Zona borde [L, L+0.25) o (U-0.25, U]", 9),
        ("d ∈ (0.00, 0.25]", 8), ("d ∈ (0.25, 0.50]", 8),
        ("d ∈ (0.50, 0.75]", 7), ("d ∈ (0.75, 1.00]", 6),
        ("d ∈ (1.00, 1.25]", 5), ("d ∈ (1.25, 1.50]", 4),
        ("d ∈ (1.50, 1.75]", 3), ("d ∈ (1.75, 2.00]", 2),
        ("d > 2.00", 1),
    ]
    for cond, sc in score_table:
        style_data_cell(ws, r, 1).value = cond
        style_data_cell(ws, r, 2).value = sc
        r += 1
    r += 1

    # Final result
    ws.cell(row=r, column=1, value="Score de Alineación por Riesgo").font = Font(bold=True, name="Arial", size=11)
    cell = ws.cell(row=r, column=2, value=f"{d.score} / 10")
    cell.font = SCORE_FONT
    cell.fill = YELLOW_FILL

    auto_width(ws)


def build_geo_sheet(wb, req: ReportRequest):
    ws = wb.create_sheet("Alineacion Geografica")
    d = req.geo_alignment
    r = 1

    ws.cell(row=r, column=1, value="ALINEACIÓN GEOGRÁFICA").font = TITLE_FONT
    r += 2

    # Benchmark
    ws.cell(row=r, column=1, value="1. Benchmark utilizado (Sabbi Cracks)").font = SECTION_FONT
    r += 1
    headers = ["Región", "Benchmark"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=r, column=i, value=h)
    style_header_row(ws, r, len(headers))
    r += 1
    for rd in d.region_details:
        style_data_cell(ws, r, 1).value = rd.region.title()
        style_data_cell(ws, r, 2, "0.00%").value = rd.benchmark_percentage / 100
        r += 1
    r += 1

    # Ranges
    ws.cell(row=r, column=1, value="2. Rangos permitidos por región").font = SECTION_FONT
    r += 1
    headers = ["Región", "Tolerancia", "Límite inferior", "Límite superior"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=r, column=i, value=h)
    style_header_row(ws, r, len(headers))
    r += 1
    for rd in d.region_details:
        style_data_cell(ws, r, 1).value = rd.region.title()
        style_data_cell(ws, r, 2).value = rd.tolerance
        style_data_cell(ws, r, 3, "0.00%").value = rd.min_limit / 100
        style_data_cell(ws, r, 4, "0.00%").value = rd.max_limit / 100
        r += 1
    r += 1

    # Comparison
    ws.cell(row=r, column=1, value="3. Distribución geográfica del portafolio").font = SECTION_FONT
    r += 1
    headers = ["Región", "Benchmark", "Lím. Inf.", "Lím. Sup.", req.inversionista.title(), "Desviación", "Penalización", "¿Dentro?"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=r, column=i, value=h)
    style_header_row(ws, r, len(headers))
    r += 1
    for rd in d.region_details:
        style_data_cell(ws, r, 1).value = rd.region.title()
        style_data_cell(ws, r, 2, "0.00%").value = rd.benchmark_percentage / 100
        style_data_cell(ws, r, 3, "0.00%").value = rd.min_limit / 100
        style_data_cell(ws, r, 4, "0.00%").value = rd.max_limit / 100
        style_data_cell(ws, r, 5, "0.00%").value = rd.portfolio_percentage / 100
        style_data_cell(ws, r, 6, "0.00%").value = rd.deviation / 100
        style_data_cell(ws, r, 7, "0.00%").value = rd.penalty / 100
        cell = style_data_cell(ws, r, 8)
        cell.value = "Sí" if rd.is_within_limits else "No"
        cell.fill = GREEN_FILL if rd.is_within_limits else RED_FILL
        r += 1
    r += 1

    # Score table
    ws.cell(row=r, column=1, value="4. Traducción a Score de Concentración Geográfica").font = SECTION_FONT
    r += 1
    headers = ["Desviación total", "Score", "Interpretación"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=r, column=i, value=h)
    style_header_row(ws, r, len(headers))
    r += 1
    score_table = [
        ("0 – 5%", 10, "Muy bien diversificado"),
        ("5 – 10%", 9, "Diversificación sólida"),
        ("10 – 20%", 8, "Desvío moderado"),
        ("20 – 30%", 6, "Riesgo relevante"),
        ("30 – 40%", 4, "Alta concentración"),
        ("40 – 55%", 3, "Riesgo elevado"),
        ("55 – 70%", 2, "Riesgo crítico"),
        ("> 70%", 1, "Riesgo extremo"),
    ]
    for dev, sc, interp in score_table:
        style_data_cell(ws, r, 1).value = dev
        style_data_cell(ws, r, 2).value = sc
        style_data_cell(ws, r, 3).value = interp
        r += 1
    r += 1

    # Final result
    ws.cell(row=r, column=1, value="Desviación geográfica total").font = NORMAL_FONT
    ws.cell(row=r, column=2, value=d.total_deviation / 100).number_format = "0.00%"
    r += 1
    ws.cell(row=r, column=1, value=f"Interpretación").font = NORMAL_FONT
    ws.cell(row=r, column=2, value=d.interpretation).font = NORMAL_FONT
    r += 1
    ws.cell(row=r, column=1, value="Score de Concentración Geográfica").font = Font(bold=True, name="Arial", size=11)
    cell = ws.cell(row=r, column=2, value=f"{d.score} / 10")
    cell.font = SCORE_FONT
    cell.fill = YELLOW_FILL

    auto_width(ws)


# ── Endpoints ────────────────────────────────────────────────────

@app.post("/generate-report")
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
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Prompt templates (from .md files) ────────────────────────────

PROMPT_ACTIVO = """Debes generar la información estructurada para la hoja "Alineacion Activo".
Basándote en los datos proporcionados, genera SOLO un JSON con la siguiente estructura exacta.
NO incluyas texto adicional, markdown, explicaciones, observaciones ni recomendaciones.
{{
  "universo": {{
    "inversionista": "<nombre>",
    "patrimonio_invertible": <number>,
    "perfil_riesgo": "<perfil>",
    "benchmark_aplicable": "<descripcion>"
  }},
  "benchmark": [
    {{"clase_activo": "<nombre>", "benchmark_pct": <number>}}
  ],
  "rangos": [
    {{"clase_activo": "<nombre>", "min_limit": <number>, "max_limit": <number>}}
  ],
  "comparacion": [
    {{"clase_activo": "<nombre>", "benchmark": <number>, "min_limit": <number>, "max_limit": <number>, "inversionista_pct": <number>, "desviacion": <number>, "penalizacion": <number>, "dentro": <boolean>}}
  ],
  "penalizacion_total": <number>,
  "distancia_estructural": <number>,
  "score": <number>
}}
Porcentajes como números (ej: 24.06, no 0.2406).

DATA:
inversionista: {inversionista}
total_patrimonio: {total_patrimonio}
perfil_riesgo: {perfil_riesgo}
{data_json}
"""

PROMPT_RIESGO = """Debes generar la información estructurada para la hoja "Alineacion Riesgo".
Basándote en los datos proporcionados, genera SOLO un JSON con la siguiente estructura exacta.
NO incluyas texto adicional, markdown, explicaciones, observaciones ni recomendaciones.
{{
  "investment_contributions": [
    {{"producto_nombre": "<nombre>", "tipo_activo": ["<tipo>"], "monto": <number>, "peso_inversion": <number>, "risk_score": <number>, "aporte": <number>}}
  ],
  "score_total_weighted": <number>,
  "perfil_riesgo": "<perfil>",
  "rango_objetivo": {{"min": <number>, "max": <number>}},
  "d_calculation": {{
    "score_total": <number>,
    "perfil_min": <number>,
    "perfil_max": <number>,
    "first_quarter": <number>,
    "third_quarter": <number>,
    "zone": "<zone_name>",
    "reference_point": <number>,
    "d_value": <number>
  }},
  "score": <number>
}}

DATA:
inversionista: {inversionista}
total_patrimonio: {total_patrimonio}
perfil_riesgo: {perfil_riesgo}
{data_json}
"""

PROMPT_GEO = """Debes generar la información estructurada para la hoja "Alineacion Geografica".
Basándote en los datos proporcionados, genera SOLO un JSON con la siguiente estructura exacta.
NO incluyas texto adicional, markdown, explicaciones, observaciones ni recomendaciones.
{{
  "benchmark": [
    {{"region": "<nombre>", "benchmark_pct": <number>}}
  ],
  "rangos": [
    {{"region": "<nombre>", "tolerancia": "<±X%>", "min_limit": <number>, "max_limit": <number>}}
  ],
  "comparacion": [
    {{"region": "<nombre>", "benchmark": <number>, "min_limit": <number>, "max_limit": <number>, "portfolio_pct": <number>, "desviacion": <number>, "penalizacion": <number>, "dentro": <boolean>}}
  ],
  "total_deviation": <number>,
  "interpretation": "<texto>",
  "score": <number>
}}
Porcentajes como números (ej: 46.5, no 0.465).

DATA:
inversionista: {inversionista}
total_patrimonio: {total_patrimonio}
perfil_riesgo: {perfil_riesgo}
{data_json}
"""


def _call_claude(prompt: str) -> dict:
    """Call Claude and parse JSON response."""
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def _build_sheet_from_ai(wb, sheet_name: str, ai_data: dict, dimension: str, req: ReportRequest):
    """Build a sheet from AI-returned structured JSON (no commentary)."""
    ws = wb.create_sheet(sheet_name)
    r = 1

    if dimension == "activo":
        ws.cell(row=r, column=1, value="ALINEACIÓN POR TIPO DE ACTIVO").font = TITLE_FONT
        r += 2
        u = ai_data["universo"]
        ws.cell(row=r, column=1, value="1. Universo del análisis").font = SECTION_FONT
        r += 1
        for label, val in [
            ("Inversionista", u["inversionista"]),
            ("Patrimonio invertible", f"USD {u['patrimonio_invertible']:,.2f}"),
            ("Perfil de riesgo", u["perfil_riesgo"]),
            ("Benchmark aplicable", u["benchmark_aplicable"]),
        ]:
            ws.cell(row=r, column=1, value=label).font = NORMAL_FONT
            ws.cell(row=r, column=2, value=val).font = NORMAL_FONT
            r += 1
        r += 1
        ws.cell(row=r, column=1, value="2. Benchmark utilizado").font = SECTION_FONT
        r += 1
        for i, h in enumerate(["Clase de activo", "Benchmark %"], 1):
            ws.cell(row=r, column=i, value=h)
        style_header_row(ws, r, 2)
        r += 1
        for item in ai_data["benchmark"]:
            style_data_cell(ws, r, 1).value = item["clase_activo"]
            style_data_cell(ws, r, 2, "0.00%").value = item["benchmark_pct"] / 100
            r += 1
        r += 1
        ws.cell(row=r, column=1, value="3. Rangos permitidos").font = SECTION_FONT
        r += 1
        for i, h in enumerate(["Clase de activo", "Límite inferior", "Límite superior"], 1):
            ws.cell(row=r, column=i, value=h)
        style_header_row(ws, r, 3)
        r += 1
        for item in ai_data["rangos"]:
            style_data_cell(ws, r, 1).value = item["clase_activo"]
            style_data_cell(ws, r, 2, "0.00%").value = item["min_limit"] / 100
            style_data_cell(ws, r, 3, "0.00%").value = item["max_limit"] / 100
            r += 1
        r += 1
        ws.cell(row=r, column=1, value="4. Comparación vs benchmark y penalizaciones").font = SECTION_FONT
        r += 1
        headers = ["Clase de activo", "Benchmark", "Rango Min", "Rango Max", req.inversionista.title(), "Desviación", "Penalización", "¿Dentro?"]
        for i, h in enumerate(headers, 1):
            ws.cell(row=r, column=i, value=h)
        style_header_row(ws, r, len(headers))
        r += 1
        for item in ai_data["comparacion"]:
            style_data_cell(ws, r, 1).value = item["clase_activo"]
            style_data_cell(ws, r, 2, "0.00%").value = item["benchmark"] / 100
            style_data_cell(ws, r, 3, "0.00%").value = item["min_limit"] / 100
            style_data_cell(ws, r, 4, "0.00%").value = item["max_limit"] / 100
            style_data_cell(ws, r, 5, "0.00%").value = item["inversionista_pct"] / 100
            style_data_cell(ws, r, 6, "0.00%").value = item["desviacion"] / 100
            style_data_cell(ws, r, 7, "0.00%").value = item["penalizacion"] / 100
            cell = style_data_cell(ws, r, 8)
            cell.value = "Sí" if item["dentro"] else "No"
            cell.fill = GREEN_FILL if item["dentro"] else RED_FILL
            r += 1
        ws.cell(row=r, column=1, value="TOTAL PENALIZACIÓN").font = Font(bold=True, name="Arial", size=10)
        ws.cell(row=r, column=7, value=ai_data["penalizacion_total"] / 100).number_format = "0.00%"
        r += 2
        ws.cell(row=r, column=1, value="Distancia estructural").font = NORMAL_FONT
        ws.cell(row=r, column=2, value=ai_data["distancia_estructural"] / 100).number_format = "0.00%"
        r += 1
        ws.cell(row=r, column=1, value="Score de Alineación por Tipo de Activo").font = Font(bold=True, name="Arial", size=11)
        cell = ws.cell(row=r, column=2, value=f"{ai_data['score']} / 10")
        cell.font = SCORE_FONT
        cell.fill = YELLOW_FILL

    elif dimension == "riesgo":
        ws.cell(row=r, column=1, value="ALINEACIÓN POR RIESGO").font = TITLE_FONT
        r += 2
        ws.cell(row=r, column=1, value="1. Score de performance ponderado por inversión").font = SECTION_FONT
        r += 1
        headers = ["Producto", "Tipo de activo", "Monto (USD)", "Peso (%)", "Risk Score", "Aporte"]
        for i, h in enumerate(headers, 1):
            ws.cell(row=r, column=i, value=h)
        style_header_row(ws, r, len(headers))
        r += 1
        for ic in ai_data["investment_contributions"]:
            style_data_cell(ws, r, 1).value = ic["producto_nombre"]
            tipos = ic["tipo_activo"]
            style_data_cell(ws, r, 2).value = ", ".join(tipos) if isinstance(tipos, list) else tipos
            style_data_cell(ws, r, 3, "#,##0.00").value = ic["monto"]
            style_data_cell(ws, r, 4, "0.00%").value = ic["peso_inversion"] / 100
            style_data_cell(ws, r, 5, "0.00").value = ic["risk_score"]
            style_data_cell(ws, r, 6, "0.0000").value = ic["aporte"]
            r += 1
        ws.cell(row=r, column=1, value="TOTAL PONDERADO").font = Font(bold=True, name="Arial", size=10)
        ws.cell(row=r, column=6, value=ai_data["score_total_weighted"]).number_format = "0.0000"
        ws.cell(row=r, column=6).font = Font(bold=True, name="Arial", size=10)
        r += 2
        ws.cell(row=r, column=1, value="2. Rango objetivo según perfil").font = SECTION_FONT
        r += 1
        profiles = [
            ("Conservador", "7 – 8"), ("Conservador & Moderado", "6 – 7"),
            ("Moderado", "5 – 6"), ("Moderado & Agresivo", "4 – 5"), ("Agresivo", "3 – 4"),
        ]
        for i, h in enumerate(["Perfil", "Rango objetivo"], 1):
            ws.cell(row=r, column=i, value=h)
        style_header_row(ws, r, 2)
        r += 1
        for prof, rng in profiles:
            c = style_data_cell(ws, r, 1)
            c.value = prof
            if prof == ai_data["perfil_riesgo"]:
                c.fill = YELLOW_FILL
                style_data_cell(ws, r, 2).fill = YELLOW_FILL
            style_data_cell(ws, r, 2).value = rng
            r += 1
        r += 1
        dc = ai_data["d_calculation"]
        ws.cell(row=r, column=1, value="3. Cálculo de distancia al rango (d)").font = SECTION_FONT
        r += 1
        for label, val in [
            ("Score total ponderado", f"{dc['score_total']:.4f}"),
            ("Rango perfil", f"[{dc['perfil_min']}, {dc['perfil_max']}]"),
            ("Zona central", f"[{dc['first_quarter']}, {dc['third_quarter']}]"),
            ("Zona del score", dc["zone"].replace("_", " ").title()),
            ("Punto de referencia", f"{dc['reference_point']}"),
            ("d value", f"{dc['d_value']}"),
        ]:
            ws.cell(row=r, column=1, value=label).font = NORMAL_FONT
            ws.cell(row=r, column=2, value=val).font = NORMAL_FONT
            r += 1
        r += 1
        ws.cell(row=r, column=1, value="Score de Alineación por Riesgo").font = Font(bold=True, name="Arial", size=11)
        cell = ws.cell(row=r, column=2, value=f"{ai_data['score']} / 10")
        cell.font = SCORE_FONT
        cell.fill = YELLOW_FILL

    elif dimension == "geografica":
        ws.cell(row=r, column=1, value="ALINEACIÓN GEOGRÁFICA").font = TITLE_FONT
        r += 2
        ws.cell(row=r, column=1, value="1. Benchmark utilizado (Sabbi Cracks)").font = SECTION_FONT
        r += 1
        for i, h in enumerate(["Región", "Benchmark"], 1):
            ws.cell(row=r, column=i, value=h)
        style_header_row(ws, r, 2)
        r += 1
        for item in ai_data["benchmark"]:
            style_data_cell(ws, r, 1).value = item["region"].title()
            style_data_cell(ws, r, 2, "0.00%").value = item["benchmark_pct"] / 100
            r += 1
        r += 1
        ws.cell(row=r, column=1, value="2. Rangos permitidos por región").font = SECTION_FONT
        r += 1
        for i, h in enumerate(["Región", "Tolerancia", "Límite inferior", "Límite superior"], 1):
            ws.cell(row=r, column=i, value=h)
        style_header_row(ws, r, 4)
        r += 1
        for item in ai_data["rangos"]:
            style_data_cell(ws, r, 1).value = item["region"].title()
            style_data_cell(ws, r, 2).value = item["tolerancia"]
            style_data_cell(ws, r, 3, "0.00%").value = item["min_limit"] / 100
            style_data_cell(ws, r, 4, "0.00%").value = item["max_limit"] / 100
            r += 1
        r += 1
        ws.cell(row=r, column=1, value="3. Distribución geográfica del portafolio").font = SECTION_FONT
        r += 1
        headers = ["Región", "Benchmark", "Lím. Inf.", "Lím. Sup.", req.inversionista.title(), "Desviación", "Penalización", "¿Dentro?"]
        for i, h in enumerate(headers, 1):
            ws.cell(row=r, column=i, value=h)
        style_header_row(ws, r, len(headers))
        r += 1
        for item in ai_data["comparacion"]:
            style_data_cell(ws, r, 1).value = item["region"].title()
            style_data_cell(ws, r, 2, "0.00%").value = item["benchmark"] / 100
            style_data_cell(ws, r, 3, "0.00%").value = item["min_limit"] / 100
            style_data_cell(ws, r, 4, "0.00%").value = item["max_limit"] / 100
            style_data_cell(ws, r, 5, "0.00%").value = item["portfolio_pct"] / 100
            style_data_cell(ws, r, 6, "0.00%").value = item["desviacion"] / 100
            style_data_cell(ws, r, 7, "0.00%").value = item["penalizacion"] / 100
            cell = style_data_cell(ws, r, 8)
            cell.value = "Sí" if item["dentro"] else "No"
            cell.fill = GREEN_FILL if item["dentro"] else RED_FILL
            r += 1
        r += 1
        ws.cell(row=r, column=1, value="Desviación geográfica total").font = NORMAL_FONT
        ws.cell(row=r, column=2, value=ai_data["total_deviation"] / 100).number_format = "0.00%"
        r += 1
        ws.cell(row=r, column=1, value="Interpretación").font = NORMAL_FONT
        ws.cell(row=r, column=2, value=ai_data["interpretation"]).font = NORMAL_FONT
        r += 1
        ws.cell(row=r, column=1, value="Score de Concentración Geográfica").font = Font(bold=True, name="Arial", size=11)
        cell = ws.cell(row=r, column=2, value=f"{ai_data['score']} / 10")
        cell.font = SCORE_FONT
        cell.fill = YELLOW_FILL

    auto_width(ws)


@app.post("/generate-report-with-ai")
async def generate_report_with_ai(req: ReportRequest):
    """
    Use Anthropic SDK to process each prompt template with the client data.
    Claude generates structured JSON only (no observations, no recommendations),
    which is then rendered into each Excel sheet.
    """
    prompts_and_data = [
        ("activo", PROMPT_ACTIVO, req.asset_alignment.model_dump()),
        ("riesgo", PROMPT_RIESGO, req.risk_alignment.model_dump()),
        ("geografica", PROMPT_GEO, req.geo_alignment.model_dump()),
    ]

    sheet_configs = {
        "activo": "Alineacion Activo",
        "riesgo": "Alineacion Riesgo",
        "geografica": "Alineacion Geografica",
    }

    ai_results = {}
    for dimension, prompt_template, data in prompts_and_data:
        prompt = prompt_template.format(
            inversionista=req.inversionista,
            total_patrimonio=req.total_patrimonio,
            perfil_riesgo=req.perfil_riesgo,
            data_json=json.dumps(data, ensure_ascii=False, indent=2),
        )
        ai_results[dimension] = _call_claude(prompt)

    wb = Workbook()
    wb.remove(wb.active)

    for dimension, sheet_name in sheet_configs.items():
        _build_sheet_from_ai(wb, sheet_name, ai_results[dimension], dimension, req)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"alineacion_{req.inversionista.replace(' ', '_').lower()}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
