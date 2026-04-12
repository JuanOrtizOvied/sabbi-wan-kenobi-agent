from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class ConcentracionItem(BaseModel):
    nombre: str
    subyacentes: List[Dict[str, Any]] = []
    monto: float
    weight: float
    weight_2: float


class ConcentracionResult(BaseModel):
    items: List[ConcentracionItem]
    hhi: float
    hhi_score: int
    hhi_interpretacion: str = ""
    max_weight: float
    max_weight_nombre: str
    max_weight_score: int
    max_weight_interpretacion: str = ""
    inversiones_totales: float
    score: float
    total: float = 0.0
    interpretacion: str = ""


class CorrelacionCell(BaseModel):
    asset_i: str
    asset_j: str
    weight_i: float
    weight_j: float
    corr: float
    value: float


class LookupWarning(BaseModel):
    dimension: str
    value: str
    item: str
    weight: float


class CorrelacionResult(BaseModel):
    subyacentes_weights: Dict[str, float]
    correlation_matrix: List[CorrelacionCell]
    total_correlation: float
    score: int
    interpretacion: str = ""
    unmatched: List[LookupWarning] = []


class EntityGroup(BaseModel):
    name: str
    weight: float
    points: float
    weighted_points: float


class EntityScoreResult(BaseModel):
    groups: List[EntityGroup]
    score: float
    unmatched: List[LookupWarning] = []


class MonedaResult(BaseModel):
    pen_total: float
    pen_pct: float
    score: int


class ReportRequest(BaseModel):
    """Full structural risk scoring payload."""
    concentracion: ConcentracionResult
    correlacion: CorrelacionResult
    gestor: EntityScoreResult
    administrador: EntityScoreResult
    moneda: MonedaResult
    global_score: float
    has_warnings: bool = False
