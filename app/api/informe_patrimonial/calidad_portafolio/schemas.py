from pydantic import BaseModel


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
    unmapped_regions: list = []
    summary: GeoSummary


class ReportRequest(BaseModel):
    inversionista: str
    total_patrimonio: float
    perfil_riesgo: str
    asset_alignment: AssetAlignmentData
    risk_alignment: RiskAlignmentData
    geo_alignment: GeoAlignmentData
