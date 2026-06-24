# schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field


class NestedItem(BaseModel):
    name: str
    percentage: float
    slugs: List[str] = Field(default_factory=list)


class PortafolioItem(BaseModel):
    cuenta_bancaria_inversion: str
    tipo_activo: str
    pertenencia: str
    moneda_invertida: Optional[str] = None
    valor_estimado_usd: float
    rendimiento_anual_porcentaje: Optional[str] = None
    name: str
    slugs: List[str] = Field(default_factory=list)
    comision_sin_igv: str
    moneda: str
    administrador: str
    gestor: str
    liquidez: str
    clase_activo: List[NestedItem] = Field(default_factory=list)
    foco_geografico: List[NestedItem] = Field(default_factory=list)
    subyacente: List[NestedItem] = Field(default_factory=list)
