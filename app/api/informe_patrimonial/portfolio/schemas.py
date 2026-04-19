from typing import List

from pydantic import BaseModel, Field


class ClaseActivo(BaseModel):
    clase: str
    porcentaje: float
    slugs: List[str] = Field(default_factory=list)


class FocoGeografico(BaseModel):
    nombre: str
    porcentaje: float
    slugs: List[str] = Field(default_factory=list)


class TipoActivo(BaseModel):
    nombre: str
    porcentaje: float
    slugs: List[str] = Field(default_factory=list)


class PortafolioItem(BaseModel):
    nombre: str
    monto: float
    clases_activo: List[ClaseActivo] = Field(default_factory=list)
    foco_geografico: List[FocoGeografico] = Field(default_factory=list)
    tipo_activo: List[TipoActivo] = Field(default_factory=list)
