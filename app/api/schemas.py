from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique id per user/conversation thread")
    message: str = Field(..., min_length=1, description="User message (one turn)")

    # ✅ Opcionales
    portafolio_promedio: Optional[List[Dict[str, Any]]] = Field(default=None, description="JSON con portafolio promedio (opcional).")
    portafolio_inversionista: Optional[List[Dict[str, Any]]] = Field(default=None, description="JSON con portafolio del inversionista (opcional).")
    mi_filosofia: Optional[str] = Field(default=None, description="Texto libre del inversionista (opcional)")

    # ✅ Club Deals (ambos opcionales)
    club_deals_concepts: Optional[str] = Field(default=None, description="Conceptos/definición de Club Deals (opcional).")
    club_deals_opinion: Optional[str] = Field(default=None, description="Qué piensa el inversionista de Club Deals (opcional).")

    # ✅ Legacy (si tu cliente aún lo envía)
    club_deals_information: Optional[str] = Field(default=None, description="LEGACY: se interpreta como club_deals_concepts.")


class ChatResponse(BaseModel):
    session_id: str
    output_text: str
