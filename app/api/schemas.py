from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique id per user/conversation thread")
    message: str = Field(..., min_length=1, description="User message (one turn)")

    # ✅ Vienen en el API como body
    portafolio_promedio: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON con portafolio promedio. Requerido al menos en el primer mensaje de la sesión.",
    )
    portafolio_inversionista: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON con portafolio del inversionista. Requerido al menos en el primer mensaje de la sesión.",
    )

    # ✅ Opcionales (también vienen en el body)
    mi_filosofia: Optional[str] = Field(default=None, description="Texto libre del inversionista (opcional)")
    club_deals_information: Optional[str] = Field(default=None, description="Definición/racional de Club Deals (opcional)")


class ChatResponse(BaseModel):
    session_id: str
    output_text: str
