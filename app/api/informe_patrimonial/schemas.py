from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    json_data: Dict[str, Any] = Field(..., description="Score JSON payload for the portfolio report")
    previous_response_id: Optional[str] = Field(None, description="Response ID for multi-turn conversations")


class ChatResponse(BaseModel):
    reply: Dict[str, Any] = Field(..., description="Score JSON payload for the portfolio report")
    response_id: str
