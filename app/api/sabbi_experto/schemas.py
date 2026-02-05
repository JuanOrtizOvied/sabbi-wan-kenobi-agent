from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_key: str = Field(..., description="Stable ID per WhatsApp contact/conversation")
    message: str = Field(..., min_length=1)
    previous_response_id: Optional[str] = Field(
        default=None,
        description="Pass the last OpenAI response.id here to continue the conversation chain"
    )


class ChatResponse(BaseModel):
    reply: str
    response_id: str
