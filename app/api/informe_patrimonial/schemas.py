from typing import Dict
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    json_data: Dict[str, any] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    message: str
