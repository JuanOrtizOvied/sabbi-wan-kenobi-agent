from fastapi import APIRouter, HTTPException
from app.api.sabbi_experto.schemas import ChatRequest, ChatResponse
from app.services.sabbi_experto.agent_service import AgentService

sabbi_experto_router = APIRouter()
agent = AgentService()


@sabbi_experto_router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = agent.reply(
            user_text=req.message,
            previous_response_id=req.previous_response_id,
        )
        return ChatResponse(reply=out.text, response_id=out.response_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
