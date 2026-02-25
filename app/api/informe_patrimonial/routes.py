from fastapi import APIRouter, HTTPException
from app.api.informe_patrimonial.schemas import ChatRequest, ChatResponse
from app.services.informe_patrimonial.agent_service import AgentService

informe_patrimonial_router = APIRouter()
agent = AgentService()


@informe_patrimonial_router.post("/calidad-portafolio", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        out = agent.reply(
            user_text=req.message,
            previous_response_id=req.previous_response_id,
        )
        return ChatResponse(reply=out.text, response_id=out.response_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
