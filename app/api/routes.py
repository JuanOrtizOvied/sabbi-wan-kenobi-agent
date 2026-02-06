# app/api/routes.py
from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    try:
        service = request.app.state.service
        output_text = await service.chat(
            session_id=req.session_id,
            message=req.message,
            portafolio_promedio=getattr(req, "portafolio_promedio", None),
            portafolio_inversionista=getattr(req, "portafolio_inversionista", None),
            mi_filosofia=getattr(req, "mi_filosofia", None),
            club_deals_concepts=getattr(req, "club_deals_concepts", None),
            club_deals_opinion=getattr(req, "club_deals_opinion", None),
            # legacy
            club_deals_information=getattr(req, "club_deals_information", None),
        )
        return ChatResponse(session_id=req.session_id, output_text=output_text)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
