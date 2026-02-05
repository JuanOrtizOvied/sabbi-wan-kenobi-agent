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
            portafolio_promedio=req.portafolio_promedio,
            portafolio_inversionista=req.portafolio_inversionista,
            mi_filosofia=req.mi_filosofia,
            club_deals_concepts=req.club_deals_concepts,
            club_deals_opinion=req.club_deals_opinion,
            club_deals_information=req.club_deals_information,
        )
        return ChatResponse(session_id=req.session_id, output_text=output_text)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
