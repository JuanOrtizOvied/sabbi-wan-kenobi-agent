from fastapi import FastAPI

from app.api.routes import router
from app.services.agent_service import AgentService

app = FastAPI(title="Sabbi WOW Philosophy Agent")

# Service singleton (engine compartido)
service = AgentService()
app.state.service = service

# Routes
app.include_router(router)


@app.on_event("startup")
async def on_startup():
    await app.state.service.startup()
