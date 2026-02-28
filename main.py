from fastapi import FastAPI

from app.api.routes import router
from app.api.sabbi_experto.routes import sabbi_experto_router
from app.api.informe_patrimonial.routes import informe_patrimonial_router
from app.services.agent_service import AgentService

app = FastAPI(title="Sabbi WOW Philosophy Agent")

# Service singleton (engine compartido)
service = AgentService()
app.state.service = service

# Routes
app.include_router(router)
app.include_router(sabbi_experto_router, prefix="/sabbi-experto")
app.include_router(informe_patrimonial_router, prefix="/informe-patrimonial")


@app.on_event("startup")
async def on_startup():
    await app.state.service.startup()
