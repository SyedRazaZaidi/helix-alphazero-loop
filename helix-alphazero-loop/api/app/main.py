from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .engine import warmup
from .routers import play


@asynccontextmanager
async def lifespan(_: FastAPI):
    warmup()
    yield


app = FastAPI(
    title="Helix",
    version="1.0.0",
    description="AlphaZero loop on four games — Connect Four, Gomoku, Hex, Othello",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(play.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"ok": "helix"}
