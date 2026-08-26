import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.core.logging import setup_logging
from backend.app.database.init_db import init_db

setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="CodeMind AI Platform - Modular AI Engine & Coding Environment",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.PROJECTS_DIR, exist_ok=True)
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)

from backend.app.api.router import api_prefix
from backend.app.api.endpoints import auth, chats, projects, code, files, models_tools, websocket

app.include_router(auth.router, prefix=api_prefix)
app.include_router(chats.router, prefix=api_prefix)
app.include_router(projects.router, prefix=api_prefix)
app.include_router(code.router, prefix=api_prefix)
app.include_router(files.router, prefix=api_prefix)
app.include_router(models_tools.router, prefix=api_prefix)
app.include_router(websocket.router)

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }

if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")
