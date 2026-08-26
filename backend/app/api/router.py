from fastapi import APIRouter
from backend.app.api.endpoints import auth, chats, projects, code, files, models_tools, websocket

api_prefix = "/api"

router = APIRouter()
router.include_router(auth.router, prefix=api_prefix)
router.include_router(chats.router, prefix=api_prefix)
router.include_router(projects.router, prefix=api_prefix)
router.include_router(code.router, prefix=api_prefix)
router.include_router(files.router, prefix=api_prefix)
router.include_router(models_tools.router, prefix=api_prefix)
