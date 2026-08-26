from fastapi import APIRouter
from backend.app.providers.model_registry import model_registry
from backend.app.tools.base import tool_registry

router = APIRouter(tags=["system"])


@router.get("/models")
async def list_models():
    return {"models": model_registry.list_models()}


@router.get("/tools")
async def list_tools():
    return {"tools": tool_registry.list_schemas()}
