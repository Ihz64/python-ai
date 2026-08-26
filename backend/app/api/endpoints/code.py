from fastapi import APIRouter
from backend.app.schemas.all_schemas import CodeRunRequest, CodeAnalyzeRequest
from backend.app.sandbox.service import sandbox_service
from backend.app.providers.model_registry import model_registry
from backend.app.providers.base import LLMMessage
from backend.app.services.diff_engine import diff_engine

router = APIRouter(prefix="/code", tags=["code"])


@router.post("/run")
async def run_code(req: CodeRunRequest):
    result = await sandbox_service.execute_code(req.code, language=req.language)
    return result.model_dump()


@router.post("/analyze")
async def analyze_code(req: CodeAnalyzeRequest):
    provider = model_registry.get_provider_for_model("local-llama3")
    messages = [
        LLMMessage(role="system", content="You are a Code Review Specialist. Analyze code for security, bugs, performance, and best practices."),
        LLMMessage(role="user", content=f"Language: {req.language}\n\nCode:\n```\n{req.code}\n```\n\nPrompt: {req.prompt}")
    ]
    resp = await provider.generate(messages, model="local-llama3")
    return {"analysis": resp.content, "model": resp.model}


@router.post("/diff")
async def create_diff(original: str, modified: str, filename: str = "main.py"):
    diff_patch = diff_engine.generate_unified_diff(original, modified, filename)
    return {
        "filename": filename,
        "diff_patch": diff_patch,
        "change_detected": len(diff_patch) > 0
    }
