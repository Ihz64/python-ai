from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.database.session import get_db
from backend.app.models.all_models import Project
from backend.app.schemas.all_schemas import ProjectCreate, ProjectResponse, ProjectFileCreate
from backend.app.services.project_service import project_service
from backend.app.core.security import get_current_user_optional, User

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=List[ProjectResponse])
async def list_projects(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_optional)
):
    stmt = select(Project).order_by(Project.created_at.desc())
    if user:
        stmt = stmt.where(Project.user_id == user.id)
    res = await db.execute(stmt)
    projs = res.scalars().all()
    return [ProjectResponse.model_validate(p) for p in projs]


@router.post("", response_model=ProjectResponse)
async def create_project(
    proj_in: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_optional)
):
    proj = Project(
        name=proj_in.name,
        description=proj_in.description,
        language=proj_in.language or "python",
        user_id=user.id if user else None
    )
    db.add(proj)
    await db.commit()
    await db.refresh(proj)

    project_service.create_project(proj.id, proj.name)
    return ProjectResponse.model_validate(proj)


@router.get("/{project_id}/files")
async def get_project_files(project_id: str):
    try:
        files = project_service.list_files(project_id)
        return {"project_id": project_id, "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/files")
async def save_project_file(project_id: str, file_in: ProjectFileCreate):
    try:
        rel_path = project_service.write_file(project_id, file_in.path, file_in.content)
        return {"status": "saved", "path": rel_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/files/content")
async def get_project_file_content(project_id: str, path: str):
    try:
        content = project_service.read_file(project_id, path)
        return {"project_id": project_id, "path": path, "content": content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
