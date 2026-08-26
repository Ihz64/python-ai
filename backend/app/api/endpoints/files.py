import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.config import settings

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_exts = {
        ".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json", ".xml", ".csv",
        ".java", ".cpp", ".cs", ".go", ".rs", ".pdf", ".png", ".jpg", ".jpeg"
    }
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' is not supported.")

    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}_{file.filename}"
    file_path = os.path.join(settings.UPLOADS_DIR, safe_name)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "saved_path": file_path,
        "size_bytes": len(content),
        "extension": ext
    }
