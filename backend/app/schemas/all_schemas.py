from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any


class UserRegister(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class MessageCreate(BaseModel):
    content: str
    role: str = "user"
    model: Optional[str] = "local-llama3"


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    model: Optional[str] = None
    created_at: Any

    model_config = ConfigDict(from_attributes=True)


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"
    model: Optional[str] = "local-llama3"
    mode: Optional[str] = "chat"
    project_id: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    title: str
    model: str
    mode: str
    project_id: Optional[str] = None
    is_archived: bool
    is_favorite: bool
    created_at: Any

    model_config = ConfigDict(from_attributes=True)


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    language: Optional[str] = "python"


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    language: Optional[str] = "python"
    created_at: Any

    model_config = ConfigDict(from_attributes=True)


class ProjectFileCreate(BaseModel):
    path: str
    content: str
    language: Optional[str] = None


class CodeRunRequest(BaseModel):
    code: str
    language: str = "python"


class CodeAnalyzeRequest(BaseModel):
    code: str
    language: str = "python"
    prompt: Optional[str] = "Analyze this code for performance, bugs, and security risks."
