from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.database.session import get_db
from backend.app.models.all_models import Conversation, Message
from backend.app.schemas.all_schemas import ConversationCreate, ConversationResponse, MessageCreate, MessageResponse
from backend.app.core.security import get_current_user_optional, User

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_optional)
):
    stmt = select(Conversation).order_by(Conversation.updated_at.desc())
    if user:
        stmt = stmt.where(Conversation.user_id == user.id)
    res = await db.execute(stmt)
    convs = res.scalars().all()
    return [ConversationResponse.model_validate(c) for c in convs]


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    conv_in: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_optional)
):
    conv = Conversation(
        title=conv_in.title or "New Conversation",
        model=conv_in.model or "local-llama3",
        mode=conv_in.mode or "chat",
        project_id=conv_in.project_id,
        user_id=user.id if user else None
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return ConversationResponse.model_validate(conv)


@router.get("/{chat_id}", response_model=ConversationResponse)
async def get_conversation(chat_id: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Conversation).where(Conversation.id == chat_id))
    conv = res.scalars().first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ConversationResponse.model_validate(conv)


@router.delete("/{chat_id}")
async def delete_conversation(chat_id: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Conversation).where(Conversation.id == chat_id))
    conv = res.scalars().first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    await db.delete(conv)
    await db.commit()
    return {"status": "success", "deleted": chat_id}


@router.get("/{chat_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(chat_id: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Message).where(Message.conversation_id == chat_id).order_by(Message.created_at.asc()))
    msgs = res.scalars().all()
    return [MessageResponse.model_validate(m) for m in msgs]
