import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from backend.app.database.session import AsyncSessionLocal
from backend.app.models.all_models import Message
from backend.app.agents.coding_agent import coding_agent
from backend.app.providers.base import LLMMessage
from backend.app.core.logging import logger

router = APIRouter()


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket chat connection established.")

    try:
        while True:
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)

            conversation_id = data.get("conversation_id")
            user_text = data.get("message", "")
            model = data.get("model", "local-llama3")
            mode = data.get("mode", "chat")
            project_context = data.get("project_context")

            if not user_text.strip():
                continue

            async with AsyncSessionLocal() as db:
                user_msg = Message(
                    conversation_id=conversation_id,
                    role="user",
                    content=user_text,
                    model=model
                )
                db.add(user_msg)
                await db.commit()

                hist_res = await db.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at.asc())
                )
                history_msgs = hist_res.scalars().all()
                llm_history = [
                    LLMMessage(role=m.role, content=m.content) for m in history_msgs
                ]

            full_assistant_reply = ""
            coding_agent.model_id = model

            async for update in coding_agent.run(
                user_prompt=user_text,
                conversation_history=llm_history,
                mode=mode,
                project_context=project_context
            ):
                if update.output_chunk:
                    full_assistant_reply += update.output_chunk

                packet = {
                    "event": "step_update",
                    "state": update.state,
                    "message": update.message,
                    "chunk": update.output_chunk
                }
                await websocket.send_text(json.dumps(packet))

            async with AsyncSessionLocal() as db:
                assistant_msg = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_assistant_reply,
                    model=model
                )
                db.add(assistant_msg)
                await db.commit()

            await websocket.send_text(json.dumps({
                "event": "done",
                "conversation_id": conversation_id,
                "full_content": full_assistant_reply
            }))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected gracefully.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"event": "error", "error": str(e)}))
        except Exception:
            pass
