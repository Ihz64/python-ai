from backend.app.database.session import engine, Base
from backend.app.models.all_models import User, Conversation, Message, Project, ProjectFile

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())
