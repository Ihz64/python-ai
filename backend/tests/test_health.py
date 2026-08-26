import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.database.init_db import init_db

@pytest.mark.asyncio
async def test_health_check():
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["app"] == "CodeMind AI"
