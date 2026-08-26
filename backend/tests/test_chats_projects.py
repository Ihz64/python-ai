import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.database.init_db import init_db

@pytest.mark.asyncio
async def test_chats_and_projects():
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        proj_resp = await client.post("/api/projects", json={
            "name": "WeatherApp",
            "description": "Weather forecasting web app",
            "language": "python"
        })
        assert proj_resp.status_code == 200
        proj = proj_resp.json()
        proj_id = proj["id"]

        file_resp = await client.post(f"/api/projects/{proj_id}/files", json={
            "path": "main.py",
            "content": "print('Hello Weather')"
        })
        assert file_resp.status_code == 200

        content_resp = await client.get(f"/api/projects/{proj_id}/files/content?path=main.py")
        assert content_resp.status_code == 200
        assert content_resp.json()["content"] == "print('Hello Weather')"

        chat_resp = await client.post("/api/chats", json={
            "title": "Weather App Discussion",
            "model": "local-llama3",
            "mode": "coding",
            "project_id": proj_id
        })
        assert chat_resp.status_code == 200
        conv = chat_resp.json()
        assert conv["mode"] == "coding"
