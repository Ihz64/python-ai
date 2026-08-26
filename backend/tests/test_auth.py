import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.database.init_db import init_db

@pytest.mark.asyncio
async def test_auth_and_user_flow():
    await init_db()
    uname = f"tester_{uuid.uuid4().hex[:6]}"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        reg_resp = await client.post("/api/auth/register", json={
            "username": uname,
            "email": f"{uname}@codemind.ai",
            "password": "securepassword123"
        })
        assert reg_resp.status_code == 200
        data = reg_resp.json()
        assert "access_token" in data
        token = data["access_token"]

        login_resp = await client.post("/api/auth/login", json={
            "username": uname,
            "password": "securepassword123"
        })
        assert login_resp.status_code == 200

        me_resp = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.status_code == 200
        user = me_resp.json()
        assert user["username"] == uname
