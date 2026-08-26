import pytest
from backend.app.sandbox.service import sandbox_service
from backend.app.tools.base import tool_registry

@pytest.mark.asyncio
async def test_sandbox_execution():
    result = await sandbox_service.execute_code("print(10 + 32)", language="python")
    assert result.exit_code == 0
    assert "42" in result.stdout

@pytest.mark.asyncio
async def test_tool_registry():
    python_tool = tool_registry.get("python_executor")
    assert python_tool is not None

    res = await tool_registry.execute("python_executor", {"code": "print('Tool test passed')"})
    assert res.success is True
    assert "Tool test passed" in res.data["stdout"]
