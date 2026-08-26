from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    data: Any
    error: Optional[str] = None


class Tool(ABC):
    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        pass

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_schemas(self) -> list:
        return [tool.get_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, data=None, error=f"Tool '{name}' not found.")
        try:
            return await tool.execute(arguments)
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Tool execution error: {str(e)}")


tool_registry = ToolRegistry()
