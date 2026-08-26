from backend.app.tools.base import Tool, ToolResult, tool_registry
from backend.app.tools.concrete_tools import (
    FileSearchTool,
    CodeSearchTool,
    PythonExecutorTool,
    TerminalTool,
    ProjectInspectorTool,
    DocumentationSearchTool
)

tool_registry.register(FileSearchTool())
tool_registry.register(CodeSearchTool())
tool_registry.register(PythonExecutorTool())
tool_registry.register(TerminalTool())
tool_registry.register(ProjectInspectorTool())
tool_registry.register(DocumentationSearchTool())

__all__ = [
    "Tool",
    "ToolResult",
    "tool_registry",
    "FileSearchTool",
    "CodeSearchTool",
    "PythonExecutorTool",
    "TerminalTool",
    "ProjectInspectorTool",
    "DocumentationSearchTool",
]
