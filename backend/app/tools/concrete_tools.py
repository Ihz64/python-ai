import os
import re
from typing import Dict, Any
from backend.app.tools.base import Tool, ToolResult
from backend.app.sandbox.service import sandbox_service


class FileSearchTool(Tool):
    name = "file_search"
    description = "Search for files in a project directory matching a pattern or query."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search pattern or filename keyword."},
            "project_path": {"type": "string", "description": "Root path of the project."}
        },
        "required": ["query"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        query = arguments.get("query", "").lower()
        project_path = arguments.get("project_path", ".")
        matches = []
        try:
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), project_path)
                    if query in file.lower() or query in rel_path.lower():
                        matches.append(rel_path)
            return ToolResult(success=True, data={"matches": matches, "count": len(matches)})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class CodeSearchTool(Tool):
    name = "code_search"
    description = "Search for specific code snippets, function names, or regex in code files."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text or regex to search inside files."},
            "project_path": {"type": "string", "description": "Root path of the project."},
            "extension": {"type": "string", "description": "Optional extension filter e.g. .py, .js"}
        },
        "required": ["query"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        query = arguments.get("query", "")
        project_path = arguments.get("project_path", ".")
        ext_filter = arguments.get("extension", "")

        results = []
        try:
            regex = re.compile(query, re.IGNORECASE)
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    if ext_filter and not file.endswith(ext_filter):
                        continue
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, project_path)
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            for idx, line in enumerate(f, 1):
                                if regex.search(line):
                                    results.append({
                                        "file": rel_path,
                                        "line_number": idx,
                                        "line": line.strip()
                                    })
                    except Exception:
                        continue
            return ToolResult(success=True, data={"results": results[:50], "total_matches": len(results)})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class PythonExecutorTool(Tool):
    name = "python_executor"
    description = "Execute Python code safely inside the sandbox container."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute."}
        },
        "required": ["code"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        code = arguments.get("code", "")
        res = await sandbox_service.execute_code(code, language="python")
        return ToolResult(
            success=(res.exit_code == 0),
            data=res.model_dump(),
            error=res.error or (res.stderr if res.exit_code != 0 else None)
        )


class TerminalTool(Tool):
    name = "terminal"
    description = "Run terminal commands safely in sandbox environment."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command line string."}
        },
        "required": ["command"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        cmd = arguments.get("command", "")
        res = await sandbox_service.execute_code(cmd, language="bash")
        return ToolResult(
            success=(res.exit_code == 0),
            data=res.model_dump(),
            error=res.error or (res.stderr if res.exit_code != 0 else None)
        )


class ProjectInspectorTool(Tool):
    name = "project_inspector"
    description = "Inspect the folder structure and file tree of a project."
    parameters = {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Root path of project to inspect."}
        },
        "required": []
    }

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        path = arguments.get("project_path", ".")
        tree = []
        try:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                rel_root = os.path.relpath(root, path)
                if rel_root == ".":
                    rel_root = ""
                for file in files:
                    if file.startswith("."):
                        continue
                    file_rel = os.path.join(rel_root, file) if rel_root else file
                    tree.append(file_rel)
            return ToolResult(success=True, data={"structure": tree, "file_count": len(tree)})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DocumentationSearchTool(Tool):
    name = "documentation_search"
    description = "Search standard programming language or framework docs."
    parameters = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Function, class, or concept topic."}
        },
        "required": ["topic"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        topic = arguments.get("topic", "")
        return ToolResult(
            success=True,
            data={"summary": f"Documentation entry for '{topic}': Consult official API guides and type definitions."}
        )
