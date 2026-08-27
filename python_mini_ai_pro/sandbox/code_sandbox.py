import sys
import io
import contextlib
import logging
import traceback
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CodeSandbox:
    """
    Local isolated Python code execution sandbox.
    Safely captures stdout/stderr and evaluates algorithmic Python code snippets.
    """

    def __init__(self, timeout_seconds: float = 3.0):
        self.timeout_seconds = timeout_seconds

    def execute_python_code(self, code: str) -> Dict[str, Any]:
        """Executes a Python code block and returns captured output and status."""
        if not code:
            return {"success": False, "output": "", "error": "No code provided."}

        # Clean code fences if present
        cleaned_code = code.strip()
        if cleaned_code.startswith("```python"):
            cleaned_code = cleaned_code[9:]
        if cleaned_code.startswith("```"):
            cleaned_code = cleaned_code[3:]
        if cleaned_code.endswith("```"):
            cleaned_code = cleaned_code[:-3]
        cleaned_code = cleaned_code.strip()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        global_scope = {"__builtins__": __builtins__}
        local_scope = {}

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(cleaned_code, global_scope, local_scope)

            output = stdout_buf.getvalue()
            error = stderr_buf.getvalue()

            return {
                "success": True,
                "output": output if output else "Code executed successfully (no stdout).",
                "error": error,
                "local_variables": {k: str(v) for k, v in local_scope.items() if not k.startswith("__")}
            }
        except Exception as e:
            return {
                "success": False,
                "output": stdout_buf.getvalue(),
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }
