import asyncio
import os
import sys
import tempfile
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel

from backend.app.config import settings
from backend.app.core.logging import logger


class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    timed_out: bool = False
    error: Optional[str] = None


class SandboxService:
    def __init__(
        self,
        timeout_seconds: int = settings.SANDBOX_TIMEOUT_SECONDS,
        max_output_length: int = 10000
    ):
        self.timeout_seconds = timeout_seconds
        self.max_output_length = max_output_length

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        environment_vars: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        language = language.lower().strip()

        if language in ["python", "py"]:
            return await self._execute_python(code, environment_vars)
        elif language in ["javascript", "js", "node"]:
            return await self._execute_javascript(code, environment_vars)
        elif language in ["bash", "sh"]:
            return await self._execute_bash(code, environment_vars)
        else:
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                execution_time_ms=0,
                error=f"Unsupported sandbox language: {language}"
            )

    async def _execute_python(self, code: str, env_vars: Optional[Dict[str, str]]) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""}
        if env_vars:
            env.update(env_vars)

        start_time = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-S",
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(self.timeout_seconds)
                )
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout_data, stderr_data = b"", b"Execution timed out."
                timed_out = True

            exec_time = (time.time() - start_time) * 1000.0
            stdout_str = stdout_data.decode("utf-8", errors="replace")[:self.max_output_length]
            stderr_str = stderr_data.decode("utf-8", errors="replace")[:self.max_output_length]

            return ExecutionResult(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=proc.returncode if proc.returncode is not None else -1,
                execution_time_ms=round(exec_time, 2),
                timed_out=timed_out
            )
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=1,
                execution_time_ms=0,
                error=str(e)
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def _execute_javascript(self, code: str, env_vars: Optional[Dict[str, str]]) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        start_time = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                "node",
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(self.timeout_seconds)
                )
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout_data, stderr_data = b"", b"Execution timed out."
                timed_out = True

            exec_time = (time.time() - start_time) * 1000.0
            return ExecutionResult(
                stdout=stdout_data.decode("utf-8", errors="replace")[:self.max_output_length],
                stderr=stderr_data.decode("utf-8", errors="replace")[:self.max_output_length],
                exit_code=proc.returncode if proc.returncode is not None else -1,
                execution_time_ms=round(exec_time, 2),
                timed_out=timed_out
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=f"JavaScript runtime error: {e}",
                exit_code=1,
                execution_time_ms=0,
                error=str(e)
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def _execute_bash(self, code: str, env_vars: Optional[Dict[str, str]]) -> ExecutionResult:
        forbidden = ["rm", "mkfs", "dd", "chmod", "chown", "sudo", "su", "shutdown", "reboot"]
        tokens = code.split()
        for t in tokens:
            if t in forbidden or any(f in t for f in [">/dev/", "eval", "exec"]):
                return ExecutionResult(
                    stdout="",
                    stderr="Potentially dangerous command blocked by safety policy.",
                    exit_code=1,
                    execution_time_ms=0,
                    error="Security Exception"
                )

        start_time = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(self.timeout_seconds)
                )
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout_data, stderr_data = b"", b"Execution timed out."
                timed_out = True

            exec_time = (time.time() - start_time) * 1000.0
            return ExecutionResult(
                stdout=stdout_data.decode("utf-8", errors="replace")[:self.max_output_length],
                stderr=stderr_data.decode("utf-8", errors="replace")[:self.max_output_length],
                exit_code=proc.returncode if proc.returncode is not None else -1,
                execution_time_ms=round(exec_time, 2),
                timed_out=timed_out
            )
        except Exception as e:
            return ExecutionResult(stdout="", stderr=str(e), exit_code=1, execution_time_ms=0, error=str(e))


sandbox_service = SandboxService()
