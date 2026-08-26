import os
import json
from enum import Enum
from typing import List, Dict, Any, AsyncGenerator, Optional
from pydantic import BaseModel

from backend.app.providers.base import LLMMessage
from backend.app.providers.model_registry import model_registry
from backend.app.tools.base import tool_registry
from backend.app.core.logging import logger


class AgentState(str, Enum):
    IDLE = "IDLE"
    ANALYZING = "ANALYZING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    TESTING = "TESTING"
    FIXING = "FIXING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class AgentStepUpdate(BaseModel):
    state: AgentState
    message: str
    tool_call: Optional[Dict[str, Any]] = None
    output_chunk: Optional[str] = None


class ContextManager:
    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @classmethod
    def prune_messages(
        cls,
        messages: List[LLMMessage],
        max_context_tokens: int = 8000
    ) -> List[LLMMessage]:
        if not messages:
            return []

        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        current_tokens = sum(cls.estimate_tokens(m.content) for m in system_msgs)
        retained: List[LLMMessage] = []

        for m in reversed(other_msgs):
            t = cls.estimate_tokens(m.content)
            if current_tokens + t > max_context_tokens and len(retained) > 2:
                break
            retained.append(m)
            current_tokens += t

        retained.reverse()
        return system_msgs + retained


class CodingAgent:
    def __init__(self, model_id: str = "local-llama3"):
        self.model_id = model_id
        self.state = AgentState.IDLE
        self.prompt_dir = os.path.join(os.path.dirname(__file__), "..", "prompts", "system")

    def _load_prompt(self, name: str) -> str:
        file_path = os.path.join(self.prompt_dir, f"{name}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "You are CodeMind AI Coding Assistant."

    async def run(
        self,
        user_prompt: str,
        conversation_history: List[LLMMessage],
        mode: str = "coding",
        project_context: Optional[str] = None
    ) -> AsyncGenerator[AgentStepUpdate, None]:
        self.state = AgentState.ANALYZING
        yield AgentStepUpdate(
            state=self.state,
            message="Analyzing user request and project context..."
        )

        system_prompt = self._load_prompt(mode if mode in ["coding", "debugging", "architecture", "testing"] else "coding")
        if project_context:
            system_prompt += f"\n\nCURRENT PROJECT CONTEXT:\n{project_context}"

        messages = [LLMMessage(role="system", content=system_prompt)] + conversation_history
        if not conversation_history or conversation_history[-1].role != "user":
            messages.append(LLMMessage(role="user", content=user_prompt))

        pruned_messages = ContextManager.prune_messages(messages, max_context_tokens=12000)

        self.state = AgentState.PLANNING
        yield AgentStepUpdate(
            state=self.state,
            message="Formulating execution plan and selecting tools..."
        )

        provider = model_registry.get_provider_for_model(self.model_id)
        tools_schema = tool_registry.list_schemas()

        lower_prompt = user_prompt.lower()
        tool_call_result = None

        if "run python" in lower_prompt or "execute python" in lower_prompt:
            tool_name = "python_executor"
            code_body = user_prompt.split("```python")[-1].split("```")[0].strip() if "```python" in user_prompt else user_prompt
            tool_args = {"code": code_body}

            self.state = AgentState.EXECUTING
            yield AgentStepUpdate(
                state=self.state,
                message=f"Executing tool '{tool_name}' in sandbox...",
                tool_call={"tool": tool_name, "arguments": tool_args}
            )

            res = await tool_registry.execute(tool_name, tool_args)
            tool_call_result = res.data

        self.state = AgentState.EXECUTING
        yield AgentStepUpdate(
            state=self.state,
            message="Generating AI response and processing actions..."
        )

        try:
            if tool_call_result:
                pruned_messages.append(LLMMessage(
                    role="system",
                    content=f"TOOL EXECUTION RESULT:\n{json.dumps(tool_call_result, indent=2)}"
                ))

            async for token in provider.generate_stream(
                messages=pruned_messages,
                model=self.model_id,
                tools=tools_schema
            ):
                yield AgentStepUpdate(
                    state=self.state,
                    message="Streaming response",
                    output_chunk=token
                )

            self.state = AgentState.VALIDATING
            yield AgentStepUpdate(
                state=self.state,
                message="Validating response output and patches..."
            )

            self.state = AgentState.COMPLETED
            yield AgentStepUpdate(
                state=self.state,
                message="Task completed successfully."
            )

        except Exception as e:
            logger.error(f"CodingAgent state execution error: {e}")
            self.state = AgentState.ERROR
            yield AgentStepUpdate(
                state=self.state,
                message=f"Execution error: {str(e)}"
            )


coding_agent = CodingAgent()
