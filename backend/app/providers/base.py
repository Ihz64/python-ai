from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional
from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMResponse(BaseModel):
    content: str
    model: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = "stop"
    usage: Optional[Dict[str, int]] = None


class AIProvider(ABC):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        pass
