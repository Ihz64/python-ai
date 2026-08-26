from typing import Dict, Any, Optional
from backend.app.providers.base import AIProvider
from backend.app.providers.concrete_providers import OpenAIProvider, LocalModelProvider, GeminiProvider, AnthropicProvider


class ModelRegistry:
    def __init__(self):
        self._providers: Dict[str, AIProvider] = {
            "openai": OpenAIProvider(),
            "local": LocalModelProvider(),
            "gemini": GeminiProvider(),
            "anthropic": AnthropicProvider(),
        }

        self._models: Dict[str, Dict[str, Any]] = {
            "local-llama3": {
                "id": "local-llama3",
                "name": "Local Llama 3",
                "provider": "local",
                "category": "coding",
                "context_window": 8192,
                "description": "Privacy-focused local model runner."
            },
            "gpt-4o": {
                "id": "gpt-4o",
                "name": "GPT-4o (Omni)",
                "provider": "openai",
                "category": "balanced",
                "context_window": 128000,
                "description": "High intelligence balanced model for general tasks."
            },
            "claude-3-5-sonnet": {
                "id": "claude-3-5-sonnet",
                "name": "Claude 3.5 Sonnet",
                "provider": "anthropic",
                "category": "balanced",
                "context_window": 200000,
                "description": "Advanced reasoning and coding intelligence model."
            },
            "gemini-1.5-pro": {
                "id": "gemini-1.5-pro",
                "name": "Gemini 1.5 Pro",
                "provider": "gemini",
                "category": "balanced",
                "context_window": 1000000,
                "description": "Ultra long context model."
            }
        }

    def register_provider(self, name: str, provider: AIProvider) -> None:
        self._providers[name] = provider

    def get_provider_for_model(self, model_id: str) -> AIProvider:
        model_info = self._models.get(model_id)
        if not model_info:
            return self._providers["local"]
        provider_name = model_info["provider"]
        return self._providers.get(provider_name, self._providers["local"])

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        return self._models


model_registry = ModelRegistry()
