import os
from collections.abc import AsyncIterator

from anthropic_client import AnthropicClient
from base_client import BaseLLMClient
from openai_client import OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider, StreamChunk

CLIENTS: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
}

API_KEY_VARS: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
}

DEFAULT_MODELS: dict[Provider, str] = {
    Provider.OPENAI: "gpt-4.1-mini",
    Provider.ANTHROPIC: "claude-haiku-4-5",
}


class AsyncLLMManager:
    def __init__(self, config: ModelConfig) -> None:
        api_key = os.getenv(API_KEY_VARS[config.provider], "")
        if not api_key:
            raise ValueError(f"Falta la variable de entorno {API_KEY_VARS[config.provider]}")
        self.config = config
        self.client = CLIENTS[config.provider](config, api_key)

    @classmethod
    def from_env(cls, provider: str | None = None) -> "AsyncLLMManager":
        selected = Provider(provider or os.getenv("LLM_PROVIDER", "openai"))
        config = ModelConfig(
            provider=selected,
            model=os.getenv("LLM_MODEL") or DEFAULT_MODELS[selected],
            temperature=os.getenv("LLM_TEMPERATURE", 0.7),
            max_tokens=os.getenv("LLM_MAX_TOKENS", 512),
        )
        return cls(config)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        async for chunk in self.client.generate_stream(messages):
            yield chunk

    async def close(self) -> None:
        await self.client.close()
