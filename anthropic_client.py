import inspect
from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from base_client import BaseLLMClient, classify_error
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Usage

SUPPORTS_TEMPERATURE = "temperature" in inspect.signature(
    anthropic.resources.messages.AsyncMessages.create
).parameters


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=api_key, timeout=config.timeout, max_retries=0)

    def _params(self, messages: list[ChatMessage]) -> dict:
        params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [m.model_dump() for m in messages if m.role != "system"],
        }
        system = "\n".join(m.content for m in messages if m.role == "system")
        if system:
            params["system"] = system
        if SUPPORTS_TEMPERATURE:
            params["temperature"] = min(self.config.temperature, 1.0)
        return params

    async def _call(self, messages: list[ChatMessage]) -> ModelResponse:
        response = await self.client.messages.create(**self._params(messages))
        return ModelResponse(
            provider=self.config.provider,
            model=response.model,
            content="".join(block.text for block in response.content if block.type == "text"),
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )

    async def _call_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with self.client.messages.stream(**self._params(messages)) as stream:
            async for text in stream.text_stream:
                yield text

    def _classify_error(self, exc: Exception) -> LLMError:
        return classify_error(anthropic, exc)

    async def close(self) -> None:
        await self.client.close()
