from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from base_client import BaseLLMClient, classify_error
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Usage


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=api_key, timeout=config.timeout, max_retries=0)

    def _params(self, messages: list[ChatMessage]) -> dict:
        return {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
        }

    async def _call(self, messages: list[ChatMessage]) -> ModelResponse:
        response = await self.client.chat.completions.create(**self._params(messages))
        usage = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return ModelResponse(
            provider=self.config.provider,
            model=response.model,
            content=response.choices[0].message.content or "",
            usage=usage,
        )

    async def _call_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(**self._params(messages), stream=True)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _classify_error(self, exc: Exception) -> LLMError:
        return classify_error(openai, exc)

    async def close(self) -> None:
        await self.client.close()
