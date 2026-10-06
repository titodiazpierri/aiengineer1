import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import (
    ChatMessage,
    ErrorType,
    LLMError,
    ModelConfig,
    ModelResponse,
    StreamChunk,
)


def classify_error(sdk, exc: Exception) -> LLMError:
    status = getattr(exc, "status_code", None)
    if isinstance(exc, sdk.AuthenticationError):
        return LLMError(type=ErrorType.AUTHENTICATION, retryable=False, status_code=status,
                        message="API key inválida o ausente.")
    if isinstance(exc, sdk.RateLimitError):
        return LLMError(type=ErrorType.RATE_LIMIT, retryable=True, status_code=status,
                        message="Límite de tasa o cuota excedido.")
    if isinstance(exc, sdk.APITimeoutError):
        return LLMError(type=ErrorType.TIMEOUT, retryable=True,
                        message="Tiempo de espera agotado.")
    if isinstance(exc, sdk.APIConnectionError):
        return LLMError(type=ErrorType.CONNECTION, retryable=True,
                        message="Error de red al conectar con la API.")
    if isinstance(exc, sdk.APIStatusError):
        return LLMError(type=ErrorType.API, retryable=(status or 0) >= 500,
                        status_code=status, message=str(exc))
    return LLMError(type=ErrorType.UNKNOWN, retryable=False,
                    message=f"{type(exc).__name__}: {exc}")


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def _call(self, messages: list[ChatMessage]) -> ModelResponse:
        pass

    @abstractmethod
    def _call_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        pass

    @abstractmethod
    def _classify_error(self, exc: Exception) -> LLMError:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        attempt = 0
        while True:
            try:
                return await self._call(messages)
            except Exception as exc:
                error = self._classify_error(exc)
                if error.retryable and attempt < self.config.max_retries:
                    attempt += 1
                    await self._wait_before_retry(attempt)
                    continue
                return ModelResponse(
                    provider=self.config.provider,
                    model=self.config.model,
                    error=error,
                )

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        attempt = 0
        while True:
            started = False
            try:
                async for text in self._call_stream(messages):
                    started = True
                    yield StreamChunk(text=text)
                return
            except Exception as exc:
                error = self._classify_error(exc)
                if error.retryable and not started and attempt < self.config.max_retries:
                    attempt += 1
                    await self._wait_before_retry(attempt)
                    continue
                yield StreamChunk(error=error)
                return

    async def _wait_before_retry(self, attempt: int) -> None:
        delay = min(2 ** (attempt - 1), 30) + random.uniform(0, 0.5)
        await asyncio.sleep(delay)
