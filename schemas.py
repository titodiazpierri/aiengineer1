from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    provider: Provider
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)
    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class ErrorType(StrEnum):
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    API = "api_error"
    UNKNOWN = "unknown"


class LLMError(BaseModel):
    type: ErrorType
    message: str
    retryable: bool
    status_code: int | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str = ""
    usage: Usage | None = None
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    text: str = ""
    error: LLMError | None = None
