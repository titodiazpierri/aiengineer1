# Unified Async LLM Client

Cliente asíncrono unificado para OpenAI y Anthropic en Python 3.12.

## Estructura

| Archivo | Contenido |
|---|---|
| `schemas.py` | Modelos Pydantic: `ChatMessage`, `ModelConfig` (temperatura 0 a 2, `max_tokens`), `ModelResponse`, `StreamChunk`, `LLMError` |
| `base_client.py` | `BaseLLMClient` (clase abstracta) con `generate()` y `generate_stream()`, reintentos con backoff exponencial y manejo de errores |
| `openai_client.py` | `OpenAIClient` con `AsyncOpenAI` |
| `anthropic_client.py` | `AnthropicClient` con `AsyncAnthropic` |
| `llm_manager.py` | `AsyncLLMManager`: carga OpenAI o Anthropic según `LLM_PROVIDER` |
| `main.py` | Script de prueba: pregunta "¿Qué es la entropía?" en modo normal y en streaming |

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows: `py -3.12 -m venv .venv` y `.venv\Scripts\activate`.

## Variables de entorno

Copiar `.env.example` a `.env` y completar:

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | `openai` o `anthropic` |
| `OPENAI_API_KEY` | API key de OpenAI |
| `ANTHROPIC_API_KEY` | API key de Anthropic |
| `LLM_MODEL` | Opcional. Por defecto `gpt-4.1-mini` o `claude-haiku-4-5` |
| `LLM_TEMPERATURE` | Opcional. Entre 0 y 2 |
| `LLM_MAX_TOKENS` | Opcional. Entero mayor a 0 |

## Ejecución

```bash
python main.py
python main.py openai
python main.py anthropic
```

## Manejo de errores

Las excepciones de los SDKs se capturan y se devuelven como `LLMError` dentro de `ModelResponse` o `StreamChunk`, sin interrumpir el programa:

| Error | Tipo | Reintento |
|---|---|---|
| API key inválida | `authentication` | No |
| Límite de tasa o cuota | `rate_limit` | Sí |
| Timeout | `timeout` | Sí |
| Error de red | `connection` | Sí |
| Error 5xx del servidor | `api_error` | Sí |
