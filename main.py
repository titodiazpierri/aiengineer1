import asyncio
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_manager import AsyncLLMManager
from schemas import ChatMessage

QUESTION = "¿Qué es la entropía?"


async def run(provider: str | None) -> None:
    try:
        manager = AsyncLLMManager.from_env(provider)
    except (ValidationError, ValueError) as exc:
        print(f"Error de configuración: {exc}")
        return

    messages = [ChatMessage(role="user", content=QUESTION)]
    print(f"Proveedor: {manager.config.provider} | Modelo: {manager.config.model}")

    print("\n--- Modo normal ---")
    response = await manager.generate(messages)
    if response.ok:
        print(response.content)
    else:
        print(f"Error [{response.error.type}]: {response.error.message}")

    print("\n--- Modo streaming ---")
    async for chunk in manager.generate_stream(messages):
        if chunk.error:
            print(f"Error [{chunk.error.type}]: {chunk.error.message}")
            break
        print(chunk.text, end="", flush=True)
    print()

    await manager.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else None))
