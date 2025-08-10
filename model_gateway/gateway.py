"""Deprecated shim. Prefer `sophie_shared.openrouter.async_client.OpenRouterAsyncClient`."""

from typing import Dict, Any
from sophie_shared.openrouter.async_client import OpenRouterAsyncClient


def dispatch_to_model(model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Keep sync path as stub only
    return {"choices": [{"message": {"content": "stub"}}]}


async def dispatch_to_model_async(model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenRouterAsyncClient()
    messages = payload.get("messages", [])
    return await client.chat.completions.create(model=model_name, messages=messages)
