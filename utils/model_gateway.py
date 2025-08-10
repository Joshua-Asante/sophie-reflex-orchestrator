"""Deprecated gateway. Prefer `sophie_shared.openrouter.async_client.OpenRouterAsyncClient`."""

from typing import Dict, Any, Optional, List

from sophie_shared.openrouter.async_client import OpenRouterAsyncClient


def call_model(model_name: str, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
    # Keep sync shim minimal
    return f"Mock response from {model_name}: {prompt[:50]}..."


async def call_model_async(model_name: str, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
    client = OpenRouterAsyncClient()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": context.get("system", "You are SOPHIE.") if context else "You are SOPHIE."},
        {"role": "user", "content": prompt},
    ]
    resp = await client.chat.completions.create(model=model_name, messages=messages)
    choice = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    return choice
