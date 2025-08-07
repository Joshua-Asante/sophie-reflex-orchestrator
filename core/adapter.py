"""
core/adapter.py
Re-usable async adapter layer.
Public API: `async execute(tool_name, parameters) -> Any`
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
from typing import Any, Dict

import httpx

from telemetry import get_logger

_logger = get_logger("adapter")

# ---------- shared resources ----------
_HTTP_CLIENT: httpx.AsyncClient = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0, connect=5.0)
)

_MODULE_CACHE: Dict[str, Any] = {}  # tool-name -> imported module


# ---------- private helpers ----------
@functools.lru_cache(maxsize=None)
def _load_tool_module(tool_name: str) -> Any:
    """Import and cache tool adapter; raise if invalid."""
    module_path = f"tools.adapters.{tool_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"No adapter module tools/adapters/{tool_name}.py"
        ) from exc

    if not (hasattr(module, "execute") and inspect.iscoroutinefunction(module.execute)):
        raise RuntimeError(
            f"tools/adapters/{tool_name}.py must expose `async def execute(params: dict) -> Any`"
        )
    return module


# ---------- public API ----------
async def execute(tool_name: str, parameters: Dict[str, Any]) -> Any:
    """Execute a tool via its adapter module."""
    module = _load_tool_module(tool_name)
    _logger.debug("Executing tool", tool=tool_name)
    try:
        # Adapters can optionally accept the shared http client
        sig = inspect.signature(module.execute)
        if "http_client" in sig.parameters:
            return await module.execute(parameters, http_client=_HTTP_CLIENT)
        return await module.execute(parameters)
    finally:
        # adapters must not close the shared client
        pass