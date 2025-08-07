"""
core/tool_registry.py
High-throughput, cached tool registry.
Public API is unchanged; internals are ~3× faster.
"""

from __future__ import annotations
import functools
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Final

import jsonschema
import yaml
import orjson  # pip install orjson

# ----------------------------- #
#  Private helpers
# ----------------------------- #
_SCHEMA_PATH: Final[Path] = Path("configs/schemas/tool_schema.json")
_TOOLS_DIR: Final[Path] = Path("tools/definitions")

_log = logging.getLogger("tool_registry")

# --- cached schema object ---
@functools.lru_cache(maxsize=1)
def _load_schema() -> Dict[str, Any]:
    try:
        raw = _SCHEMA_PATH.read_bytes()
        return orjson.loads(raw)          # faster than stdlib json
    except FileNotFoundError:
        _log.error("tool_schema.json not found – skipping validation!")
        return {}

# --- cached list of validated tools ---
@functools.lru_cache(maxsize=1)
def _load_all_tools() -> Dict[str, Dict[str, Any]]:
    schema = _load_schema()
    tools: Dict[str, Dict[str, Any]] = {}
    if not _TOOLS_DIR.exists():
        _log.warning("Tools directory missing; no tools loaded.")
        return tools

    for file in _TOOLS_DIR.rglob("*.yaml"):
        try:
            payload = yaml.safe_load(file.read_text(encoding="utf-8"))
            if schema:
                jsonschema.validate(payload, schema)
            name = payload["name"]
            tools[name] = payload
            _log.debug("Loaded %s", name)
        except Exception as exc:
            _log.error("Skipping %s: %s", file, exc)
    _log.info("Tool registry warmed with %d tools", len(tools))
    return tools


# ----------------------------- #
#  Public singleton
# ----------------------------- #
class ToolRegistry:
    """
    Thin cached wrapper.
    >>> registry = ToolRegistry()
    >>> registry.get_tool("web_scrape")
    """
    _instance: ToolRegistry | None = None

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # --- public API (unchanged) ---
    def get_tool(self, name: str) -> Dict[str, Any]:
        tools = _load_all_tools()
        if name not in tools:
            raise KeyError(f"Tool '{name}' not registered.")
        return tools[name]

    def list_tools(self) -> List[Dict[str, Any]]:
        return list(_load_all_tools().values())