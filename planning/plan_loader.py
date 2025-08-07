"""
core/plan_loader.py
Fast, cached plan loader with orjson schema parsing.
Public signature unchanged.
"""

from __future__ import annotations
import functools
import json
import logging
from pathlib import Path
from typing import Dict, Any

import jsonschema
import yaml
import orjson

_log = logging.getLogger("plan_loader")

_PLANS_DIR: Path = Path("plans")
_SCHEMA_PATH: Path = Path("configs/schemas/plan_schema.json")


# ----------------------------- #
#  Cached schema
# ----------------------------- #
@functools.lru_cache(maxsize=1)
def _get_plan_schema() -> Dict[str, Any]:
    try:
        return orjson.loads(_SCHEMA_PATH.read_bytes())
    except FileNotFoundError as exc:
        raise RuntimeError("plan_schema.json missing") from exc


# ----------------------------- #
#  Cached per-plan load
# ----------------------------- #
@functools.lru_cache(maxsize=64)
def load_plan(plan_name: str) -> Dict[str, Any]:
    """
    Load and validate a plan YAML.
    Cached by plan_name (case-sensitive).
    """
    plan_path = _PLANS_DIR / f"{plan_name}.yaml"
    if not plan_path.exists():
        raise FileNotFoundError(str(plan_path))

    payload = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    schema = _get_plan_schema()
    jsonschema.validate(payload, schema)
    _log.debug("Loaded plan %s", plan_name)
    return payload