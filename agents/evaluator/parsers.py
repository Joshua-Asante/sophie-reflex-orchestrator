"""
Parsers for evaluator responses (JSON-first with fallback parsing).
"""
from __future__ import annotations

from typing import Dict, Any
import json


def parse_json_response(text: str) -> Dict[str, Any] | None:
    try:
        return json.loads(text) if text.strip().startswith("{") else None
    except Exception:
        return None


