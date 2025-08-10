"""
Collaboration enhancement utilities for Prover variants.
"""
from __future__ import annotations

from typing import List, Dict, Any


def extract_insight(variant: Dict[str, Any]) -> str | None:
    """Extract key insight from a variant.

    Simple heuristic: return first notable line or a substantial line.
    """
    content = variant.get("content", "")
    if not content:
        return None
    lines = content.split("\n")
    for line in lines:
        s = line.strip()
        if any(k in s.lower() for k in ["key", "important", "critical", "essential", "notable"]):
            return s
    for line in lines:
        s = line.strip()
        if len(s) > 50:
            return s[:200] + "..."
    return None


def format_insights(insights: List[str]) -> str:
    """Format insights for prompt inclusion."""
    if not insights:
        return "No additional insights available."
    return "\n".join(f"{idx+1}. {insight}" for idx, insight in enumerate(insights))


