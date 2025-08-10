"""
service/judge_model.py
Stub for judge scoring across model variants.
"""

from __future__ import annotations

from typing import Dict, Any, List


class Judge:
    @staticmethod
    def score(variants: List[Dict[str, Any]]) -> Dict[str, float]:
        """Return per-variant scores (stub)."""
        return {str(i): 0.0 for i, _ in enumerate(variants)}




