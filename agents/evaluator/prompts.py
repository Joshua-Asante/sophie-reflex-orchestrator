"""
Prompt builders for EvaluatorAgent categories and overall feedback.
"""
from __future__ import annotations

from typing import Dict, Any


def build_category_prompt(category: str, task: str, solution: str, config: Dict[str, Any]) -> str:
    """Return a category-specific evaluation prompt (stub)."""
    return f"Evaluate category {category} for task: {task}\nSolution:\n{solution}\n"


def build_overall_feedback_prompt(task: str, breakdown: str) -> str:
    return f"Overall feedback for task: {task}\n{breakdown}"


