"""
Quality assessment helpers for Prover variants.
"""
from __future__ import annotations

from typing import Dict


def assess_quality(content: str, task: str, task_type: str, strategy_name: str) -> float:
    """Return a quality score [0,1] for the content.

    Uses the same simple heuristics as the inline scorer for now.
    """
    # Heuristic components
    length_score = min(1.0, len(content) / 1000)

    # Structure
    lines = content.split("\n")
    has_numbered = any(line.strip().startswith(("1.", "2.", "3.")) for line in lines)
    has_bullets = any(line.strip().startswith(("-", "•", "*")) for line in lines)
    has_sections = len([line for line in lines if line.strip().isupper()]) > 0
    structure_score = 0.0
    if has_numbered:
        structure_score += 0.4
    if has_bullets:
        structure_score += 0.3
    if has_sections:
        structure_score += 0.3

    # Relevance
    task_keywords = set(task.lower().split())
    content_keywords = set(content.lower().split())
    if not task_keywords:
        relevance_score = 0.5
    else:
        overlap = len(task_keywords.intersection(content_keywords))
        relevance_score = min(1.0, overlap / len(task_keywords))

    # Completeness by task type
    indicators = {
        "problem_solving": ["solution", "approach", "analysis", "result"],
        "creative_writing": ["narrative", "story", "character", "plot"],
        "technical_design": ["design", "architecture", "implementation", "specification"],
        "strategic_planning": ["strategy", "plan", "timeline", "milestone"],
    }.get(task_type, ["solution", "approach"])
    lower = content.lower()
    indicator_count = sum(1 for ind in indicators if ind in lower)
    completeness_score = min(1.0, indicator_count / len(indicators))

    weights = {"length": 0.2, "structure": 0.3, "relevance": 0.3, "completeness": 0.2}
    score = (
        length_score * weights["length"]
        + structure_score * weights["structure"]
        + relevance_score * weights["relevance"]
        + completeness_score * weights["completeness"]
    )
    return float(max(0.0, min(1.0, score)))


