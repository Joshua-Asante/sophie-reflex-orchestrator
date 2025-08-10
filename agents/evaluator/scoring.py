"""
Scoring utilities and evaluation metrics for EvaluatorAgent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvaluationStats:
    total: int = 0
    category_scores: Dict[str, List[float]] = field(default_factory=dict)


def update_stats(stats: EvaluationStats, category: str, score: float) -> None:
    stats.total += 1
    stats.category_scores.setdefault(category, []).append(score)


