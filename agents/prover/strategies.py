"""
Strategies for Prover variant generation.

Define abstractions for selecting strategies and building strategy-specific parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class StrategySpec:
    name: str
    temperature_adjustment: float
    focus: str


def get_strategy_for_variant(variant_id: int, task_type: str) -> StrategySpec:
    """Return a strategy spec for a given variant and task type.

    Mirrors the inline strategy mapping in ProverAgent, centralized here.
    """
    strategies = [
        "creative_innovative",
        "practical_feasible",
        "analytical_rigorous",
        "efficient_optimized",
        "sustainable_scalable",
    ]
    strategy = strategies[variant_id % len(strategies)]

    adaptations: Dict[str, Dict[str, Dict[str, Any]]] = {
        "problem_solving": {
            "creative_innovative": {"temperature": 0.8, "focus": "out-of-the-box thinking"},
            "practical_feasible": {"temperature": 0.6, "focus": "implementable solutions"},
            "analytical_rigorous": {"temperature": 0.4, "focus": "logical analysis"},
            "efficient_optimized": {"temperature": 0.5, "focus": "performance optimization"},
            "sustainable_scalable": {"temperature": 0.7, "focus": "long-term viability"},
        },
        "creative_writing": {
            "creative_innovative": {"temperature": 0.9, "focus": "artistic expression"},
            "practical_feasible": {"temperature": 0.7, "focus": "readable content"},
            "analytical_rigorous": {"temperature": 0.5, "focus": "structured narrative"},
            "efficient_optimized": {"temperature": 0.6, "focus": "concise writing"},
            "sustainable_scalable": {"temperature": 0.8, "focus": "engaging storytelling"},
        },
    }

    task_adaptations = adaptations.get(task_type, {})
    cfg = task_adaptations.get(strategy, {"temperature": 0.7, "focus": "balanced approach"})
    return StrategySpec(name=strategy, temperature_adjustment=cfg["temperature"], focus=cfg["focus"])


def get_strategy_instructions(strategy_name: str) -> str:
    """Return instructions guidance for a given strategy name."""
    strategy_instructions = {
        "creative_innovative": (
            """
Focus on creative and innovative approaches. Think outside the box and consider unconventional solutions.
Emphasize originality and novel perspectives while maintaining relevance to the task.
"""
        ),
        "practical_feasible": (
            """
Emphasize practicality and feasibility. Focus on implementable solutions that can be executed effectively.
Consider real-world constraints and practical considerations.
"""
        ),
        "analytical_rigorous": (
            """
Balance creativity with analytical rigor. Provide well-reasoned solutions with clear logical structure.
Include detailed analysis and systematic approach to problem-solving.
"""
        ),
        "efficient_optimized": (
            """
Prioritize efficiency and optimization. Focus on solutions that maximize effectiveness while minimizing resource usage.
Consider performance, scalability, and optimization opportunities.
"""
        ),
        "sustainable_scalable": (
            """
Consider long-term sustainability and scalability. Focus on solutions that can grow and adapt over time.
Emphasize maintainability, extensibility, and future-proofing.
"""
        ),
    }
    return strategy_instructions.get(strategy_name, "Provide a balanced and comprehensive solution.")


