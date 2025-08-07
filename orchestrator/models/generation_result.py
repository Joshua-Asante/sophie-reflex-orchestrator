"""
Generation Result Models
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional


@dataclass
class GenerationResult:
    """Result of a single generation in the GA loop."""
    generation: int
    task: str
    agents: List[Any]
    best_solution: Optional[Dict[str, Any]]
    best_score: float
    average_score: float
    execution_time: float
    interventions: List[Dict[str, Any]]
    trust_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self) 