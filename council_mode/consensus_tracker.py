"""
Consensus Tracker Component

Tracks consensus results for reflective pause evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class IntentConfidence:
    """Represents an intent with its confidence score."""
    intent: str
    confidence: float


@dataclass
class ConsensusResult:
    """Represents consensus evaluation results."""
    
    intents: Dict[str, Any]  # Intent details
    sorted_confidence: List[IntentConfidence]  # Sorted by confidence
    mean_confidence: float  # Average confidence across all intents
    consensus_achieved: bool  # Whether consensus was reached
    divergence_score: float  # Measure of divergence between top intents
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if not self.sorted_confidence:
            self.mean_confidence = 0.0
            self.consensus_achieved = False
            self.divergence_score = 0.0
        else:
            # Calculate mean confidence
            self.mean_confidence = sum(i.confidence for i in self.sorted_confidence) / len(self.sorted_confidence)
            
            # Calculate divergence between top 2 intents
            if len(self.sorted_confidence) >= 2:
                self.divergence_score = self.sorted_confidence[0].confidence - self.sorted_confidence[1].confidence
            else:
                self.divergence_score = 0.0
            
            # Determine consensus (simplified logic)
            self.consensus_achieved = self.mean_confidence > 0.7 and self.divergence_score < 0.2


class ConsensusTracker:
    """Tracks and manages consensus evaluation."""
    
    def __init__(self):
        self.results: List[ConsensusResult] = []
    
    def add_result(self, result: ConsensusResult) -> None:
        """Add a consensus result to the tracker."""
        self.results.append(result)
    
    def get_latest_result(self) -> Optional[ConsensusResult]:
        """Get the most recent consensus result."""
        return self.results[-1] if self.results else None
    
    def get_results(self) -> List[ConsensusResult]:
        """Get all consensus results."""
        return self.results.copy() 