"""
consensus_tracker.py

Evaluates agreement, dissent, and quorum within Council Mode.
Implements trust-weighted consensus logic for decision resolution.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

ConsensusResult = Dict[str, float]

@dataclass
class ModelResponse:
    intent: Optional[str]
    confidence: float
    trust: float

class ConsensusTracker:
    def __init__(self, quorum_threshold: float = 0.67):
        self.responses: Dict[str, List[ModelResponse]] = {}
        self.quorum_threshold = quorum_threshold

    def register_response(self, source: str, intent: Optional[str], confidence: float, trust: float):
        if intent is None:
            return  # Null intent, skip
        if intent not in self.responses:
            self.responses[intent] = []
        self.responses[intent].append(ModelResponse(intent, confidence, trust))

    def compute_consensus(self) -> Optional[ConsensusResult]:
        consensus_scores = {
            intent: sum(r.trust * r.confidence for r in responses)
            for intent, responses in self.responses.items()
        }

        total_score = sum(consensus_scores.values())
        if total_score == 0:
            return None

        normalized_scores = {
            intent: score / total_score
            for intent, score in consensus_scores.items()
        }

        return normalized_scores

    def primary_intent(self) -> Optional[str]:
        scores = self.compute_consensus()
        if not scores:
            return None

        top_intent, top_score = max(scores.items(), key=lambda x: x[1])
        if top_score >= self.quorum_threshold:
            return top_intent
        return None

    def detect_dissent(self) -> bool:
        scores = self.compute_consensus()
        if not scores:
            return False
        return len(scores) > 1

    def has_quorum(self) -> bool:
        scores = self.compute_consensus()
        if not scores:
            return False
        return max(scores.values()) >= self.quorum_threshold
