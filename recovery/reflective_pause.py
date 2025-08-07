"""
reflective_pause.py

Triggers a system-wide Reflective Pause if constitutional safety thresholds are breached.
Evaluates consensus output for nullity, divergence, or ambiguity.
"""

from dataclasses import dataclass
from typing import Optional
from council_mode.debug_trace import DebugTrace
from council_mode.consensus_tracker import ConsensusResult

# Safety thresholds – can be adjusted via config if needed
DELTA_DIVERGENCE_THRESHOLD = 0.20  # Δ: Gap between top competing intents
EPSILON_AMBIGUITY_THRESHOLD = 0.75  # ε: Confidence drop below this triggers pause

tracer = DebugTrace()

@dataclass
class PauseSignal:
    reason: str
    severity: str  # e.g., "critical", "warning", "info"

class ReflectivePause:
    def __init__(self):
        self.tracer = tracer

    def check(self, consensus: ConsensusResult) -> Optional[PauseSignal]:
        # Check for ⊥ nullity
        for intent, details in consensus.intents.items():
            if intent in ("⊥", "null", "undefined"):
                signal = PauseSignal(reason="Null intent detected", severity="critical")
                self.tracer.log("reflective_pause_triggered", signal.__dict__)
                return signal

        # Check for Δ divergence
        if len(consensus.sorted_confidence) >= 2:
            top, second = consensus.sorted_confidence[:2]
            gap = top.confidence - second.confidence
            if gap > DELTA_DIVERGENCE_THRESHOLD:
                signal = PauseSignal(reason="High divergence between top intents (Δ)", severity="warning")
                self.tracer.log("reflective_pause_triggered", signal.__dict__)
                return signal

        # Check for ε ambiguity
        mean_conf = sum(i.confidence for i in consensus.sorted_confidence) / len(consensus.sorted_confidence)
        if mean_conf < EPSILON_AMBIGUITY_THRESHOLD:
            signal = PauseSignal(reason="Low mean confidence across intents (ε)", severity="warning")
            self.tracer.log("reflective_pause_triggered", signal.__dict__)
            return signal

        return None
