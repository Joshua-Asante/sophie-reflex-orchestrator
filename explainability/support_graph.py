# 3. support_graph.py
# Path: explainability/support_graph.py

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Evidence:
    source: str
    trust: float
    confidence: float
    output: Any

@dataclass
class SupportGraph:
    primary_intent: str
    competing_intents: Dict[str, List[Evidence]] = field(default_factory=dict)

    def add_evidence(self, source: str, trust: float, confidence: float, intent: str):
        if intent not in self.competing_intents:
            self.competing_intents[intent] = []
        self.competing_intents[intent].append(
            Evidence(source=source, trust=trust, confidence=confidence, output=intent)
        )

    def to_json(self) -> str:
        graph_data = {
            "primary_intent": self.primary_intent,
            "decision_context": "Supported by sources with the highest weighted evidence.",
            "intents": {}
        }
        for intent, evidences in self.competing_intents.items():
            graph_data["intents"][intent] = {
                "is_primary": intent == self.primary_intent,
                "evidence": [
                    {"source": ev.source, "trust": ev.trust, "confidence": ev.confidence}
                    for ev in evidences
                ]
            }
        return json.dumps(graph_data, indent=2)

