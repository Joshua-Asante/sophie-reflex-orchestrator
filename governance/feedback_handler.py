# core/feedback_handler.py
from typing import Dict, Any, Set, Optional
from orchestrator.components.trust_manager import TrustManager
from governance.audit_log import AuditLog as TrustAuditLog

TRUST_UPDATE_FACTOR = 0.05

def process_human_feedback(
    explanation_graph: Dict[str, Any],
    confirmed_intent: str,
    trust_manager: TrustManager,
    audit_log: TrustAuditLog,
    session_id: str,
    protected_attributes: Optional[Dict[str, str]] = None,
    outcome: Optional[str] = None
) -> Dict[str, float]:
    """
    Updates trust scores based on human-confirmed intent and logs audit data.
    Optionally captures bias-monitoring metadata.
    """
    updated_scores: Dict[str, float] = {}
    correct_sources: Set[str] = set()
    intents = explanation_graph.get("intents", {})

    if confirmed_intent in intents:
        for evidence in intents[confirmed_intent].get("evidence", []):
            correct_sources.add(evidence["source"])

    processed_sources: Set[str] = set()

    for intent, details in intents.items():
        for evidence in details.get("evidence", []):
            source = evidence["source"]
            if source in processed_sources:
                continue

            old_score = trust_manager.get_trust(source)

            if source in correct_sources:
                change_factor = TRUST_UPDATE_FACTOR
                reason = "Confirmation"
            else:
                change_factor = -TRUST_UPDATE_FACTOR
                reason = "Refutation"

            new_score = trust_manager.update_trust(source, change_factor)
            updated_scores[source] = new_score
            processed_sources.add(source)

            audit_log.log_update(
                source=source,
                old_score=old_score,
                new_score=new_score,
                reason=reason,
                session_id=session_id,
                protected_attributes=protected_attributes,
                outcome=outcome
            )

    return updated_scores

import json
import yaml
from typing import Dict

TRUST_UPDATE_FACTOR = 0.05
TRUST_SCORES_PATH = 'cost_optimization/trust_scores.yaml'

def _update_trust_scores(updates: Dict[str, float]):
    with open(TRUST_SCORES_PATH, 'r') as f:
        data = yaml.safe_load(f)
    for source, change in updates.items():
        if source in data['sources']:
            current_trust = data['sources'][source]['trust']
            new_trust = max(0.0, min(1.0, current_trust + change))
            data['sources'][source]['trust'] = round(new_trust, 4)
    with open(TRUST_SCORES_PATH, 'w') as f:
        yaml.dump(data, f, indent=2)

def process_human_feedback_simple(correct_intent: str, support_graph_json: str):
    """Simplified version of human feedback processing."""
    graph = json.loads(support_graph_json)
    trust_updates = {}
    for intent, details in graph['intents'].items():
        for evidence in details['evidence']:
            source = evidence['source']
            trust_updates[source] = (
                TRUST_UPDATE_FACTOR if intent == correct_intent else -TRUST_UPDATE_FACTOR
            )
    if trust_updates:
        _update_trust_scores(trust_updates)

