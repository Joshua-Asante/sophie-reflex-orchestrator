"""
Defines the various reasoning strategies available to the SOPHIE Council. Each mode maps to a procedural method of intent evaluation.
"""

from enum import Enum

class ReasoningMode(Enum):
    """Enumeration of the available reasoning strategies for the SOPHIE Council."""
    CONSENSUS = "consensus"
    REFLECTIVE_PAUSE = "reflective_pause"
    DISSENT_TRACKING = "dissent_tracking"
    ETHICAL_ESCALATION = "ethical_escalation"
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"
    ADVERSARIAL_PROBE = "adversarial_probe" # Corrected: Added missing enum member
    FAST_PATH = "fast_path"


MODE_DESCRIPTIONS = {
    ReasoningMode.CONSENSUS: "Standard mode where council members converge on shared intent.",
    ReasoningMode.REFLECTIVE_PAUSE: "Triggered when high divergence or ambiguity is detected, requiring deliberate reflection.",
    ReasoningMode.DISSENT_TRACKING: "Logs and weighs dissenting views for minority reporting or downstream consideration.",
    ReasoningMode.ETHICAL_ESCALATION: "Routes decision to ethical review when value conflict or harm potential is present.",
    ReasoningMode.UNCERTAINTY_ANALYSIS: "Used when confidence scores are low across all models, triggering clarification or user input.",
    ReasoningMode.ADVERSARIAL_PROBE: "Activates stress-testing via Grok to detect structural or semantic weaknesses in reasoning.",
    ReasoningMode.FAST_PATH: "Shortcut for low-stakes, high-certainty requests where consensus is pre-established."
}

def describe_mode(mode: ReasoningMode) -> str:
    """Returns the textual description for a given reasoning mode."""
    return MODE_DESCRIPTIONS.get(mode, "Unknown reasoning mode.")