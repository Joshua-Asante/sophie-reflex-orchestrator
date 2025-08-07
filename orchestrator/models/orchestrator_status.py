"""
Orchestrator Status Models
"""

from enum import Enum


class OrchestratorStatus(Enum):
    """Status of the orchestrator."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error" 