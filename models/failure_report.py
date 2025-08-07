"""
Failure Report Model

Structured representation of task execution failures for recovery and analysis.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class FailureReport:
    """Structured report of a task execution failure."""
    
    step_name: str
    agent: str  # Tool/agent that failed
    args: Dict[str, Any]  # Arguments passed to the tool
    error_type: str  # Type of error (e.g., "ValueError", "ConnectionError")
    error_message: str  # Human-readable error message
    traceback: str  # Full traceback for debugging
    context: Optional[Dict[str, Any]] = None  # Additional context
    timestamp: Optional[str] = None  # When the failure occurred
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_name": self.step_name,
            "agent": self.agent,
            "args": self.args,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "context": self.context or {},
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureReport":
        """Create from dictionary."""
        return cls(
            step_name=data["step_name"],
            agent=data["agent"],
            args=data["args"],
            error_type=data["error_type"],
            error_message=data["error_message"],
            traceback=data["traceback"],
            context=data.get("context"),
            timestamp=data.get("timestamp")
        ) 