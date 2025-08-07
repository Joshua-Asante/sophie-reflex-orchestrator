"""
Sophie Reflex Orchestrator - Modular Architecture

A modular implementation of the genetic algorithm orchestrator with
separated concerns and improved maintainability.
"""

from .core import SophieReflexOrchestrator
from .models.orchestrator_status import OrchestratorStatus
from .models.orchestrator_config import OrchestratorConfig
from .models.generation_result import GenerationResult

__all__ = [
    'SophieReflexOrchestrator',
    'OrchestratorStatus', 
    'OrchestratorConfig',
    'GenerationResult'
] 