"""
Test Modules Package

This package contains modularized test suites for the Sophie Reflexive Orchestrator.
Each module focuses on testing specific components of the system.
"""

from .unit_tests import UnitTestSuite
from .integration_tests import IntegrationTestSuite
from .agent_tests import AgentTestSuite
from .orchestrator_tests import OrchestratorTestSuite
from .memory_tests import MemoryTestSuite
from .governance_tests import GovernanceTestSuite
from .performance_tests import PerformanceTestSuite
from .e2e_tests import E2ETestSuite

__all__ = [
    "UnitTestSuite",
    "IntegrationTestSuite", 
    "AgentTestSuite",
    "OrchestratorTestSuite",
    "MemoryTestSuite",
    "GovernanceTestSuite",
    "PerformanceTestSuite",
    "E2ETestSuite"
] 