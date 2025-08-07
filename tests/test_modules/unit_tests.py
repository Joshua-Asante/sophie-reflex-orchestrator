#!/usr/bin/env python3
"""
Unit Tests Module

Tests individual components in isolation without external dependencies.
"""

import asyncio
import sys
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import structlog

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agents.base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus
    from agents.prover import ProverAgent
    from agents.evaluator import EvaluatorAgent
    from agents.refiner import RefinerAgent
    from orchestrator.models.orchestrator_config import OrchestratorConfig
    from orchestrator.models.orchestrator_status import OrchestratorStatus
    from memory.trust_tracker import TrustTracker, TrustEventType
    from governance.policy_engine import PolicyEngine
    from governance.audit_log import AuditLog, AuditEventType
    from configs.config_manager import ConfigManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

logger = structlog.get_logger()


class UnitTestSuite:
    """Unit test suite for individual components."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.test_results = []
        
    async def run_all_tests(self) -> bool:
        """Run all unit tests and return success status."""
        print("🧪 Running Unit Tests")
        print("-" * 40)
        
        test_functions = [
            ("Agent Config Creation", self._test_agent_config_creation),
            ("Base Agent Initialization", self._test_base_agent_initialization),
            ("Prover Agent Creation", self._test_prover_agent_creation),
            ("Evaluator Agent Creation", self._test_evaluator_agent_creation),
            ("Refiner Agent Creation", self._test_refiner_agent_creation),
            ("Orchestrator Config Loading", self._test_orchestrator_config_loading),
            ("Trust Tracker Initialization", self._test_trust_tracker_initialization),
            ("Policy Engine Initialization", self._test_policy_engine_initialization),
            ("Audit Log Initialization", self._test_audit_log_initialization),
            ("Config Manager Loading", self._test_config_manager_loading),
            ("Agent Result Creation", self._test_agent_result_creation),
            ("Status Enum Values", self._test_status_enum_values)
        ]
        
        all_passed = True
        for test_name, test_func in test_functions:
            try:
                result = await test_func()
                if result:
                    print(f"✅ {test_name}: PASSED")
                    self.test_results.append((test_name, "PASSED", None))
                else:
                    print(f"❌ {test_name}: FAILED")
                    self.test_results.append((test_name, "FAILED", "Test returned False"))
                    all_passed = False
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {str(e)}")
                self.test_results.append((test_name, "ERROR", str(e)))
                all_passed = False
        
        return all_passed
    
    async def _test_agent_config_creation(self) -> bool:
        """Test AgentConfig creation and validation."""
        try:
            # Test basic config creation
            config = AgentConfig(
                name="test_agent",
                prompt="You are a test agent.",
                model="openai",
                temperature=0.7,
                max_tokens=1000
            )
            
            # Verify all fields are set correctly
            assert config.name == "test_agent"
            assert config.prompt == "You are a test agent."
            assert config.model == "openai"
            assert config.temperature == 0.7
            assert config.max_tokens == 1000
            assert config.timeout == 30  # Default value
            assert config.max_retries == 3  # Default value
            
            # Test with hyperparameters
            config_with_hypers = AgentConfig(
                name="test_agent_with_hypers",
                prompt="You are a test agent with hyperparameters.",
                model="anthropic",
                temperature=0.8,
                max_tokens=2000,
                hyperparameters={"creativity": 0.9, "detail_level": 0.7}
            )
            
            assert config_with_hypers.hyperparameters["creativity"] == 0.9
            assert config_with_hypers.hyperparameters["detail_level"] == 0.7
            
            return True
            
        except Exception as e:
            logger.error("Agent config creation test failed", error=str(e))
            return False
    
    async def _test_base_agent_initialization(self) -> bool:
        """Test BaseAgent initialization (abstract class)."""
        try:
            # Create a concrete implementation for testing
            class TestAgent(BaseAgent):
                async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
                    return AgentResult(
                        agent_id=self.agent_id,
                        agent_name=self.config.name,
                        result={"test": "result"},
                        confidence_score=0.8,
                        execution_time=0.1,
                        status=AgentStatus.COMPLETED
                    )
                
                async def generate_prompt(self, task: str, context: Dict[str, Any] = None) -> str:
                    return f"Test prompt for task: {task}"
            
            # Test agent initialization
            config = AgentConfig(
                name="test_base_agent",
                prompt="You are a test base agent.",
                model="openai",
                temperature=0.7,
                max_tokens=1000
            )
            
            agent = TestAgent(config, agent_id="test_agent_001")
            
            # Verify initialization
            assert agent.agent_id == "test_agent_001"
            assert agent.config.name == "test_base_agent"
            assert agent.trust_score == 0.5  # Default value
            assert agent.execution_count == 0
            
            # Test basic methods
            assert agent.get_success_rate() == 0.0  # No executions yet
            assert agent.get_performance_metrics()["total_executions"] == 0
            
            return True
            
        except Exception as e:
            logger.error("Base agent initialization test failed", error=str(e))
            return False
    
    async def _test_prover_agent_creation(self) -> bool:
        """Test ProverAgent creation and basic functionality."""
        try:
            config = AgentConfig(
                name="test_prover",
                prompt="You are a test prover agent.",
                model="openai",
                temperature=0.7,
                max_tokens=1000,
                hyperparameters={
                    "max_variants": 3,
                    "creativity": 0.8,
                    "detail_level": 0.7
                }
            )
            
            prover = ProverAgent(config, agent_id="test_prover_001")
            
            # Verify initialization
            assert prover.agent_id == "test_prover_001"
            assert prover.max_variants == 3
            assert prover.creativity == 0.8
            assert prover.detail_level == 0.7
            assert prover.variant_generator is not None
            
            # Test specialization info
            specialization_info = prover.get_specialization_info()
            assert "specialization" in specialization_info
            assert "capabilities" in specialization_info
            
            return True
            
        except Exception as e:
            logger.error("Prover agent creation test failed", error=str(e))
            return False
    
    async def _test_evaluator_agent_creation(self) -> bool:
        """Test EvaluatorAgent creation and basic functionality."""
        try:
            config = AgentConfig(
                name="test_evaluator",
                prompt="You are a test evaluator agent.",
                model="openai",
                temperature=0.7,
                max_tokens=1000,
                hyperparameters={
                    "consensus_enabled": True,
                    "consensus_threshold": 0.2
                }
            )
            
            evaluator = EvaluatorAgent(config, agent_id="test_evaluator_001")
            
            # Verify initialization
            assert evaluator.agent_id == "test_evaluator_001"
            assert evaluator.consensus_enabled is True
            assert evaluator.consensus_threshold == 0.2
            assert evaluator.evaluation_metrics is not None
            
            # Test evaluation criteria
            criteria = evaluator.get_evaluation_criteria()
            assert isinstance(criteria, dict)
            
            return True
            
        except Exception as e:
            logger.error("Evaluator agent creation test failed", error=str(e))
            return False
    
    async def _test_refiner_agent_creation(self) -> bool:
        """Test RefinerAgent creation and basic functionality."""
        try:
            config = AgentConfig(
                name="test_refiner",
                prompt="You are a test refiner agent.",
                model="openai",
                temperature=0.7,
                max_tokens=1000,
                hyperparameters={
                    "mutation_strength": 0.3,
                    "focus_areas": ["clarity", "efficiency"],
                    "crossover_points": 3,
                    "balance_ratio": 0.6
                }
            )
            
            refiner = RefinerAgent(config, agent_id="test_refiner_001")
            
            # Verify initialization
            assert refiner.agent_id == "test_refiner_001"
            assert refiner.mutation_strength == 0.3
            assert refiner.focus_areas == ["clarity", "efficiency"]
            assert refiner.crossover_points == 3
            assert refiner.balance_ratio == 0.6
            assert refiner.population_manager is not None
            assert refiner.adaptive_mutation is not None
            
            # Test refinement stats
            stats = refiner.get_refinement_stats()
            assert "generation_count" in stats
            assert "mutation_strength" in stats
            assert "focus_areas" in stats
            
            return True
            
        except Exception as e:
            logger.error("Refiner agent creation test failed", error=str(e))
            return False
    
    async def _test_orchestrator_config_loading(self) -> bool:
        """Test OrchestratorConfig loading from file."""
        try:
            # Test loading from default config
            config = OrchestratorConfig.from_file("configs/system.yaml")

            # Verify essential fields are loaded
            assert config.max_generations is not None
            assert config.population_size is not None
            assert config.max_agents is not None

            # Test specific fields
            assert config.max_agents >= 0
            assert config.population_size >= 0
            assert config.mutation_rate >= 0.0

            return True

        except Exception as e:
            logger.error("Orchestrator config loading test failed", error=str(e))
            return False
    
    async def _test_trust_tracker_initialization(self) -> bool:
        """Test TrustTracker initialization."""
        try:
            config = {
                "db_path": ":memory:",
                "cache_size": 100,
                "decay_rate": 0.1,
                "min_score": 0.0,
                "max_score": 1.0
            }
            tracker = TrustTracker(config)

            # Test basic functionality
            await tracker.register_agent("agent_001", 0.8)
            trust_score = await tracker.get_agent_trust_score("agent_001")
            assert trust_score == 0.8

            # Test trust history
            await tracker.record_event(
                "agent_001", 
                TrustEventType.TASK_SUCCESS, 
                "Test event",
                custom_score_change=0.1
            )
            profile = await tracker.get_agent_profile("agent_001")
            assert profile is not None
            assert profile.total_events > 0

            return True

        except Exception as e:
            logger.error("Trust tracker initialization test failed", error=str(e))
            return False
    
    async def _test_policy_engine_initialization(self) -> bool:
        """Test PolicyEngine initialization."""
        try:
            # Create test policies
            test_policies = {
                "hitl": {
                    "enabled": True,
                    "approval_threshold": 0.7
                },
                "trust": {
                    "min_trust_score": 0.3,
                    "max_trust_score": 1.0
                }
            }
            
            policy_engine = PolicyEngine(test_policies)
            
            # Test policy engine initialization
            assert policy_engine.policies_config == test_policies
            assert policy_engine.hitl_policies["enabled"] is True
            assert policy_engine.trust_policies["min_trust_score"] == 0.3
            
            return True
            
        except Exception as e:
            logger.error("Policy engine initialization test failed", error=str(e))
            return False
    
    async def _test_audit_log_initialization(self) -> bool:
        """Test AuditLog initialization."""
        try:
            audit_log = AuditLog()
            
            # Test logging functionality
            event_id = audit_log.log_event(
                event_type=AuditEventType.TASK_SUBMITTED,
                description="Test audit event",
                details={"test": "data"},
                agent_id="test_agent"
            )
            
            # Verify log entry was created
            assert event_id is not None
            
            # Test audit log initialization
            assert audit_log.db_path is not None
            assert audit_log.current_session is not None
            
            return True
            
        except Exception as e:
            logger.error("Audit log initialization test failed", error=str(e))
            return False
    
    async def _test_config_manager_loading(self) -> bool:
        """Test ConfigManager loading and validation."""
        try:
            config_manager = ConfigManager()
            
            # Test loading configurations
            await config_manager.load_and_validate_all()
            
            # Verify configurations are loaded
            assert config_manager.system_config is not None
            assert config_manager.agents_config is not None
            assert config_manager.policies_config is not None
            
            return True
            
        except Exception as e:
            logger.error("Config manager loading test failed", error=str(e))
            return False
    
    async def _test_agent_result_creation(self) -> bool:
        """Test AgentResult creation and validation."""
        try:
            # Test basic result creation
            result = AgentResult(
                agent_id="test_agent_001",
                agent_name="Test Agent",
                result={"test": "data"},
                confidence_score=0.8,
                execution_time=1.5,
                status=AgentStatus.COMPLETED
            )
            
            # Verify all fields
            assert result.agent_id == "test_agent_001"
            assert result.agent_name == "Test Agent"
            assert result.result["test"] == "data"
            assert result.confidence_score == 0.8
            assert result.execution_time == 1.5
            assert result.status == AgentStatus.COMPLETED
            assert result.error_message is None
            
            # Test with error
            error_result = AgentResult(
                agent_id="test_agent_002",
                agent_name="Test Agent Error",
                result=None,
                confidence_score=0.0,
                execution_time=0.1,
                status=AgentStatus.FAILED,
                error_message="Test error"
            )
            
            assert error_result.status == AgentStatus.FAILED
            assert error_result.error_message == "Test error"
            
            return True
            
        except Exception as e:
            logger.error("Agent result creation test failed", error=str(e))
            return False
    
    async def _test_status_enum_values(self) -> bool:
        """Test AgentStatus enum values."""
        try:
            # Test all status values
            statuses = [
                AgentStatus.IDLE,
                AgentStatus.RUNNING,
                AgentStatus.COMPLETED,
                AgentStatus.FAILED,
                AgentStatus.BLOCKED,
                AgentStatus.RATE_LIMITED,
                AgentStatus.CIRCUIT_OPEN
            ]
            
            # Verify all statuses are valid
            for status in statuses:
                assert isinstance(status, AgentStatus)
                assert status.value in [
                    "idle", "running", "completed", "failed", 
                    "blocked", "rate_limited", "circuit_open"
                ]
            
            # Test status comparison
            assert AgentStatus.COMPLETED != AgentStatus.FAILED
            assert AgentStatus.IDLE == AgentStatus.IDLE
            
            return True
            
        except Exception as e:
            logger.error("Status enum values test failed", error=str(e))
            return False 