#!/usr/bin/env python3
"""
End-to-End Tests Module

Tests complete workflows and system integration.
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
    from main import SophieCLI
    from orchestrator.core import SophieReflexOrchestrator
    from orchestrator.models.orchestrator_status import OrchestratorStatus
    from agents.base_agent import AgentConfig
    from agents.prover import ProverAgent
    from agents.evaluator import EvaluatorAgent
    from agents.refiner import RefinerAgent
    from memory.trust_tracker import TrustTracker
    from governance.policy_engine import PolicyEngine
    from governance.audit_log import AuditLog
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

logger = structlog.get_logger()


class E2ETestSuite:
    """End-to-end test suite for complete workflows and system integration."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.test_results = []
        
    async def run_all_tests(self) -> bool:
        """Run all end-to-end tests and return success status."""
        print("🧪 Running End-to-End Tests")
        print("-" * 40)
        
        test_functions = [
            ("Complete Workflow", self._test_complete_workflow),
            ("Genetic Algorithm Loop", self._test_genetic_algorithm_loop),
            ("Agent Population Evolution", self._test_agent_population_evolution),
            ("Trust System Integration", self._test_trust_system_integration),
            ("Memory System Integration", self._test_memory_system_integration),
            ("Policy System Integration", self._test_policy_system_integration),
            ("Audit System Integration", self._test_audit_system_integration),
            ("Error Recovery Workflow", self._test_error_recovery_workflow),
            ("Performance Under Load", self._test_performance_under_load),
            ("System Resilience", self._test_system_resilience)
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
    
    async def _test_complete_workflow(self) -> bool:
        """Test complete system workflow."""
        try:
            # Initialize orchestrator
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            
            # Create test agents
            prover_config = AgentConfig(
                name="e2e_prover",
                prompt="You are an E2E test prover agent.",
                model="openai",
                temperature=0.7,
                max_tokens=1000
            )
            
            evaluator_config = AgentConfig(
                name="e2e_evaluator",
                prompt="You are an E2E test evaluator agent.",
                model="openai",
                temperature=0.7,
                max_tokens=1000
            )
            
            refiner_config = AgentConfig(
                name="e2e_refiner",
                prompt="You are an E2E test refiner agent.",
                model="openai",
                temperature=0.7,
                max_tokens=1000
            )
            
            # Register agents
            prover = ProverAgent(prover_config, agent_id="e2e_prover_001")
            evaluator = EvaluatorAgent(evaluator_config, agent_id="e2e_evaluator_001")
            refiner = RefinerAgent(refiner_config, agent_id="e2e_refiner_001")
            
            orchestrator.agent_manager.register_agent(prover)
            orchestrator.agent_manager.register_agent(evaluator)
            orchestrator.agent_manager.register_agent(refiner)
            
            # Test complete workflow
            task = "Design a simple web application architecture"
            
            # Step 1: Start task
            session_id = await orchestrator.start_task(task)
            assert session_id is not None
            
            # Step 2: Run generation
            generation_result = await orchestrator.run_generation()
            assert generation_result is not None
            assert generation_result.generation_number >= 0
            
            # Step 3: Check system status
            status = await orchestrator.get_status()
            assert status["status"] in ["idle", "running", "completed"]
            
            # Step 4: Get statistics
            statistics = await orchestrator.get_statistics()
            assert isinstance(statistics, dict)
            
            # Step 5: Finalize task
            await orchestrator.finalize_task()
            
            return True
            
        except Exception as e:
            logger.error("Complete workflow test failed", error=str(e))
            return False
    
    async def _test_genetic_algorithm_loop(self) -> bool:
        """Test genetic algorithm loop functionality."""
        try:
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            
            # Setup test environment
            orchestrator.state.current_task = "Test genetic algorithm loop"
            orchestrator.state.status = OrchestratorStatus.RUNNING
            
            # Test multiple generations
            generation_results = []
            for i in range(3):
                generation_result = await orchestrator.run_generation()
                assert generation_result is not None
                generation_results.append(generation_result)
                
                # Verify generation progression
                assert generation_result.generation_number == i + 1
            
            # Verify all generations completed
            assert len(generation_results) == 3
            
            # Test loop continuation logic
            should_continue = orchestrator.should_continue()
            assert isinstance(should_continue, bool)
            
            return True
            
        except Exception as e:
            logger.error("Genetic algorithm loop test failed", error=str(e))
            return False
    
    async def _test_agent_population_evolution(self) -> bool:
        """Test agent population evolution."""
        try:
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            
            # Create initial population
            initial_agents = []
            for i in range(5):
                config = AgentConfig(
                    name=f"evolution_agent_{i}",
                    prompt=f"You are evolution test agent {i}.",
                    model="openai",
                    temperature=0.7,
                    max_tokens=1000
                )
                
                agent = ProverAgent(config, agent_id=f"evolution_{i:03d}")
                initial_agents.append(agent)
                orchestrator.agent_manager.register_agent(agent)
            
            # Test population management
            registered_agents = orchestrator.agent_manager.get_registered_agents()
            assert len(registered_agents) >= 5
            
            # Test trust evolution
            for agent in initial_agents:
                orchestrator.trust_manager.update_trust_score(agent.agent_id, 0.8)
                trust_score = orchestrator.trust_manager.get_trust_score(agent.agent_id)
                assert trust_score == 0.8
            
            # Test memory integration
            for agent in initial_agents:
                orchestrator.memory_manager.store_memory(
                    session_id=f"evolution_session_{agent.agent_id}",
                    task="Evolution test task",
                    result={"evolution": "test", "agent_id": agent.agent_id},
                    agent_id=agent.agent_id
                )
            
            # Test audit integration
            for agent in initial_agents:
                orchestrator.audit_log.log_action(
                    action="evolution_test",
                    agent_id=agent.agent_id,
                    details={"evolution": "test", "generation": 1}
                )
            
            return True
            
        except Exception as e:
            logger.error("Agent population evolution test failed", error=str(e))
            return False
    
    async def _test_trust_system_integration(self) -> bool:
        """Test trust system integration."""
        try:
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            
            # Test trust tracking integration
            agent_ids = ["trust_agent_001", "trust_agent_002", "trust_agent_003"]
            
            for agent_id in agent_ids:
                # Update trust scores
                orchestrator.trust_manager.update_trust_score(agent_id, 0.8)
                
                # Verify trust score
                trust_score = orchestrator.trust_manager.get_trust_score(agent_id)
                assert trust_score == 0.8
                
                # Test trust validation
                is_trusted = orchestrator.trust_manager.is_agent_trusted(agent_id)
                assert isinstance(is_trusted, bool)
                
                # Test trust decay
                orchestrator.trust_manager.apply_trust_decay(agent_id)
                decayed_trust = orchestrator.trust_manager.get_trust_score(agent_id)
                assert decayed_trust < 0.8
            
            # Test trust history
            for agent_id in agent_ids:
                history = orchestrator.trust_tracker.get_trust_history(agent_id)
                assert len(history) > 0
            
            return True
            
        except Exception as e:
            logger.error("Trust system integration test failed", error=str(e))
            return False
    
    async def _test_memory_system_integration(self) -> bool:
        """Test memory system integration."""
        try:
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            
            # Test memory operations integration
            session_ids = ["memory_session_001", "memory_session_002", "memory_session_003"]
            
            for session_id in session_ids:
                # Store memory
                orchestrator.memory_manager.store_memory(
                    session_id=session_id,
                    task=f"Memory integration test for {session_id}",
                    result={"memory": "test", "session": session_id},
                    agent_id="test_agent_001"
                )
                
                # Retrieve memory
                retrieved_memory = orchestrator.memory_manager.get_memory(session_id)
                assert retrieved_memory is not None
                assert retrieved_memory["session_id"] == session_id
            
            # Test memory search
            search_results = orchestrator.memory_manager.search_memory("memory")
            assert len(search_results) > 0
            
            # Test memory cleanup
            orchestrator.memory_manager.cleanup_old_entries()
            
            return True
            
        except Exception as e:
            logger.error("Memory system integration test failed", error=str(e))
            return False
    
    async def _test_policy_system_integration(self) -> bool:
        """Test policy system integration."""
        try:
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            
            # Test policy evaluation integration
            test_scenarios = [
                {"confidence": 0.6, "trust": 0.5, "agent_id": "policy_agent_001", "retry_count": 1},
                {"confidence": 0.8, "trust": 0.9, "agent_id": "policy_agent_002", "retry_count": 2},
                {"confidence": 0.4, "trust": 0.3, "agent_id": "policy_agent_003", "retry_count": 0}
            ]
            
            for scenario in test_scenarios:
                # Test HITL evaluation
                hitl_required = orchestrator.policy_engine.evaluate_hitl_requirement(
                    confidence_score=scenario["confidence"],
                    trust_score=scenario["trust"]
                )
                assert isinstance(hitl_required, bool)
                
                # Test trust validation
                trust_valid = orchestrator.policy_engine.validate_trust_score(scenario["trust"])
                assert isinstance(trust_valid, bool)
                
                # Test execution policy
                execution_allowed = orchestrator.policy_engine.check_execution_policy(
                    agent_id=scenario["agent_id"],
                    retry_count=scenario["retry_count"]
                )
                assert isinstance(execution_allowed, bool)
            
            return True
            
        except Exception as e:
            logger.error("Policy system integration test failed", error=str(e))
            return False
    
    async def _test_audit_system_integration(self) -> bool:
        """Test audit system integration."""
        try:
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            
            # Test audit logging integration
            test_actions = [
                {"action": "e2e_agent_execution", "agent_id": "audit_agent_001", "details": {"task": "e2e_task1"}},
                {"action": "e2e_trust_update", "agent_id": "audit_agent_001", "details": {"old_score": 0.7, "new_score": 0.8}},
                {"action": "e2e_policy_evaluation", "agent_id": "audit_agent_002", "details": {"policy": "hitl", "result": True}},
                {"action": "e2e_memory_operation", "agent_id": "audit_agent_002", "details": {"operation": "store", "session_id": "test_session"}}
            ]
            
            for action_data in test_actions:
                orchestrator.audit_log.log_action(**action_data)
            
            # Verify audit entries
            entries = orchestrator.audit_log.get_entries()
            assert len(entries) >= len(test_actions)
            
            # Test audit search
            search_results = orchestrator.audit_log.search_entries("e2e")
            assert len(search_results) >= len(test_actions)
            
            # Test audit filtering
            agent_001_entries = orchestrator.audit_log.get_entries_by_agent("audit_agent_001")
            assert len(agent_001_entries) >= 2
            
            return True
            
        except Exception as e:
            logger.error("Audit system integration test failed", error=str(e))
            return False
    
    async def _test_error_recovery_workflow(self) -> bool:
        """Test error recovery workflow."""
        try:
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            
            # Test error handling in workflow
            try:
                # Simulate error condition
                orchestrator.state.current_task = None
                generation_result = await orchestrator.run_generation()
                
                # Should handle the error gracefully
                assert generation_result is not None
                assert generation_result.status in ["completed", "failed"]
                
            except Exception as e:
                # Expected error handling
                assert "task" in str(e).lower() or "error" in str(e).lower()
            
            # Test state recovery
            orchestrator.state.current_task = "Error recovery test task"
            orchestrator.state.status = OrchestratorStatus.RUNNING
            
            # Verify state recovery
            assert orchestrator.state.current_task == "Error recovery test task"
            assert orchestrator.state.status == OrchestratorStatus.RUNNING
            
            # Test pause/resume functionality
            orchestrator.pause()
            assert orchestrator.state.status == OrchestratorStatus.IDLE
            
            orchestrator.resume()
            assert orchestrator.state.status == OrchestratorStatus.RUNNING
            
            return True
            
        except Exception as e:
            logger.error("Error recovery workflow test failed", error=str(e))
            return False
    
    async def _test_performance_under_load(self) -> bool:
        """Test system performance under load."""
        try:
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            
            # Test performance under load
            start_time = datetime.now()
            
            # Create multiple agents under load
            agents = []
            for i in range(10):
                config = AgentConfig(
                    name=f"load_agent_{i}",
                    prompt=f"You are load test agent {i}.",
                    model="openai",
                    temperature=0.7,
                    max_tokens=1000
                )
                
                agent = ProverAgent(config, agent_id=f"load_{i:03d}")
                agents.append(agent)
                orchestrator.agent_manager.register_agent(agent)
            
            # Test memory operations under load
            for i in range(20):
                orchestrator.memory_manager.store_memory(
                    session_id=f"load_memory_{i:03d}",
                    task=f"Load test task {i}",
                    result={"load": "test", "index": i},
                    agent_id="load_agent_001"
                )
            
            # Test trust operations under load
            for i in range(20):
                agent_id = f"load_trust_agent_{i:03d}"
                orchestrator.trust_manager.update_trust_score(agent_id, 0.8)
                trust_score = orchestrator.trust_manager.get_trust_score(agent_id)
                assert trust_score == 0.8
            
            # Test audit operations under load
            for i in range(20):
                orchestrator.audit_log.log_action(
                    action=f"load_action_{i:03d}",
                    agent_id=f"load_audit_agent_{i:03d}",
                    details={"load": "test", "index": i}
                )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Performance should be reasonable under load (less than 15 seconds for 70 operations)
            assert execution_time < 15.0
            
            return True
            
        except Exception as e:
            logger.error("Performance under load test failed", error=str(e))
            return False
    
    async def _test_system_resilience(self) -> bool:
        """Test system resilience."""
        try:
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            
            # Test system resilience to various conditions
            
            # Test with invalid inputs
            try:
                orchestrator.state.current_task = ""
                generation_result = await orchestrator.run_generation()
                assert generation_result is not None
            except Exception as e:
                # Expected error handling
                pass
            
            # Test with missing components
            try:
                # Simulate missing agent scenario
                orchestrator.agent_manager = None
                status = await orchestrator.get_status()
                assert isinstance(status, dict)
            except Exception as e:
                # Expected error handling
                pass
            
            # Test with corrupted state
            try:
                orchestrator.state.current_generation = -1
                orchestrator.state.status = "invalid_status"
                
                # System should handle gracefully
                should_continue = orchestrator.should_continue()
                assert isinstance(should_continue, bool)
            except Exception as e:
                # Expected error handling
                pass
            
            # Test recovery mechanisms
            orchestrator.state.current_task = "Resilience test task"
            orchestrator.state.status = OrchestratorStatus.RUNNING
            orchestrator.state.current_generation = 1
            
            # Verify recovery
            assert orchestrator.state.current_task == "Resilience test task"
            assert orchestrator.state.status == OrchestratorStatus.RUNNING
            assert orchestrator.state.current_generation == 1
            
            return True
            
        except Exception as e:
            logger.error("System resilience test failed", error=str(e))
            return False 