#!/usr/bin/env python3
"""
Performance Tests Module

Tests system performance and scalability.
"""

import asyncio
import sys
import os
import json
import tempfile
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import structlog

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from orchestrator.core import SophieReflexOrchestrator
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


class PerformanceTestSuite:
    """Performance test suite for system performance and scalability."""

    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.test_results = []

    async def run_all_tests(self) -> bool:
        """Run all performance tests and return success status."""
        print("🧪 Running Performance Tests")
        print("-" * 40)

        test_functions = [
            ("Agent Creation Performance", self._test_agent_creation_performance),
            ("Agent Execution Performance", self._test_agent_execution_performance),
            ("Memory Operations Performance", self._test_memory_operations_performance),
            ("Trust Tracking Performance", self._test_trust_tracking_performance),
            ("Audit Logging Performance", self._test_audit_logging_performance),
            ("Policy Evaluation Performance", self._test_policy_evaluation_performance),
            ("Orchestrator Performance", self._test_orchestrator_performance),
            ("Concurrent Operations", self._test_concurrent_operations),
            ("Memory Usage", self._test_memory_usage),
            ("Response Time", self._test_response_time)
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

    async def _test_agent_creation_performance(self) -> bool:
        """Test agent creation performance."""
        try:
            start_time = time.time()

            # Create multiple agents
            agents = []
            for i in range(10):
                config = AgentConfig(
                    name=f"perf_agent_{i}",
                    prompt=f"You are performance test agent {i}.",
                    model="capability:general_agentic",
                    temperature=0.7,
                    max_tokens=1000
                )

                if i % 3 == 0:
                    agent = ProverAgent(config, agent_id=f"prover_perf_{i}")
                elif i % 3 == 1:
                    agent = EvaluatorAgent(config, agent_id=f"evaluator_perf_{i}")
                else:
                    agent = RefinerAgent(config, agent_id=f"refiner_perf_{i}")

                agents.append(agent)

            end_time = time.time()
            creation_time = end_time - start_time

            # Performance should be reasonable (less than 2 seconds for 10 agents)
            assert creation_time < 2.0
            assert len(agents) == 10

            return True

        except Exception as e:
            logger.error("Agent creation performance test failed", error=str(e))
            return False

    async def _test_agent_execution_performance(self) -> bool:
        """Test agent execution performance."""
        try:
            # Create test agent
            config = AgentConfig(
                name="perf_execution_agent",
                prompt="You are a performance test agent for execution testing.",
                model="capability:general_agentic",
                temperature=0.7,
                max_tokens=1000
            )

            agent = ProverAgent(config, agent_id="perf_exec_001")

            # Test execution performance
            start_time = time.time()

            # Execute multiple tasks
            tasks = [
                "Simple task 1",
                "Simple task 2",
                "Simple task 3",
                "Simple task 4",
                "Simple task 5"
            ]

            for task in tasks:
                result = await agent.execute(task)
                assert result.status.value in ["completed", "failed"]

            end_time = time.time()
            execution_time = end_time - start_time

            # Performance should be reasonable (less than 10 seconds for 5 tasks)
            assert execution_time < 10.0

            return True

        except Exception as e:
            logger.error("Agent execution performance test failed", error=str(e))
            return False

    async def _test_memory_operations_performance(self) -> bool:
        """Test memory operations performance."""
        try:
            from orchestrator.components.memory_manager import MemoryManager

            memory_manager = MemoryManager()

            start_time = time.time()

            # Test memory operations performance
            for i in range(50):
                memory_manager.store_memory(
                    session_id=f"perf_memory_{i:03d}",
                    task=f"Performance test task {i}",
                    result={"performance": "test", "index": i},
                    agent_id="test_agent_001"
                )

            # Test retrieval performance
            for i in range(50):
                memory = memory_manager.get_memory(f"perf_memory_{i:03d}")
                assert memory is not None

            # Test search performance
            search_results = memory_manager.search_memory("performance")
            assert len(search_results) > 0

            end_time = time.time()
            operation_time = end_time - start_time

            # Performance should be reasonable (less than 5 seconds for 150 operations)
            assert operation_time < 5.0

            return True

        except Exception as e:
            logger.error("Memory operations performance test failed", error=str(e))
            return False

    async def _test_trust_tracking_performance(self) -> bool:
        """Test trust tracking performance."""
        try:
            trust_tracker = TrustTracker()

            start_time = time.time()

            # Test trust tracking performance
            for i in range(100):
                agent_id = f"perf_trust_agent_{i:03d}"
                trust_score = 0.5 + (i % 10) * 0.05  # Vary trust scores

                trust_tracker.update_trust(agent_id, trust_score)

                # Test retrieval
                retrieved_trust = trust_tracker.get_trust(agent_id)
                assert retrieved_trust == trust_score

            end_time = time.time()
            operation_time = end_time - start_time

            # Performance should be reasonable (less than 2 seconds for 100 operations)
            assert operation_time < 2.0

            return True

        except Exception as e:
            logger.error("Trust tracking performance test failed", error=str(e))
            return False

    async def _test_audit_logging_performance(self) -> bool:
        """Test audit logging performance."""
        try:
            audit_log = AuditLog()

            start_time = time.time()

            # Test audit logging performance
            for i in range(100):
                audit_log.log_action(
                    action=f"perf_action_{i:03d}",
                    agent_id=f"perf_agent_{i:03d}",
                    details={"performance": "test", "index": i}
                )

            # Test retrieval performance
            entries = audit_log.get_entries()
            assert len(entries) >= 100

            # Test search performance
            search_results = audit_log.search_entries("perf_action")
            assert len(search_results) > 0

            end_time = time.time()
            operation_time = end_time - start_time

            # Performance should be reasonable (less than 3 seconds for 100 operations)
            assert operation_time < 3.0

            return True

        except Exception as e:
            logger.error("Audit logging performance test failed", error=str(e))
            return False

    async def _test_policy_evaluation_performance(self) -> bool:
        """Test policy evaluation performance."""
        try:
            test_policies = {
                "hitl": {"enabled": True, "approval_threshold": 0.7},
                "trust": {"min_trust_score": 0.3, "max_trust_score": 1.0},
                "execution": {"max_retries": 3, "timeout": 30}
            }

            policy_engine = PolicyEngine(test_policies)

            start_time = time.time()

            # Test policy evaluation performance
            for i in range(100):
                confidence = 0.1 + (i % 10) * 0.1
                trust = 0.1 + (i % 10) * 0.1

                # Test HITL evaluation
                hitl_required = policy_engine.evaluate_hitl_requirement(
                    confidence_score=confidence,
                    trust_score=trust
                )
                assert isinstance(hitl_required, bool)

                # Test trust validation
                trust_valid = policy_engine.validate_trust_score(trust)
                assert isinstance(trust_valid, bool)

                # Test execution policy
                execution_allowed = policy_engine.check_execution_policy(
                    agent_id=f"perf_agent_{i:03d}",
                    retry_count=i % 5
                )
                assert isinstance(execution_allowed, bool)

            end_time = time.time()
            operation_time = end_time - start_time

            # Performance should be reasonable (less than 2 seconds for 300 operations)
            assert operation_time < 2.0

            return True

        except Exception as e:
            logger.error("Policy evaluation performance test failed", error=str(e))
            return False

    async def _test_orchestrator_performance(self) -> bool:
        """Test orchestrator performance."""
        try:
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")

            start_time = time.time()

            # Test orchestrator operations performance
            for i in range(10):
                # Test status retrieval
                status = await orchestrator.get_status()
                assert isinstance(status, dict)

                # Test statistics generation
                statistics = await orchestrator.get_statistics()
                assert isinstance(statistics, dict)

            end_time = time.time()
            operation_time = end_time - start_time

            # Performance should be reasonable (less than 5 seconds for 20 operations)
            assert operation_time < 5.0

            return True

        except Exception as e:
            logger.error("Orchestrator performance test failed", error=str(e))
            return False

    async def _test_concurrent_operations(self) -> bool:
        """Test concurrent operations performance."""
        try:
            # Test concurrent agent creation
            async def create_agent(agent_id: str):
                config = AgentConfig(
                    name=f"concurrent_agent_{agent_id}",
                    prompt=f"You are concurrent test agent {agent_id}.",
                    model="capability:general_agentic",
                    temperature=0.7,
                    max_tokens=1000
                )
                return ProverAgent(config, agent_id=f"concurrent_{agent_id}")

            start_time = time.time()

            # Create agents concurrently
            tasks = [create_agent(f"agent_{i:03d}") for i in range(10)]
            agents = await asyncio.gather(*tasks)

            end_time = time.time()
            creation_time = end_time - start_time

            # Concurrent creation should be faster than sequential
            assert creation_time < 3.0
            assert len(agents) == 10

            return True

        except Exception as e:
            logger.error("Concurrent operations test failed", error=str(e))
            return False

    async def _test_memory_usage(self) -> bool:
        """Test memory usage patterns."""
        try:
            from orchestrator.components.memory_manager import MemoryManager

            memory_manager = MemoryManager()

            # Test memory usage with large datasets
            large_dataset = []
            for i in range(100):
                large_dataset.append({
                    "session_id": f"memory_usage_{i:03d}",
                    "task": f"Memory usage test task {i}",
                    "result": {"data": "x" * 1000, "index": i},  # Large result
                    "agent_id": "test_agent_001"
                })

            start_time = time.time()

            # Store large dataset
            for data in large_dataset:
                memory_manager.store_memory(**data)

            # Test retrieval performance with large dataset
            for data in large_dataset:
                memory = memory_manager.get_memory(data["session_id"])
                assert memory is not None

            end_time = time.time()
            operation_time = end_time - start_time

            # Performance should be reasonable even with large dataset
            assert operation_time < 10.0

            return True

        except Exception as e:
            logger.error("Memory usage test failed", error=str(e))
            return False

    async def _test_response_time(self) -> bool:
        """Test system response time."""
        try:
            # Test various system operations response time
            operations = []

            # Test agent creation response time
            start_time = time.time()
            config = AgentConfig(
                name="response_time_agent",
                prompt="You are a response time test agent.",
                model="capability:general_agentic",
                temperature=0.7,
                max_tokens=1000
            )
            agent = ProverAgent(config, agent_id="response_001")
            creation_time = time.time() - start_time
            operations.append(("agent_creation", creation_time))

            # Test agent execution response time
            start_time = time.time()
            result = await agent.execute("Test response time")
            execution_time = time.time() - start_time
            operations.append(("agent_execution", execution_time))

            # Test trust tracking response time
            trust_tracker = TrustTracker()
            start_time = time.time()
            trust_tracker.update_trust("response_agent", 0.8)
            trust_time = time.time() - start_time
            operations.append(("trust_update", trust_time))

            # Test audit logging response time
            audit_log = AuditLog()
            start_time = time.time()
            audit_log.log_action(
                action="response_test",
                agent_id="response_agent",
                details={"test": "response_time"}
            )
            audit_time = time.time() - start_time
            operations.append(("audit_logging", audit_time))

            # Verify all operations completed within reasonable time
            for operation_name, operation_time in operations:
                assert operation_time < 5.0, f"{operation_name} took too long: {operation_time}s"

            return True

        except Exception as e:
            logger.error("Response time test failed", error=str(e))
            return False
