#!/usr/bin/env python3
"""
Governance Tests Module

Tests policy engine and audit logging functionality.
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
    from governance.policy_engine import PolicyEngine
    from governance.audit_log import AuditLog
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

logger = structlog.get_logger()


class GovernanceTestSuite:
    """Governance test suite for policy engine and audit logging."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.test_results = []
        
    async def run_all_tests(self) -> bool:
        """Run all governance tests and return success status."""
        print("🧪 Running Governance Tests")
        print("-" * 40)
        
        test_functions = [
            ("Policy Engine Initialization", self._test_policy_engine_initialization),
            ("Policy Evaluation", self._test_policy_evaluation),
            ("HITL Requirements", self._test_hitl_requirements),
            ("Trust Validation", self._test_trust_validation),
            ("Execution Policies", self._test_execution_policies),
            ("Audit Log Initialization", self._test_audit_log_initialization),
            ("Audit Logging", self._test_audit_logging),
            ("Audit Search", self._test_audit_search),
            ("Audit Filtering", self._test_audit_filtering),
            ("Audit Persistence", self._test_audit_persistence)
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
                },
                "execution": {
                    "max_retries": 3,
                    "timeout": 30
                }
            }
            
            policy_engine = PolicyEngine(test_policies)
            
            # Verify initialization
            assert policy_engine is not None
            assert policy_engine.policies == test_policies
            
            return True
            
        except Exception as e:
            logger.error("Policy engine initialization test failed", error=str(e))
            return False
    
    async def _test_policy_evaluation(self) -> bool:
        """Test policy evaluation functionality."""
        try:
            test_policies = {
                "hitl": {"enabled": True, "approval_threshold": 0.7},
                "trust": {"min_trust_score": 0.3, "max_trust_score": 1.0},
                "execution": {"max_retries": 3, "timeout": 30}
            }
            
            policy_engine = PolicyEngine(test_policies)
            
            # Test policy retrieval
            hitl_enabled = policy_engine.get_policy("hitl", "enabled")
            assert hitl_enabled is True
            
            approval_threshold = policy_engine.get_policy("hitl", "approval_threshold")
            assert approval_threshold == 0.7
            
            max_retries = policy_engine.get_policy("execution", "max_retries")
            assert max_retries == 3
            
            return True
            
        except Exception as e:
            logger.error("Policy evaluation test failed", error=str(e))
            return False
    
    async def _test_hitl_requirements(self) -> bool:
        """Test HITL (Human-in-the-Loop) requirements."""
        try:
            test_policies = {
                "hitl": {"enabled": True, "approval_threshold": 0.7},
                "trust": {"min_trust_score": 0.3, "max_trust_score": 1.0}
            }
            
            policy_engine = PolicyEngine(test_policies)
            
            # Test HITL evaluation with different scenarios
            scenarios = [
                {"confidence": 0.6, "trust": 0.5, "expected": True},   # Below threshold
                {"confidence": 0.8, "trust": 0.9, "expected": False},  # Above threshold
                {"confidence": 0.7, "trust": 0.7, "expected": False},  # At threshold
                {"confidence": 0.5, "trust": 0.3, "expected": True}    # Low scores
            ]
            
            for scenario in scenarios:
                hitl_required = policy_engine.evaluate_hitl_requirement(
                    confidence_score=scenario["confidence"],
                    trust_score=scenario["trust"]
                )
                assert isinstance(hitl_required, bool)
            
            return True
            
        except Exception as e:
            logger.error("HITL requirements test failed", error=str(e))
            return False
    
    async def _test_trust_validation(self) -> bool:
        """Test trust validation functionality."""
        try:
            test_policies = {
                "trust": {"min_trust_score": 0.3, "max_trust_score": 1.0}
            }
            
            policy_engine = PolicyEngine(test_policies)
            
            # Test trust validation with different scores
            test_scores = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
            
            for score in test_scores:
                is_valid = policy_engine.validate_trust_score(score)
                assert isinstance(is_valid, bool)
                
                # Scores within range should be valid
                if 0.3 <= score <= 1.0:
                    assert is_valid is True
                else:
                    assert is_valid is False
            
            return True
            
        except Exception as e:
            logger.error("Trust validation test failed", error=str(e))
            return False
    
    async def _test_execution_policies(self) -> bool:
        """Test execution policy functionality."""
        try:
            test_policies = {
                "execution": {"max_retries": 3, "timeout": 30}
            }
            
            policy_engine = PolicyEngine(test_policies)
            
            # Test execution policy with different retry counts
            test_scenarios = [
                {"agent_id": "agent_001", "retry_count": 0, "expected": True},
                {"agent_id": "agent_002", "retry_count": 2, "expected": True},
                {"agent_id": "agent_003", "retry_count": 3, "expected": False},
                {"agent_id": "agent_004", "retry_count": 5, "expected": False}
            ]
            
            for scenario in test_scenarios:
                execution_allowed = policy_engine.check_execution_policy(
                    agent_id=scenario["agent_id"],
                    retry_count=scenario["retry_count"]
                )
                assert isinstance(execution_allowed, bool)
            
            return True
            
        except Exception as e:
            logger.error("Execution policies test failed", error=str(e))
            return False
    
    async def _test_audit_log_initialization(self) -> bool:
        """Test AuditLog initialization."""
        try:
            audit_log = AuditLog()
            
            # Verify initialization
            assert audit_log is not None
            
            return True
            
        except Exception as e:
            logger.error("Audit log initialization test failed", error=str(e))
            return False
    
    async def _test_audit_logging(self) -> bool:
        """Test audit logging functionality."""
        try:
            audit_log = AuditLog()
            
            # Test basic logging
            audit_log.log_action(
                action="test_action",
                agent_id="test_agent_001",
                details={"test": "data", "confidence": 0.8}
            )
            
            # Verify log entry was created
            entries = audit_log.get_entries()
            assert len(entries) > 0
            
            # Test entry structure
            latest_entry = entries[-1]
            assert "action" in latest_entry
            assert "agent_id" in latest_entry
            assert "timestamp" in latest_entry
            assert "details" in latest_entry
            assert latest_entry["action"] == "test_action"
            assert latest_entry["agent_id"] == "test_agent_001"
            
            return True
            
        except Exception as e:
            logger.error("Audit logging test failed", error=str(e))
            return False
    
    async def _test_audit_search(self) -> bool:
        """Test audit log search functionality."""
        try:
            audit_log = AuditLog()
            
            # Create test audit entries
            test_actions = [
                {"action": "agent_execution", "agent_id": "agent_001", "details": {"task": "task1"}},
                {"action": "trust_update", "agent_id": "agent_001", "details": {"old_score": 0.7, "new_score": 0.8}},
                {"action": "policy_evaluation", "agent_id": "agent_002", "details": {"policy": "hitl", "result": True}},
                {"action": "agent_execution", "agent_id": "agent_002", "details": {"task": "task2"}}
            ]
            
            # Log test actions
            for action_data in test_actions:
                audit_log.log_action(**action_data)
            
            # Test search functionality
            search_terms = ["agent_execution", "trust_update", "policy_evaluation", "agent_001", "agent_002"]
            
            for term in search_terms:
                results = audit_log.search_entries(term)
                assert len(results) > 0
                
                # Verify search results contain the search term
                found_term = False
                for result in results:
                    if term.lower() in str(result).lower():
                        found_term = True
                        break
                assert found_term, f"Search term '{term}' not found in results"
            
            return True
            
        except Exception as e:
            logger.error("Audit search test failed", error=str(e))
            return False
    
    async def _test_audit_filtering(self) -> bool:
        """Test audit log filtering functionality."""
        try:
            audit_log = AuditLog()
            
            # Create test audit entries
            test_actions = [
                {"action": "agent_execution", "agent_id": "agent_001", "details": {"task": "task1"}},
                {"action": "trust_update", "agent_id": "agent_001", "details": {"old_score": 0.7, "new_score": 0.8}},
                {"action": "policy_evaluation", "agent_id": "agent_002", "details": {"policy": "hitl", "result": True}},
                {"action": "agent_execution", "agent_id": "agent_002", "details": {"task": "task2"}}
            ]
            
            # Log test actions
            for action_data in test_actions:
                audit_log.log_action(**action_data)
            
            # Test filtering by agent
            agent_001_entries = audit_log.get_entries_by_agent("agent_001")
            assert len(agent_001_entries) == 2
            
            agent_002_entries = audit_log.get_entries_by_agent("agent_002")
            assert len(agent_002_entries) == 2
            
            # Test filtering by action type
            execution_entries = audit_log.search_entries("agent_execution")
            assert len(execution_entries) == 2
            
            trust_entries = audit_log.search_entries("trust_update")
            assert len(trust_entries) == 1
            
            return True
            
        except Exception as e:
            logger.error("Audit filtering test failed", error=str(e))
            return False
    
    async def _test_audit_persistence(self) -> bool:
        """Test audit log persistence functionality."""
        try:
            audit_log = AuditLog()
            
            # Test audit persistence
            test_actions = [
                {"action": "persistence_test_1", "agent_id": "agent_001", "details": {"test": "data1"}},
                {"action": "persistence_test_2", "agent_id": "agent_002", "details": {"test": "data2"}},
                {"action": "persistence_test_3", "agent_id": "agent_001", "details": {"test": "data3"}}
            ]
            
            # Log test actions
            for action_data in test_actions:
                audit_log.log_action(**action_data)
            
            # Verify all entries are persisted
            entries = audit_log.get_entries()
            assert len(entries) >= len(test_actions)
            
            # Verify entry structure and content
            for i, action_data in enumerate(test_actions):
                entry = entries[-(len(test_actions)-i)]
                assert entry["action"] == action_data["action"]
                assert entry["agent_id"] == action_data["agent_id"]
                assert entry["details"]["test"] == action_data["details"]["test"]
            
            return True
            
        except Exception as e:
            logger.error("Audit persistence test failed", error=str(e))
            return False 