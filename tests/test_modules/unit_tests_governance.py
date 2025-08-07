#!/usr/bin/env python3
"""
Governance Unit Tests Module

Tests governance components (trust tracker, policy engine, audit log).
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
    from memory.trust_tracker import TrustTracker, TrustEventType
    from governance.policy_engine import PolicyEngine
    from governance.audit_log import AuditLog, AuditEventType
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

logger = structlog.get_logger()


class GovernanceUnitTestSuite:
    """Unit test suite for governance components."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.test_results = []
        
    async def run_all_tests(self) -> bool:
        """Run all governance unit tests and return success status."""
        print("🧪 Running Governance Unit Tests")
        print("-" * 40)
        
        test_functions = [
            ("Trust Tracker Initialization", self._test_trust_tracker_initialization),
            ("Policy Engine Initialization", self._test_policy_engine_initialization),
            ("Audit Log Initialization", self._test_audit_log_initialization)
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
    
    async def _test_trust_tracker_initialization(self) -> bool:
        """Test TrustTracker initialization."""
        try:
            print("Starting TrustTracker test...")
            config = {
                "db_path": os.path.join(self.temp_dir, "trust_tracker_test.db"),
                "cache_size": 100,
                "decay_rate": 0.1,
                "min_score": 0.0,
                "max_score": 1.0
            }
            print("Creating TrustTracker...")
            tracker = TrustTracker(config)
            print("TrustTracker created successfully")

            # Test basic functionality
            print("Registering agent...")
            await tracker.register_agent("agent_001", 0.8)
            print("Agent registered successfully")
            
            print("Getting trust score...")
            trust_score = await tracker.get_agent_trust_score("agent_001")
            print(f"Trust score: {trust_score}")
            assert trust_score == 0.8

            # Test trust history
            print("Recording event...")
            await tracker.record_event(
                "agent_001", 
                TrustEventType.TASK_SUCCESS, 
                "Test event",
                custom_score_change=0.1
            )
            print("Event recorded successfully")
            
            print("Getting agent profile...")
            profile = await tracker.get_agent_profile("agent_001")
            print(f"Profile: {profile}")
            assert profile is not None
            assert profile.total_events > 0

            print("TrustTracker test completed successfully")
            return True

        except Exception as e:
            print(f"TrustTracker test failed with error: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
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
            print("Starting AuditLog test...")
            print("Creating AuditLog...")
            audit_log = AuditLog()
            print("AuditLog created successfully")
            
            # Initialize the database
            print("Initializing database...")
            await audit_log.initialize()
            print("Database initialized successfully")
            
            # Start a session
            print("Starting session...")
            audit_log.start_session("test_session")
            print("Session started successfully")
            
            # Test logging functionality
            print("Logging event...")
            event_id = audit_log.log_event(
                event_type=AuditEventType.TASK_SUBMITTED,
                description="Test audit event",
                details={"test": "data"},
                agent_id="test_agent"
            )
            print(f"Event logged with ID: {event_id}")
            
            # Verify log entry was created
            assert event_id is not None
            
            # Test audit log initialization
            assert audit_log.db_path is not None
            assert audit_log.current_session is not None
            
            print("AuditLog test completed successfully")
            return True
            
        except Exception as e:
            print(f"AuditLog test failed with error: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            logger.error("Audit log initialization test failed", error=str(e))
            return False 