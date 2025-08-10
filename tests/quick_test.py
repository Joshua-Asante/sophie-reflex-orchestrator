#!/usr/bin/env python3
"""
Quick Test Script
Simple script to check if basic imports and components are working.
"""

import sys
import os
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test basic imports."""
    print("🧪 Testing basic imports...")

    try:
        # Test agent imports
        from agents.base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus
        print("✅ Agent imports: OK")

        # Test orchestrator imports
        from orchestrator.models.orchestrator_config import OrchestratorConfig
        print("✅ Orchestrator config: OK")

        # Test memory imports
        from memory.trust_tracker import TrustTracker, TrustEventType
        print("✅ Trust tracker: OK")

        # Test governance imports
        from governance.policy_engine import PolicyEngine
        from governance.audit_log import AuditLog, AuditEventType
        print("✅ Governance components: OK")

        # Test config imports
        from configs.config_manager import ConfigManager
        print("✅ Config manager: OK")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

async def test_basic_components():
    """Test basic component creation."""
    print("\n🧪 Testing basic component creation...")

    try:
        # Test AgentConfig creation
        from agents.base_agent import AgentConfig
        config = AgentConfig(
            name="test_agent",
            prompt="You are a test agent.",
            model="capability:general_agentic",
            temperature=0.7,
            max_tokens=1000
        )
        print("✅ AgentConfig creation: OK")

        # Test AgentResult creation
        from agents.base_agent import AgentResult, AgentStatus
        result = AgentResult(
            agent_id="test_001",
            agent_name="Test Agent",
            result={"test": "data"},
            confidence_score=0.8,
            execution_time=1.0,
            status=AgentStatus.COMPLETED
        )
        print("✅ AgentResult creation: OK")

        # Test TrustTracker creation
        from memory.trust_tracker import TrustTracker
        tracker_config = {
            "db_path": ":memory:",
            "cache_size": 100,
            "decay_rate": 0.1,
            "min_score": 0.0,
            "max_score": 1.0
        }
        tracker = TrustTracker(tracker_config)
        print("✅ TrustTracker creation: OK")

        # Test PolicyEngine creation
        from governance.policy_engine import PolicyEngine
        policies = {
            "hitl": {"enabled": True, "approval_threshold": 0.7},
            "trust": {"min_trust_score": 0.3, "max_trust_score": 1.0}
        }
        policy_engine = PolicyEngine(policies)
        print("✅ PolicyEngine creation: OK")

        # Test AuditLog creation (async)
        from governance.audit_log import AuditLog, AuditEventType
        audit_log = AuditLog()
        # Wait for async initialization
        await audit_log._initialize_database_async()
        print("✅ AuditLog creation: OK")

        return True

    except Exception as e:
        print(f"❌ Component creation error: {e}")
        return False

async def main():
    """Main test function."""
    print("🚀 Sophie Reflexive Orchestrator - Quick Test")
    print("=" * 50)

    # Test imports
    imports_ok = test_imports()

    if imports_ok:
        # Test component creation
        components_ok = await test_basic_components()

        if components_ok:
            print("\n🎉 All basic tests passed!")
            print("✅ The system appears to be working correctly.")
            return True
        else:
            print("\n❌ Component creation tests failed.")
            return False
    else:
        print("\n❌ Import tests failed.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
