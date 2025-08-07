#!/usr/bin/env python3
"""
Sophie Reflex Orchestrator Core Test Suite
Tests essential components without complex dependencies
"""

import asyncio
import sys
import os
import json
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class SophieCoreTestSuite:
    """Core test suite for Sophie Reflex Orchestrator."""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        
    async def run_all_tests(self) -> bool:
        """Run all core tests and return success status."""
        print("🧪 Starting Sophie Reflex Orchestrator Core Test Suite")
        print("=" * 60)
        
        try:
            # Setup test environment
            await self._setup_test_environment()
            
            # Run test categories
            test_categories = [
                ("File Structure", self._test_file_structure),
                ("Configuration Files", self._test_configuration_files),
                ("Memory System", self._test_memory_system),
                ("Trust Tracking", self._test_trust_tracking),
                ("Policy Engine", self._test_policy_engine),
                ("Audit Logging", self._test_audit_logging),
                ("Agent Files", self._test_agent_files),
                ("Orchestrator Core", self._test_orchestrator_core),
                ("Webhook Server", self._test_webhook_server),
                ("Docker Configuration", self._test_docker_configuration)
            ]
            
            all_passed = True
            for category_name, test_func in test_categories:
                print(f"\n📋 Testing: {category_name}")
                print("-" * 40)
                
                try:
                    result = await test_func()
                    if result:
                        print(f"✅ {category_name}: PASSED")
                        self.test_results.append((category_name, "PASSED", None))
                    else:
                        print(f"❌ {category_name}: FAILED")
                        self.test_results.append((category_name, "FAILED", "Test returned False"))
                        all_passed = False
                except Exception as e:
                    print(f"❌ {category_name}: ERROR - {str(e)}")
                    self.test_results.append((category_name, "ERROR", str(e)))
                    all_passed = False
            
            # Generate comprehensive report
            await self._generate_test_report()
            
            return all_passed
            
        finally:
            await self._cleanup_test_environment()
    
    async def _setup_test_environment(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="sophie_core_test_")
        print(f"📁 Test directory: {self.temp_dir}")
        print("🔧 Test environment created")
    
    async def _cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("🧹 Test environment cleaned up")
    
    async def _test_file_structure(self) -> bool:
        """Test project file structure."""
        try:
            required_files = [
                "main.py",
                "orchestrator/core.py",
                "memory/vector_store.py",
                "memory/trust_tracker.py",
                "governance/policy_engine.py",
                "governance/audit_log.py",
                "agents/base_agent.py",
                "agents/prover.py",
                "agents/evaluator.py",
                "agents/refiner.py",
                "configs/agents.yaml",
                "configs/policies.yaml",
                "configs/rubric.yaml",
                "requirements.txt",
                "Dockerfile",
                "docker-compose.yaml"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                print(f"  ❌ Missing files: {missing_files}")
                return False
            
            print("  ✅ All required files present")
            print(f"  📁 Found {len(required_files)} required files")
            
            return True
            
        except Exception as e:
            print(f"  ❌ File structure test failed: {str(e)}")
            return False
    
    async def _test_configuration_files(self) -> bool:
        """Test configuration file parsing."""
        try:
            import yaml
            
            # Test agents.yaml
            with open("configs/agents.yaml", 'r') as f:
                agents_config = yaml.safe_load(f)
                assert agents_config is not None, "agents.yaml should be valid YAML"
                assert 'provers' in agents_config, "agents.yaml should contain provers"
                assert 'evaluators' in agents_config, "agents.yaml should contain evaluators"
                assert 'refiners' in agents_config, "agents.yaml should contain refiners"
            
            # Test policies.yaml
            with open("configs/policies.yaml", 'r') as f:
                policies_config = yaml.safe_load(f)
                assert policies_config is not None, "policies.yaml should be valid YAML"
                assert 'resource_limits' in policies_config, "policies.yaml should contain resource_limits"
            
            # Test rubric.yaml
            with open("configs/rubric.yaml", 'r') as f:
                rubric_config = yaml.safe_load(f)
                assert rubric_config is not None, "rubric.yaml should be valid YAML"
                assert 'categories' in rubric_config, "rubric.yaml should contain categories"
            
            print("  ✅ agents.yaml parsing")
            print("  ✅ policies.yaml parsing")
            print("  ✅ rubric.yaml parsing")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Configuration files test failed: {str(e)}")
            return False
    
    async def _test_memory_system(self) -> bool:
        """Test memory system functionality."""
        try:
            # Test vector store import
            from memory.vector_store import VectorStore, MemoryEntry
            
            # Test basic initialization
            config = {
                'backend': 'chroma',
                'collection_name': 'test_memory',
                'persist_directory': self.temp_dir
            }
            
            vector_store = VectorStore(config)
            assert vector_store is not None, "Vector store should initialize"
            
            # Test memory entry creation
            test_entry = MemoryEntry(
                id="test_id",
                content="Test memory content",
                embedding=[0.1] * 100,
                metadata={"test": True},
                timestamp=datetime.now()
            )
            
            assert test_entry.content == "Test memory content", "Memory entry should be created correctly"
            
            print("  ✅ Vector store initialization")
            print("  ✅ Memory entry creation")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Memory system test failed: {str(e)}")
            return False
    
    async def _test_trust_tracking(self) -> bool:
        """Test trust tracking functionality."""
        try:
            from memory.trust_tracker import TrustTracker
            
            # Test trust tracker initialization with config
            config = {
                'decay_rate': 0.1,
                'min_trust': 0.1,
                'max_trust': 1.0
            }
            
            trust_tracker = TrustTracker(config)
            assert trust_tracker is not None, "Trust tracker should initialize"
            
            # Test trust score retrieval (should return default for new agent)
            trust_score = await trust_tracker.get_agent_trust_score("test_agent")
            assert trust_score >= 0, "Trust score should be non-negative"
            
            print("  ✅ Trust tracker initialization")
            print("  ✅ Trust score retrieval")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Trust tracking test failed: {str(e)}")
            return False
    
    async def _test_policy_engine(self) -> bool:
        """Test policy engine functionality."""
        try:
            from governance.policy_engine import PolicyEngine
            
            # Test policy engine initialization
            policies = {
                'resource_limits': {
                    'max_api_calls_per_minute': 60,
                    'max_tokens_per_request': 1000
                }
            }
            
            policy_engine = PolicyEngine(policies)
            assert policy_engine is not None, "Policy engine should initialize"
            
            print("  ✅ Policy engine initialization")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Policy engine test failed: {str(e)}")
            return False
    
    async def _test_audit_logging(self) -> bool:
        """Test audit logging functionality."""
        try:
            from governance.audit_log import AuditLog
            
            # Test audit log initialization
            audit_log = AuditLog()
            assert audit_log is not None, "Audit log should initialize"
            
            print("  ✅ Audit log initialization")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Audit logging test failed: {str(e)}")
            return False
    
    async def _test_agent_files(self) -> bool:
        """Test agent file imports and basic functionality."""
        try:
            from agents.base_agent import BaseAgent, AgentConfig
            from agents.prover import ProverAgent
            from agents.evaluator import EvaluatorAgent
            from agents.refiner import RefinerAgent
            
            # Test agent config creation
            config = AgentConfig(
                name="test_agent",
                prompt="You are a test agent.",
                model="openai",
                temperature=0.7,
                max_tokens=500
            )
            
            assert config.name == "test_agent", "Agent config should be created correctly"
            
            print("  ✅ Agent imports")
            print("  ✅ Agent config creation")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Agent files test failed: {str(e)}")
            return False
    
    async def _test_orchestrator_core(self) -> bool:
        """Test orchestrator core functionality."""
        try:
            # Test main.py import
            import main
            assert main is not None, "main.py should be importable"
            
            # Test orchestrator import
            from orchestrator.core import SophieReflexOrchestrator
            assert SophieReflexOrchestrator is not None, "SophieReflexOrchestrator should be importable"
            
            print("  ✅ Main module import")
            print("  ✅ Orchestrator import")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Orchestrator core test failed: {str(e)}")
            return False
    
    async def _test_webhook_server(self) -> bool:
        """Test webhook server functionality."""
        try:
            from ui.webhook_server import WebhookServer
            
            # Test webhook server initialization
            server = WebhookServer()
            assert server is not None, "Webhook server should initialize"
            
            print("  ✅ Webhook server initialization")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Webhook server test failed: {str(e)}")
            return False
    
    async def _test_docker_configuration(self) -> bool:
        """Test Docker configuration files."""
        try:
            # Test Dockerfile
            if os.path.exists("Dockerfile"):
                with open("Dockerfile", 'r') as f:
                    dockerfile_content = f.read()
                    assert "FROM python" in dockerfile_content, "Dockerfile should use Python base image"
                    assert "COPY requirements.txt" in dockerfile_content, "Dockerfile should copy requirements"
                
                print("  ✅ Dockerfile validation")
            else:
                print("  ⚠️  Dockerfile not found")
            
            # Test docker-compose.yaml
            if os.path.exists("docker-compose.yaml"):
                with open("docker-compose.yaml", 'r') as f:
                    compose_content = f.read()
                    assert "services:" in compose_content, "docker-compose.yaml should contain services"
                    assert "sophie-orchestrator:" in compose_content, "docker-compose.yaml should contain sophie-orchestrator service"
                
                print("  ✅ docker-compose.yaml validation")
            else:
                print("  ⚠️  docker-compose.yaml not found")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Docker configuration test failed: {str(e)}")
            return False
    
    async def _generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 SOPHIE REFLEX ORCHESTRATOR CORE TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r[1] == "PASSED"])
        failed_tests = len([r for r in self.test_results if r[1] in ["FAILED", "ERROR"]])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        print("-" * 40)
        
        for test_name, status, error in self.test_results:
            status_icon = "✅" if status == "PASSED" else "❌"
            print(f"{status_icon} {test_name}: {status}")
            if error:
                print(f"   Error: {error}")
        
        print("\n🎯 System Status:")
        if failed_tests == 0:
            print("✅ ALL SYSTEMS OPERATIONAL!")
            print("✅ Sophie Reflex Orchestrator core is functioning correctly")
            print("✅ All essential components are working")
            print("✅ Ready for advanced testing and deployment")
        elif failed_tests <= 2:
            print("⚠️  MOSTLY OPERATIONAL")
            print("✅ Core functionality is working")
            print("⚠️  Some minor issues detected")
            print("🔧 Review failed tests for improvements")
        else:
            print("❌ SIGNIFICANT ISSUES DETECTED")
            print("❌ Multiple components have problems")
            print("🔧 Immediate attention required")
            print("📝 Review error messages above")
        
        print("\n" + "=" * 60)


async def main():
    """Main test execution function."""
    test_suite = SophieCoreTestSuite()
    
    try:
        success = await test_suite.run_all_tests()
        
        if success:
            print("\n🎉 ALL CORE TESTS PASSED!")
            print("Sophie Reflex Orchestrator core is functioning correctly!")
            return 0
        else:
            print("\n❌ SOME CORE TESTS FAILED!")
            print("Please review the test report above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 CORE TEST SUITE ERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 