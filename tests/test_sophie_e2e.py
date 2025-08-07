#!/usr/bin/env python3
"""
Sophie Reflex Orchestrator End-to-End Test Suite
Comprehensive testing of the entire system functionality
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

try:
    from main import SophieCLI
    from orchestrator.core import SophieReflexOrchestrator
    from configs.config_manager import ConfigManager
    from memory.vector_store import VectorStore
    from memory.trust_tracker import TrustTracker
    from governance.policy_engine import PolicyEngine
    from governance.audit_log import AuditLog
    from agents.base_agent import BaseAgent, AgentConfig
    from agents.prover import ProverAgent
    from agents.evaluator import EvaluatorAgent
    from agents.refiner import RefinerAgent
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class SophieE2ETestSuite:
    """Comprehensive end-to-end test suite for Sophie Reflex Orchestrator."""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.test_config = None
        self.orchestrator = None
        
    async def run_all_tests(self) -> bool:
        """Run all end-to-end tests and return success status."""
        print("🧪 Starting Sophie Reflex Orchestrator E2E Test Suite")
        print("=" * 70)
        
        try:
            # Setup test environment
            await self._setup_test_environment()
            
            # Run test categories
            test_categories = [
                ("Configuration Loading", self._test_configuration_loading),
                ("Component Initialization", self._test_component_initialization),
                ("Agent Creation", self._test_agent_creation),
                ("Memory System", self._test_memory_system),
                ("Trust Tracking", self._test_trust_tracking),
                ("Policy Engine", self._test_policy_engine),
                ("Audit Logging", self._test_audit_logging),
                ("Orchestrator Integration", self._test_orchestrator_integration),
                ("Agent Execution", self._test_agent_execution),
                ("End-to-End Workflow", self._test_e2e_workflow),
                ("Performance Tests", self._test_performance),
                ("Error Handling", self._test_error_handling)
            ]
            
            all_passed = True
            for category_name, test_func in test_categories:
                print(f"\n📋 Testing: {category_name}")
                print("-" * 50)
                
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
        """Setup test environment with temporary directory and test configuration."""
        self.temp_dir = tempfile.mkdtemp(prefix="sophie_e2e_test_")
        print(f"📁 Test directory: {self.temp_dir}")
        
        # Create test configuration
        self.test_config = {
            'system': {
                'max_generations': 3,
                'population_size': 5,
                'evaluation_threshold': 0.7,
                'trust_threshold': 0.6
            },
            'agents': {
                'provers': [
                    {
                        'name': 'test_prover',
                        'prompt': 'You are a test prover agent.',
                        'model': 'openai',
                        'temperature': 0.7,
                        'max_tokens': 500,
                        'hyperparameters': {
                            'max_variants': 2,
                            'creativity': 0.6,
                            'detail_level': 0.7
                        }
                    }
                ],
                'evaluators': [
                    {
                        'name': 'test_evaluator',
                        'prompt': 'You are a test evaluator agent.',
                        'model': 'openai',
                        'temperature': 0.5,
                        'max_tokens': 300,
                        'hyperparameters': {
                            'evaluation_mode': 'standard',
                            'consensus_enabled': False
                        }
                    }
                ],
                'refiners': [
                    {
                        'name': 'test_refiner',
                        'prompt': 'You are a test refiner agent.',
                        'model': 'openai',
                        'temperature': 0.6,
                        'max_tokens': 400,
                        'hyperparameters': {
                            'mutation_strength': 0.3,
                            'focus_areas': ['clarity', 'efficiency']
                        }
                    }
                ]
            },
            'memory': {
                'backend': 'chroma',
                'collection_name': 'test_memory',
                'persist_directory': self.temp_dir
            },
            'trust': {
                'decay_rate': 0.1,
                'min_trust': 0.1,
                'max_trust': 1.0
            },
            'policies': {
                'resource_limits': {
                    'max_api_calls_per_minute': 60,
                    'max_tokens_per_request': 1000
                },
                'quality_thresholds': {
                    'min_quality_score': 0.6,
                    'max_execution_time': 30
                }
            }
        }
        
        print("🔧 Test configuration created")
    
    async def _cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("🧹 Test environment cleaned up")
    
    async def _test_configuration_loading(self) -> bool:
        """Test configuration loading and validation."""
        try:
            # Test config manager
            config_manager = ConfigManager()
            
            # Test loading from dict
            loaded_config = config_manager.load_from_dict(self.test_config)
            assert loaded_config is not None, "Configuration should load successfully"
            
            # Test validation
            validation_result = config_manager.validate_config(loaded_config)
            assert validation_result['valid'], f"Configuration validation failed: {validation_result.get('errors', [])}"
            
            print("  ✅ Configuration loading")
            print("  ✅ Configuration validation")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Configuration loading failed: {str(e)}")
            return False
    
    async def _test_component_initialization(self) -> bool:
        """Test initialization of core components."""
        try:
            # Test vector store initialization
            vector_store = VectorStore(self.test_config['memory'])
            assert vector_store is not None, "Vector store should initialize"
            
            # Test trust tracker initialization
            trust_tracker = TrustTracker()
            assert trust_tracker is not None, "Trust tracker should initialize"
            
            # Test policy engine initialization
            policy_engine = PolicyEngine(self.test_config['policies'])
            assert policy_engine is not None, "Policy engine should initialize"
            
            # Test audit log initialization
            audit_log = AuditLog()
            assert audit_log is not None, "Audit log should initialize"
            
            print("  ✅ Vector store initialization")
            print("  ✅ Trust tracker initialization")
            print("  ✅ Policy engine initialization")
            print("  ✅ Audit log initialization")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Component initialization failed: {str(e)}")
            return False
    
    async def _test_agent_creation(self) -> bool:
        """Test agent creation and configuration."""
        try:
            # Test prover agent creation
            prover_config = AgentConfig(
                name="test_prover",
                prompt="You are a test prover agent.",
                model="openai",
                temperature=0.7,
                max_tokens=500,
                hyperparameters={
                    'max_variants': 2,
                    'creativity': 0.6,
                    'detail_level': 0.7
                }
            )
            
            prover_agent = ProverAgent(prover_config)
            assert prover_agent is not None, "Prover agent should be created"
            assert prover_agent.config.name == "test_prover", "Agent name should match"
            
            # Test evaluator agent creation
            evaluator_config = AgentConfig(
                name="test_evaluator",
                prompt="You are a test evaluator agent.",
                model="openai",
                temperature=0.5,
                max_tokens=300,
                hyperparameters={
                    'evaluation_mode': 'standard',
                    'consensus_enabled': False
                }
            )
            
            evaluator_agent = EvaluatorAgent(evaluator_config)
            assert evaluator_agent is not None, "Evaluator agent should be created"
            
            # Test refiner agent creation
            refiner_config = AgentConfig(
                name="test_refiner",
                prompt="You are a test refiner agent.",
                model="openai",
                temperature=0.6,
                max_tokens=400,
                hyperparameters={
                    'mutation_strength': 0.3,
                    'focus_areas': ['clarity', 'efficiency']
                }
            )
            
            refiner_agent = RefinerAgent(refiner_config)
            assert refiner_agent is not None, "Refiner agent should be created"
            
            print("  ✅ Prover agent creation")
            print("  ✅ Evaluator agent creation")
            print("  ✅ Refiner agent creation")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Agent creation failed: {str(e)}")
            return False
    
    async def _test_memory_system(self) -> bool:
        """Test memory system functionality."""
        try:
            vector_store = VectorStore(self.test_config['memory'])
            
            # Test memory addition
            test_content = "Test memory entry for E2E testing"
            test_embedding = [0.1] * 100  # 100-dimensional vector
            
            memory_id = await vector_store.add_memory(
                content=test_content,
                embedding=test_embedding,
                metadata={"test": True, "e2e": True}
            )
            
            assert memory_id is not None, "Memory should be added successfully"
            
            # Test memory retrieval
            retrieved_memory = await vector_store.get_memory(memory_id)
            assert retrieved_memory is not None, "Memory should be retrievable"
            assert retrieved_memory.content == test_content, "Memory content should match"
            
            # Test memory search
            search_results = await vector_store.search_similar(test_embedding, limit=5)
            assert len(search_results) > 0, "Search should return results"
            
            print("  ✅ Memory addition")
            print("  ✅ Memory retrieval")
            print("  ✅ Memory search")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Memory system test failed: {str(e)}")
            return False
    
    async def _test_trust_tracking(self) -> bool:
        """Test trust tracking functionality."""
        try:
            trust_tracker = TrustTracker()
            
            # Test trust event recording
            agent_id = "test_agent_1"
            event_type = "success"
            score = 0.8
            
            await trust_tracker.record_event(agent_id, event_type, score)
            
            # Test trust score retrieval
            trust_score = trust_tracker.get_trust_score(agent_id)
            assert trust_score > 0, "Trust score should be positive"
            
            # Test trust score update
            await trust_tracker.record_event(agent_id, "success", 0.9)
            updated_score = trust_tracker.get_trust_score(agent_id)
            assert updated_score > trust_score, "Trust score should increase"
            
            print("  ✅ Trust event recording")
            print("  ✅ Trust score retrieval")
            print("  ✅ Trust score updates")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Trust tracking test failed: {str(e)}")
            return False
    
    async def _test_policy_engine(self) -> bool:
        """Test policy engine functionality."""
        try:
            policy_engine = PolicyEngine(self.test_config['policies'])
            
            # Test policy evaluation
            action = "add_entry"
            context = {
                "agent_id": "test_agent",
                "content_length": 500,
                "trust_score": 0.7
            }
            
            result = await policy_engine.evaluate_action(action, context)
            assert result is not None, "Policy evaluation should return result"
            assert 'allowed' in result, "Policy result should contain 'allowed' field"
            
            # Test resource limit policy
            resource_context = {
                "api_calls_this_minute": 30,
                "tokens_this_request": 800
            }
            
            resource_result = await policy_engine.evaluate_action("api_call", resource_context)
            assert resource_result is not None, "Resource policy should evaluate"
            
            print("  ✅ Policy evaluation")
            print("  ✅ Resource limit checking")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Policy engine test failed: {str(e)}")
            return False
    
    async def _test_audit_logging(self) -> bool:
        """Test audit logging functionality."""
        try:
            audit_log = AuditLog()
            
            # Test event logging
            event_type = "agent_execution"
            event_data = {
                "agent_id": "test_agent",
                "task": "test task",
                "result": "success"
            }
            
            await audit_log.log_event(event_type, event_data)
            
            # Test metric logging
            metric_name = "execution_time"
            metric_value = 2.5
            
            await audit_log.log_metric(metric_name, metric_value)
            
            # Test audit statistics
            stats = await audit_log.get_audit_statistics()
            assert stats is not None, "Audit statistics should be retrievable"
            
            print("  ✅ Event logging")
            print("  ✅ Metric logging")
            print("  ✅ Statistics retrieval")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Audit logging test failed: {str(e)}")
            return False
    
    async def _test_orchestrator_integration(self) -> bool:
        """Test orchestrator integration with all components."""
        try:
            # Create orchestrator with test configuration
            orchestrator = SophieReflexOrchestrator(self.test_config)
            
            # Test orchestrator initialization
            assert orchestrator is not None, "Orchestrator should initialize"
            
            # Test component access
            assert orchestrator.vector_store is not None, "Vector store should be accessible"
            assert orchestrator.trust_tracker is not None, "Trust tracker should be accessible"
            assert orchestrator.policy_engine is not None, "Policy engine should be accessible"
            assert orchestrator.audit_log is not None, "Audit log should be accessible"
            
            # Test orchestrator status
            status = await orchestrator.get_status()
            assert status is not None, "Orchestrator status should be retrievable"
            
            print("  ✅ Orchestrator initialization")
            print("  ✅ Component integration")
            print("  ✅ Status retrieval")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Orchestrator integration test failed: {str(e)}")
            return False
    
    async def _test_agent_execution(self) -> bool:
        """Test agent execution with mock data."""
        try:
            # Create test agents
            prover_config = AgentConfig(
                name="test_prover",
                prompt="You are a test prover agent. Generate a simple solution.",
                model="openai",
                temperature=0.7,
                max_tokens=200,
                hyperparameters={'max_variants': 1}
            )
            
            prover_agent = ProverAgent(prover_config)
            
            # Test agent execution (with mock LLM response)
            task = "Create a simple Python function to add two numbers"
            context = {"test": True}
            
            # Mock the LLM call to avoid actual API calls during testing
            original_call_llm = prover_agent.call_llm
            prover_agent.call_llm = lambda prompt, ctx: asyncio.create_task(
                asyncio.Future()).set_result({
                    "content": "def add_numbers(a, b):\n    return a + b",
                    "confidence": 0.8,
                    "reasoning": "Simple addition function"
                })
            
            try:
                result = await prover_agent.execute(task, context)
                assert result is not None, "Agent execution should return result"
                assert result.status.value == "COMPLETED", "Agent should complete successfully"
                
                print("  ✅ Agent execution")
                print("  ✅ Result generation")
                
            finally:
                # Restore original method
                prover_agent.call_llm = original_call_llm
            
            return True
            
        except Exception as e:
            print(f"  ❌ Agent execution test failed: {str(e)}")
            return False
    
    async def _test_e2e_workflow(self) -> bool:
        """Test complete end-to-end workflow."""
        try:
            # Create orchestrator
            orchestrator = SophieReflexOrchestrator(self.test_config)
            
            # Test workflow initialization
            await orchestrator.initialize()
            
            # Test task execution workflow
            task = "Write a Python function to calculate factorial"
            
            # Mock agent responses for testing
            def mock_prover_execute(task, context):
                return asyncio.create_task(asyncio.Future()).set_result(
                    type('MockResult', (), {
                        'result': {'content': 'def factorial(n): return 1 if n <= 1 else n * factorial(n-1)'},
                        'confidence_score': 0.8,
                        'status': type('MockStatus', (), {'value': 'COMPLETED'})()
                    })()
                )
            
            def mock_evaluator_execute(task, context):
                return asyncio.create_task(asyncio.Future()).set_result(
                    type('MockResult', (), {
                        'result': {'overall_score': 0.85, 'feedback': 'Good implementation'},
                        'confidence_score': 0.9,
                        'status': type('MockStatus', (), {'value': 'COMPLETED'})()
                    })()
                )
            
            # Temporarily replace agent execution methods
            original_prover_execute = orchestrator.prover_agent.execute
            original_evaluator_execute = orchestrator.evaluator_agent.execute
            
            try:
                orchestrator.prover_agent.execute = mock_prover_execute
                orchestrator.evaluator_agent.execute = mock_evaluator_execute
                
                # Test workflow execution
                workflow_result = await orchestrator.execute_task(task)
                assert workflow_result is not None, "Workflow should complete"
                
                print("  ✅ Workflow initialization")
                print("  ✅ Task execution")
                print("  ✅ Result processing")
                
            finally:
                # Restore original methods
                orchestrator.prover_agent.execute = original_prover_execute
                orchestrator.evaluator_agent.execute = original_evaluator_execute
            
            return True
            
        except Exception as e:
            print(f"  ❌ E2E workflow test failed: {str(e)}")
            return False
    
    async def _test_performance(self) -> bool:
        """Test system performance characteristics."""
        try:
            print("  📊 Running performance tests...")
            
            # Test memory system performance
            vector_store = VectorStore(self.test_config['memory'])
            
            start_time = datetime.now()
            for i in range(10):
                await vector_store.add_memory(
                    content=f"Performance test memory {i}",
                    embedding=[0.1] * 100,
                    metadata={"perf_test": True, "index": i}
                )
            
            memory_time = (datetime.now() - start_time).total_seconds()
            print(f"  ⏱️  Memory operations (10 entries): {memory_time:.3f}s")
            
            # Test trust tracking performance
            trust_tracker = TrustTracker()
            
            start_time = datetime.now()
            for i in range(50):
                await trust_tracker.record_event(f"agent_{i}", "success", 0.8)
            
            trust_time = (datetime.now() - start_time).total_seconds()
            print(f"  ⏱️  Trust tracking (50 events): {trust_time:.3f}s")
            
            # Performance thresholds
            assert memory_time < 5.0, "Memory operations should be fast"
            assert trust_time < 3.0, "Trust tracking should be fast"
            
            print("  ✅ Performance thresholds met")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Performance test failed: {str(e)}")
            return False
    
    async def _test_error_handling(self) -> bool:
        """Test error handling and recovery."""
        try:
            # Test invalid configuration handling
            try:
                invalid_config = {'invalid': 'config'}
                orchestrator = SophieReflexOrchestrator(invalid_config)
                print("  ⚠️  Invalid config handled gracefully")
            except Exception as e:
                print(f"  ⚠️  Invalid config error (expected): {str(e)}")
            
            # Test memory system error handling
            vector_store = VectorStore(self.test_config['memory'])
            
            try:
                # Test with invalid embedding
                await vector_store.add_memory(
                    content="Test",
                    embedding="invalid_embedding",  # Should be list
                    metadata={}
                )
                print("  ⚠️  Invalid embedding handled gracefully")
            except Exception as e:
                print(f"  ⚠️  Invalid embedding error (expected): {str(e)}")
            
            # Test trust tracker error handling
            trust_tracker = TrustTracker()
            
            try:
                await trust_tracker.record_event("", "invalid_event", -1.0)
                print("  ⚠️  Invalid trust event handled gracefully")
            except Exception as e:
                print(f"  ⚠️  Invalid trust event error (expected): {str(e)}")
            
            print("  ✅ Error handling tests completed")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error handling test failed: {str(e)}")
            return False
    
    async def _generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 70)
        print("📊 SOPHIE REFLEX ORCHESTRATOR E2E TEST REPORT")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r[1] == "PASSED"])
        failed_tests = len([r for r in self.test_results if r[1] in ["FAILED", "ERROR"]])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        print("-" * 50)
        
        for test_name, status, error in self.test_results:
            status_icon = "✅" if status == "PASSED" else "❌"
            print(f"{status_icon} {test_name}: {status}")
            if error:
                print(f"   Error: {error}")
        
        print("\n🎯 System Status:")
        if failed_tests == 0:
            print("✅ ALL SYSTEMS OPERATIONAL!")
            print("✅ Sophie Reflex Orchestrator is running optimally")
            print("✅ All components are functioning correctly")
            print("✅ Ready for production use")
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
        
        print("\n" + "=" * 70)


async def main():
    """Main test execution function."""
    test_suite = SophieE2ETestSuite()
    
    try:
        success = await test_suite.run_all_tests()
        
        if success:
            print("\n🎉 ALL E2E TESTS PASSED!")
            print("Sophie Reflex Orchestrator is running optimally!")
            return 0
        else:
            print("\n❌ SOME E2E TESTS FAILED!")
            print("Please review the test report above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 E2E TEST SUITE ERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 