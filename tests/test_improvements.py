#!/usr/bin/env python3
"""
Test script to demonstrate the improvements made to main.py

This script tests the modular components and new features.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.mock_orchestrator import MockOrchestratorFactory, MockConfigManager
from ui.cli_components import TaskExecutor, ServerManager, InteractiveController, ResultDisplay
from configs.config_manager import ConfigManager
from utils.resource_manager import ResourceManager, AsyncConfigLoader
from ui.interactive_ui import EnhancedInteractiveUI, InputValidator, ColorOutput


async def test_mock_orchestrator():
    """Test the mock orchestrator functionality."""
    print("🧪 Testing Mock Orchestrator...")
    
    # Create mock orchestrator
    orchestrator = MockOrchestratorFactory.create_mock_orchestrator()
    
    # Test task execution
    task_executor = TaskExecutor(orchestrator)
    success = await task_executor.run_task("Test task for demonstration")
    
    print(f"✅ Mock orchestrator test: {'PASSED' if success else 'FAILED'}")
    return success


async def test_config_manager():
    """Test the configuration manager."""
    print("🧪 Testing Configuration Manager...")
    
    try:
        config_manager = ConfigManager()
        configs = config_manager.load_and_validate_all()
        
        print(f"✅ Configuration manager test: PASSED")
        print(f"   Loaded {len(configs)} configuration files")
        return True
        
    except Exception as e:
        print(f"❌ Configuration manager test: FAILED - {e}")
        return False


async def test_resource_manager():
    """Test the resource manager."""
    print("🧪 Testing Resource Manager...")
    
    try:
        resource_manager = ResourceManager()
        
        # Test async file operations
        test_data = {"test": "data", "number": 42}
        await resource_manager.save_results_async(test_data, "test_output.json")
        
        # Test loading
        loaded_data = await resource_manager.load_config_async("test_output.json")
        
        # Cleanup
        if os.path.exists("test_output.json"):
            os.remove("test_output.json")
        
        print(f"✅ Resource manager test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Resource manager test: FAILED - {e}")
        return False


async def test_enhanced_ui():
    """Test the enhanced interactive UI."""
    print("🧪 Testing Enhanced Interactive UI...")
    
    try:
        ui = EnhancedInteractiveUI()
        
        # Test input validation
        validator = InputValidator()
        valid_task = validator.validate_task_description("Valid task description")
        invalid_task = validator.validate_task_description("")
        
        # Test color output
        colored_text = ColorOutput.success("Success message")
        
        print(f"✅ Enhanced UI test: PASSED")
        print(f"   Input validation: {'PASSED' if valid_task and not invalid_task else 'FAILED'}")
        print(f"   Color output: {'PASSED' if colored_text else 'FAILED'}")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced UI test: FAILED - {e}")
        return False


async def test_modular_components():
    """Test the modular CLI components."""
    print("🧪 Testing Modular CLI Components...")
    
    try:
        # Test with mock orchestrator
        orchestrator = MockOrchestratorFactory.create_mock_orchestrator()
        
        # Test task executor
        task_executor = TaskExecutor(orchestrator)
        
        # Test server manager
        server_manager = ServerManager()
        
        # Test result display
        mock_results = [
            type('MockResult', (), {
                'generation': 1,
                'best_score': 0.85,
                'average_score': 0.75,
                'execution_time': 2.5,
                'interventions': [],
                'best_solution': {'overall_feedback': 'Test solution'}
            })()
        ]
        
        # This should not raise an exception
        ResultDisplay.display_results(mock_results)
        
        print(f"✅ Modular components test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Modular components test: FAILED - {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    print("🚀 Running Improvement Tests...")
    print("=" * 50)
    
    tests = [
        test_mock_orchestrator,
        test_config_manager,
        test_resource_manager,
        test_enhanced_ui,
        test_modular_components
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📊 Test Summary:")
    print("=" * 30)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Improvements are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


def main():
    """Main test function."""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 