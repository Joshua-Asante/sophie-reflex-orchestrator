#!/usr/bin/env python3
"""
Core SOPHIE Modules Test Script
Tests core SOPHIE orchestrator modules and basic functionality
"""

import sys
import os
import importlib
from typing import List, Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_module_import(module_name: str, display_name: str = None) -> Dict[str, Any]:
    """Test importing a SOPHIE module."""
    try:
        module = importlib.import_module(module_name)
        return {
            "success": True,
            "module": module_name,
            "display_name": display_name or module_name,
            "error": None
        }
    except ImportError as e:
        return {
            "success": False,
            "module": module_name,
            "display_name": display_name or module_name,
            "error": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "module": module_name,
            "display_name": display_name or module_name,
            "error": f"Unexpected error: {str(e)}"
        }

def test_basic_functionality():
    """Test basic functionality of core modules."""
    print("\n🔧 Testing Basic Functionality")
    print("=" * 40)
    
    tests = []
    
    # Test 1: Configuration loading
    try:
        from configs.config_manager import ConfigManager
        config = ConfigManager()
        print("✅ ConfigManager: Basic initialization successful")
        tests.append(("ConfigManager", True))
    except Exception as e:
        print(f"❌ ConfigManager: {e}")
        tests.append(("ConfigManager", False))
    
    # Test 2: Database models
    try:
        from models.data_models import TaskRequest, TaskResponse
        task_request = TaskRequest(mode="engineering", prompt="test")
        print("✅ Data Models: Basic model creation successful")
        tests.append(("Data Models", True))
    except Exception as e:
        print(f"❌ Data Models: {e}")
        tests.append(("Data Models", False))
    
    # Test 3: Core telemetry
    try:
        from core.telemetry import telemetry_manager, log_telemetry_event
        log_telemetry_event("test", "test_component", success=True)
        print("✅ Telemetry: Basic logging successful")
        tests.append(("Telemetry", True))
    except Exception as e:
        print(f"❌ Telemetry: {e}")
        tests.append(("Telemetry", False))
    
    # Test 4: Agent base class
    try:
        from agents.base_agent import BaseAgent
        print("✅ Base Agent: Class import successful")
        tests.append(("Base Agent", True))
    except Exception as e:
        print(f"❌ Base Agent: {e}")
        tests.append(("Base Agent", False))
    
    # Test 5: Orchestrator core
    try:
        from orchestrator.core import Orchestrator
        print("✅ Orchestrator Core: Class import successful")
        tests.append(("Orchestrator Core", True))
    except Exception as e:
        print(f"❌ Orchestrator Core: {e}")
        tests.append(("Orchestrator Core", False))
    
    return tests

def main():
    """Test all core SOPHIE modules."""
    print("🧪 Testing SOPHIE Core Modules")
    print("=" * 50)
    
    # Core modules to test
    modules = [
        # Core modules
        ("core.telemetry", "Telemetry System"),
        ("core.autonomous_executor", "Autonomous Executor"),
        ("core.batch_processor", "Batch Processor"),
        
        # Configuration
        ("configs.config_manager", "Config Manager"),
        ("configs.schemas.plan_schema", "Plan Schema"),
        
        # Models
        ("models.data_models", "Data Models"),
        ("models.failure_report", "Failure Report"),
        
        # Agents
        ("agents.base_agent", "Base Agent"),
        ("agents.evaluator", "Evaluator Agent"),
        ("agents.prover", "Prover Agent"),
        
        # Orchestrator
        ("orchestrator.core", "Orchestrator Core"),
        ("orchestrator.components.agent_manager", "Agent Manager"),
        ("orchestrator.components.evaluation_engine", "Evaluation Engine"),
        
        # Tools
        ("tools.adapters.generative_ai", "Generative AI Adapter"),
        ("tools.adapters.web_scrape", "Web Scrape Adapter"),
        
        # Recovery
        ("recovery.recovery_manager", "Recovery Manager"),
        ("recovery.revision_engine", "Revision Engine"),
        
        # Governance
        ("governance.audit_log", "Audit Log"),
        ("governance.feedback_handler", "Feedback Handler"),
        
        # Security
        ("security.security_manager", "Security Manager"),
        ("security.store_credentials", "Credential Store"),
    ]
    
    results = []
    success_count = 0
    total_count = len(modules)
    
    for module_name, display_name in modules:
        print(f"Testing {display_name}...", end=" ")
        result = test_module_import(module_name, display_name)
        results.append(result)
        
        if result["success"]:
            print("✅")
            success_count += 1
        else:
            print(f"❌ {result['error']}")
    
    print("\n" + "=" * 50)
    print(f"📊 Module Import Results: {success_count}/{total_count} successful")
    
    # Test basic functionality
    functionality_tests = test_basic_functionality()
    functionality_success = sum(1 for _, success in functionality_tests if success)
    functionality_total = len(functionality_tests)
    
    print(f"📊 Functionality Results: {functionality_success}/{functionality_total} successful")
    
    # Overall assessment
    total_success = success_count + functionality_success
    total_tests = total_count + functionality_total
    
    print(f"\n🎯 Overall Results: {total_success}/{total_tests} tests passed")
    
    if total_success == total_tests:
        print("🎉 All core modules are working!")
        return True
    else:
        print("⚠️  Some modules need attention:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['display_name']}: {result['error']}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
