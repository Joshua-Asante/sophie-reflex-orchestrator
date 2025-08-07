#!/usr/bin/env python3
"""
Sophie Reflexive Orchestrator - Test Runner

Simple script to run the comprehensive test suite.
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_suite_orchestrator import TestSuiteOrchestrator


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Sophie Reflexive Orchestrator Test Runner")
    parser.add_argument(
        "--module", 
        choices=["unit_agents", "unit_config", "unit_governance", "integration", "agent", "orchestrator", "memory", "governance", "performance", "e2e", "all"],
        default="all",
        help="Specific test module to run (default: all)"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run tests in parallel (default: True)"
    )
    parser.add_argument(
        "--timeout", 
        type=int,
        default=900,
        help="Timeout per test module in seconds (default: 900)"
    )
    parser.add_argument(
        "--save-artifacts", 
        action="store_true",
        help="Save test artifacts (default: True)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--mock-agents", 
        action="store_true",
        default=True,
        help="Use mock agents (no real API calls) - default: True"
    )
    parser.add_argument(
        "--real-agents", 
        action="store_true",
        help="Use real agents (with API calls) - overrides --mock-agents"
    )
    
    args = parser.parse_args()
    
    print("🧪 Sophie Reflexive Orchestrator - Test Runner")
    print("=" * 60)
    # Determine agent type
    use_mock_agents = args.mock_agents and not args.real_agents
    
    print(f"Module: {args.module}")
    print(f"Parallel: {args.parallel}")
    print(f"Timeout: {args.timeout}s")
    print(f"Save Artifacts: {args.save_artifacts}")
    print(f"Verbose: {args.verbose}")
    print(f"Agent Type: {'Mock' if use_mock_agents else 'Real'}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    try:
        # Create test orchestrator
        orchestrator = TestSuiteOrchestrator()
        
        # Configure based on arguments
        if args.module != "all":
            # Enable only the specified module
            for module_name in orchestrator.config["test_modules"]:
                orchestrator.config["test_modules"][module_name]["enabled"] = (
                    module_name == args.module
                )
        
        orchestrator.config["parallel_execution"] = args.parallel
        orchestrator.config["timeout_per_test"] = args.timeout
        orchestrator.config["save_artifacts"] = args.save_artifacts
        orchestrator.config["use_mock_agents"] = use_mock_agents
        
        # Run tests
        success = asyncio.run(orchestrator.run_all_tests())
        
        if success:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 