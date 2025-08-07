#!/usr/bin/env python3
"""
Test script to run only the State Management test
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_modules.orchestrator_tests import OrchestratorTestSuite
from tempfile import mkdtemp


async def test_state_management_only():
    """Run only the State Management test."""
    temp_dir = mkdtemp()
    test_suite = OrchestratorTestSuite(temp_dir)
    
    print("🧪 Running State Management Test Only")
    print("-" * 40)
    
    try:
        result = await test_suite._test_state_management()
        if result:
            print("✅ State Management: PASSED")
        else:
            print("❌ State Management: FAILED")
    except Exception as e:
        print(f"❌ State Management: ERROR - {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
        # Also try to get more info about the orchestrator state
        try:
            from orchestrator.core import SophieReflexOrchestrator
            orchestrator = SophieReflexOrchestrator("configs/system.yaml")
            print(f"Orchestrator state: {orchestrator.state}")
            print(f"State status: {orchestrator.state.status}")
            print(f"State current_task: {orchestrator.state.current_task}")
            print(f"State current_generation: {orchestrator.state.current_generation}")
        except Exception as debug_e:
            print(f"Debug info error: {debug_e}")


if __name__ == "__main__":
    asyncio.run(test_state_management_only()) 