#!/usr/bin/env python3
"""
Test script to run orchestrator tests with mock agents (no real API calls)
"""

import asyncio
import sys
import os
from tempfile import mkdtemp

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_modules.orchestrator_tests import OrchestratorTestSuite


async def test_orchestrator_with_mock_agents():
    """Run orchestrator tests with mock agents."""
    temp_dir = mkdtemp()
    test_suite = OrchestratorTestSuite(temp_dir, use_mock_agents=True)

    print("🧪 Running Orchestrator Tests with Mock Agents")
    print("=" * 50)

    try:
        result = await test_suite.run_all_tests()
        if result:
            print("\n🎉 All tests PASSED!")
            return True
        else:
            print("\n❌ Some tests FAILED!")
            return False
    except Exception as e:
        print(f"\n💥 Test suite ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_orchestrator_with_mock_agents())
    sys.exit(0 if success else 1) 