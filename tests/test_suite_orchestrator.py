#!/usr/bin/env python3
"""
Sophie Reflexive Orchestrator - Comprehensive Test Suite
Modularized testing framework for end-to-end system validation
"""

import asyncio
import sys
import os
import json
import tempfile
import shutil
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import structlog

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import test modules
from test_modules.unit_tests_agents import AgentUnitTestSuite
from test_modules.unit_tests_config import ConfigUnitTestSuite
from test_modules.unit_tests_governance import GovernanceUnitTestSuite
from test_modules.integration_tests import IntegrationTestSuite
from test_modules.agent_tests import AgentTestSuite
from test_modules.orchestrator_tests import OrchestratorTestSuite
from test_modules.memory_tests import MemoryTestSuite
from test_modules.governance_tests import GovernanceTestSuite
from test_modules.performance_tests import PerformanceTestSuite
from test_modules.e2e_tests import E2ETestSuite

logger = structlog.get_logger()


class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    module_name: str
    test_name: str
    status: TestStatus
    execution_time: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class TestSuiteOrchestrator:
    """Main orchestrator for running all test modules."""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.temp_dir = None
        self.start_time = None
        self.test_modules = {}
        self.config = self._load_test_config()
        
    def _load_test_config(self) -> Dict[str, Any]:
        """Load test configuration."""
        return {
            "parallel_execution": True,
            "timeout_per_test": 300,  # 5 minutes
            "max_retries": 2,
            "generate_reports": True,
            "save_artifacts": True,
            "test_modules": {
                "unit_agents": {"enabled": True, "timeout": 30},
                "unit_config": {"enabled": True, "timeout": 30},
                "unit_governance": {"enabled": True, "timeout": 30},
                "integration": {"enabled": True, "timeout": 120},
                "agent": {"enabled": True, "timeout": 180},
                "orchestrator": {"enabled": True, "timeout": 300},
                "memory": {"enabled": True, "timeout": 90},
                "governance": {"enabled": True, "timeout": 120},
                "performance": {"enabled": True, "timeout": 600},
                "e2e": {"enabled": True, "timeout": 900}
            }
        }
    
    async def run_all_tests(self) -> bool:
        """Run all test modules and return overall success status."""
        print("🧪 Sophie Reflexive Orchestrator - Comprehensive Test Suite")
        print("=" * 80)
        
        self.start_time = datetime.now()
        
        try:
            # Setup test environment
            await self._setup_test_environment()
            
            # Initialize test modules
            await self._initialize_test_modules()
            
            # Run test modules
            all_passed = await self._run_test_modules()
            
            # Generate comprehensive report
            await self._generate_comprehensive_report()
            
            return all_passed
            
        finally:
            await self._cleanup_test_environment()
    
    async def _setup_test_environment(self):
        """Setup test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp(prefix="sophie_test_suite_")
        print(f"📁 Test directory: {self.temp_dir}")
        
        # Create subdirectories for different test types
        os.makedirs(os.path.join(self.temp_dir, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "logs"), exist_ok=True)
        
        print("🔧 Test environment created")
    
    async def _initialize_test_modules(self):
        """Initialize all test modules."""
        print("\n🔧 Initializing test modules...")
        
        module_configs = {
            "unit_agents": AgentUnitTestSuite,
            "unit_config": ConfigUnitTestSuite,
            "unit_governance": GovernanceUnitTestSuite,
            "integration": IntegrationTestSuite,
            "agent": AgentTestSuite,
            "orchestrator": OrchestratorTestSuite,
            "memory": MemoryTestSuite,
            "governance": GovernanceTestSuite,
            "performance": PerformanceTestSuite,
            "e2e": E2ETestSuite
        }
        
        for module_name, module_class in module_configs.items():
            if self.config["test_modules"][module_name]["enabled"]:
                try:
                    module_instance = module_class(self.temp_dir)
                    self.test_modules[module_name] = module_instance
                    print(f"  ✅ {module_name}: Initialized")
                except Exception as e:
                    print(f"  ❌ {module_name}: Failed to initialize - {str(e)}")
                    logger.error(f"Failed to initialize {module_name} module", error=str(e))
    
    async def _run_test_modules(self) -> bool:
        """Run all test modules."""
        print(f"\n🚀 Running {len(self.test_modules)} test modules...")
        
        if self.config["parallel_execution"]:
            return await self._run_test_modules_parallel()
        else:
            return await self._run_test_modules_sequential()
    
    async def _run_test_modules_parallel(self) -> bool:
        """Run test modules in parallel."""
        tasks = []
        for module_name, module_instance in self.test_modules.items():
            task = asyncio.create_task(
                self._run_single_test_module(module_name, module_instance)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_passed = True
        for i, (module_name, _) in enumerate(self.test_modules.items()):
            result = results[i]
            if isinstance(result, Exception):
                print(f"❌ {module_name}: ERROR - {str(result)}")
                all_passed = False
            elif not result:
                print(f"❌ {module_name}: FAILED")
                all_passed = False
            else:
                print(f"✅ {module_name}: PASSED")
        
        return all_passed
    
    async def _run_test_modules_sequential(self) -> bool:
        """Run test modules sequentially."""
        all_passed = True
        
        for module_name, module_instance in self.test_modules.items():
            try:
                result = await self._run_single_test_module(module_name, module_instance)
                if result:
                    print(f"✅ {module_name}: PASSED")
                else:
                    print(f"❌ {module_name}: FAILED")
                    all_passed = False
            except Exception as e:
                print(f"❌ {module_name}: ERROR - {str(e)}")
                all_passed = False
        
        return all_passed
    
    async def _run_single_test_module(self, module_name: str, module_instance) -> bool:
        """Run a single test module with timeout and retries."""
        timeout = self.config["test_modules"][module_name]["timeout"]
        max_retries = self.config["max_retries"]
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Run the test module with timeout
                result = await asyncio.wait_for(
                    module_instance.run_all_tests(),
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                # Record test result
                test_result = TestResult(
                    module_name=module_name,
                    test_name="all_tests",
                    status=TestStatus.PASSED if result else TestStatus.FAILED,
                    execution_time=execution_time,
                    details={"attempt": attempt + 1}
                )
                self.test_results.append(test_result)
                
                return result
                
            except asyncio.TimeoutError:
                print(f"⏰ {module_name}: TIMEOUT (attempt {attempt + 1})")
                if attempt == max_retries:
                    test_result = TestResult(
                        module_name=module_name,
                        test_name="all_tests",
                        status=TestStatus.ERROR,
                        execution_time=timeout,
                        error_message="Timeout exceeded"
                    )
                    self.test_results.append(test_result)
                    return False
                    
            except Exception as e:
                print(f"❌ {module_name}: ERROR (attempt {attempt + 1}) - {str(e)}")
                if attempt == max_retries:
                    test_result = TestResult(
                        module_name=module_name,
                        test_name="all_tests",
                        status=TestStatus.ERROR,
                        execution_time=0.0,
                        error_message=str(e)
                    )
                    self.test_results.append(test_result)
                    return False
        
        return False
    
    async def _generate_comprehensive_report(self):
        """Generate comprehensive test report."""
        print("\n📊 Generating comprehensive test report...")
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in self.test_results if r.status == TestStatus.FAILED])
        error_tests = len([r for r in self.test_results if r.status == TestStatus.ERROR])
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        # Generate report
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_execution_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": [
                {
                    "module_name": r.module_name,
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "details": r.details
                }
                for r in self.test_results
            ],
            "module_summary": self._generate_module_summary()
        }
        
        # Save report
        report_path = os.path.join(self.temp_dir, "reports", "comprehensive_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📈 Test Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Errors: {error_tests}")
        print(f"  Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Report saved to: {report_path}")
    
    def _generate_module_summary(self) -> Dict[str, Any]:
        """Generate summary by module."""
        module_summary = {}
        
        for result in self.test_results:
            module = result.module_name
            if module not in module_summary:
                module_summary[module] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "errors": 0,
                    "total_time": 0.0
                }
            
            module_summary[module]["total"] += 1
            module_summary[module]["total_time"] += result.execution_time
            
            if result.status == TestStatus.PASSED:
                module_summary[module]["passed"] += 1
            elif result.status == TestStatus.FAILED:
                module_summary[module]["failed"] += 1
            elif result.status == TestStatus.ERROR:
                module_summary[module]["errors"] += 1
        
        return module_summary
    
    async def _cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            if self.config["save_artifacts"]:
                print(f"📁 Test artifacts preserved at: {self.temp_dir}")
            else:
                shutil.rmtree(self.temp_dir)
                print("🧹 Test environment cleaned up")


async def main():
    """Main entry point for the test suite."""
    orchestrator = TestSuiteOrchestrator()
    
    try:
        success = await orchestrator.run_all_tests()
        
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
        print(f"\n💥 Test suite error: {str(e)}")
        logger.error("Test suite error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 