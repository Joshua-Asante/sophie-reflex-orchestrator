#!/usr/bin/env python3
"""
Execution Test Suite for SOPHIE

This module provides comprehensive testing for SOPHIE's execution engine functionality
including unified executor, constitutional AI, and risk assessment capabilities.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import structlog

# Add the parent directory to the path to import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.unified_executor import UnifiedExecutor, ExecutionType, RiskLevel
from core.constitutional_executor import ConstitutionalExecutor

logger = structlog.get_logger()


class ExecutionTestSuite:
    """
    Comprehensive test suite for SOPHIE's execution functionality.
    """
    
    def __init__(self):
        self.test_results = []
        self.unified_executor = UnifiedExecutor()
        self.constitutional_executor = ConstitutionalExecutor()
        
        # Test intents for different execution types
        self.test_intents = {
            "cli": "Create a new directory called 'test_project'",
            "api": "Send a GET request to https://api.example.com/users",
            "filesystem": "Create a file called 'test.txt' with content 'Hello World'",
            "database": "Create a new table called 'users' with columns id, name, email",
            "cloud": "Deploy a Docker container to AWS ECS",
            "shell": "List all running processes",
            "python": "Calculate the factorial of 10"
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all execution tests and return results."""
        print("⚡ SOPHIE Execution Test Suite")
        print("=" * 50)
        
        test_suites = [
            ("Execution Type Classification", self.test_execution_type_classification),
            ("Plan Generation", self.test_plan_generation),
            ("Risk Assessment", self.test_risk_assessment),
            ("Constitutional Validation", self.test_constitutional_validation),
            ("Execution Flow", self.test_execution_flow),
            ("Error Recovery", self.test_error_recovery),
            ("Performance", self.test_performance)
        ]
        
        for suite_name, test_func in test_suites:
            print(f"\n🧪 Running {suite_name} Tests")
            print("-" * 30)
            
            try:
                result = await test_func()
                self.test_results.append({
                    "suite": suite_name,
                    "status": "PASSED" if result["success"] else "FAILED",
                    "details": result
                })
                
                if result["success"]:
                    print(f"✅ {suite_name}: PASSED")
                else:
                    print(f"❌ {suite_name}: FAILED - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ {suite_name}: ERROR - {str(e)}")
                self.test_results.append({
                    "suite": suite_name,
                    "status": "ERROR",
                    "details": {"error": str(e)}
                })
        
        return self._generate_test_report()
    
    async def test_execution_type_classification(self) -> Dict[str, Any]:
        """Test execution type classification accuracy."""
        try:
            classification_results = []
            
            for intent_type, intent in self.test_intents.items():
                # Test classification
                classified_type = await self.unified_executor._classify_execution_type(intent)
                
                # Check if classification matches expected type
                expected_type = ExecutionType(intent_type.upper())
                classification_correct = classified_type == expected_type
                
                classification_results.append({
                    "intent": intent,
                    "expected_type": intent_type,
                    "classified_type": classified_type.value,
                    "correct": classification_correct
                })
            
            # Calculate accuracy
            correct_classifications = sum(1 for r in classification_results if r["correct"])
            total_classifications = len(classification_results)
            accuracy = correct_classifications / total_classifications if total_classifications > 0 else 0
            
            overall_success = accuracy >= 0.8  # At least 80% accuracy
            
            return {
                "success": overall_success,
                "metrics": {
                    "classification_accuracy": accuracy,
                    "correct_classifications": correct_classifications,
                    "total_classifications": total_classifications
                },
                "details": {
                    "classification_results": classification_results
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_plan_generation(self) -> Dict[str, Any]:
        """Test execution plan generation."""
        try:
            plan_results = []
            
            for intent_type, intent in self.test_intents.items():
                # Generate execution plan
                execution_type = ExecutionType(intent_type.upper())
                plan = await self.unified_executor._generate_execution_plan(
                    intent, execution_type
                )
                
                # Validate plan structure
                plan_valid = (
                    hasattr(plan, 'id') and
                    hasattr(plan, 'user_intent') and
                    hasattr(plan, 'execution_type') and
                    hasattr(plan, 'commands') and
                    hasattr(plan, 'trust_score') and
                    hasattr(plan, 'risk_level')
                )
                
                plan_results.append({
                    "intent": intent,
                    "execution_type": intent_type,
                    "plan_valid": plan_valid,
                    "trust_score": getattr(plan, 'trust_score', 0),
                    "risk_level": getattr(plan, 'risk_level', RiskLevel.LOW).value,
                    "commands_count": len(getattr(plan, 'commands', []))
                })
            
            # Calculate success metrics
            valid_plans = sum(1 for r in plan_results if r["plan_valid"])
            total_plans = len(plan_results)
            plan_success_rate = valid_plans / total_plans if total_plans > 0 else 0
            
            # Check trust score distribution
            avg_trust_score = sum(r["trust_score"] for r in plan_results) / len(plan_results)
            trust_score_acceptable = avg_trust_score >= 0.5  # At least 50% average trust
            
            overall_success = plan_success_rate >= 0.9 and trust_score_acceptable
            
            return {
                "success": overall_success,
                "metrics": {
                    "plan_success_rate": plan_success_rate,
                    "avg_trust_score": avg_trust_score,
                    "valid_plans": valid_plans,
                    "total_plans": total_plans
                },
                "details": {
                    "plan_results": plan_results
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_risk_assessment(self) -> Dict[str, Any]:
        """Test risk assessment functionality."""
        try:
            risk_results = []
            
            # Test different risk scenarios
            risk_scenarios = [
                ("low_risk", "List current directory contents"),
                ("medium_risk", "Create a backup of important files"),
                ("high_risk", "Delete all files in the system"),
                ("critical_risk", "Format the entire hard drive")
            ]
            
            for scenario_name, intent in risk_scenarios:
                # Generate plan and assess risk
                execution_type = await self.unified_executor._classify_execution_type(intent)
                plan = await self.unified_executor._generate_execution_plan(intent, execution_type)
                
                # Assess risk level
                risk_level = getattr(plan, 'risk_level', RiskLevel.LOW)
                
                # Validate risk assessment logic
                risk_assessment_correct = (
                    (scenario_name == "low_risk" and risk_level == RiskLevel.LOW) or
                    (scenario_name == "medium_risk" and risk_level == RiskLevel.MEDIUM) or
                    (scenario_name == "high_risk" and risk_level == RiskLevel.HIGH) or
                    (scenario_name == "critical_risk" and risk_level == RiskLevel.CRITICAL)
                )
                
                risk_results.append({
                    "scenario": scenario_name,
                    "intent": intent,
                    "assessed_risk": risk_level.value,
                    "expected_risk": scenario_name.split('_')[0].upper(),
                    "correct": risk_assessment_correct
                })
            
            # Calculate accuracy
            correct_assessments = sum(1 for r in risk_results if r["correct"])
            total_assessments = len(risk_results)
            risk_accuracy = correct_assessments / total_assessments if total_assessments > 0 else 0
            
            overall_success = risk_accuracy >= 0.75  # At least 75% accuracy
            
            return {
                "success": overall_success,
                "metrics": {
                    "risk_assessment_accuracy": risk_accuracy,
                    "correct_assessments": correct_assessments,
                    "total_assessments": total_assessments
                },
                "details": {
                    "risk_results": risk_results
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_constitutional_validation(self) -> Dict[str, Any]:
        """Test constitutional AI validation."""
        try:
            validation_results = []
            
            # Test different validation scenarios
            validation_scenarios = [
                ("safe_operation", "Create a new user account", True),
                ("potentially_unsafe", "Delete user account", False),
                ("system_operation", "Restart the server", False),
                ("data_operation", "Export user data", True)
            ]
            
            for scenario_name, intent, expected_approval in validation_scenarios:
                # Test constitutional validation
                directive = await self.constitutional_executor._interpret_directive_navigator(intent)
                plan = await self.constitutional_executor._generate_plan_integrator(directive)
                validation = await self.constitutional_executor._validate_plan_constitutional(plan)
                
                validation_approved = validation.get("approved", False)
                validation_correct = validation_approved == expected_approval
                
                validation_results.append({
                    "scenario": scenario_name,
                    "intent": intent,
                    "expected_approval": expected_approval,
                    "actual_approval": validation_approved,
                    "correct": validation_correct
                })
            
            # Calculate accuracy
            correct_validations = sum(1 for r in validation_results if r["correct"])
            total_validations = len(validation_results)
            validation_accuracy = correct_validations / total_validations if total_validations > 0 else 0
            
            overall_success = validation_accuracy >= 0.8  # At least 80% accuracy
            
            return {
                "success": overall_success,
                "metrics": {
                    "validation_accuracy": validation_accuracy,
                    "correct_validations": correct_validations,
                    "total_validations": total_validations
                },
                "details": {
                    "validation_results": validation_results
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_execution_flow(self) -> Dict[str, Any]:
        """Test complete execution flow."""
        try:
            flow_results = []
            
            # Test simple execution flow
            test_intent = "Create a test file with content 'Hello World'"
            
            # Execute the command
            result = await self.unified_executor.execute_command(
                test_intent, 
                auto_approve=True  # Auto-approve for testing
            )
            
            # Validate execution result
            execution_successful = (
                result.get("status") == "completed" and
                "plan_id" in result and
                "execution_result" in result
            )
            
            flow_results.append({
                "intent": test_intent,
                "status": result.get("status"),
                "successful": execution_successful,
                "plan_id": result.get("plan_id"),
                "execution_result": result.get("execution_result")
            })
            
            overall_success = execution_successful
            
            return {
                "success": overall_success,
                "metrics": {
                    "execution_success": execution_successful,
                    "flow_completed": execution_successful
                },
                "details": {
                    "flow_results": flow_results
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_error_recovery(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        try:
            error_tests = []
            
            # Test invalid execution type
            try:
                invalid_type = ExecutionType("INVALID")
                error_tests.append({"test": "invalid_execution_type", "success": False})
            except ValueError:
                error_tests.append({"test": "invalid_execution_type", "success": True})
            
            # Test empty intent
            try:
                empty_result = await self.unified_executor.execute_command("")
                error_tests.append({"test": "empty_intent", "success": empty_result.get("status") == "rejected"})
            except Exception:
                error_tests.append({"test": "empty_intent", "success": True})
            
            # Test malformed intent
            try:
                malformed_result = await self.unified_executor.execute_command("INVALID_COMMAND_THAT_SHOULD_FAIL")
                error_tests.append({"test": "malformed_intent", "success": malformed_result.get("status") == "rejected"})
            except Exception:
                error_tests.append({"test": "malformed_intent", "success": True})
            
            # Calculate success rate
            error_handling_success_rate = sum(1 for test in error_tests if test["success"]) / len(error_tests)
            overall_success = error_handling_success_rate >= 0.8
            
            return {
                "success": overall_success,
                "metrics": {
                    "error_handling_success_rate": error_handling_success_rate,
                    "tests_run": len(error_tests)
                },
                "details": {
                    "error_tests": error_tests
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test execution performance."""
        try:
            performance_metrics = {}
            
            # Test plan generation performance
            start_time = time.time()
            for i in range(5):
                intent = f"Create test file {i}"
                execution_type = await self.unified_executor._classify_execution_type(intent)
                plan = await self.unified_executor._generate_execution_plan(intent, execution_type)
            plan_generation_time = time.time() - start_time
            performance_metrics["plans_per_second"] = 5 / plan_generation_time
            
            # Test classification performance
            start_time = time.time()
            for i in range(10):
                intent = f"Test intent {i}"
                await self.unified_executor._classify_execution_type(intent)
            classification_time = time.time() - start_time
            performance_metrics["classifications_per_second"] = 10 / classification_time
            
            # Test constitutional validation performance
            start_time = time.time()
            for i in range(3):
                intent = f"Test constitutional validation {i}"
                directive = await self.constitutional_executor._interpret_directive_navigator(intent)
                plan = await self.constitutional_executor._generate_plan_integrator(directive)
                await self.constitutional_executor._validate_plan_constitutional(plan)
            validation_time = time.time() - start_time
            performance_metrics["validations_per_second"] = 3 / validation_time
            
            # Define performance thresholds
            performance_thresholds = {
                "plans_per_second": 1.0,  # At least 1 plan/sec
                "classifications_per_second": 5.0,  # At least 5 classifications/sec
                "validations_per_second": 0.5  # At least 0.5 validations/sec
            }
            
            # Check if performance meets thresholds
            performance_success = True
            for metric, threshold in performance_thresholds.items():
                if metric in performance_metrics:
                    if performance_metrics[metric] < threshold:
                        performance_success = False
            
            return {
                "success": performance_success,
                "metrics": performance_metrics,
                "details": {
                    "performance_thresholds": performance_thresholds
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["status"] == "PASSED")
        failed_tests = sum(1 for result in self.test_results if result["status"] == "FAILED")
        error_tests = sum(1 for result in self.test_results if result["status"] == "ERROR")
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": success_rate
            },
            "results": self.test_results,
            "overall_status": "PASSED" if success_rate >= 0.8 else "FAILED"
        }


async def run_execution_tests():
    """Run the complete execution test suite."""
    test_suite = ExecutionTestSuite()
    results = await test_suite.run_all_tests()
    
    print(f"\n📊 Execution Test Results")
    print("=" * 50)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1%}")
    print(f"Tests Passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_execution_tests()) 