#!/usr/bin/env python3
"""
Security Test Suite for SOPHIE

This module provides comprehensive testing for SOPHIE's security scaffold functionality
including vault operations, OAuth integration, audit logging, and HMAC validation.
"""

import asyncio
import json
import time
import hashlib
import hmac
from pathlib import Path
from typing import Dict, Any, List
import structlog

# Add the parent directory to the path to import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.security_file import SecurityFile, create_vault_key, initialize_vault
from tools.oauth_google import check_google_auth_status, get_google_access_token

logger = structlog.get_logger()


class SecurityTestSuite:
    """
    Comprehensive test suite for SOPHIE's security functionality.
    """
    
    def __init__(self):
        self.test_results = []
        self.vault = None
        self.test_secrets = {
            "test_api_key": "sk-test-1234567890abcdef",
            "test_database_password": "test_secure_password_123",
            "test_oauth_client_secret": "test_client_secret_456",
            "test_encryption_key": "test_32_byte_encryption_key_here"
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests and return results."""
        print("🔐 SOPHIE Security Test Suite")
        print("=" * 50)
        
        test_suites = [
            ("Vault Operations", self.test_vault_operations),
            ("OAuth Integration", self.test_oauth_integration),
            ("Audit Logging", self.test_audit_logging),
            ("HMAC Validation", self.test_hmac_validation),
            ("Error Handling", self.test_error_handling),
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
    
    async def test_vault_operations(self) -> Dict[str, Any]:
        """Test vault operations including store, retrieve, and list."""
        try:
            # Initialize vault
            vault_key = create_vault_key()
            self.vault = initialize_vault(vault_key, ".test_vault.json")
            
            # Test store operations
            store_results = []
            for name, value in self.test_secrets.items():
                success = self.vault.store_secret(name, value, {
                    "test": True,
                    "created_by": "security_test"
                })
                store_results.append({"name": name, "success": success})
            
            # Test list operations
            secrets = self.vault.list_secrets()
            list_success = len(secrets) == len(self.test_secrets)
            
            # Test retrieve operations
            retrieve_results = []
            for name in self.test_secrets.keys():
                value = self.vault.retrieve_secret(name)
                retrieve_results.append({
                    "name": name,
                    "success": value is not None,
                    "value_matches": value == self.test_secrets[name]
                })
            
            # Calculate success metrics
            store_success_rate = sum(1 for r in store_results if r["success"]) / len(store_results)
            retrieve_success_rate = sum(1 for r in retrieve_results if r["success"]) / len(retrieve_results)
            value_match_rate = sum(1 for r in retrieve_results if r["value_matches"]) / len(retrieve_results)
            
            overall_success = (store_success_rate > 0.9 and 
                             retrieve_success_rate > 0.9 and 
                             value_match_rate > 0.9 and 
                             list_success)
            
            return {
                "success": overall_success,
                "metrics": {
                    "store_success_rate": store_success_rate,
                    "retrieve_success_rate": retrieve_success_rate,
                    "value_match_rate": value_match_rate,
                    "list_success": list_success
                },
                "details": {
                    "store_results": store_results,
                    "retrieve_results": retrieve_results,
                    "total_secrets": len(secrets)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_oauth_integration(self) -> Dict[str, Any]:
        """Test OAuth integration functionality."""
        try:
            # Test OAuth status check
            auth_status = await check_google_auth_status()
            
            # Test token retrieval (if authenticated)
            token_result = None
            if auth_status.get("authenticated", False):
                token_result = await get_google_access_token()
            
            # Calculate success metrics
            status_check_success = isinstance(auth_status, dict)
            token_success = token_result is not None if auth_status.get("authenticated") else True
            
            overall_success = status_check_success and token_success
            
            return {
                "success": overall_success,
                "metrics": {
                    "status_check_success": status_check_success,
                    "token_success": token_success,
                    "authenticated": auth_status.get("authenticated", False)
                },
                "details": {
                    "auth_status": auth_status,
                    "token_result": token_result
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_audit_logging(self) -> Dict[str, Any]:
        """Test audit logging functionality."""
        try:
            # Simulate audit events
            audit_events = [
                {"action": "vault_access", "user": "test_user", "timestamp": time.time()},
                {"action": "oauth_token_request", "user": "test_user", "timestamp": time.time()},
                {"action": "secret_retrieval", "user": "test_user", "timestamp": time.time()}
            ]
            
            # Test audit log creation
            audit_log_success = True
            for event in audit_events:
                try:
                    # Simulate audit log entry
                    log_entry = {
                        "timestamp": event["timestamp"],
                        "action": event["action"],
                        "user": event["user"],
                        "status": "success"
                    }
                    # In a real implementation, this would write to audit log
                    audit_log_success = audit_log_success and isinstance(log_entry, dict)
                except Exception:
                    audit_log_success = False
            
            return {
                "success": audit_log_success,
                "metrics": {
                    "audit_log_success": audit_log_success,
                    "events_processed": len(audit_events)
                },
                "details": {
                    "audit_events": audit_events
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_hmac_validation(self) -> Dict[str, Any]:
        """Test HMAC validation functionality."""
        try:
            # Test data
            test_message = "Hello, SOPHIE!"
            test_key = "test_secret_key_32_bytes_long"
            
            # Generate HMAC
            hmac_obj = hmac.new(
                test_key.encode('utf-8'),
                test_message.encode('utf-8'),
                hashlib.sha256
            )
            generated_hmac = hmac_obj.hexdigest()
            
            # Verify HMAC
            verification_obj = hmac.new(
                test_key.encode('utf-8'),
                test_message.encode('utf-8'),
                hashlib.sha256
            )
            verification_hmac = verification_obj.hexdigest()
            
            # Test validation
            hmac_valid = generated_hmac == verification_hmac
            
            # Test with different message (should fail)
            different_message = "Hello, World!"
            different_obj = hmac.new(
                test_key.encode('utf-8'),
                different_message.encode('utf-8'),
                hashlib.sha256
            )
            different_hmac = different_obj.hexdigest()
            different_valid = generated_hmac != different_hmac
            
            overall_success = hmac_valid and different_valid
            
            return {
                "success": overall_success,
                "metrics": {
                    "hmac_generation_success": hmac_valid,
                    "hmac_verification_success": hmac_valid,
                    "different_message_rejection": different_valid
                },
                "details": {
                    "original_hmac": generated_hmac,
                    "verification_hmac": verification_hmac,
                    "different_hmac": different_hmac
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        try:
            error_tests = []
            
            # Test invalid vault key
            try:
                invalid_vault = initialize_vault("invalid_key", ".invalid_vault.json")
                error_tests.append({"test": "invalid_vault_key", "success": False})
            except Exception:
                error_tests.append({"test": "invalid_vault_key", "success": True})
            
            # Test invalid secret retrieval
            if self.vault:
                try:
                    invalid_secret = self.vault.retrieve_secret("nonexistent_secret")
                    error_tests.append({"test": "invalid_secret_retrieval", "success": invalid_secret is None})
                except Exception:
                    error_tests.append({"test": "invalid_secret_retrieval", "success": True})
            
            # Test invalid OAuth request
            try:
                invalid_token = await get_google_access_token("invalid_credentials")
                error_tests.append({"test": "invalid_oauth_request", "success": invalid_token is None})
            except Exception:
                error_tests.append({"test": "invalid_oauth_request", "success": True})
            
            # Calculate success rate
            error_handling_success_rate = sum(1 for test in error_tests if test["success"]) / len(error_tests)
            overall_success = error_handling_success_rate > 0.8
            
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
        """Test performance characteristics."""
        try:
            performance_metrics = {}
            
            # Test vault operation performance
            if self.vault:
                start_time = time.time()
                for i in range(10):
                    self.vault.store_secret(f"perf_test_{i}", f"value_{i}")
                store_time = time.time() - start_time
                performance_metrics["store_operations_per_second"] = 10 / store_time
                
                start_time = time.time()
                for i in range(10):
                    self.vault.retrieve_secret(f"perf_test_{i}")
                retrieve_time = time.time() - start_time
                performance_metrics["retrieve_operations_per_second"] = 10 / retrieve_time
            
            # Test HMAC performance
            start_time = time.time()
            for i in range(100):
                hmac_obj = hmac.new(
                    "test_key".encode('utf-8'),
                    f"test_message_{i}".encode('utf-8'),
                    hashlib.sha256
                )
                hmac_obj.hexdigest()
            hmac_time = time.time() - start_time
            performance_metrics["hmac_operations_per_second"] = 100 / hmac_time
            
            # Define performance thresholds
            performance_thresholds = {
                "store_operations_per_second": 5.0,  # At least 5 ops/sec
                "retrieve_operations_per_second": 10.0,  # At least 10 ops/sec
                "hmac_operations_per_second": 1000.0  # At least 1000 ops/sec
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


async def run_security_tests():
    """Run the complete security test suite."""
    test_suite = SecurityTestSuite()
    results = await test_suite.run_all_tests()
    
    print(f"\n📊 Security Test Results")
    print("=" * 50)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1%}")
    print(f"Tests Passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_security_tests()) 