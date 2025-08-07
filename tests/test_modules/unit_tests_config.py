#!/usr/bin/env python3
"""
Configuration Unit Tests Module

Tests configuration loading and validation components.
"""

import asyncio
import sys
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import structlog

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from orchestrator.models.orchestrator_config import OrchestratorConfig
    from configs.config_manager import ConfigManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

logger = structlog.get_logger()


class ConfigUnitTestSuite:
    """Unit test suite for configuration components."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.test_results = []
        
    async def run_all_tests(self) -> bool:
        """Run all configuration unit tests and return success status."""
        print("🧪 Running Configuration Unit Tests")
        print("-" * 40)
        
        test_functions = [
            ("Orchestrator Config Loading", self._test_orchestrator_config_loading),
            ("Config Manager Loading", self._test_config_manager_loading)
        ]
        
        all_passed = True
        for test_name, test_func in test_functions:
            try:
                result = await test_func()
                if result:
                    print(f"✅ {test_name}: PASSED")
                    self.test_results.append((test_name, "PASSED", None))
                else:
                    print(f"❌ {test_name}: FAILED")
                    self.test_results.append((test_name, "FAILED", "Test returned False"))
                    all_passed = False
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {str(e)}")
                self.test_results.append((test_name, "ERROR", str(e)))
                all_passed = False
        
        return all_passed
    
    async def _test_orchestrator_config_loading(self) -> bool:
        """Test OrchestratorConfig loading from file."""
        try:
            # Test loading from default config
            config = OrchestratorConfig.from_file("configs/system.yaml")

            # Verify essential fields are loaded
            assert config.max_generations is not None
            assert config.population_size is not None
            assert config.max_agents is not None

            # Test specific fields
            assert config.max_agents >= 0
            assert config.population_size >= 0
            assert config.mutation_rate >= 0.0

            return True

        except Exception as e:
            logger.error("Orchestrator config loading test failed", error=str(e))
            return False
    
    async def _test_config_manager_loading(self) -> bool:
        """Test ConfigManager loading and validation."""
        try:
            config_manager = ConfigManager()
            
            # Test loading configurations
            config_manager.load_and_validate_all()
            
            # Verify configurations are loaded
            assert config_manager.get_system_config() is not None
            assert config_manager.get_agent_config() is not None
            assert config_manager.get_policy_config() is not None
            
            return True
            
        except Exception as e:
            logger.error("Config manager loading test failed", error=str(e))
            return False 