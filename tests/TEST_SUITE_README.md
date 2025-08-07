# Sophie Reflexive Orchestrator - Comprehensive Test Suite

This directory contains a comprehensive, modularized test suite for the Sophie Reflexive Orchestrator system. The test suite is designed to validate all aspects of the system, from individual components to complete end-to-end workflows.

## 🧪 Test Modules

The test suite is organized into 8 specialized modules:

### 1. **Unit Tests** (`test_modules/unit_tests.py`)
- Tests individual components in isolation
- Validates agent configurations, initialization, and basic functionality
- Tests data structures, enums, and core classes
- **Focus**: Component-level validation

### 2. **Integration Tests** (`test_modules/integration_tests.py`)
- Tests component interactions and workflows
- Validates agent manager, evaluation engine, trust manager integration
- Tests component communication patterns
- **Focus**: Component interaction validation

### 3. **Agent Tests** (`test_modules/agent_tests.py`)
- Tests individual agent types and their specific functionalities
- Validates ProverAgent variant generation, quality assessment, collaboration
- Tests EvaluatorAgent scoring, consensus, category evaluation
- Tests RefinerAgent population analysis, mutation, crossover, creation
- **Focus**: Agent-specific functionality validation

### 4. **Orchestrator Tests** (`test_modules/orchestrator_tests.py`)
- Tests main orchestrator functionality and genetic algorithm loop
- Validates configuration loading, agent population management
- Tests generation execution, trust management integration
- Tests performance monitoring and state management
- **Focus**: Core system orchestration validation

### 5. **Memory Tests** (`test_modules/memory_tests.py`)
- Tests memory systems and persistence functionality
- Validates vector store operations, trust tracking
- Tests memory manager operations, persistence, search, cleanup
- **Focus**: Memory system validation

### 6. **Governance Tests** (`test_modules/governance_tests.py`)
- Tests policy engine and audit logging functionality
- Validates policy evaluation, HITL requirements, trust validation
- Tests audit logging, search, filtering, persistence
- **Focus**: Governance and compliance validation

### 7. **Performance Tests** (`test_modules/performance_tests.py`)
- Tests system performance and scalability
- Validates agent creation/execution performance, memory operations
- Tests trust tracking, audit logging, policy evaluation performance
- Tests concurrent operations, memory usage, response time
- **Focus**: Performance and scalability validation

### 8. **End-to-End Tests** (`test_modules/e2e_tests.py`)
- Tests complete workflows and system integration
- Validates complete workflow, genetic algorithm loop
- Tests agent population evolution, system integration
- Tests error recovery, performance under load, system resilience
- **Focus**: Complete system validation

## 🚀 Quick Start

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test Module
```bash
python run_tests.py --module unit
python run_tests.py --module integration
python run_tests.py --module agent
python run_tests.py --module orchestrator
python run_tests.py --module memory
python run_tests.py --module governance
python run_tests.py --module performance
python run_tests.py --module e2e
```

### Run Tests with Custom Options
```bash
# Run tests sequentially (not parallel)
python run_tests.py --module all

# Set custom timeout
python run_tests.py --timeout 1800

# Save test artifacts
python run_tests.py --save-artifacts

# Verbose output
python run_tests.py --verbose
```

## 📊 Test Results

The test suite generates comprehensive reports including:

- **Test Summary**: Total tests, passed/failed/error counts, success rate
- **Execution Time**: Total execution time and per-module timing
- **Detailed Results**: Individual test results with error messages
- **Module Summary**: Performance breakdown by test module
- **Artifacts**: Test artifacts saved to temporary directory

### Sample Output
```
🧪 Sophie Reflexive Orchestrator - Comprehensive Test Suite
================================================================================

🔧 Initializing test modules...
  ✅ unit: Initialized
  ✅ integration: Initialized
  ✅ agent: Initialized
  ✅ orchestrator: Initialized
  ✅ memory: Initialized
  ✅ governance: Initialized
  ✅ performance: Initialized
  ✅ e2e: Initialized

🚀 Running 8 test modules...
✅ unit: PASSED
✅ integration: PASSED
✅ agent: PASSED
✅ orchestrator: PASSED
✅ memory: PASSED
✅ governance: PASSED
✅ performance: PASSED
✅ e2e: PASSED

📊 Generating comprehensive test report...

📈 Test Summary:
  Total Tests: 8
  Passed: 8
  Failed: 0
  Errors: 0
  Success Rate: 100.0%
  Total Time: 45.23s
  Report saved to: /tmp/sophie_test_suite_xxx/reports/comprehensive_report.json

🎉 All tests passed!
```

## 🔧 Configuration

The test suite can be configured through the `TestSuiteOrchestrator` class:

```python
# Default configuration
config = {
    "parallel_execution": True,
    "timeout_per_test": 300,  # 5 minutes
    "max_retries": 2,
    "generate_reports": True,
    "save_artifacts": True,
    "test_modules": {
        "unit": {"enabled": True, "timeout": 60},
        "integration": {"enabled": True, "timeout": 120},
        "agent": {"enabled": True, "timeout": 180},
        "orchestrator": {"enabled": True, "timeout": 300},
        "memory": {"enabled": True, "timeout": 90},
        "governance": {"enabled": True, "timeout": 120},
        "performance": {"enabled": True, "timeout": 600},
        "e2e": {"enabled": True, "timeout": 900}
    }
}
```

## 📁 File Structure

```
test_modules/
├── __init__.py              # Package initialization
├── unit_tests.py            # Unit tests
├── integration_tests.py     # Integration tests
├── agent_tests.py          # Agent tests
├── orchestrator_tests.py   # Orchestrator tests
├── memory_tests.py         # Memory tests
├── governance_tests.py     # Governance tests
├── performance_tests.py    # Performance tests
└── e2e_tests.py           # End-to-end tests

test_suite_orchestrator.py  # Main test orchestrator
run_tests.py                # Simple test runner
TEST_SUITE_README.md        # This file
```

## 🛠️ Development

### Adding New Tests

To add new tests to a module:

1. **Unit Tests**: Add test methods to `UnitTestSuite` class
2. **Integration Tests**: Add test methods to `IntegrationTestSuite` class
3. **Agent Tests**: Add test methods to `AgentTestSuite` class
4. **Orchestrator Tests**: Add test methods to `OrchestratorTestSuite` class
5. **Memory Tests**: Add test methods to `MemoryTestSuite` class
6. **Governance Tests**: Add test methods to `GovernanceTestSuite` class
7. **Performance Tests**: Add test methods to `PerformanceTestSuite` class
8. **E2E Tests**: Add test methods to `E2ETestSuite` class

### Test Method Pattern

```python
async def _test_example_functionality(self) -> bool:
    """Test example functionality."""
    try:
        # Setup test environment
        # Execute test logic
        # Verify results
        assert condition, "Error message"
        return True
    except Exception as e:
        logger.error("Test failed", error=str(e))
        return False
```

### Running Individual Test Modules

You can also run individual test modules directly:

```python
import asyncio
from test_modules.unit_tests import UnitTestSuite

async def run_unit_tests():
    suite = UnitTestSuite("/tmp/test_dir")
    success = await suite.run_all_tests()
    return success

# Run
asyncio.run(run_unit_tests())
```

## 🔍 Debugging

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Timeout Errors**: Increase timeout for slow tests
   ```bash
   python run_tests.py --timeout 1800
   ```

3. **Memory Issues**: Run tests sequentially
   ```bash
   python run_tests.py --module all
   ```

### Verbose Output

Enable verbose output for detailed debugging:

```bash
python run_tests.py --verbose
```

### Test Artifacts

Test artifacts are saved to a temporary directory and can be preserved:

```bash
python run_tests.py --save-artifacts
```

## 📈 Performance Considerations

- **Unit Tests**: Fastest (1-2 minutes)
- **Integration Tests**: Medium (2-3 minutes)
- **Agent Tests**: Medium (3-5 minutes)
- **Orchestrator Tests**: Medium (3-5 minutes)
- **Memory Tests**: Fast (1-2 minutes)
- **Governance Tests**: Fast (1-2 minutes)
- **Performance Tests**: Slow (5-10 minutes)
- **E2E Tests**: Slowest (10-15 minutes)

**Total Runtime**: ~30-45 minutes for all tests

## 🎯 Test Coverage

The test suite provides comprehensive coverage:

- ✅ **Component Initialization**: All major components
- ✅ **Configuration Loading**: System and component configs
- ✅ **Agent Functionality**: All agent types and features
- ✅ **Orchestrator Logic**: Genetic algorithm and workflow
- ✅ **Memory Operations**: Storage, retrieval, search
- ✅ **Trust Management**: Scoring, validation, decay
- ✅ **Policy Evaluation**: HITL, trust validation, execution
- ✅ **Audit Logging**: Logging, search, filtering
- ✅ **Performance**: Scalability and response times
- ✅ **Error Handling**: Recovery and resilience
- ✅ **End-to-End**: Complete workflows

## 🤝 Contributing

When adding new features to the Sophie Reflexive Orchestrator:

1. **Add Unit Tests**: Test individual components
2. **Add Integration Tests**: Test component interactions
3. **Add Agent Tests**: Test agent-specific functionality
4. **Add E2E Tests**: Test complete workflows
5. **Update Performance Tests**: If performance-critical
6. **Run Full Suite**: Ensure all tests pass

## 📝 License

This test suite is part of the Sophie Reflexive Orchestrator project and follows the same license terms. 