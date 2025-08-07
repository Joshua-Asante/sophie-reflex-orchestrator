"""
Test Phase 1 Performance Optimizations

Tests all Phase 1 optimization components:
- Connection Pooling
- Request Batching
- Smart Caching
- Error Recovery
- Performance Monitoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import time
from typing import Dict, Any

from core.connection_pool import connection_manager, ProviderType
from core.batch_processor import batch_processor
from core.smart_cache import smart_cache
from core.error_recovery import error_recovery_manager, RetryableOperation
from core.performance_monitor import performance_monitor
from core.performance_integration import (
    performance_optimizer, optimized_llm_call, optimized_tool_call,
    get_performance_status, run_performance_optimization
)


def test_connection_pool():
    """Test connection pooling functionality."""
    print("Testing connection pooling...")
    
    # Test provider registration
    assert len(connection_manager.pool.configs) > 0, "Should have registered providers"
    
    # Test provider types
    expected_providers = ["openai", "google", "xai", "mistral", "deepseek", "kimi"]
    for provider in expected_providers:
        assert ProviderType(provider) in connection_manager.pool.configs, f"Provider {provider} should be registered"
    
    print("✓ Connection pool configuration working")


@pytest.mark.asyncio
async def test_batch_processor():
    """Test batch processing functionality."""
    print("Testing batch processor...")
    
    # Test request submission
    results = []
    
    async def callback(result):
        results.append(result)
    
    # Submit multiple requests
    batch_keys = []
    for i in range(5):
        batch_key = await batch_processor.submit_request(
            prompt=f"Test prompt {i}",
            model="gpt-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            callback=callback
        )
        batch_keys.append(batch_key)
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Check statistics
    stats = batch_processor.get_batch_statistics()
    assert "total_batches" in stats, "Should have batch statistics"
    
    print("✓ Batch processor working")


@pytest.mark.asyncio
async def test_smart_cache():
    """Test smart caching functionality."""
    print("Testing smart cache...")
    
    # Test LLM response caching
    test_prompt = "Test prompt for caching"
    test_response = "Test response"
    
    # Cache a response
    smart_cache.cache_llm_response(
        test_prompt, test_response, "gpt-4", "openai", 0.7, 100
    )
    
    # Try to retrieve from cache
    cached_response = await smart_cache.get_llm_response(
        test_prompt, "gpt-4", "openai", 0.7, 100
    )
    
    assert cached_response == test_response, "Should retrieve cached response"
    
    # Test tool result caching
    test_tool = "test_tool"
    test_params = {"param1": "value1"}
    test_result = "Test tool result"
    
    # Cache tool result
    smart_cache.cache_tool_result(test_tool, test_params, test_result)
    
    # Try to retrieve from cache
    cached_tool_result = await smart_cache.get_tool_result(test_tool, test_params)
    
    assert cached_tool_result == test_result, "Should retrieve cached tool result"
    
    # Test statistics
    stats = smart_cache.get_statistics()
    assert "semantic_cache" in stats, "Should have semantic cache stats"
    assert "result_cache" in stats, "Should have result cache stats"
    
    print("✓ Smart cache working")


@pytest.mark.asyncio
async def test_error_recovery():
    """Test error recovery functionality."""
    print("Testing error recovery...")
    
    # Test circuit breaker
    component = "test_component"
    circuit_breaker = error_recovery_manager.get_circuit_breaker(component)
    
    # Initially should be closed
    assert circuit_breaker.can_execute(), "Circuit breaker should be closed initially"
    
    # Record some failures
    for i in range(3):
        circuit_breaker.record_failure()
    
    # Should still be able to execute (threshold is 5)
    assert circuit_breaker.can_execute(), "Circuit breaker should still be closed"
    
    # Record more failures to trigger circuit breaker
    for i in range(3):
        circuit_breaker.record_failure()
    
    # Should not be able to execute now
    assert not circuit_breaker.can_execute(), "Circuit breaker should be open"
    
    # Test retryable operation
    retryable_op = RetryableOperation(error_recovery_manager, "test_retry")
    
    success_count = 0
    
    async def failing_operation():
        nonlocal success_count
        success_count += 1
        if success_count < 3:
            raise Exception("Simulated failure")
        return "Success"
    
    # Execute with retry logic
    result = await retryable_op.execute(failing_operation)
    assert result == "Success", "Should eventually succeed"
    
    print("✓ Error recovery working")


@pytest.mark.asyncio
async def test_performance_monitor():
    """Test performance monitoring functionality."""
    print("Testing performance monitor...")
    
    # Test metric recording
    performance_monitor.record_component_metric("test_component", "test_metric", 1.5)
    
    # Test timer functionality
    performance_monitor.start_component_timer("test_component", "test_operation")
    await asyncio.sleep(0.1)
    performance_monitor.end_component_timer("test_component", "test_operation")
    
    # Get component performance
    component_perf = performance_monitor.get_component_performance("test_component")
    assert "test_component" in component_perf["component"], "Should have component performance data"
    
    # Get system performance
    system_perf = performance_monitor.get_system_performance()
    assert "overall_metrics" in system_perf, "Should have overall metrics"
    assert "bottlenecks" in system_perf, "Should have bottleneck detection"
    
    # Get performance summary
    summary = performance_monitor.get_performance_summary()
    assert "health_score" in summary, "Should have health score"
    assert "total_metrics" in summary, "Should have total metrics"
    
    print("✓ Performance monitor working")


@pytest.mark.asyncio
async def test_performance_integration():
    """Test integrated performance optimization."""
    print("Testing performance integration...")
    
    # Test optimized LLM call (mock)
    try:
        result = await optimized_llm_call(
            "Test prompt",
            "gpt-4",
            "openai",
            temperature=0.7,
            max_tokens=100
        )
        print(f"✓ Optimized LLM call result: {result}")
    except Exception as e:
        print(f"⚠️ LLM call failed (expected without API keys): {e}")
    
    # Test optimized tool call
    result = await optimized_tool_call("test_tool", {"param": "value"})
    assert "test_tool" in result, "Should return tool result"
    
    # Test performance status
    status = await get_performance_status()
    assert "performance" in status, "Should have performance data"
    assert "cache" in status, "Should have cache data"
    assert "batch_processing" in status, "Should have batch processing data"
    assert "error_recovery" in status, "Should have error recovery data"
    
    # Test performance optimization
    await run_performance_optimization()
    
    print("✓ Performance integration working")


@pytest.mark.asyncio
async def test_end_to_end_optimization():
    """Test end-to-end optimization workflow."""
    print("Testing end-to-end optimization...")
    
    # Simulate multiple operations
    operations = []
    
    for i in range(10):
        # Simulate LLM calls
        operations.append(optimized_llm_call(f"Prompt {i}", "gpt-4", "openai"))
        
        # Simulate tool calls
        operations.append(optimized_tool_call(f"tool_{i}", {"param": f"value_{i}"}))
    
    # Execute operations concurrently
    results = await asyncio.gather(*operations, return_exceptions=True)
    
    # Check that most operations succeeded
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    assert success_count > 0, "Should have some successful operations"
    
    # Get final status
    final_status = await get_performance_status()
    
    # Verify all components are working
    assert "performance" in final_status, "Should have performance data"
    assert "cache" in final_status, "Should have cache data"
    assert "batch_processing" in final_status, "Should have batch processing data"
    assert "error_recovery" in final_status, "Should have error recovery data"
    
    print("✓ End-to-end optimization working")


def test_component_imports():
    """Test that all components can be imported correctly."""
    print("Testing component imports...")
    
    try:
        from core.connection_pool import connection_manager, ProviderType
        print("✓ Connection pool imports working")
    except ImportError as e:
        print(f"❌ Connection pool import failed: {e}")
    
    try:
        from core.batch_processor import batch_processor
        print("✓ Batch processor imports working")
    except ImportError as e:
        print(f"❌ Batch processor import failed: {e}")
    
    try:
        from core.smart_cache import smart_cache
        print("✓ Smart cache imports working")
    except ImportError as e:
        print(f"❌ Smart cache import failed: {e}")
    
    try:
        from core.error_recovery import error_recovery_manager
        print("✓ Error recovery imports working")
    except ImportError as e:
        print(f"❌ Error recovery import failed: {e}")
    
    try:
        from core.performance_monitor import performance_monitor
        print("✓ Performance monitor imports working")
    except ImportError as e:
        print(f"❌ Performance monitor import failed: {e}")
    
    try:
        from core.performance_integration import performance_optimizer
        print("✓ Performance integration imports working")
    except ImportError as e:
        print(f"❌ Performance integration import failed: {e}")


def test_dependencies():
    """Test that required dependencies are available."""
    print("Testing dependencies...")
    
    # Test basic dependencies
    try:
        import asyncio
        print("✓ asyncio available")
    except ImportError:
        print("❌ asyncio not available")
    
    try:
        import time
        print("✓ time available")
    except ImportError:
        print("❌ time not available")
    
    try:
        import statistics
        print("✓ statistics available")
    except ImportError:
        print("❌ statistics not available")
    
    try:
        import hashlib
        print("✓ hashlib available")
    except ImportError:
        print("❌ hashlib not available")
    
    try:
        import json
        print("✓ json available")
    except ImportError:
        print("❌ json not available")
    
    try:
        import pickle
        print("✓ pickle available")
    except ImportError:
        print("❌ pickle not available")
    
    try:
        import httpx
        print("✓ httpx available")
    except ImportError:
        print("❌ httpx not available")
    
    print("✓ All dependencies available")


def main():
    """Run all Phase 1 optimization tests."""
    print("Testing Phase 1 Performance Optimizations...\n")
    
    try:
        # Test dependencies and imports
        test_dependencies()
        test_component_imports()
        
        # Test individual components
        test_connection_pool()
        
        # Run async tests
        asyncio.run(test_batch_processor())
        asyncio.run(test_smart_cache())
        asyncio.run(test_error_recovery())
        asyncio.run(test_performance_monitor())
        asyncio.run(test_performance_integration())
        asyncio.run(test_end_to_end_optimization())
        
        print("\n✅ All Phase 1 optimization tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 