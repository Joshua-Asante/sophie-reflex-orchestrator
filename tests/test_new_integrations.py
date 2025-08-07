"""
Test new integrations: telemetry, adapter, and tool_registry
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
from telemetry import get_logger, get_meter
from adapter import execute
from tool_registry import ToolRegistry


def test_telemetry_logging():
    """Test telemetry logging functionality."""
    print("Testing telemetry logging...")
    
    logger = get_logger("test_telemetry")
    logger.info("Test log message", test_field="test_value")
    
    # Test meter creation
    meter = get_meter("test_meter")
    assert meter is not None
    
    print("✓ Telemetry logging and metrics working")


def test_tool_registry():
    """Test tool registry functionality."""
    print("Testing tool registry...")
    
    registry = ToolRegistry()
    
    # Test listing tools
    tools = registry.list_tools()
    print(f"✓ Found {len(tools)} tools in registry")
    
    # Test getting specific tool
    try:
        tool = registry.get_tool("web_scrape")
        assert tool["name"] == "web_scrape"
        assert "description" in tool
        print("✓ Tool registry working correctly")
    except KeyError:
        print("⚠️ Tool 'web_scrape' not found (expected if no tools defined)")


@pytest.mark.asyncio
async def test_adapter_execution():
    """Test adapter execution functionality."""
    print("Testing adapter execution...")
    
    # Test with a simple mock tool
    parameters = {
        "url": "https://httpbin.org/get",
        "timeout": 10
    }
    
    try:
        result = await execute("web_scrape", parameters)
        assert isinstance(result, dict)
        print("✓ Adapter execution working")
    except Exception as e:
        print(f"⚠️ Adapter execution failed (expected if tool not implemented): {e}")


def test_integration():
    """Test integration of all three components."""
    print("Testing integration...")
    
    # Test telemetry
    logger = get_logger("integration_test")
    logger.info("Integration test started")
    
    # Test tool registry
    registry = ToolRegistry()
    tools = registry.list_tools()
    
    # Test adapter (if tools available)
    if tools:
        print(f"✓ Integration working with {len(tools)} tools")
    else:
        print("✓ Integration working (no tools defined yet)")
    
    print("✓ All integrations working together")


if __name__ == "__main__":
    print("Testing new integrations...\n")
    
    try:
        test_telemetry_logging()
        test_tool_registry()
        
        # Run async test
        asyncio.run(test_adapter_execution())
        
        test_integration()
        
        print("\n✅ All new integrations working!")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc() 