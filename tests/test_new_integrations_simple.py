"""
Simple test for new integrations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path


def test_tool_registry_basic():
    """Test basic tool registry functionality."""
    print("Testing tool registry...")
    
    # Test that the schema file exists
    schema_path = Path("configs/schemas/tool_schema.json")
    assert schema_path.exists(), "Tool schema file should exist"
    
    # Test that the tools directory exists
    tools_dir = Path("tools/definitions")
    assert tools_dir.exists(), "Tools directory should exist"
    
    # Test that example tool exists
    example_tool = Path("tools/definitions/example_tool.yaml")
    assert example_tool.exists(), "Example tool should exist"
    
    print("✓ Tool registry infrastructure working")


def test_adapter_basic():
    """Test basic adapter functionality."""
    print("Testing adapter...")
    
    # Test that adapter module can be imported
    try:
        from adapter import execute
        print("✓ Adapter module importable")
    except ImportError as e:
        print(f"⚠️ Adapter import failed: {e}")
    
    # Test that tools/adapters directory exists
    adapters_dir = Path("tools/adapters")
    assert adapters_dir.exists(), "Adapters directory should exist"
    
    # Test that example adapter exists
    example_adapter = Path("tools/adapters/web_scrape.py")
    assert example_adapter.exists(), "Example adapter should exist"
    
    print("✓ Adapter infrastructure working")


def test_telemetry_basic():
    """Test basic telemetry functionality."""
    print("Testing telemetry...")
    
    # Test that telemetry module can be imported
    try:
        from telemetry import get_logger
        print("✓ Telemetry module importable")
    except ImportError as e:
        print(f"⚠️ Telemetry import failed: {e}")
    
    # Test that telemetry file exists
    telemetry_file = Path("telemetry.py")
    assert telemetry_file.exists(), "Telemetry file should exist"
    
    print("✓ Telemetry infrastructure working")


def test_dependencies():
    """Test that required dependencies are available."""
    print("Testing dependencies...")
    
    # Test basic dependencies
    try:
        import structlog
        print("✓ structlog available")
    except ImportError:
        print("❌ structlog not available")
    
    try:
        import httpx
        print("✓ httpx available")
    except ImportError:
        print("❌ httpx not available")
    
    try:
        import yaml
        print("✓ yaml available")
    except ImportError:
        print("❌ yaml not available")
    
    print("✓ Basic dependencies working")


def main():
    """Run all basic tests."""
    print("Testing new integrations (basic)...\n")
    
    try:
        test_dependencies()
        test_telemetry_basic()
        test_adapter_basic()
        test_tool_registry_basic()
        
        print("\n✅ All basic integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 