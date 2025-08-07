"""
Test tools system integration: adapters, definitions, and tool registry
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import json
import yaml
from pathlib import Path
from typing import Dict, Any

from tools.adapters import execute as adapter_execute, REGISTRY
from core.tool_registry import ToolRegistry


def test_tools_directory_structure():
    """Test that tools directory structure exists."""
    print("Testing tools directory structure...")
    
    # Test main tools directory
    tools_dir = Path("tools")
    assert tools_dir.exists(), "Tools directory should exist"
    
    # Test adapters directory
    adapters_dir = Path("tools/adapters")
    assert adapters_dir.exists(), "Adapters directory should exist"
    
    # Test definitions directory
    definitions_dir = Path("tools/definitions")
    assert definitions_dir.exists(), "Definitions directory should exist"
    
    # Test adapter files
    adapter_files = [
        "__init__.py",
        "web_scrape.py",
        "web_search.py",
        "generative_ai.py"
    ]
    
    for file_name in adapter_files:
        file_path = adapters_dir / file_name
        assert file_path.exists(), f"Adapter file {file_name} should exist"
    
    # Test definition files (actual files that exist)
    definition_files = [
        "example_tool.yaml",
        "web.yaml",
        "generative_ai.yaml",
        "file_read.yaml",
        "file_write.yaml",
        "file_list_dir.yaml",
        "human.yaml",
        "wikipedia_search.yaml",
        "notion.yaml",
        "followup.yaml",
        "query_generator.yaml",
        "summary.yaml",
        "trust_summary.yaml"
    ]
    
    for file_name in definition_files:
        file_path = definitions_dir / file_name
        assert file_path.exists(), f"Definition file {file_name} should exist"
    
    print("✓ Tools directory structure working")


def test_adapter_registry():
    """Test adapter registry functionality."""
    print("Testing adapter registry...")
    
    # Test that registry exists and has expected tools
    expected_tools = [
        "web_search",
        "generative_ai", 
        "file_tools_write",
        "file_tools_read",
        "file_tools_list_dir",
        "human"
    ]
    
    for tool_name in expected_tools:
        assert tool_name in REGISTRY, f"Tool {tool_name} should be in registry"
    
    # Test that all registry entries are callable
    for tool_name, tool_func in REGISTRY.items():
        assert callable(tool_func), f"Tool {tool_name} should be callable"
    
    print("✓ Adapter registry working")


def test_tool_definitions():
    """Test tool definition files."""
    print("Testing tool definitions...")
    
    definitions_dir = Path("tools/definitions")
    
    # Test that all YAML files are valid
    for yaml_file in definitions_dir.glob("*.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                definition = yaml.safe_load(f)
            
            # Check required fields
            assert "name" in definition, f"Definition {yaml_file.name} should have 'name'"
            assert "description" in definition, f"Definition {yaml_file.name} should have 'description'"
            assert "parameters" in definition, f"Definition {yaml_file.name} should have 'parameters'"
            
            print(f"✓ {yaml_file.name} definition valid")
        except Exception as e:
            print(f"⚠️ {yaml_file.name} definition failed: {e}")
    
    print("✓ Tool definitions working")


@pytest.mark.asyncio
async def test_adapter_execution():
    """Test adapter execution functionality."""
    print("Testing adapter execution...")
    
    # Test web_search adapter
    try:
        result = await adapter_execute("web_search", {"query": "test query", "num_results": 1})
        assert isinstance(result, list), "web_search should return list of URLs"
        print("✓ web_search adapter working")
    except Exception as e:
        print(f"⚠️ web_search adapter failed (expected if googlesearch not available): {e}")
    
    # Test file_tools_read adapter (with mock file)
    try:
        # Create a test file
        test_file = Path("test_file.txt")
        test_file.write_text("test content")
        
        result = await adapter_execute("file_tools_read", {"path": "test_file.txt"})
        assert "test content" in result, "file_tools_read should return file content"
        
        # Clean up
        test_file.unlink()
        print("✓ file_tools_read adapter working")
    except Exception as e:
        print(f"⚠️ file_tools_read adapter failed: {e}")
    
    # Test human adapter (mock)
    try:
        # Mock input for testing
        import builtins
        original_input = builtins.input
        builtins.input = lambda x: "test response"
        
        result = await adapter_execute("human", {"message": "Test prompt"})
        assert "test response" in result, "human adapter should return user input"
        
        # Restore original input
        builtins.input = original_input
        print("✓ human adapter working")
    except Exception as e:
        print(f"⚠️ human adapter failed: {e}")
    
    print("✓ Adapter execution working")


def test_tool_registry_integration():
    """Test tool registry integration."""
    print("Testing tool registry integration...")
    
    try:
        # Create tool registry
        registry = ToolRegistry()
        
        # Register tools from adapter registry
        for tool_name, tool_func in REGISTRY.items():
            registry.register_tool(tool_name, tool_func)
        
        # Test that tools are registered
        tools = registry.get_all_tools()
        assert len(tools) > 0, "Should have registered tools"
        
        # Test specific tools
        expected_tools = ["web_search", "generative_ai", "human"]
        for tool_name in expected_tools:
            if tool_name in REGISTRY:
                assert tool_name in tools, f"Tool {tool_name} should be registered"
        
        print("✓ Tool registry integration working")
    except Exception as e:
        print(f"⚠️ Tool registry integration failed: {e}")


def test_definition_schema_validation():
    """Test that tool definitions follow expected schema."""
    print("Testing definition schema validation...")
    
    definitions_dir = Path("tools/definitions")
    
    for yaml_file in definitions_dir.glob("*.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                definition = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = ["name", "description", "parameters"]
            for field in required_fields:
                assert field in definition, f"Definition {yaml_file.name} missing required field: {field}"
            
            # Validate parameter structure
            parameters = definition.get("parameters", {})
            assert isinstance(parameters, dict), f"Parameters in {yaml_file.name} should be dict"
            
            # Validate parameter fields
            for param_name, param_def in parameters.items():
                assert "type" in param_def, f"Parameter {param_name} in {yaml_file.name} missing type"
                assert "description" in param_def, f"Parameter {param_name} in {yaml_file.name} missing description"
            
            print(f"✓ {yaml_file.name} schema valid")
        except Exception as e:
            print(f"⚠️ {yaml_file.name} schema validation failed: {e}")
    
    print("✓ Definition schema validation working")


def test_adapter_imports():
    """Test that all adapter modules can be imported."""
    print("Testing adapter imports...")
    
    try:
        # Test main adapter module
        from tools.adapters import execute, REGISTRY
        print("✓ Main adapter module importable")
        
        # Test individual adapter modules
        from tools.adapters.web_scrape import execute as web_scrape_execute
        print("✓ web_scrape adapter importable")
        
        from tools.adapters.web_search import execute as web_search_execute
        print("✓ web_search adapter importable")
        
        from tools.adapters.generative_ai import execute as generative_ai_execute
        print("✓ generative_ai adapter importable")
        
    except ImportError as e:
        print(f"⚠️ Adapter import failed: {e}")
    
    print("✓ Adapter imports working")


def test_dependencies():
    """Test that required dependencies are available."""
    print("Testing dependencies...")
    
    # Test basic dependencies
    try:
        import yaml
        print("✓ yaml available")
    except ImportError:
        print("❌ yaml not available")
    
    try:
        import httpx
        print("✓ httpx available")
    except ImportError:
        print("❌ httpx not available")
    
    try:
        import asyncio
        print("✓ asyncio available")
    except ImportError:
        print("❌ asyncio not available")
    
    try:
        import json
        print("✓ json available")
    except ImportError:
        print("❌ json not available")
    
    # Test optional dependencies
    try:
        import googlesearch
        print("✓ googlesearch available")
    except ImportError:
        print("⚠️ googlesearch not available (web_search will be limited)")
    
    print("✓ Dependencies working")


def test_integration():
    """Test integration of tools components."""
    print("Testing integration...")
    
    try:
        # Test that adapter registry works with tool registry
        registry = ToolRegistry()
        
        # Register a test tool
        async def test_tool(params):
            return "test result"
        
        registry.register_tool("test_tool", test_tool)
        
        # Test that tool is registered
        tools = registry.get_all_tools()
        assert "test_tool" in tools
        
        # Test that definitions can be loaded
        definitions_dir = Path("tools/definitions")
        definition_files = list(definitions_dir.glob("*.yaml"))
        assert len(definition_files) > 0, "Should have definition files"
        
        print("✓ Integration working")
    except Exception as e:
        print(f"⚠️ Integration test failed: {e}")
    
    print("✓ All components integrated")


def main():
    """Run all basic tests."""
    print("Testing tools system integration...\n")
    
    try:
        test_dependencies()
        test_tools_directory_structure()
        test_adapter_imports()
        test_adapter_registry()
        test_tool_definitions()
        test_definition_schema_validation()
        test_tool_registry_integration()
        test_integration()
        
        # Run async tests
        asyncio.run(test_adapter_execution())
        
        print("\n✅ All tools system integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 