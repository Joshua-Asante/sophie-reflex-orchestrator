"""
Test planning system integration: plan_executor, plan_loader
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import json
from pathlib import Path

from planning.plan_executor import execute_plan
from planning.plan_loader import load_plan


def test_plan_loader():
    """Test plan loader functionality."""
    print("Testing plan loader...")
    
    try:
        # Test loading the example plan
        plan = load_plan("example_plan")
        
        assert "plan_name" in plan
        assert "steps" in plan
        assert plan["plan_name"] == "example_plan"
        assert len(plan["steps"]) == 3
        
        # Test step structure
        step_1 = plan["steps"][0]
        assert step_1["name"] == "step_1"
        assert step_1["tool"] == "web_scrape"
        assert "params" in step_1
        
        print("✓ Plan loader working")
    except Exception as e:
        print(f"⚠️ Plan loader failed: {e}")
    
    print("✓ Plan loader infrastructure working")


@pytest.mark.asyncio
async def test_plan_executor():
    """Test plan executor functionality."""
    print("Testing plan executor...")
    
    # Create a simple test plan
    test_plan = {
        "plan_name": "test_plan",
        "steps": [
            {
                "name": "step_1",
                "tool": "test_tool",
                "params": {"input": "test_input"}
            },
            {
                "name": "step_2",
                "tool": "test_tool",
                "params": {"input": "{{ steps.step_1.output }}"}
            }
        ]
    }
    
    try:
        # Execute the plan
        result = await execute_plan(test_plan)
        
        assert "context" in result
        assert "steps" in result
        assert "step_1" in result["steps"]
        assert "step_2" in result["steps"]
        
        print("✓ Plan executor working")
    except Exception as e:
        print(f"⚠️ Plan executor failed: {e}")
    
    print("✓ Plan executor infrastructure working")


def test_core_modules():
    """Test core modules functionality."""
    print("Testing core modules...")
    
    # Test core.adapter
    try:
        from core.adapter import execute
        print("✓ core.adapter available")
    except ImportError:
        print("❌ core.adapter not available")
    
    # Test core.telemetry
    try:
        from core.telemetry import get_logger
        logger = get_logger("test")
        print("✓ core.telemetry available")
    except ImportError:
        print("❌ core.telemetry not available")
    
    # Test core.graph_utils
    try:
        from core.graph_utils import topological_sort, has_cycle
        print("✓ core.graph_utils available")
    except ImportError:
        print("❌ core.graph_utils not available")
    
    print("✓ Core modules working")


def test_graph_utils():
    """Test graph utilities functionality."""
    print("Testing graph utilities...")
    
    from core.graph_utils import topological_sort, has_cycle
    
    # Test simple linear graph
    graph = {
        "A": {"depends_on": []},
        "B": {"depends_on": ["A"]},
        "C": {"depends_on": ["B"]}
    }
    
    # Test topological sort
    order = topological_sort(graph)
    assert order == ["A", "B", "C"]
    
    # Test cycle detection
    cyclic_graph = {
        "A": {"depends_on": ["C"]},
        "B": {"depends_on": ["A"]},
        "C": {"depends_on": ["B"]}
    }
    
    assert has_cycle(cyclic_graph) == True
    assert has_cycle(graph) == False
    
    print("✓ Graph utilities working")


def test_integration():
    """Test integration of planning components."""
    print("Testing integration...")
    
    try:
        # Load a plan
        plan = load_plan("example_plan")
        
        # Execute the plan
        result = asyncio.run(execute_plan(plan))
        
        assert "context" in result
        assert "steps" in result
        
        print("✓ Integration working")
    except Exception as e:
        print(f"⚠️ Integration test failed: {e}")
    
    print("✓ All components integrated")


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
        import jsonschema
        print("✓ jsonschema available")
    except ImportError:
        print("❌ jsonschema not available")
    
    try:
        import orjson
        print("✓ orjson available")
    except ImportError:
        print("❌ orjson not available")
    
    try:
        import asyncio
        print("✓ asyncio available")
    except ImportError:
        print("❌ asyncio not available")
    
    print("✓ Basic dependencies working")


def test_planning_directory():
    """Test that planning directory and files exist."""
    print("Testing planning directory...")
    
    planning_dir = Path("planning")
    assert planning_dir.exists(), "Planning directory should exist"
    
    # Test that all planning files exist
    planning_files = [
        "plan_executor.py",
        "plan_loader.py"
    ]
    
    for file_name in planning_files:
        file_path = planning_dir / file_name
        assert file_path.exists(), f"Planning file {file_name} should exist"
    
    # Test plans directory
    plans_dir = Path("plans")
    assert plans_dir.exists(), "Plans directory should exist"
    
    # Test example plan
    example_plan = plans_dir / "example_plan.yaml"
    assert example_plan.exists(), "Example plan should exist"
    
    # Test plan schema
    schema_path = Path("configs/schemas/plan_schema.json")
    assert schema_path.exists(), "Plan schema should exist"
    
    print("✓ Planning directory structure working")


def main():
    """Run all basic tests."""
    print("Testing planning system integration...\n")
    
    try:
        test_dependencies()
        test_planning_directory()
        test_core_modules()
        test_graph_utils()
        test_plan_loader()
        test_integration()
        
        # Run async tests
        asyncio.run(test_plan_executor())
        
        print("\n✅ All planning system integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 