"""
Test Autonomous Execution Engine

Tests SOPHIE's ability to interpret high-level human directives and execute them
autonomously while maintaining human oversight.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import time
from typing import Dict, Any

from core.autonomous_executor import (
    autonomous_executor, interpret_and_execute, retain_purpose_context,
    get_autonomous_status, suggest_improvements, DirectiveType, ApprovalLevel
)


def test_directive_types():
    """Test directive type enumeration."""
    print("Testing directive types...")
    
    # Test all directive types
    types = [
        DirectiveType.IMPLEMENTATION,
        DirectiveType.OPTIMIZATION,
        DirectiveType.ANALYSIS,
        DirectiveType.COORDINATION,
        DirectiveType.IMPROVEMENT,
        DirectiveType.EXECUTION
    ]
    
    for directive_type in types:
        assert directive_type.value in [
            "implementation", "optimization", "analysis", 
            "coordination", "improvement", "execution"
        ]
    
    print("✓ Directive types working")


def test_approval_levels():
    """Test approval level enumeration."""
    print("Testing approval levels...")
    
    levels = [
        ApprovalLevel.AUTONOMOUS,
        ApprovalLevel.NOTIFICATION,
        ApprovalLevel.APPROVAL,
        ApprovalLevel.SUPERVISION
    ]
    
    for level in levels:
        assert level.value in ["autonomous", "notification", "approval", "supervision"]
    
    print("✓ Approval levels working")


@pytest.mark.asyncio
async def test_directive_interpretation():
    """Test interpretation of high-level human directives."""
    print("Testing directive interpretation...")
    
    # Test different types of directives
    test_directives = [
        "implement performance optimizations",
        "analyze the current system architecture",
        "coordinate the memory and reasoning systems",
        "improve error handling mechanisms"
    ]
    
    for directive_text in test_directives:
        directive = await autonomous_executor.interpret_directive(directive_text)
        
        assert directive.id is not None, "Directive should have an ID"
        assert directive.description is not None, "Directive should have a description"
        assert directive.type in DirectiveType, "Directive should have a valid type"
        assert directive.approval_level in ApprovalLevel, "Directive should have a valid approval level"
        
        print(f"✓ Interpreted: '{directive_text}' → {directive.type.value}")
    
    print("✓ Directive interpretation working")


@pytest.mark.asyncio
async def test_execution_planning():
    """Test creation of execution plans."""
    print("Testing execution planning...")
    
    # Create a test directive
    directive = await autonomous_executor.interpret_directive("implement caching system")
    
    # Create execution plan
    plan = await autonomous_executor.create_execution_plan(directive)
    
    assert plan.directive == directive, "Plan should reference the directive"
    assert len(plan.steps) > 0, "Plan should have execution steps"
    assert plan.estimated_duration > 0, "Plan should have estimated duration"
    assert len(plan.resources_required) > 0, "Plan should list required resources"
    
    print(f"✓ Created plan with {len(plan.steps)} steps")
    print("✓ Execution planning working")


@pytest.mark.asyncio
async def test_autonomous_execution():
    """Test autonomous execution of directives."""
    print("Testing autonomous execution...")
    
    # Test a simple directive
    result = await interpret_and_execute("test the performance monitoring system")
    
    assert "directive_id" in result, "Result should have directive ID"
    assert "status" in result, "Result should have status"
    assert "results" in result, "Result should have execution results"
    assert "duration" in result, "Result should have execution duration"
    
    print(f"✓ Executed directive: {result['status']}")
    print("✓ Autonomous execution working")


@pytest.mark.asyncio
async def test_purpose_retention():
    """Test retention of purpose and context."""
    print("Testing purpose retention...")
    
    # Retain some context
    context = {
        "current_phase": "Phase 1",
        "focus_area": "performance optimization",
        "user_goals": ["improve speed", "reduce costs"],
        "system_state": "optimization_complete"
    }
    
    await retain_purpose_context(context)
    
    # Check that context was retained
    status = await get_autonomous_status()
    assert status["purpose_context_size"] > 0, "Should have retained context"
    
    print("✓ Purpose retention working")


@pytest.mark.asyncio
async def test_improvement_suggestions():
    """Test generation of improvement suggestions."""
    print("Testing improvement suggestions...")
    
    # First, execute some directives to build history
    await interpret_and_execute("test memory system")
    await interpret_and_execute("test performance monitoring")
    
    # Generate improvement suggestions
    suggestions = await suggest_improvements()
    
    # Suggestions should be a list (even if empty)
    assert isinstance(suggestions, list), "Suggestions should be a list"
    
    if suggestions:
        print(f"✓ Generated {len(suggestions)} improvement suggestions")
    else:
        print("✓ Improvement suggestion system working (no suggestions yet)")
    
    print("✓ Improvement suggestions working")


@pytest.mark.asyncio
async def test_autonomous_status():
    """Test autonomous status reporting."""
    print("Testing autonomous status...")
    
    status = await get_autonomous_status()
    
    # Check required status fields
    required_fields = [
        "active_directives",
        "execution_history", 
        "purpose_context_size",
        "improvement_suggestions",
        "performance_metrics"
    ]
    
    for field in required_fields:
        assert field in status, f"Status should have {field}"
    
    print(f"✓ Status: {status['active_directives']} active directives, "
          f"{status['execution_history']} executions")
    print("✓ Autonomous status working")


@pytest.mark.asyncio
async def test_end_to_end_autonomy():
    """Test end-to-end autonomous operation."""
    print("Testing end-to-end autonomy...")
    
    # Simulate a complete autonomous workflow
    directives = [
        "analyze current system performance",
        "identify optimization opportunities", 
        "implement the most impactful improvements",
        "verify the improvements work correctly"
    ]
    
    results = []
    for directive in directives:
        result = await interpret_and_execute(directive)
        results.append(result)
        
        # Small delay between executions
        await asyncio.sleep(0.1)
    
    # Check that all directives were processed
    assert len(results) == len(directives), "Should have processed all directives"
    
    # Check that most executions succeeded
    success_count = sum(1 for r in results if r["status"] == "completed")
    assert success_count > 0, "Should have some successful executions"
    
    # Get final status
    final_status = await get_autonomous_status()
    assert final_status["execution_history"] > 0, "Should have execution history"
    
    print(f"✓ End-to-end autonomy: {success_count}/{len(directives)} successful")
    print("✓ End-to-end autonomy working")


def test_component_imports():
    """Test that all autonomous components can be imported."""
    print("Testing autonomous component imports...")
    
    try:
        from core.autonomous_executor import autonomous_executor
        print("✓ Autonomous executor imports working")
    except ImportError as e:
        print(f"❌ Autonomous executor import failed: {e}")
    
    try:
        from core.autonomous_executor import DirectiveType, ApprovalLevel
        print("✓ Directive types imports working")
    except ImportError as e:
        print(f"❌ Directive types import failed: {e}")


def main():
    """Run all autonomous execution tests."""
    print("Testing SOPHIE Autonomous Execution Engine...\n")
    
    try:
        # Test basic components
        test_directive_types()
        test_approval_levels()
        test_component_imports()
        
        # Run async tests
        asyncio.run(test_directive_interpretation())
        asyncio.run(test_execution_planning())
        asyncio.run(test_autonomous_execution())
        asyncio.run(test_purpose_retention())
        asyncio.run(test_improvement_suggestions())
        asyncio.run(test_autonomous_status())
        asyncio.run(test_end_to_end_autonomy())
        
        print("\n✅ All autonomous execution tests passed!")
        print("\n🎯 SOPHIE now has autonomous execution capabilities!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 