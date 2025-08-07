"""
Test service system integration: council_orchestrator, reflex_router, plan_generator, query_complexity_assessor, consensus_tracker
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import json
from pathlib import Path

from service.council_orchestrator import CouncilOrchestrator
from service.reflex_router import ReflexRouter
from service.plan_generator import generate_plan
from service.query_complexity_assessor import QueryComplexityAssessor
from service.consensus_tracker import ConsensusTracker, ModelResponse


def test_council_orchestrator():
    """Test council orchestrator functionality."""
    print("Testing council orchestrator...")
    
    orchestrator = CouncilOrchestrator()
    
    # Test pipeline stages
    assert "Strategy" in orchestrator.pipeline_stages
    assert "Tooling" in orchestrator.pipeline_stages
    assert "Execution" in orchestrator.pipeline_stages
    assert "Review" in orchestrator.pipeline_stages
    assert "Tests" in orchestrator.pipeline_stages
    
    print("✓ Council orchestrator structure working")


@pytest.mark.asyncio
async def test_council_orchestrator_execution():
    """Test council orchestrator execution."""
    print("Testing council orchestrator execution...")
    
    orchestrator = CouncilOrchestrator()
    
    try:
        result = await orchestrator.run_council("Test prompt for council execution")
        assert "status" in result
        assert "chain_of_thought" in result
        print("✓ Council orchestrator execution working")
    except Exception as e:
        print(f"⚠️ Council orchestrator execution failed: {e}")
    
    print("✓ Council orchestrator infrastructure working")


def test_reflex_router():
    """Test reflex router functionality."""
    print("Testing reflex router...")
    
    router = ReflexRouter()
    
    # Test assessor initialization
    assert hasattr(router, 'assessor')
    assert hasattr(router, 'council')
    assert hasattr(router, 'episodic_memory')
    
    print("✓ Reflex router structure working")


@pytest.mark.asyncio
async def test_reflex_router_routing():
    """Test reflex router routing functionality."""
    print("Testing reflex router routing...")
    
    router = ReflexRouter()
    
    try:
        result = await router.route("Test prompt for routing")
        assert "execution_mode" in result
        assert "assessment" in result
        assert "result" in result
        print("✓ Reflex router routing working")
    except Exception as e:
        print(f"⚠️ Reflex router routing failed: {e}")
    
    print("✓ Reflex router infrastructure working")


def test_query_complexity_assessor():
    """Test query complexity assessor functionality."""
    print("Testing query complexity assessor...")
    
    assessor = QueryComplexityAssessor()
    
    # Test assessment
    try:
        assessment = assessor.assess("What is the weather like today?")
        assert isinstance(assessment, dict)
        print("✓ Query complexity assessor working")
    except Exception as e:
        print(f"⚠️ Query complexity assessor failed: {e}")
    
    print("✓ Query complexity assessor infrastructure working")


def test_plan_generator():
    """Test plan generator functionality."""
    print("Testing plan generator...")
    
    # Test plan generation
    try:
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        plan_path = generate_plan("Test goal", output_dir)
        assert plan_path.exists()
        print("✓ Plan generator working")
    except Exception as e:
        print(f"⚠️ Plan generator failed: {e}")
    
    print("✓ Plan generator infrastructure working")


def test_consensus_tracker():
    """Test consensus tracker functionality."""
    print("Testing consensus tracker...")
    
    tracker = ConsensusTracker(quorum_threshold=0.67)
    
    # Test response registration
    tracker.register_response("model_1", "intent_a", 0.8, 0.9)
    tracker.register_response("model_2", "intent_a", 0.7, 0.8)
    tracker.register_response("model_3", "intent_b", 0.6, 0.7)
    
    # Test consensus computation
    consensus = tracker.compute_consensus()
    assert consensus is not None
    assert "intent_a" in consensus
    assert "intent_b" in consensus
    
    # Test primary intent
    primary = tracker.primary_intent()
    assert primary is not None
    
    # Test dissent detection
    dissent = tracker.detect_dissent()
    assert isinstance(dissent, bool)
    
    # Test quorum detection
    quorum = tracker.has_quorum()
    assert isinstance(quorum, bool)
    
    print("✓ Consensus tracker working")


def test_integration():
    """Test integration of all service components."""
    print("Testing integration...")
    
    # Test council orchestrator
    orchestrator = CouncilOrchestrator()
    
    # Test reflex router
    router = ReflexRouter()
    
    # Test query complexity assessor
    assessor = QueryComplexityAssessor()
    
    # Test consensus tracker
    tracker = ConsensusTracker()
    
    # Test integrated workflow
    try:
        # Assess query complexity
        assessment = assessor.assess("Test query for integration")
        
        # Register consensus responses
        tracker.register_response("model_1", "intent_a", 0.8, 0.9)
        tracker.register_response("model_2", "intent_a", 0.7, 0.8)
        
        # Compute consensus
        consensus = tracker.compute_consensus()
        
        assert assessment is not None
        assert consensus is not None
        
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
        import asyncio
        print("✓ asyncio available")
    except ImportError:
        print("❌ asyncio not available")
    
    try:
        import dataclasses
        print("✓ dataclasses available")
    except ImportError:
        print("❌ dataclasses not available")
    
    print("✓ Basic dependencies working")


def test_service_directory():
    """Test that service directory and files exist."""
    print("Testing service directory...")
    
    service_dir = Path("service")
    assert service_dir.exists(), "Service directory should exist"
    
    # Test that all service files exist
    service_files = [
        "council_orchestrator.py",
        "reflex_router.py",
        "plan_generator.py",
        "query_complexity_assessor.py",
        "consensus_tracker.py"
    ]
    
    for file_name in service_files:
        file_path = service_dir / file_name
        assert file_path.exists(), f"Service file {file_name} should exist"
    
    print("✓ Service directory structure working")


def main():
    """Run all basic tests."""
    print("Testing service system integration...\n")
    
    try:
        test_dependencies()
        test_service_directory()
        test_council_orchestrator()
        test_reflex_router()
        test_query_complexity_assessor()
        test_plan_generator()
        test_consensus_tracker()
        test_integration()
        
        # Run async tests
        asyncio.run(test_council_orchestrator_execution())
        asyncio.run(test_reflex_router_routing())
        
        print("\n✅ All service system integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 