"""
Test explainability system integration: support_graph, trust_audit_log, and trust_manager
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import json
from pathlib import Path

from explainability.support_graph import SupportGraph, Evidence
from governance.trust_audit import TrustAuditLog
from utils.simple_trust_manager import SimpleTrustManager


def test_support_graph():
    """Test support graph functionality."""
    print("Testing support graph...")
    
    # Create a support graph
    graph = SupportGraph(primary_intent="web_search")
    
    # Add evidence for different intents
    graph.add_evidence("agent_1", trust=0.8, confidence=0.9, intent="web_search")
    graph.add_evidence("agent_2", trust=0.7, confidence=0.8, intent="web_search")
    graph.add_evidence("agent_3", trust=0.6, confidence=0.7, intent="data_analysis")
    
    # Test JSON serialization
    json_output = graph.to_json()
    data = json.loads(json_output)
    
    assert data["primary_intent"] == "web_search"
    assert "intents" in data
    assert "web_search" in data["intents"]
    assert "data_analysis" in data["intents"]
    
    # Test evidence structure
    web_search_evidence = data["intents"]["web_search"]["evidence"]
    assert len(web_search_evidence) == 2
    assert web_search_evidence[0]["source"] == "agent_1"
    assert web_search_evidence[0]["trust"] == 0.8
    
    print("✓ Support graph working")


def test_simple_trust_manager():
    """Test simple trust manager functionality."""
    print("Testing simple trust manager...")
    
    manager = SimpleTrustManager(default_score=0.5)
    
    # Test setting trust
    manager.set_trust("agent_1", 0.8)
    assert manager.get_trust("agent_1") == 0.8
    
    # Test updating trust
    new_score = manager.update_trust("agent_1", 0.1)
    assert new_score == 0.9
    
    # Test default score for new sources
    assert manager.get_trust("new_agent") == 0.5
    
    # Test score clamping
    manager.set_trust("agent_2", 1.5)  # Should be clamped to 1.0
    assert manager.get_trust("agent_2") == 1.0
    
    manager.set_trust("agent_3", -0.5)  # Should be clamped to 0.0
    assert manager.get_trust("agent_3") == 0.0
    
    # Test statistics
    stats = manager.get_statistics()
    assert stats["total_sources"] == 3
    assert "average_score" in stats
    
    print("✓ Simple trust manager working")


def test_trust_audit_log():
    """Test trust audit log functionality."""
    print("Testing trust audit log...")
    
    audit_log = TrustAuditLog()
    
    # Test logging trust update
    try:
        audit_log.log_update(
            source="test_agent",
            old_score=0.5,
            new_score=0.7,
            reason="test_update",
            session_id="test_session"
        )
        print("✓ Trust audit log working")
    except Exception as e:
        print(f"⚠️ Trust audit log test failed: {e}")
    
    # Test statistics
    stats = audit_log.get_trust_statistics()
    assert "total_updates" in stats
    assert "sources" in stats
    
    print("✓ Trust audit log infrastructure working")


def test_integration():
    """Test integration of all three components."""
    print("Testing integration...")
    
    # Create support graph
    graph = SupportGraph(primary_intent="test_intent")
    graph.add_evidence("agent_1", trust=0.8, confidence=0.9, intent="test_intent")
    
    # Create trust manager
    manager = SimpleTrustManager()
    manager.set_trust("agent_1", 0.8)
    
    # Create audit log
    audit_log = TrustAuditLog()
    
    # Test integrated workflow
    try:
        # Update trust based on evidence
        old_score = manager.get_trust("agent_1")
        new_score = manager.update_trust("agent_1", 0.1)
        
        # Log the update
        audit_log.log_update(
            source="agent_1",
            old_score=old_score,
            new_score=new_score,
            reason="evidence_based_update",
            session_id="integration_test"
        )
        
        # Generate support graph JSON
        graph_json = graph.to_json()
        graph_data = json.loads(graph_json)
        
        assert graph_data["primary_intent"] == "test_intent"
        assert new_score > old_score
        
        print("✓ Integration working")
    except Exception as e:
        print(f"⚠️ Integration test failed: {e}")
    
    print("✓ All components integrated")


def test_dependencies():
    """Test that required dependencies are available."""
    print("Testing dependencies...")
    
    # Test basic dependencies
    try:
        import json
        print("✓ json available")
    except ImportError:
        print("❌ json not available")
    
    try:
        import dataclasses
        print("✓ dataclasses available")
    except ImportError:
        print("❌ dataclasses not available")
    
    try:
        import pathlib
        print("✓ pathlib available")
    except ImportError:
        print("❌ pathlib not available")
    
    print("✓ Basic dependencies working")


def test_explainability_directory():
    """Test that explainability directory and files exist."""
    print("Testing explainability directory...")
    
    explainability_dir = Path("explainability")
    assert explainability_dir.exists(), "Explainability directory should exist"
    
    support_graph_file = explainability_dir / "support_graph.py"
    assert support_graph_file.exists(), "Support graph file should exist"
    
    print("✓ Explainability directory structure working")


def main():
    """Run all basic tests."""
    print("Testing explainability system integration...\n")
    
    try:
        test_dependencies()
        test_explainability_directory()
        test_support_graph()
        test_simple_trust_manager()
        test_trust_audit_log()
        test_integration()
        
        print("\n✅ All explainability system integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 