"""
Basic memory component tests for Sophie Reflex Orchestrator
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.episodic_memory import EpisodicMemory
from memory.longitudinal_memory import LongitudinalMemory
from memory.norm_memory import NormMemory
from memory.semantic_memory import SemanticMemory
from memory.working_memory import WorkingMemory


def test_episodic_memory():
    """Test episodic memory functionality."""
    print("Testing episodic memory...")
    
    episodic = EpisodicMemory()
    
    # Test storing an episode
    episode_id = episodic.store_episode(
        user_input="Test user input",
        system_output="Test system output",
        models_used=["test-model"]
    )
    
    assert episode_id is not None
    print(f"✓ Stored episode with ID: {episode_id}")
    
    # Test loading episodes
    episodes = episodic.load_episodes(10)
    assert len(episodes) >= 1
    assert episodes[-1]['user_input'] == "Test user input"
    print(f"✓ Loaded {len(episodes)} episodes")
    
    return True


def test_longitudinal_memory():
    """Test longitudinal memory functionality."""
    print("Testing longitudinal memory...")
    
    longitudinal = LongitudinalMemory()
    
    # Test recording an event
    longitudinal.record_event(
        event_type="test_event",
        content={"test": "data"},
        source="test",
        tags=["test"]
    )
    
    # Test reading events
    events = longitudinal.read_all()
    assert len(events) >= 1
    assert events[-1]['event_type'] == "test_event"
    print(f"✓ Recorded and read {len(events)} events")
    
    return True


def test_norm_memory():
    """Test norm memory functionality."""
    print("Testing norm memory...")
    
    norm = NormMemory()
    
    # Test promoting an entry
    test_entry = {
        "summary": "Test summary",
        "category": "test",
        "tone": "neutral",
        "vector": [0.1, 0.2, 0.3]
    }
    
    key = norm.promote(
        entry=test_entry,
        confirmed_by="test_user"
    )
    
    assert key is not None
    print(f"✓ Promoted entry with key: {key}")
    
    # Test listing norms
    norms = norm.list_norms()
    assert len(norms) >= 1
    print(f"✓ Listed {len(norms)} norms")
    
    return True


def test_working_memory():
    """Test working memory functionality."""
    print("Testing working memory...")
    
    working = WorkingMemory()
    
    # Test adding user input
    working.add_user_input("Hello, how are you?")
    assert len(working.user_inputs) == 1
    
    # Test adding agent output
    working.add_agent_output("I'm doing well, thank you!")
    assert len(working.agent_outputs) == 1
    
    # Test getting recent history
    history = working.get_recent_history(1)
    assert len(history) == 1
    assert history[0]['user'] == "Hello, how are you?"
    assert history[0]['agent'] == "I'm doing well, thank you!"
    print(f"✓ Working memory operations successful")
    
    # Test temp variables
    working.set_temp_var("test_key", "test_value")
    value = working.get_temp_var("test_key")
    assert value == "test_value"
    print(f"✓ Temp variable operations successful")
    
    return True


def test_semantic_memory():
    """Test semantic memory functionality."""
    print("Testing semantic memory...")
    
    semantic = SemanticMemory()
    
    # Test summarizing intent
    result = semantic.summarize_intent(
        text="This is a test input for semantic processing"
    )
    
    # Check that result has expected structure
    assert 'summary' in result
    assert 'category' in result
    assert 'semantic_hash' in result
    print(f"✓ Semantic processing successful: {result['category']}")
    
    return True


def main():
    """Run all basic memory tests."""
    print("Running basic memory component tests...\n")
    
    try:
        test_episodic_memory()
        test_longitudinal_memory()
        test_norm_memory()
        test_working_memory()
        test_semantic_memory()
        
        print("\n✅ All basic memory tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    main() 