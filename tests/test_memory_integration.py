"""
Test memory integration for Sophie Reflex Orchestrator
"""

import pytest
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_manager import MemoryManager


class TestMemoryIntegration:
    """Test the integration of all memory components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'trust_tracker': {},
            'vector_store': {}
        }
        self.memory_manager = MemoryManager(self.config)
    
    def test_memory_manager_initialization(self):
        """Test that memory manager initializes all components."""
        assert self.memory_manager.episodic is not None
        assert self.memory_manager.longitudinal is not None
        assert self.memory_manager.norm is not None
        assert self.memory_manager.semantic is not None
        assert self.memory_manager.working is not None
        assert self.memory_manager.trust_tracker is not None
        assert self.memory_manager.vector_store is not None
    
    def test_working_memory_basic_operations(self):
        """Test basic working memory operations."""
        # Test adding user input
        self.memory_manager.working.add_user_input("Hello, how are you?")
        assert len(self.memory_manager.working.user_inputs) == 1
        
        # Test adding agent output
        self.memory_manager.working.add_agent_output("I'm doing well, thank you!")
        assert len(self.memory_manager.working.agent_outputs) == 1
        
        # Test getting recent history
        history = self.memory_manager.working.get_recent_history(1)
        assert len(history) == 1
        assert history[0]['user'] == "Hello, how are you?"
        assert history[0]['agent'] == "I'm doing well, thank you!"
    
    def test_episodic_memory_storage(self):
        """Test episodic memory storage."""
        episode_id = self.memory_manager.episodic.store_episode(
            user_input="Test input",
            system_output="Test output",
            models_used=["test-model"]
        )
        assert episode_id is not None
        
        # Test loading episodes
        episodes = self.memory_manager.episodic.load_episodes(10)
        assert len(episodes) >= 1
        assert episodes[-1]['user_input'] == "Test input"
    
    def test_longitudinal_memory_recording(self):
        """Test longitudinal memory recording."""
        self.memory_manager.longitudinal.record_event(
            event_type="test_event",
            content={"test": "data"},
            source="test",
            tags=["test"]
        )
        
        # Test reading events
        events = self.memory_manager.longitudinal.read_all()
        assert len(events) >= 1
        assert events[-1]['event_type'] == "test_event"
    
    def test_norm_memory_operations(self):
        """Test norm memory operations."""
        test_entry = {
            "summary": "Test summary",
            "category": "test",
            "tone": "neutral",
            "vector": [0.1, 0.2, 0.3]
        }
        
        # Test promoting to norm
        key = self.memory_manager.norm.promote(
            entry=test_entry,
            confirmed_by="test_user"
        )
        assert key is not None
        
        # Test listing norms
        norms = self.memory_manager.norm.list_norms()
        assert len(norms) >= 1
    
    def test_semantic_memory_processing(self):
        """Test semantic memory processing."""
        result = self.memory_manager.semantic.summarize_intent(
            text="This is a test input for semantic processing"
        )
        
        # Check that result has expected structure
        assert 'summary' in result
        assert 'category' in result
        assert 'semantic_hash' in result
    
    def test_memory_stats(self):
        """Test memory statistics collection."""
        stats = self.memory_manager.get_memory_stats()
        
        # Check that stats contain expected keys
        expected_keys = ['episodic_count', 'longitudinal_count', 'norm_count', 'working_size', 'vector_count', 'trust_agents']
        for key in expected_keys:
            assert key in stats
    
    @pytest.mark.asyncio
    async def test_memory_stats_async(self):
        """Test async memory statistics collection."""
        stats = await self.memory_manager.get_memory_stats_async()
        
        # Check that stats contain expected keys
        expected_keys = ['episodic_count', 'longitudinal_count', 'norm_count', 'working_size', 'vector_count', 'trust_agents']
        for key in expected_keys:
            assert key in stats
    
    @pytest.mark.asyncio
    async def test_async_memory_operations(self):
        """Test async memory operations."""
        # Test recording interaction
        result = await self.memory_manager.record_interaction(
            user_input="Async test input",
            system_output="Async test output",
            agent_id="test_agent",
            task_id="test_task"
        )
        
        # Check that result contains expected keys
        expected_keys = ['episode_id', 'vector_id', 'semantic_hash']
        for key in expected_keys:
            assert key in result
    
    def test_working_memory_temp_vars(self):
        """Test working memory temporary variables."""
        # Test setting temp var
        self.memory_manager.working.set_temp_var("test_key", "test_value")
        
        # Test getting temp var
        value = self.memory_manager.working.get_temp_var("test_key")
        assert value == "test_value"
        
        # Test deleting temp var
        self.memory_manager.working.delete_temp_var("test_key")
        value = self.memory_manager.working.get_temp_var("test_key")
        assert value is None


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestMemoryIntegration()
    test_instance.setup_method()
    
    print("Testing memory integration...")
    
    test_instance.test_memory_manager_initialization()
    print("✓ Memory manager initialization")
    
    test_instance.test_working_memory_basic_operations()
    print("✓ Working memory operations")
    
    test_instance.test_episodic_memory_storage()
    print("✓ Episodic memory storage")
    
    test_instance.test_longitudinal_memory_recording()
    print("✓ Longitudinal memory recording")
    
    test_instance.test_norm_memory_operations()
    print("✓ Norm memory operations")
    
    test_instance.test_semantic_memory_processing()
    print("✓ Semantic memory processing")
    
    test_instance.test_memory_stats()
    print("✓ Memory statistics")
    
    # Run async test
    import asyncio
    asyncio.run(test_instance.test_memory_stats_async())
    print("✓ Async memory statistics")
    
    test_instance.test_working_memory_temp_vars()
    print("✓ Working memory temp variables")
    
    print("\nAll memory integration tests passed!") 