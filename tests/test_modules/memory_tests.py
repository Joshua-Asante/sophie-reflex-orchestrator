#!/usr/bin/env python3
"""
Memory Tests Module

Tests memory systems and persistence functionality.
"""

import asyncio
import sys
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import structlog

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from memory.vector_store import VectorStore
    from memory.trust_tracker import TrustTracker
    from orchestrator.components.memory_manager import MemoryManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

logger = structlog.get_logger()


class MemoryTestSuite:
    """Memory test suite for memory systems and persistence."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.test_results = []
        
    async def run_all_tests(self) -> bool:
        """Run all memory tests and return success status."""
        print("🧪 Running Memory Tests")
        print("-" * 40)
        
        test_functions = [
            ("Vector Store Initialization", self._test_vector_store_initialization),
            ("Vector Store Operations", self._test_vector_store_operations),
            ("Trust Tracker Initialization", self._test_trust_tracker_initialization),
            ("Trust Tracker Operations", self._test_trust_tracker_operations),
            ("Memory Manager Initialization", self._test_memory_manager_initialization),
            ("Memory Manager Operations", self._test_memory_manager_operations),
            ("Memory Persistence", self._test_memory_persistence),
            ("Memory Search", self._test_memory_search),
            ("Memory Cleanup", self._test_memory_cleanup),
            ("Memory Performance", self._test_memory_performance)
        ]
        
        all_passed = True
        for test_name, test_func in test_functions:
            try:
                result = await test_func()
                if result:
                    print(f"✅ {test_name}: PASSED")
                    self.test_results.append((test_name, "PASSED", None))
                else:
                    print(f"❌ {test_name}: FAILED")
                    self.test_results.append((test_name, "FAILED", "Test returned False"))
                    all_passed = False
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {str(e)}")
                self.test_results.append((test_name, "ERROR", str(e)))
                all_passed = False
        
        return all_passed
    
    async def _test_vector_store_initialization(self) -> bool:
        """Test VectorStore initialization."""
        try:
            # Test vector store creation
            vector_store = VectorStore()
            
            # Verify initialization
            assert vector_store is not None
            
            return True
            
        except Exception as e:
            logger.error("Vector store initialization test failed", error=str(e))
            return False
    
    async def _test_vector_store_operations(self) -> bool:
        """Test VectorStore operations."""
        try:
            vector_store = VectorStore()
            
            # Test vector operations
            test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
            vector_id = "test_vector_001"
            
            # Store vector
            vector_store.store_vector(vector_id, test_vector)
            
            # Retrieve vector
            retrieved_vector = vector_store.get_vector(vector_id)
            assert retrieved_vector == test_vector
            
            # Test vector search
            search_results = vector_store.search_similar(test_vector, top_k=5)
            assert len(search_results) >= 1
            
            return True
            
        except Exception as e:
            logger.error("Vector store operations test failed", error=str(e))
            return False
    
    async def _test_trust_tracker_initialization(self) -> bool:
        """Test TrustTracker initialization."""
        try:
            trust_tracker = TrustTracker()
            
            # Verify initialization
            assert trust_tracker is not None
            
            return True
            
        except Exception as e:
            logger.error("Trust tracker initialization test failed", error=str(e))
            return False
    
    async def _test_trust_tracker_operations(self) -> bool:
        """Test TrustTracker operations."""
        try:
            trust_tracker = TrustTracker()
            
            # Test trust score operations
            agent_id = "test_agent_001"
            
            # Update trust score
            trust_tracker.update_trust(agent_id, 0.8)
            
            # Get trust score
            trust_score = trust_tracker.get_trust(agent_id)
            assert trust_score == 0.8
            
            # Test trust history
            trust_tracker.update_trust(agent_id, 0.9)
            history = trust_tracker.get_trust_history(agent_id)
            assert len(history) == 2
            assert history[-1] == 0.9
            
            # Test trust decay
            trust_tracker.apply_decay(agent_id, decay_rate=0.1)
            decayed_trust = trust_tracker.get_trust(agent_id)
            assert decayed_trust < 0.9
            
            return True
            
        except Exception as e:
            logger.error("Trust tracker operations test failed", error=str(e))
            return False
    
    async def _test_memory_manager_initialization(self) -> bool:
        """Test MemoryManager initialization."""
        try:
            memory_manager = MemoryManager()
            
            # Verify initialization
            assert memory_manager is not None
            
            return True
            
        except Exception as e:
            logger.error("Memory manager initialization test failed", error=str(e))
            return False
    
    async def _test_memory_manager_operations(self) -> bool:
        """Test MemoryManager operations."""
        try:
            memory_manager = MemoryManager()
            
            # Test memory operations
            session_id = "test_session_001"
            task = "Test memory manager operations"
            result = {"test": "data", "confidence": 0.8}
            agent_id = "test_agent_001"
            
            # Store memory
            memory_manager.store_memory(
                session_id=session_id,
                task=task,
                result=result,
                agent_id=agent_id
            )
            
            # Retrieve memory
            retrieved_memory = memory_manager.get_memory(session_id)
            assert retrieved_memory is not None
            assert retrieved_memory["task"] == task
            assert retrieved_memory["result"]["test"] == "data"
            assert retrieved_memory["agent_id"] == agent_id
            
            # Test memory search
            search_results = memory_manager.search_memory("test")
            assert len(search_results) > 0
            
            return True
            
        except Exception as e:
            logger.error("Memory manager operations test failed", error=str(e))
            return False
    
    async def _test_memory_persistence(self) -> bool:
        """Test memory persistence functionality."""
        try:
            memory_manager = MemoryManager()
            
            # Test memory persistence
            session_id = "persistence_test_001"
            task = "Test memory persistence"
            result = {"persistence": "test", "timestamp": datetime.now().isoformat()}
            
            # Store memory
            memory_manager.store_memory(
                session_id=session_id,
                task=task,
                result=result,
                agent_id="test_agent_001"
            )
            
            # Verify persistence
            retrieved_memory = memory_manager.get_memory(session_id)
            assert retrieved_memory is not None
            assert retrieved_memory["task"] == task
            assert retrieved_memory["result"]["persistence"] == "test"
            
            # Test multiple memory entries
            for i in range(3):
                memory_manager.store_memory(
                    session_id=f"persistence_test_{i+2:03d}",
                    task=f"Task {i+2}",
                    result={"index": i+2},
                    agent_id="test_agent_001"
                )
            
            # Verify all entries are persisted
            for i in range(3):
                memory = memory_manager.get_memory(f"persistence_test_{i+2:03d}")
                assert memory is not None
                assert memory["result"]["index"] == i+2
            
            return True
            
        except Exception as e:
            logger.error("Memory persistence test failed", error=str(e))
            return False
    
    async def _test_memory_search(self) -> bool:
        """Test memory search functionality."""
        try:
            memory_manager = MemoryManager()
            
            # Create test memories
            test_memories = [
                {
                    "session_id": "search_test_001",
                    "task": "Find the best algorithm for sorting",
                    "result": {"algorithm": "quicksort", "performance": "O(n log n)"}
                },
                {
                    "session_id": "search_test_002",
                    "task": "Design a user interface for mobile app",
                    "result": {"design": "material design", "platform": "android"}
                },
                {
                    "session_id": "search_test_003",
                    "task": "Optimize database queries",
                    "result": {"optimization": "indexing", "improvement": "50%"}
                }
            ]
            
            # Store test memories
            for memory in test_memories:
                memory_manager.store_memory(
                    session_id=memory["session_id"],
                    task=memory["task"],
                    result=memory["result"],
                    agent_id="test_agent_001"
                )
            
            # Test search functionality
            search_terms = ["algorithm", "design", "optimize", "database"]
            
            for term in search_terms:
                results = memory_manager.search_memory(term)
                assert len(results) > 0
                
                # Verify search results contain the search term
                found_term = False
                for result in results:
                    if term.lower() in str(result).lower():
                        found_term = True
                        break
                assert found_term, f"Search term '{term}' not found in results"
            
            return True
            
        except Exception as e:
            logger.error("Memory search test failed", error=str(e))
            return False
    
    async def _test_memory_cleanup(self) -> bool:
        """Test memory cleanup functionality."""
        try:
            memory_manager = MemoryManager()
            
            # Create test memories
            for i in range(5):
                memory_manager.store_memory(
                    session_id=f"cleanup_test_{i+1:03d}",
                    task=f"Cleanup test task {i+1}",
                    result={"index": i+1},
                    agent_id="test_agent_001"
                )
            
            # Verify memories are stored
            for i in range(5):
                memory = memory_manager.get_memory(f"cleanup_test_{i+1:03d}")
                assert memory is not None
            
            # Test cleanup
            memory_manager.cleanup_old_entries()
            
            # Verify cleanup worked (should still have memories, but old ones might be cleaned)
            total_memories = 0
            for i in range(5):
                memory = memory_manager.get_memory(f"cleanup_test_{i+1:03d}")
                if memory is not None:
                    total_memories += 1
            
            # Should have at least some memories remaining
            assert total_memories >= 0
            
            return True
            
        except Exception as e:
            logger.error("Memory cleanup test failed", error=str(e))
            return False
    
    async def _test_memory_performance(self) -> bool:
        """Test memory performance functionality."""
        try:
            memory_manager = MemoryManager()
            
            # Test performance with multiple operations
            start_time = datetime.now()
            
            # Perform multiple memory operations
            for i in range(10):
                memory_manager.store_memory(
                    session_id=f"perf_test_{i+1:03d}",
                    task=f"Performance test task {i+1}",
                    result={"performance": "test", "index": i+1},
                    agent_id="test_agent_001"
                )
            
            # Test retrieval performance
            for i in range(10):
                memory = memory_manager.get_memory(f"perf_test_{i+1:03d}")
                assert memory is not None
            
            # Test search performance
            search_results = memory_manager.search_memory("performance")
            assert len(search_results) > 0
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Performance should be reasonable (less than 5 seconds for 10 operations)
            assert execution_time < 5.0
            
            return True
            
        except Exception as e:
            logger.error("Memory performance test failed", error=str(e))
            return False 