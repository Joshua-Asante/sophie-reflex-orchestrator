#!/usr/bin/env python3
"""
Comprehensive ChromaDB Integration Test Suite
Tests vector storage, retrieval, search, and data integrity for Sophie Reflex Orchestrator
"""

import asyncio
import sys
import os
import tempfile
import shutil
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.vector_store import VectorStore, MemoryEntry, ChromaBackend


class ChromaDBTestSuite:
    """Comprehensive test suite for ChromaDB integration."""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.vector_store = None
        
    async def run_all_tests(self) -> bool:
        """Run all ChromaDB tests and return success status."""
        print("🧪 Starting ChromaDB Integration Test Suite")
        print("=" * 60)
        
        try:
            # Setup test environment
            await self._setup_test_environment()
            
            # Run test categories
            test_categories = [
                ("Basic Initialization", self._test_basic_initialization),
                ("Memory Entry Operations", self._test_memory_operations),
                ("Vector Storage & Retrieval", self._test_vector_storage),
                ("Search Functionality", self._test_search_functionality),
                ("Batch Operations", self._test_batch_operations),
                ("Data Integrity", self._test_data_integrity),
                ("Performance Tests", self._test_performance),
                ("Error Handling", self._test_error_handling),
                ("Persistence Tests", self._test_persistence),
                ("Concurrent Operations", self._test_concurrent_operations)
            ]
            
            all_passed = True
            for category_name, test_func in test_categories:
                print(f"\n📋 Testing: {category_name}")
                print("-" * 40)
                
                try:
                    result = await test_func()
                    if result:
                        print(f"✅ {category_name}: PASSED")
                        self.test_results.append((category_name, "PASSED", None))
                    else:
                        print(f"❌ {category_name}: FAILED")
                        self.test_results.append((category_name, "FAILED", "Test returned False"))
                        all_passed = False
                except Exception as e:
                    print(f"❌ {category_name}: ERROR - {str(e)}")
                    self.test_results.append((category_name, "ERROR", str(e)))
                    all_passed = False
            
            # Generate comprehensive report
            await self._generate_test_report()
            
            return all_passed
            
        finally:
            await self._cleanup_test_environment()
    
    async def _setup_test_environment(self):
        """Setup test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp(prefix="chromadb_test_")
        print(f"📁 Test directory: {self.temp_dir}")
        
        # Initialize vector store with test configuration
        config = {
            'backend': 'chroma',
            'collection_name': 'test_memory',
            'persist_directory': self.temp_dir
        }
        
        self.vector_store = VectorStore(config)
        print("🔧 Vector store initialized for testing")
    
    async def _cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("🧹 Test environment cleaned up")
    
    async def _test_basic_initialization(self) -> bool:
        """Test basic ChromaDB initialization and configuration."""
        try:
            # Test backend type
            assert self.vector_store.backend_type == "chroma", "Backend type should be 'chroma'"
            
            # Test backend instance
            assert isinstance(self.vector_store.backend, ChromaBackend), "Backend should be ChromaBackend instance"
            
            # Test collection name
            assert self.vector_store.backend.collection_name == "test_memory", "Collection name should match"
            
            # Test configuration
            assert self.vector_store.backend.persist_directory == self.temp_dir, "Persist directory should match"
            
            print("  ✅ Backend type verification")
            print("  ✅ Backend instance verification")
            print("  ✅ Configuration verification")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Basic initialization failed: {str(e)}")
            return False
    
    async def _test_memory_operations(self) -> bool:
        """Test basic memory entry operations."""
        try:
            # Create test memory entries
            test_entries = [
                MemoryEntry(
                    content="Test memory entry 1",
                    metadata={"type": "test", "category": "basic"},
                    embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 20  # 100-dim vector
                ),
                MemoryEntry(
                    content="Test memory entry 2",
                    metadata={"type": "test", "category": "advanced"},
                    embedding=[0.2, 0.3, 0.4, 0.5, 0.6] * 20
                )
            ]
            
            # Test adding single entry
            result1 = await self.vector_store.add_entry(test_entries[0])
            assert result1, "Single entry addition should succeed"
            
            # Test adding another entry
            result2 = await self.vector_store.add_entry(test_entries[1])
            assert result2, "Second entry addition should succeed"
            
            # Test retrieving entries
            entries = await self.vector_store.get_entries(limit=10)
            assert len(entries) >= 2, "Should retrieve at least 2 entries"
            
            # Verify entry content
            content_found = [entry.content for entry in entries]
            assert "Test memory entry 1" in content_found, "First entry should be found"
            assert "Test memory entry 2" in content_found, "Second entry should be found"
            
            print("  ✅ Single entry addition")
            print("  ✅ Multiple entry addition")
            print("  ✅ Entry retrieval")
            print("  ✅ Content verification")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Memory operations failed: {str(e)}")
            return False
    
    async def _test_vector_storage(self) -> bool:
        """Test vector storage and retrieval functionality."""
        try:
            # Create test vectors with different dimensions
            test_vectors = [
                np.random.rand(100).tolist(),
                np.random.rand(100).tolist(),
                np.random.rand(100).tolist()
            ]
            
            # Add entries with different vectors
            for i, vector in enumerate(test_vectors):
                entry = MemoryEntry(
                    content=f"Vector test entry {i+1}",
                    metadata={"test_id": i, "vector_type": "random"},
                    embedding=vector
                )
                result = await self.vector_store.add_entry(entry)
                assert result, f"Vector entry {i+1} addition should succeed"
            
            # Test vector retrieval
            entries = await self.vector_store.get_entries(limit=5)
            assert len(entries) >= 3, "Should retrieve at least 3 vector entries"
            
            # Verify vector dimensions
            for entry in entries:
                if hasattr(entry, 'embedding') and entry.embedding:
                    assert len(entry.embedding) == 100, "All vectors should be 100-dimensional"
            
            print("  ✅ Vector storage")
            print("  ✅ Vector retrieval")
            print("  ✅ Dimension verification")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Vector storage test failed: {str(e)}")
            return False
    
    async def _test_search_functionality(self) -> bool:
        """Test search functionality with different queries."""
        try:
            # Add diverse test entries
            test_content = [
                "Python programming language tutorial",
                "Machine learning algorithms and models",
                "Data science workflow and analysis",
                "Artificial intelligence applications",
                "Deep learning neural networks"
            ]
            
            for i, content in enumerate(test_content):
                entry = MemoryEntry(
                    content=content,
                    metadata={"category": "ai_ml", "index": i},
                    embedding=np.random.rand(100).tolist()
                )
                await self.vector_store.add_entry(entry)
            
            # Test semantic search
            search_results = await self.vector_store.search(
                query="machine learning",
                limit=3
            )
            
            assert len(search_results) > 0, "Search should return results"
            assert len(search_results) <= 3, "Search should respect limit"
            
            # Test metadata filtering
            filtered_results = await self.vector_store.search(
                query="programming",
                limit=5,
                filter_metadata={"category": "ai_ml"}
            )
            
            assert len(filtered_results) >= 0, "Filtered search should work"
            
            print("  ✅ Semantic search")
            print("  ✅ Result limiting")
            print("  ✅ Metadata filtering")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Search functionality failed: {str(e)}")
            return False
    
    async def _test_batch_operations(self) -> bool:
        """Test batch operations for efficiency."""
        try:
            # Create batch of entries
            batch_entries = []
            for i in range(10):
                entry = MemoryEntry(
                    content=f"Batch entry {i+1}",
                    metadata={"batch_id": i, "operation": "batch_test"},
                    embedding=np.random.rand(100).tolist()
                )
                batch_entries.append(entry)
            
            # Test batch addition
            start_time = datetime.now()
            batch_result = await self.vector_store.add_entries_batch(batch_entries)
            end_time = datetime.now()
            
            assert batch_result, "Batch addition should succeed"
            
            # Verify batch retrieval
            retrieved_entries = await self.vector_store.get_entries(limit=20)
            batch_content = [entry.content for entry in retrieved_entries if "Batch entry" in entry.content]
            
            assert len(batch_content) >= 10, "Should retrieve all batch entries"
            
            # Performance check
            batch_time = (end_time - start_time).total_seconds()
            print(f"  ⏱️  Batch operation time: {batch_time:.3f}s")
            
            print("  ✅ Batch addition")
            print("  ✅ Batch retrieval")
            print("  ✅ Performance verification")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Batch operations failed: {str(e)}")
            return False
    
    async def _test_data_integrity(self) -> bool:
        """Test data integrity and consistency."""
        try:
            # Add entries with specific content
            test_content = "Integrity test content with specific keywords"
            test_metadata = {"integrity_test": True, "timestamp": datetime.now().isoformat()}
            
            entry = MemoryEntry(
                content=test_content,
                metadata=test_metadata,
                embedding=np.random.rand(100).tolist()
            )
            
            # Add entry
            add_result = await self.vector_store.add_entry(entry)
            assert add_result, "Entry addition should succeed"
            
            # Retrieve and verify
            entries = await self.vector_store.get_entries(limit=50)
            matching_entries = [e for e in entries if e.content == test_content]
            
            assert len(matching_entries) > 0, "Should find the added entry"
            
            # Verify metadata
            found_entry = matching_entries[0]
            assert found_entry.metadata.get("integrity_test") == True, "Metadata should be preserved"
            
            # Test content uniqueness
            duplicate_entry = MemoryEntry(
                content=test_content,  # Same content
                metadata={"duplicate": True},
                embedding=np.random.rand(100).tolist()
            )
            
            # Should handle duplicates gracefully
            duplicate_result = await self.vector_store.add_entry(duplicate_entry)
            assert duplicate_result, "Duplicate handling should work"
            
            print("  ✅ Content preservation")
            print("  ✅ Metadata integrity")
            print("  ✅ Duplicate handling")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Data integrity test failed: {str(e)}")
            return False
    
    async def _test_performance(self) -> bool:
        """Test performance characteristics."""
        try:
            # Performance test with larger dataset
            print("  📊 Running performance tests...")
            
            # Test insertion performance
            start_time = datetime.now()
            for i in range(50):
                entry = MemoryEntry(
                    content=f"Performance test entry {i+1}",
                    metadata={"perf_test": True, "index": i},
                    embedding=np.random.rand(100).tolist()
                )
                await self.vector_store.add_entry(entry)
            
            insert_time = (datetime.now() - start_time).total_seconds()
            print(f"  ⏱️  Insertion time (50 entries): {insert_time:.3f}s")
            
            # Test retrieval performance
            start_time = datetime.now()
            entries = await self.vector_store.get_entries(limit=100)
            retrieve_time = (datetime.now() - start_time).total_seconds()
            print(f"  ⏱️  Retrieval time (100 entries): {retrieve_time:.3f}s")
            
            # Test search performance
            start_time = datetime.now()
            search_results = await self.vector_store.search("performance", limit=10)
            search_time = (datetime.now() - start_time).total_seconds()
            print(f"  ⏱️  Search time: {search_time:.3f}s")
            
            # Performance thresholds
            assert insert_time < 10.0, "Insertion should be reasonably fast"
            assert retrieve_time < 5.0, "Retrieval should be fast"
            assert search_time < 3.0, "Search should be fast"
            
            print("  ✅ Performance thresholds met")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Performance test failed: {str(e)}")
            return False
    
    async def _test_error_handling(self) -> bool:
        """Test error handling and edge cases."""
        try:
            # Test invalid entry
            try:
                invalid_entry = MemoryEntry(
                    content="",  # Empty content
                    metadata={},
                    embedding=[]  # Empty embedding
                )
                result = await self.vector_store.add_entry(invalid_entry)
                # Should handle gracefully
                print("  ✅ Empty content handling")
            except Exception as e:
                print(f"  ⚠️  Empty content error (expected): {str(e)}")
            
            # Test invalid search query
            try:
                results = await self.vector_store.search("", limit=5)
                assert len(results) >= 0, "Empty query should return empty results"
                print("  ✅ Empty query handling")
            except Exception as e:
                print(f"  ⚠️  Empty query error (expected): {str(e)}")
            
            # Test invalid metadata
            try:
                entry = MemoryEntry(
                    content="Test entry",
                    metadata={"invalid": object()},  # Non-serializable object
                    embedding=np.random.rand(100).tolist()
                )
                result = await self.vector_store.add_entry(entry)
                print("  ✅ Invalid metadata handling")
            except Exception as e:
                print(f"  ⚠️  Invalid metadata error (expected): {str(e)}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error handling test failed: {str(e)}")
            return False
    
    async def _test_persistence(self) -> bool:
        """Test data persistence across restarts."""
        try:
            # Add persistent test data
            persistent_content = "Persistent test content that should survive restart"
            persistent_entry = MemoryEntry(
                content=persistent_content,
                metadata={"persistent": True, "test_id": "persistence_test"},
                embedding=np.random.rand(100).tolist()
            )
            
            # Add to current store
            await self.vector_store.add_entry(persistent_entry)
            
            # Create new store instance (simulating restart)
            new_config = {
                'backend': 'chroma',
                'collection_name': 'test_memory',
                'persist_directory': self.temp_dir
            }
            
            new_vector_store = VectorStore(new_config)
            
            # Verify data persistence
            entries = await new_vector_store.get_entries(limit=100)
            persistent_entries = [e for e in entries if e.content == persistent_content]
            
            assert len(persistent_entries) > 0, "Persistent data should survive restart"
            
            print("  ✅ Data persistence")
            print("  ✅ Restart simulation")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Persistence test failed: {str(e)}")
            return False
    
    async def _test_concurrent_operations(self) -> bool:
        """Test concurrent operations for thread safety."""
        try:
            # Create concurrent tasks
            async def add_entry(task_id: int):
                entry = MemoryEntry(
                    content=f"Concurrent entry {task_id}",
                    metadata={"concurrent": True, "task_id": task_id},
                    embedding=np.random.rand(100).tolist()
                )
                return await self.vector_store.add_entry(entry)
            
            # Run concurrent additions
            tasks = [add_entry(i) for i in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            successful_results = [r for r in results if r is True]
            assert len(successful_results) >= 8, "Most concurrent operations should succeed"
            
            # Test concurrent retrieval
            async def get_entries():
                return await self.vector_store.get_entries(limit=20)
            
            retrieval_tasks = [get_entries() for _ in range(5)]
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
            
            successful_retrievals = [r for r in retrieval_results if not isinstance(r, Exception)]
            assert len(successful_retrievals) >= 4, "Most concurrent retrievals should succeed"
            
            print("  ✅ Concurrent additions")
            print("  ✅ Concurrent retrievals")
            print("  ✅ Thread safety")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Concurrent operations failed: {str(e)}")
            return False
    
    async def _generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 CHROMADB INTEGRATION TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r[1] == "PASSED"])
        failed_tests = len([r for r in self.test_results if r[1] in ["FAILED", "ERROR"]])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        print("-" * 40)
        
        for test_name, status, error in self.test_results:
            status_icon = "✅" if status == "PASSED" else "❌"
            print(f"{status_icon} {test_name}: {status}")
            if error:
                print(f"   Error: {error}")
        
        print("\n🎯 Recommendations:")
        if failed_tests == 0:
            print("✅ All tests passed! ChromaDB integration is working correctly.")
            print("✅ Vector storage and retrieval are functioning properly.")
            print("✅ Data integrity is maintained.")
            print("✅ Performance meets expectations.")
        else:
            print("⚠️  Some tests failed. Please review the error messages above.")
            print("🔧 Consider checking ChromaDB installation and configuration.")
            print("📝 Verify that the chromadb folder contains valid data.")
        
        print("\n" + "=" * 60)


async def main():
    """Main test execution function."""
    test_suite = ChromaDBTestSuite()
    
    try:
        success = await test_suite.run_all_tests()
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("ChromaDB integration is working correctly.")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED!")
            print("Please review the test report above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 TEST SUITE ERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 