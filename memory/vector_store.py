"""
Vector Store Module for Sophie Reflex Orchestrator

Handles semantic memory storage and retrieval using vector embeddings.
Supports multiple backends including ChromaDB and SQLite-VSS.
Optimized for performance, scalability, and reliability.
"""

import os
import json
import numpy as np
import asyncio
import structlog
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import gzip

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Initialize logger first
logger = structlog.get_logger()

# SQLite VSS is not available for Windows, so we'll use ChromaDB as the primary backend
SQLITE_VSS_AVAILABLE = False

@dataclass
class MemoryEntry:
    """Represents a single memory entry with vector embedding"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    compressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def compress_embedding(self) -> bytes:
        """Compress embedding for storage"""
        if self.compressed:
            return self.embedding
        return gzip.compress(pickle.dumps(self.embedding))

    def decompress_embedding(self, compressed_data: bytes) -> List[float]:
        """Decompress embedding from storage"""
        return pickle.loads(gzip.decompress(compressed_data))

class VectorStoreMetrics:
    """Performance metrics for vector store operations"""
    
    def __init__(self):
        self.operation_times = {}
        self.operation_counts = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record operation metrics"""
        with self._lock:
            if operation not in self.operation_times:
                self.operation_times[operation] = []
                self.operation_counts[operation] = 0
            
            self.operation_times[operation].append(duration)
            self.operation_counts[operation] += 1
            
            if not success:
                self.errors += 1
    
    def record_cache_hit(self):
        """Record cache hit"""
        with self._lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        with self._lock:
            self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            stats = {
                'uptime_seconds': time.time() - self.start_time,
                'total_operations': sum(self.operation_counts.values()),
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                'errors': self.errors,
                'operations': {}
            }
            
            for op, times in self.operation_times.items():
                if times:
                    stats['operations'][op] = {
                        'count': self.operation_counts[op],
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times)
                    }
            
            return stats

class VectorBackend:
    """Abstract base class for vector storage backends"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = VectorStoreMetrics()
        self.cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes
        self._cache_lock = threading.Lock()

    async def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry to the vector store"""
        raise NotImplementedError

    async def add_entries_batch(self, entries: List[MemoryEntry]) -> None:
        """Add multiple memory entries in batch"""
        raise NotImplementedError

    async def search(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar entries by embedding"""
        raise NotImplementedError

    async def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID"""
        raise NotImplementedError

    async def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries in the store"""
        raise NotImplementedError

    async def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry by ID"""
        raise NotImplementedError

    async def clear(self) -> None:
        """Clear all entries from the store"""
        raise NotImplementedError

    def _get_cache_key(self, key: str) -> str:
        """Generate cache key"""
        return f"vector_store:{key}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._get_cache_key(key)
        with self._cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if time.time() - entry['timestamp'] < self.cache_ttl:
                    self.metrics.record_cache_hit()
                    return entry['value']
                else:
                    del self.cache[cache_key]
            self.metrics.record_cache_miss()
            return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache"""
        cache_key = self._get_cache_key(key)
        with self._cache_lock:
            self.cache[cache_key] = {
                'value': value,
                'timestamp': time.time()
            }

    def _clear_cache(self) -> None:
        """Clear all cache entries"""
        with self._cache_lock:
            self.cache.clear()

class ChromaBackend(VectorBackend):
    """ChromaDB backend for vector storage with async support"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")

        collection_name = config.get('collection_name', 'sophie_memory')
        persist_directory = config.get('persist_directory', './chroma_db')

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create embedding function
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))

    async def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry to ChromaDB asynchronously"""
        start_time = time.time()
        try:
            # Compress embedding for storage
            compressed_embedding = entry.compress_embedding()
            
            # Add metadata including all entry data except the embedding
            metadata = {
                'content': entry.content,
                'timestamp': entry.timestamp.isoformat(),
                'agent_id': entry.agent_id or '',
                'task_id': entry.task_id or '',
                'compressed': True,
                **entry.metadata
            }

            # Run in thread pool
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._add_entry_sync,
                entry.id,
                compressed_embedding,
                metadata
            )

            # Update cache
            self._set_cache(entry.id, entry)
            
            duration = time.time() - start_time
            self.metrics.record_operation('add_entry', duration, True)
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('add_entry', duration, False)
            logger.error("Failed to add entry to ChromaDB", error=str(e))
            raise

    def _add_entry_sync(self, entry_id: str, embedding: bytes, metadata: Dict[str, Any]):
        """Synchronous add entry for ChromaDB"""
        self.collection.add(
            ids=[entry_id],
            embeddings=[pickle.loads(gzip.decompress(embedding))],  # Decompress for ChromaDB
            metadatas=[metadata]
        )

    async def add_entries_batch(self, entries: List[MemoryEntry]) -> None:
        """Add multiple memory entries in batch"""
        start_time = time.time()
        try:
            # Prepare batch data
            ids = []
            embeddings = []
            metadatas = []
            
            for entry in entries:
                ids.append(entry.id)
                embeddings.append(entry.compress_embedding())
                metadata = {
                    'content': entry.content,
                    'timestamp': entry.timestamp.isoformat(),
                    'agent_id': entry.agent_id or '',
                    'task_id': entry.task_id or '',
                    'compressed': True,
                    **entry.metadata
                }
                metadatas.append(metadata)

            # Run in thread pool
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._add_entries_batch_sync,
                ids,
                embeddings,
                metadatas
            )

            # Update cache
            for entry in entries:
                self._set_cache(entry.id, entry)
            
            duration = time.time() - start_time
            self.metrics.record_operation('add_entries_batch', duration, True)
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('add_entries_batch', duration, False)
            logger.error("Failed to add entries batch to ChromaDB", error=str(e))
            raise

    def _add_entries_batch_sync(self, ids: List[str], embeddings: List[bytes], metadatas: List[Dict[str, Any]]):
        """Synchronous batch add for ChromaDB"""
        # Decompress embeddings for ChromaDB
        decompressed_embeddings = [pickle.loads(gzip.decompress(emb)) for emb in embeddings]
        
        self.collection.add(
            ids=ids,
            embeddings=decompressed_embeddings,
            metadatas=metadatas
        )

    async def search(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar entries in ChromaDB asynchronously"""
        start_time = time.time()
        try:
            # Run in thread pool
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._search_sync,
                query_embedding,
                limit
            )

            duration = time.time() - start_time
            self.metrics.record_operation('search', duration, True)
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('search', duration, False)
            logger.error("Failed to search ChromaDB", error=str(e))
            raise

    def _search_sync(self, query_embedding: List[float], limit: int) -> List[Tuple[MemoryEntry, float]]:
        """Synchronous search for ChromaDB"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        entries = []
        for i in range(len(results['ids'][0])):
            entry_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]

            # Reconstruct MemoryEntry
            entry = MemoryEntry(
                id=entry_id,
                content=metadata['content'],
                embedding=[],  # ChromaDB doesn't return embeddings by default
                metadata={k: v for k, v in metadata.items()
                    if k not in ['content', 'timestamp', 'agent_id', 'task_id', 'compressed']},
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                agent_id=metadata.get('agent_id') or None,
                task_id=metadata.get('task_id') or None,
                compressed=metadata.get('compressed', False)
            )

            # Convert distance to similarity score (lower distance = higher similarity)
            similarity = 1.0 / (1.0 + distance)
            entries.append((entry, similarity))

        return entries

    async def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID from ChromaDB asynchronously"""
        # Check cache first
        cached = self._get_from_cache(entry_id)
        if cached:
            return cached

        start_time = time.time()
        try:
            # Run in thread pool
            entry = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._get_entry_sync,
                entry_id
            )

            if entry:
                self._set_cache(entry_id, entry)

            duration = time.time() - start_time
            self.metrics.record_operation('get_entry', duration, True)
            return entry
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('get_entry', duration, False)
            logger.error("Failed to get entry from ChromaDB", error=str(e))
            raise

    def _get_entry_sync(self, entry_id: str) -> Optional[MemoryEntry]:
        """Synchronous get entry for ChromaDB"""
        results = self.collection.get(ids=[entry_id])

        if not results['ids']:
            return None

        metadata = results['metadatas'][0]
        return MemoryEntry(
            id=entry_id,
            content=metadata['content'],
            embedding=[],  # Not returned by get operation
            metadata={k: v for k, v in metadata.items()
                  if k not in ['content', 'timestamp', 'agent_id', 'task_id', 'compressed']},
            timestamp=datetime.fromisoformat(metadata['timestamp']),
            agent_id=metadata.get('agent_id') or None,
            task_id=metadata.get('task_id') or None,
            compressed=metadata.get('compressed', False)
        )

    async def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries from ChromaDB asynchronously"""
        start_time = time.time()
        try:
            # Run in thread pool
            entries = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._get_all_entries_sync
            )

            duration = time.time() - start_time
            self.metrics.record_operation('get_all_entries', duration, True)
            return entries
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('get_all_entries', duration, False)
            logger.error("Failed to get all entries from ChromaDB", error=str(e))
            raise

    def _get_all_entries_sync(self) -> List[MemoryEntry]:
        """Synchronous get all entries for ChromaDB"""
        results = self.collection.get()

        entries = []
        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i]
            entry = MemoryEntry(
                id=results['ids'][i],
                content=metadata['content'],
                embedding=[],
                metadata={k: v for k, v in metadata.items()
                      if k not in ['content', 'timestamp', 'agent_id', 'task_id', 'compressed']},
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                agent_id=metadata.get('agent_id') or None,
                task_id=metadata.get('task_id') or None,
                compressed=metadata.get('compressed', False)
            )
            entries.append(entry)

        return entries

    async def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry by ID from ChromaDB asynchronously"""
        start_time = time.time()
        try:
            # Run in thread pool
            success = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._delete_entry_sync,
                entry_id
            )

            if success:
                # Remove from cache
                cache_key = self._get_cache_key(entry_id)
                with self._cache_lock:
                    if cache_key in self.cache:
                        del self.cache[cache_key]

            duration = time.time() - start_time
            self.metrics.record_operation('delete_entry', duration, success)
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('delete_entry', duration, False)
            logger.error("Failed to delete entry from ChromaDB", error=str(e))
            raise

    def _delete_entry_sync(self, entry_id: str) -> bool:
        """Synchronous delete entry for ChromaDB"""
        try:
            self.collection.delete(ids=[entry_id])
            return True
        except Exception:
            return False

    async def clear(self) -> None:
        """Clear all entries from ChromaDB asynchronously"""
        start_time = time.time()
        try:
            # Run in thread pool
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._clear_sync
            )

            # Clear cache
            self._clear_cache()

            duration = time.time() - start_time
            self.metrics.record_operation('clear', duration, True)
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('clear', duration, False)
            logger.error("Failed to clear ChromaDB", error=str(e))
            raise

    def _clear_sync(self) -> None:
        """Synchronous clear for ChromaDB"""
        # Delete and recreate collection
        collection_name = self.collection.name
        self.client.delete_collection(name=collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

# SQLite backend removed - ChromaDB is the primary backend for Windows compatibility

class VectorStore:
    """Main vector store interface that manages multiple backends with async support"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend_type = config.get('backend', 'sqlite')
        self.backend = self._create_backend()
        self.metrics = self.backend.metrics

    def _create_backend(self) -> VectorBackend:
        """Create the appropriate backend based on configuration"""
        if self.backend_type == 'chroma':
            return ChromaBackend(self.config)
        else:
            # Default to ChromaDB since SQLite VSS is not available for Windows
            logger.info("Using ChromaDB as the vector store backend")
            return ChromaBackend(self.config)

    async def add_memory(self, content: str, embedding: List[float],
                   metadata: Optional[Dict[str, Any]] = None,
                   agent_id: Optional[str] = None,
                   task_id: Optional[str] = None) -> str:
        """Add a new memory entry asynchronously"""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=datetime.now(),
            agent_id=agent_id,
            task_id=task_id
        )

        await self.backend.add_entry(entry)
        return entry.id

    async def add_memories_batch(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Add multiple memory entries in batch"""
        entries = []
        for memory in memories:
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=memory['content'],
                embedding=memory['embedding'],
                metadata=memory.get('metadata', {}),
                timestamp=datetime.now(),
                agent_id=memory.get('agent_id'),
                task_id=memory.get('task_id')
            )
            entries.append(entry)

        await self.backend.add_entries_batch(entries)
        return [entry.id for entry in entries]

    async def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """Search for memories similar to the query embedding asynchronously"""
        return await self.backend.search(query_embedding, limit)

    async def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID asynchronously"""
        return await self.backend.get_entry(memory_id)

    async def get_all_memories(self) -> List[MemoryEntry]:
        """Get all memories in the store asynchronously"""
        return await self.backend.get_all_entries()

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID asynchronously"""
        return await self.backend.delete_entry(memory_id)

    async def clear_all_memories(self) -> None:
        """Clear all memories from the store asynchronously"""
        await self.backend.clear()

    async def get_memories_by_agent(self, agent_id: str) -> List[MemoryEntry]:
        """Get all memories created by a specific agent asynchronously"""
        all_memories = await self.get_all_memories()
        return [m for m in all_memories if m.agent_id == agent_id]

    async def get_memories_by_task(self, task_id: str) -> List[MemoryEntry]:
        """Get all memories related to a specific task asynchronously"""
        all_memories = await self.get_all_memories()
        return [m for m in all_memories if m.task_id == task_id]

    async def get_recent_memories(self, limit: int = 10) -> List[MemoryEntry]:
        """Get the most recent memories asynchronously"""
        all_memories = await self.get_all_memories()
        return sorted(all_memories, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_stats()

    def clear_cache(self) -> None:
        """Clear the cache"""
        self.backend._clear_cache()