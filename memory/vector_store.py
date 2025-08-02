"""
Vector Store Module for Sophie Reflex Orchestrator

Handles semantic memory storage and retrieval using vector embeddings.
Supports multiple backends including ChromaDB and SQLite-VSS.
"""

import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import sqlite_vss
    SQLITE_VSS_AVAILABLE = True
except ImportError:
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


class VectorBackend:
    """Abstract base class for vector storage backends"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry to the vector store"""
        raise NotImplementedError

    def search(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar entries by embedding"""
        raise NotImplementedError

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID"""
        raise NotImplementedError

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries in the store"""
        raise NotImplementedError

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry by ID"""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all entries from the store"""
        raise NotImplementedError


class ChromaBackend(VectorBackend):
    """ChromaDB backend for vector storage"""

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

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry to ChromaDB"""
        # Add metadata including all entry data except the embedding
        metadata = {
            'content': entry.content,
            'timestamp': entry.timestamp.isoformat(),
            'agent_id': entry.agent_id or '',
            'task_id': entry.task_id or '',
            **entry.metadata
        }

        self.collection.add(
            ids=[entry.id],
            embeddings=[entry.embedding],
            metadatas=[metadata]
        )

    def search(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar entries in ChromaDB"""
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
                    if k not in ['content', 'timestamp', 'agent_id', 'task_id']},
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                agent_id=metadata.get('agent_id') or None,
                task_id=metadata.get('task_id') or None
            )

            # Convert distance to similarity score (lower distance = higher similarity)
            similarity = 1.0 / (1.0 + distance)
            entries.append((entry, similarity))

        return entries

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID from ChromaDB"""
        results = self.collection.get(ids=[entry_id])

        if not results['ids']:
            return None

        metadata = results['metadatas'][0]
        return MemoryEntry(
            id=entry_id,
            content=metadata['content'],
            embedding=[],  # Not returned by get operation
            metadata={k: v for k, v in metadata.items()
                  if k not in ['content', 'timestamp', 'agent_id', 'task_id']},
            timestamp=datetime.fromisoformat(metadata['timestamp']),
            agent_id=metadata.get('agent_id') or None,
            task_id=metadata.get('task_id') or None
        )


    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries from ChromaDB"""
        results = self.collection.get()

        entries = []
        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i]
            entry = MemoryEntry(
                id=results['ids'][i],
                content=metadata['content'],
                embedding=[],
                metadata={k: v for k, v in metadata.items()
                      if k not in ['content', 'timestamp', 'agent_id', 'task_id']},
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                agent_id=metadata.get('agent_id') or None,
                task_id=metadata.get('task_id') or None
            )
            entries.append(entry)

        return entries

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry by ID from ChromaDB"""
        try:
            self.collection.delete(ids=[entry_id])
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all entries from ChromaDB"""
        # Delete and recreate collection
        collection_name = self.collection.name
        self.client.delete_collection(name=collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

class SQLiteBackend(VectorBackend):
    """SQLite with VSS backend for vector storage"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not SQLITE_VSS_AVAILABLE:
            raise ImportError("SQLite-VSS is not installed. Install with: pip install sqlite-vss")

        db_path = config.get('db_path', './memory.db')
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with VSS support"""
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)

        # Load VSS extension
        sqlite_vss.load(conn)

        # Create tables
        conn.execute('''
                     CREATE TABLE IF NOT EXISTS memory_entries
                     (
                         id
                         TEXT
                         PRIMARY
                         KEY,
                         content
                         TEXT
                         NOT
                         NULL,
                         timestamp
                         TEXT
                         NOT
                         NULL,
                         agent_id
                         TEXT,
                         task_id
                         TEXT,
                         metadata
                         TEXT
                     )
                     ''')

        conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors 
                USING vss0(embedding(1536))
            ''')

        conn.commit()
        conn.close()

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry to SQLite"""
        conn = sqlite3.connect(self.db_path)

        # Insert entry metadata
        conn.execute('''
            INSERT OR REPLACE INTO memory_entries 
            (id, content, timestamp, agent_id, task_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            entry.id,
            entry.content,
            entry.timestamp.isoformat(),
            entry.agent_id,
            entry.task_id,
            json.dumps(entry.metadata)
        ))

        # Insert vector
        conn.execute('''
            INSERT OR REPLACE INTO memory_vectors (rowid, embedding)
            VALUES ((SELECT rowid FROM memory_entries WHERE id = ?), ?)
        ''', (entry.id, json.dumps(entry.embedding)))

        conn.commit()
        conn.close()

    def search(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar entries in SQLite"""
        conn = sqlite3.connect(self.db_path)

        # Perform vector search
        cursor = conn.execute('''
            SELECT me.id, me.content, me.timestamp, me.agent_id, me.task_id, 
                   me.metadata, vss_distance(mv.embedding, ?) as distance
            FROM memory_entries me
            JOIN memory_vectors mv ON me.rowid = mv.rowid
            ORDER BY distance
            LIMIT ?
        ''', (json.dumps(query_embedding), limit))

        entries = []
        for row in cursor.fetchall():
            entry_id, content, timestamp, agent_id, task_id, metadata_json, distance = row

            entry = MemoryEntry(
                id=entry_id,
                content=content,
                embedding=[],  # Not returned by search
                metadata=json.loads(metadata_json) if metadata_json else {},
                timestamp=datetime.fromisoformat(timestamp),
                agent_id=agent_id or None,
                task_id=task_id or None
            )

            # Convert distance to similarity score
            similarity = 1.0 / (1.0 + distance)
            entries.append((entry, similarity))

        conn.close()
        return entries


    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID from SQLite"""
        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute('''
            SELECT id, content, timestamp, agent_id, task_id, metadata
            FROM memory_entries
            WHERE id = ?
        ''', (entry_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        entry_id, content, timestamp, agent_id, task_id, metadata_json = row

        return MemoryEntry(
            id=entry_id,
            content=content,
            embedding=[],  # Not returned by get operation
            metadata=json.loads(metadata_json) if metadata_json else {},
            timestamp=datetime.fromisoformat(timestamp),
            agent_id=agent_id or None,
            task_id=task_id or None
        )

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries from SQLite"""
        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute('''
            SELECT id, content, timestamp, agent_id, task_id, metadata
            FROM memory_entries
            ORDER BY timestamp DESC
        ''')

        entries = []
        for row in cursor.fetchall():
            entry_id, content, timestamp, agent_id, task_id, metadata_json = row

            entry = MemoryEntry(
                id=entry_id,
                content=content,
                embedding=[],
                metadata=json.loads(metadata_json) if metadata_json else {},
                timestamp=datetime.fromisoformat(timestamp),
                agent_id=agent_id or None,
                task_id=task_id or None
            )
            entries.append(entry)

        conn.close()
        return entries

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry by ID from SQLite"""
        conn = sqlite3.connect(self.db_path)


        try:
            # Delete from both tables
            conn.execute('DELETE FROM memory_vectors WHERE rowid = (SELECT rowid FROM memory_entries WHERE id = ?)',
                     (entry_id,))
            conn.execute('DELETE FROM memory_entries WHERE id = ?', (entry_id,))
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False
        finally:
            conn.close()

    def clear(self) -> None:
        """Clear all entries from SQLite"""
        conn = sqlite3.connect(self.db_path)

        conn.execute('DELETE FROM memory_vectors')
        conn.execute('DELETE FROM memory_entries')
        conn.commit()
        conn.close()


class VectorStore:
    """Main vector store interface that manages multiple backends"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend_type = config.get('backend', 'sqlite')
        self.backend = self._create_backend()

    def _create_backend(self) -> VectorBackend:
        """Create the appropriate backend based on configuration"""
        if self.backend_type == 'chroma':
            return ChromaBackend(self.config)
        elif self.backend_type == 'sqlite':
            return SQLiteBackend(self.config)
        else:
            raise ValueError(f"Unsupported backend: {self.backend_type}")

    def add_memory(self, content: str, embedding: List[float],
                   metadata: Optional[Dict[str, Any]] = None,
                   agent_id: Optional[str] = None,
                   task_id: Optional[str] = None) -> str:
        """Add a new memory entry"""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=datetime.now(),
            agent_id=agent_id,
            task_id=task_id
        )

        self.backend.add_entry(entry)
        return entry.id

    def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """Search for memories similar to the query embedding"""
        return self.backend.search(query_embedding, limit)

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID"""
        return self.backend.get_entry(memory_id)

    def get_all_memories(self) -> List[MemoryEntry]:
        """Get all memories in the store"""
        return self.backend.get_all_entries()

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        return self.backend.delete_entry(memory_id)

    def clear_all_memories(self) -> None:
        """Clear all memories from the store"""
        self.backend.clear()

    def get_memories_by_agent(self, agent_id: str) -> List[MemoryEntry]:
        """Get all memories created by a specific agent"""
        all_memories = self.get_all_memories()
        return [m for m in all_memories if m.agent_id == agent_id]

    def get_memories_by_task(self, task_id: str) -> List[MemoryEntry]:
        """Get all memories related to a specific task"""
        all_memories = self.get_all_memories()
        return [m for m in all_memories if m.task_id == task_id]

    def get_recent_memories(self, limit: int = 10) -> List[MemoryEntry]:
        """Get the most recent memories"""
        all_memories = self.get_all_memories()
        return sorted(all_memories, key=lambda x: x.timestamp, reverse=True)[:limit]