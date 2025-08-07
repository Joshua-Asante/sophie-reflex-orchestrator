"""
Trust Tracker Module for Sophie Reflex Orchestrator

Manages dynamic trust scoring and performance tracking for agents.
Tracks historical performance and adjusts trust scores based on outcomes.
Optimized for performance, scalability, and reliability.
"""

import json
import sqlite3
import asyncio
import structlog
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
from contextlib import asynccontextmanager
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

logger = structlog.get_logger()

class TrustEventType(Enum):
    """Types of trust events that can affect agent trust scores"""
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    HUMAN_APPROVAL = "human_approval"
    HUMAN_REJECTION = "human_rejection"
    POLICY_VIOLATION = "policy_violation"
    QUALITY_HIGH = "quality_high"
    QUALITY_LOW = "quality_low"
    TIMEOUT = "timeout"
    ERROR = "error"
    IMPROVEMENT = "improvement"
    COLLABORATION_SUCCESS = "collaboration_success"


@dataclass
class TrustEvent:
    """Represents a single trust event"""
    id: str
    agent_id: str
    event_type: TrustEventType
    score_change: float
    description: str
    timestamp: datetime
    task_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrustEvent':
        """Create from dictionary"""
        data['event_type'] = TrustEventType(data['event_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class AgentTrustProfile:
    """Trust profile for a single agent"""
    agent_id: str
    current_score: float
    max_score: float
    min_score: float
    total_events: int
    success_rate: float
    average_score_change: float
    last_updated: datetime
    performance_history: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentTrustProfile':
        """Create from dictionary"""
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


class TrustTrackerMetrics:
    """Performance metrics for trust tracker operations"""
    
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


class TrustTracker:
    """Main trust tracking system with async support and optimizations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('db_path', './trust_tracker.db')
        self.default_score = config.get('default_score', 0.5)
        self.min_score = config.get('min_score', 0.0)
        self.max_score = config.get('max_score', 1.0)
        self.decay_rate = config.get('decay_rate', 0.01)  # Daily decay rate
        self.event_weights = config.get('event_weights', {
            TrustEventType.TASK_SUCCESS: 0.1,
            TrustEventType.TASK_FAILURE: -0.15,
            TrustEventType.HUMAN_APPROVAL: 0.2,
            TrustEventType.HUMAN_REJECTION: -0.25,
            TrustEventType.POLICY_VIOLATION: -0.3,
            TrustEventType.QUALITY_HIGH: 0.15,
            TrustEventType.QUALITY_LOW: -0.1,
            TrustEventType.TIMEOUT: -0.05,
            TrustEventType.ERROR: -0.2,
            TrustEventType.IMPROVEMENT: 0.1,
            TrustEventType.COLLABORATION_SUCCESS: 0.05
        })

        # Performance optimizations
        self.metrics = TrustTrackerMetrics()
        self.cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes
        self._cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))

        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for trust tracking"""
        conn = sqlite3.connect(self.db_path)

        # Create tables - agent_profiles first due to foreign key constraint
        conn.execute('''
            CREATE TABLE IF NOT EXISTS agent_profiles (
                agent_id TEXT PRIMARY KEY,
                current_score REAL NOT NULL,
                max_score REAL NOT NULL,
                min_score REAL NOT NULL,
                total_events INTEGER NOT NULL,
                success_rate REAL NOT NULL,
                average_score_change REAL NOT NULL,
                last_updated TEXT NOT NULL,
                performance_history TEXT
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS trust_events(
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                score_change REAL NOT NULL,
                description TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                task_id TEXT,
                metadata TEXT,
                FOREIGN KEY (agent_id) REFERENCES agent_profiles(agent_id)
            )
        ''')

        # Create indexes for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_agent ON trust_events (agent_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON trust_events (timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON trust_events (event_type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_profiles_score ON agent_profiles (current_score)')

        conn.commit()
        conn.close()

    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Ensure tables exist for this connection (especially important for :memory: databases)
            self._ensure_tables_exist(conn)
            yield conn
        finally:
            conn.close()
    
    def _ensure_tables_exist(self, conn: sqlite3.Connection):
        """Ensure tables exist in the database"""
        # Create tables - agent_profiles first due to foreign key constraint
        conn.execute('''
            CREATE TABLE IF NOT EXISTS agent_profiles (
                agent_id TEXT PRIMARY KEY,
                current_score REAL NOT NULL,
                max_score REAL NOT NULL,
                min_score REAL NOT NULL,
                total_events INTEGER NOT NULL,
                success_rate REAL NOT NULL,
                average_score_change REAL NOT NULL,
                last_updated TEXT NOT NULL,
                performance_history TEXT
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS trust_events(
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                score_change REAL NOT NULL,
                description TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                task_id TEXT,
                metadata TEXT,
                FOREIGN KEY (agent_id) REFERENCES agent_profiles(agent_id)
            )
        ''')

        # Create indexes for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_agent ON trust_events (agent_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON trust_events (timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON trust_events (event_type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_profiles_score ON agent_profiles (current_score)')

    def _get_cache_key(self, key: str) -> str:
        """Generate cache key"""
        return f"trust_tracker:{key}"

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

    async def register_agent(self, agent_id: str, initial_score: Optional[float] = None) -> None:
        """Register a new agent with the trust tracker asynchronously"""
        start_time = time.time()
        try:
            score = initial_score if initial_score is not None else self.default_score

            async with self._get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO agent_profiles 
                    (agent_id, current_score, max_score, min_score, total_events, 
                     success_rate, average_score_change, last_updated, performance_history)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    agent_id,
                    score,
                    score,  # Initial max_score
                    score,  # Initial min_score
                    0,  # total_events
                    0.0,  # success_rate
                    0.0,  # average_score_change
                    datetime.now().isoformat(),
                    json.dumps([score])  # performance_history
                ))
                conn.commit()

            # Update cache
            self._set_cache(f"profile:{agent_id}", {
                'current_score': score,
                'max_score': score,
                'min_score': score,
                'total_events': 0,
                'success_rate': 0.0,
                'average_score_change': 0.0,
                'last_updated': datetime.now(),
                'performance_history': [score]
            })

            duration = time.time() - start_time
            self.metrics.record_operation('register_agent', duration, True)
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('register_agent', duration, False)
            logger.error("Failed to register agent", error=str(e))
            raise

    async def record_event(self, agent_id: str, event_type: TrustEventType,
                          description: str, task_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          custom_score_change: Optional[float] = None) -> float:
        """Record a trust event and update agent's trust score asynchronously"""
        start_time = time.time()
        try:
            # Ensure agent exists
            await self.register_agent(agent_id)

            # Calculate score change
            if custom_score_change is not None:
                score_change = custom_score_change
            else:
                score_change = self.event_weights.get(event_type, 0.0)

            # Create event
            event = TrustEvent(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                event_type=event_type,
                score_change=score_change,
                description=description,
                timestamp=datetime.now(),
                task_id=task_id,
                metadata=metadata
            )

            async with self._get_connection() as conn:
                # Record event
                conn.execute('''
                    INSERT INTO trust_events
                    (id, agent_id, event_type, score_change, description, timestamp, task_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.id,
                    event.agent_id,
                    event.event_type.value,
                    event.score_change,
                    event.description,
                    event.timestamp.isoformat(),
                    event.task_id,
                    json.dumps(event.metadata) if event.metadata else None
                ))

                # Update agent profile
                await self._update_agent_profile(conn, agent_id, event)

                conn.commit()

            # Clear cache for this agent
            cache_key = self._get_cache_key(f"profile:{agent_id}")
            with self._cache_lock:
                if cache_key in self.cache:
                    del self.cache[cache_key]

            # Return new score
            new_score = await self.get_agent_trust_score(agent_id)

            duration = time.time() - start_time
            self.metrics.record_operation('record_event', duration, True)
            return new_score

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('record_event', duration, False)
            logger.error("Failed to record event", error=str(e))
            raise

    async def record_events_batch(self, events: List[Dict[str, Any]]) -> List[float]:
        """Record multiple trust events in batch"""
        start_time = time.time()
        try:
            results = []
            async with self._get_connection() as conn:
                for event_data in events:
                    agent_id = event_data['agent_id']
                    event_type = TrustEventType(event_data['event_type'])
                    description = event_data['description']
                    task_id = event_data.get('task_id')
                    metadata = event_data.get('metadata')
                    custom_score_change = event_data.get('custom_score_change')

                    # Ensure agent exists
                    await self.register_agent(agent_id)

                    # Calculate score change
                    if custom_score_change is not None:
                        score_change = custom_score_change
                    else:
                        score_change = self.event_weights.get(event_type, 0.0)

                    # Create event
                    event = TrustEvent(
                        id=str(uuid.uuid4()),
                        agent_id=agent_id,
                        event_type=event_type,
                        score_change=score_change,
                        description=description,
                        timestamp=datetime.now(),
                        task_id=task_id,
                        metadata=metadata
                    )

                    # Record event
                    conn.execute('''
                        INSERT INTO trust_events
                        (id, agent_id, event_type, score_change, description, timestamp, task_id, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event.id,
                        event.agent_id,
                        event.event_type.value,
                        event.score_change,
                        event.description,
                        event.timestamp.isoformat(),
                        event.task_id,
                        json.dumps(event.metadata) if event.metadata else None
                    ))

                    # Update agent profile
                    await self._update_agent_profile(conn, agent_id, event)

                conn.commit()

            # Clear cache for all affected agents
            affected_agents = list(set(event['agent_id'] for event in events))
            for agent_id in affected_agents:
                cache_key = self._get_cache_key(f"profile:{agent_id}")
                with self._cache_lock:
                    if cache_key in self.cache:
                        del self.cache[cache_key]

            # Get new scores
            for event_data in events:
                new_score = await self.get_agent_trust_score(event_data['agent_id'])
                results.append(new_score)

            duration = time.time() - start_time
            self.metrics.record_operation('record_events_batch', duration, True)
            return results

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('record_events_batch', duration, False)
            logger.error("Failed to record events batch", error=str(e))
            raise

    async def _update_agent_profile(self, conn: sqlite3.Connection, agent_id: str, event: TrustEvent) -> None:
        """Update agent profile based on new event asynchronously"""
        # Get current profile
        cursor = conn.execute('''
            SELECT current_score, max_score, min_score, total_events,
                   success_rate, average_score_change, performance_history
            FROM agent_profiles WHERE agent_id = ?
        ''', (agent_id,))

        row = cursor.fetchone()
        if not row:
            return

        current_score, max_score, min_score, total_events, success_rate, avg_change, history_json = row

        # Parse performance history
        performance_history = json.loads(history_json) if history_json else []

        # Calculate new score with bounds
        new_score = current_score + event.score_change
        new_score = max(self.min_score, min(self.max_score, new_score))

        # Update statistics
        total_events += 1
        max_score = max(max_score, new_score)
        min_score = min(min_score, new_score)

        # Update performance history (keep last 100 scores)
        performance_history.append(new_score)
        if len(performance_history) > 100:
            performance_history = performance_history[-100:]

        # Calculate success rate (events with positive score changes)
        cursor = conn.execute('''
            SELECT COUNT(*) FROM trust_events
            WHERE agent_id = ? AND score_change > 0
        ''', (agent_id,))
        success_count = cursor.fetchone()[0]
        success_rate = success_count / total_events if total_events > 0 else 0.0

        # Calculate average score change
        cursor = conn.execute('''
            SELECT AVG(score_change) FROM trust_events WHERE agent_id = ?
        ''', (agent_id,))
        avg_change = cursor.fetchone()[0] or 0.0

        # Apply time decay if needed
        last_updated = datetime.now()
        time_decay = await self._calculate_time_decay(conn, agent_id)
        if time_decay > 0:
            new_score = max(self.min_score, new_score - time_decay)

        # Update profile
        conn.execute('''
            UPDATE agent_profiles
            SET current_score = ?, max_score = ?, min_score = ?, 
                total_events = ?, success_rate = ?, average_score_change = ?,
                last_updated = ?, performance_history = ?
            WHERE agent_id = ?
        ''', (
            new_score,
            max_score,
            min_score,
            total_events,
            success_rate,
            avg_change,
            last_updated.isoformat(),
            json.dumps(performance_history),
            agent_id
        ))

    async def _calculate_time_decay(self, conn: sqlite3.Connection, agent_id: str) -> float:
        """Calculate time-based decay for agent trust score asynchronously"""
        cursor = conn.execute('''
            SELECT last_updated FROM agent_profiles WHERE agent_id = ?
        ''', (agent_id,))

        row = cursor.fetchone()
        if not row:
            return 0.0

        last_updated = datetime.fromisoformat(row[0])
        days_inactive = (datetime.now() - last_updated).days

        if days_inactive <= 0:
            return 0.0

        # Apply exponential decay
        return self.decay_rate * days_inactive

    async def get_agent_trust_score(self, agent_id: str) -> float:
        """Get current trust score for an agent asynchronously"""
        # Check cache first
        cached = self._get_from_cache(f"profile:{agent_id}")
        if cached:
            return cached['current_score']

        start_time = time.time()
        try:
            async with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT current_score FROM agent_profiles WHERE agent_id = ?
                ''', (agent_id,))

                row = cursor.fetchone()
                if row:
                    score = row[0]
                    # Cache the result
                    self._set_cache(f"profile:{agent_id}", {'current_score': score})
                    
                    duration = time.time() - start_time
                    self.metrics.record_operation('get_agent_trust_score', duration, True)
                    return score
                else:
                    # Agent not found, register with default score
                    await self.register_agent(agent_id)
                    
                    duration = time.time() - start_time
                    self.metrics.record_operation('get_agent_trust_score', duration, True)
                    return self.default_score

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('get_agent_trust_score', duration, False)
            logger.error("Failed to get agent trust score", error=str(e))
            raise

    async def get_agent_profile(self, agent_id: str) -> Optional[AgentTrustProfile]:
        """Get complete trust profile for an agent asynchronously"""
        start_time = time.time()
        try:
            async with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT current_score, max_score, min_score, total_events,
                           success_rate, average_score_change, last_updated, performance_history
                    FROM agent_profiles WHERE agent_id = ?
                ''', (agent_id,))

                row = cursor.fetchone()
                if not row:
                    duration = time.time() - start_time
                    self.metrics.record_operation('get_agent_profile', duration, True)
                    return None

                profile = AgentTrustProfile(
                    agent_id=agent_id,
                    current_score=row[0],
                    max_score=row[1],
                    min_score=row[2],
                    total_events=row[3],
                    success_rate=row[4],
                    average_score_change=row[5],
                    last_updated=datetime.fromisoformat(row[6]),
                    performance_history=json.loads(row[7]) if row[7] else []
                )

                duration = time.time() - start_time
                self.metrics.record_operation('get_agent_profile', duration, True)
                return profile

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('get_agent_profile', duration, False)
            logger.error("Failed to get agent profile", error=str(e))
            raise

    async def get_agent_events(self, agent_id: str, limit: int = 50) -> List[TrustEvent]:
        """Get recent trust events for an agent asynchronously"""
        start_time = time.time()
        try:
            async with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT id, event_type, score_change, description, timestamp, task_id, metadata
                    FROM trust_events
                    WHERE agent_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (agent_id, limit))

                events = []
                for row in cursor.fetchall():
                    event_id, event_type, score_change, description, timestamp, task_id, metadata_json = row

                    event = TrustEvent(
                        id=event_id,
                        agent_id=agent_id,
                        event_type=TrustEventType(event_type),
                        score_change=score_change,
                        description=description,
                        timestamp=datetime.fromisoformat(timestamp),
                        task_id=task_id,
                        metadata=json.loads(metadata_json) if metadata_json else None
                    )
                    events.append(event)

                duration = time.time() - start_time
                self.metrics.record_operation('get_agent_events', duration, True)
                return events

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('get_agent_events', duration, False)
            logger.error("Failed to get agent events", error=str(e))
            raise

    async def get_all_agents(self) -> List[str]:
        """Get list of all registered agents asynchronously"""
        start_time = time.time()
        try:
            async with self._get_connection() as conn:
                cursor = conn.execute('SELECT agent_id FROM agent_profiles')
                agents = [row[0] for row in cursor.fetchall()]

                duration = time.time() - start_time
                self.metrics.record_operation('get_all_agents', duration, True)
                return agents

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('get_all_agents', duration, False)
            logger.error("Failed to get all agents", error=str(e))
            raise

    async def get_top_agents(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top agents by trust score asynchronously"""
        start_time = time.time()
        try:
            async with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT agent_id, current_score FROM agent_profiles
                    ORDER BY current_score DESC 
                    LIMIT ?
                ''', (limit,))

                results = [(row[0], row[1]) for row in cursor.fetchall()]

                duration = time.time() - start_time
                self.metrics.record_operation('get_top_agents', duration, True)
                return results

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('get_top_agents', duration, False)
            logger.error("Failed to get top agents", error=str(e))
            raise

    async def get_trust_statistics(self) -> Dict[str, Any]:
        """Get overall trust statistics asynchronously"""
        start_time = time.time()
        try:
            async with self._get_connection() as conn:
                # Basic statistics
                cursor = conn.execute('SELECT COUNT(*) FROM agent_profiles')
                total_agents = cursor.fetchone()[0]

                cursor = conn.execute('SELECT COUNT(*) FROM trust_events')
                total_events = cursor.fetchone()[0]

                # Average trust score
                cursor = conn.execute('SELECT AVG(current_score) FROM agent_profiles')
                avg_score = cursor.fetchone()[0] or 0.0

                # Event type distribution
                cursor = conn.execute('''
                    SELECT event_type, COUNT(*) FROM trust_events GROUP BY event_type
                ''')
                event_distribution = {row[0]: row[1] for row in cursor.fetchall()}

                # Recent activity
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM trust_events
                    WHERE timestamp > datetime('now', '-24 hours')
                ''')
                recent_events = cursor.fetchone()[0]

                stats = {
                    'total_agents': total_agents,
                    'total_events': total_events,
                    'average_trust_score': avg_score,
                    'event_distribution': event_distribution,
                    'recent_events_24h': recent_events
                }

                duration = time.time() - start_time
                self.metrics.record_operation('get_trust_statistics', duration, True)
                return stats

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('get_trust_statistics', duration, False)
            logger.error("Failed to get trust statistics", error=str(e))
            raise

    async def apply_decay_to_all_agents(self) -> None:
        """Apply time-based decay to all inactive agents asynchronously"""
        start_time = time.time()
        try:
            async with self._get_connection() as conn:
                cursor = conn.execute('SELECT agent_id FROM agent_profiles')
                agent_ids = [row[0] for row in cursor.fetchall()]

                for agent_id in agent_ids:
                    decay = await self._calculate_time_decay(conn, agent_id)
                    if decay > 0:
                        current_score = await self.get_agent_trust_score(agent_id)
                        new_score = max(self.min_score, current_score - decay)

                        conn.execute('''
                            UPDATE agent_profiles
                            SET current_score = ?, last_updated = ?
                            WHERE agent_id = ?
                        ''', (new_score, datetime.now().isoformat(), agent_id))

                conn.commit()

            # Clear cache for all agents
            self._clear_cache()

            duration = time.time() - start_time
            self.metrics.record_operation('apply_decay_to_all_agents', duration, True)

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('apply_decay_to_all_agents', duration, False)
            logger.error("Failed to apply decay to all agents", error=str(e))
            raise

    async def reset_agent_trust(self, agent_id: str, new_score: Optional[float] = None) -> None:
        """Reset an agent's trust score asynchronously"""
        start_time = time.time()
        try:
            score = new_score if new_score is not None else self.default_score

            async with self._get_connection() as conn:
                conn.execute('''
                    UPDATE agent_profiles
                    SET current_score = ?, max_score = ?, min_score = ?,
                        total_events = 0, success_rate = 0, average_score_change = 0,
                        last_updated = ?, performance_history = ?
                    WHERE agent_id = ?
                ''', (
                    score,
                    score,
                    score,
                    datetime.now().isoformat(),
                    json.dumps([score]),
                    agent_id
                ))

                conn.commit()

            # Clear cache for this agent
            cache_key = self._get_cache_key(f"profile:{agent_id}")
            with self._cache_lock:
                if cache_key in self.cache:
                    del self.cache[cache_key]

            duration = time.time() - start_time
            self.metrics.record_operation('reset_agent_trust', duration, True)

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('reset_agent_trust', duration, False)
            logger.error("Failed to reset agent trust", error=str(e))
            raise

    async def export_data(self, filepath: str) -> None:
        """Export all trust data to JSON file asynchronously"""
        start_time = time.time()
        try:
            data = {
                'agents': {},
                'events': [],
                'statistics': await self.get_trust_statistics()
            }

            # Export agent profiles
            for agent_id in await self.get_all_agents():
                profile = await self.get_agent_profile(agent_id)
                if profile:
                    data['agents'][agent_id] = profile.to_dict()

            # Export events
            async with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT id, agent_id, event_type, score_change, description, 
                        timestamp, task_id, metadata
                    FROM trust_events
                ''')

                for row in cursor.fetchall():
                    event_id, agent_id, event_type, score_change, description, timestamp, task_id, metadata_json = row

                    event = TrustEvent(
                        id=event_id,
                        agent_id=agent_id,
                        event_type=TrustEventType(event_type),
                        score_change=score_change,
                        description=description,
                        timestamp=datetime.fromisoformat(timestamp),
                        task_id=task_id,
                        metadata=json.loads(metadata_json) if metadata_json else None
                    )
                    data['events'].append(event.to_dict())

            # Write to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            duration = time.time() - start_time
            self.metrics.record_operation('export_data', duration, True)

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('export_data', duration, False)
            logger.error("Failed to export data", error=str(e))
            raise

    async def import_data(self, filepath: str) -> None:
        """Import trust data from JSON file asynchronously"""
        start_time = time.time()
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            async with self._get_connection() as conn:
                # Clear existing data
                conn.execute('DELETE FROM trust_events')
                conn.execute('DELETE FROM agent_profiles')

                # Import agent profiles
                for agent_id, profile_data in data.get('agents', {}).items():
                    profile = AgentTrustProfile.from_dict(profile_data)

                    conn.execute('''
                        INSERT INTO agent_profiles
                        (agent_id, current_score, max_score, min_score, total_events,
                        success_rate, average_score_change, last_updated, performance_history)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                     ''', (
                         profile.agent_id,
                         profile.current_score,
                         profile.max_score,
                         profile.min_score,
                         profile.total_events,
                         profile.success_rate,
                         profile.average_score_change,
                         profile.last_updated.isoformat(),
                         json.dumps(profile.performance_history)
                     ))

                # Import events
                for event_data in data.get('events', []):
                    event = TrustEvent.from_dict(event_data)

                    conn.execute('''
                        INSERT INTO trust_events
                        (id, agent_id, event_type, score_change, description,
                        timestamp, task_id, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event.id,
                        event.agent_id,
                        event.event_type.value,
                        event.score_change,
                        event.description,
                        event.timestamp.isoformat(),
                        event.task_id,
                        json.dumps(event.metadata) if event.metadata else None
                    ))

                conn.commit()

            # Clear cache
            self._clear_cache()

            duration = time.time() - start_time
            self.metrics.record_operation('import_data', duration, True)

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation('import_data', duration, False)
            logger.error("Failed to import data", error=str(e))
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_stats()

    def clear_cache(self) -> None:
        """Clear the cache"""
        self._clear_cache()