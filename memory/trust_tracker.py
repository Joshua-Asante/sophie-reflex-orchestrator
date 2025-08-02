"""
Trust Tracker Module for Sophie Reflex Orchestrator

Manages dynamic trust scoring and performance tracking for agents.
Tracks historical performance and adjusts trust scores based on outcomes.
"""

import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

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


class TrustTracker:
    """Main trust tracking system"""

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

        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for trust tracking"""
        conn = sqlite3.connect(self.db_path)

        # Create tables
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

        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_agent ON trust_events (agent_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON trust_events (timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON trust_events (event_type)')

        conn.commit()
        conn.close()

    def register_agent(self, agent_id: str, initial_score: Optional[float] = None) -> None:
        """Register a new agent with the trust tracker"""
        conn = sqlite3.connect(self.db_path)

        score = initial_score if initial_score is not None else self.default_score

        try:
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
        finally:
            conn.close()

    def record_event(self, agent_id: str, event_type: TrustEventType,
                    description: str, task_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     custom_score_change: Optional[float] = None) -> float:
        """Record a trust event and update agent's trust score"""
        # Ensure agent exists
        self.register_agent(agent_id)

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

        conn = sqlite3.connect(self.db_path)

        try:
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
            self._update_agent_profile(conn, agent_id, event)

            conn.commit()

            # Return new score
            return self.get_agent_trust_score(agent_id)

        finally:
            conn.close()

    def _update_agent_profile(self, conn: sqlite3.Connection, agent_id: str, event: TrustEvent) -> None:
        """Update agent profile based on new event"""
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
        time_decay = self._calculate_time_decay(conn, agent_id)
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

    def _calculate_time_decay(self, conn: sqlite3.Connection, agent_id: str) -> float:
        """Calculate time-based decay for agent trust score"""
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

    def get_agent_trust_score(self, agent_id: str) -> float:
        """Get current trust score for an agent"""
        conn = sqlite3.connect(self.db_path)

        try:
            cursor = conn.execute('''
                SELECT current_score FROM agent_profiles WHERE agent_id = ?
            ''', (agent_id,))

            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                # Agent not found, register with default score
                conn.close()
                self.register_agent(agent_id)
                return self.default_score

        finally:
            conn.close()

    def get_agent_profile(self, agent_id: str) -> Optional[AgentTrustProfile]:
        """Get complete trust profile for an agent"""
        conn = sqlite3.connect(self.db_path)

        try:
            cursor = conn.execute('''
                SELECT current_score, max_score, min_score, total_events,
                       success_rate, average_score_change, last_updated, performance_history
                FROM agent_profiles WHERE agent_id = ?
            ''', (agent_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return AgentTrustProfile(
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

        finally:
            conn.close()

    def get_agent_events(self, agent_id: str, limit: int = 50) -> List[TrustEvent]:
        """Get recent trust events for an agent"""
        conn = sqlite3.connect(self.db_path)

        try:
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

            return events

        finally:
            conn.close()

    def get_all_agents(self) -> List[str]:
        """Get list of all registered agents"""
        conn = sqlite3.connect(self.db_path)

        try:
            cursor = conn.execute('SELECT agent_id FROM agent_profiles')
            return [row[0] for row in cursor.fetchall()]

        finally:
            conn.close()

    def get_top_agents(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top agents by trust score"""
        conn = sqlite3.connect(self.db_path)

        try:
            cursor = conn.execute('''
                SELECT agent_id, current_score FROM agent_profiles
                ORDER BY current_score DESC 
                LIMIT ?
            ''', (limit,))

            return [(row[0], row[1]) for row in cursor.fetchall()]

        finally:
            conn.close()

    def get_trust_statistics(self) -> Dict[str, Any]:
        """Get overall trust statistics"""
        conn = sqlite3.connect(self.db_path)

        try:
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

            return {
                'total_agents': total_agents,
                'total_events': total_events,
                'average_trust_score': avg_score,
                'event_distribution': event_distribution,
                'recent_events_24h': recent_events
            }

        finally:
            conn.close()

    def apply_decay_to_all_agents(self) -> None:
        """Apply time-based decay to all inactive agents"""
        conn = sqlite3.connect(self.db_path)

        try:
            cursor = conn.execute('SELECT agent_id FROM agent_profiles')
            agent_ids = [row[0] for row in cursor.fetchall()]

            for agent_id in agent_ids:
                decay = self._calculate_time_decay(conn, agent_id)
                if decay > 0:
                    current_score = self.get_agent_trust_score(agent_id)
                    new_score = max(self.min_score, current_score - decay)

                    conn.execute('''
                        UPDATE agent_profiles
                        SET current_score = ?, last_updated  = ?
                        WHERE agent_id = ?
                    ''', (new_score, datetime.now().isoformat(), agent_id))

            conn.commit()

        finally:
            conn.close()

    def reset_agent_trust(self, agent_id: str, new_score: Optional[float] = None) -> None:
        """Reset an agent's trust score"""
        score = new_score if new_score is not None else self.default_score

        conn = sqlite3.connect(self.db_path)

        try:
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

        finally:
            conn.close()

    def export_data(self, filepath: str) -> None:
        """Export all trust data to JSON file"""
        data = {
            'agents': {},
            'events': [],
            'statistics': self.get_trust_statistics()
        }

        # Export agent profiles
        for agent_id in self.get_all_agents():
            profile = self.get_agent_profile(agent_id)
            if profile:
                data['agents'][agent_id] = profile.to_dict()

        # Export events
        conn = sqlite3.connect(self.db_path)
        try:
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

        finally:
            conn.close()

        # Write to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_data(self, filepath: str) -> None:
        """Import trust data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        conn = sqlite3.connect(self.db_path)

        try:
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

        finally:
            conn.close()