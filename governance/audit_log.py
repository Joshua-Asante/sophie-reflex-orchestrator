from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
import structlog
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import os
import difflib
import hashlib
from concurrent.futures import ThreadPoolExecutor
import aiosqlite
from contextlib import asynccontextmanager

logger = structlog.get_logger()


class AuditEventType(Enum):
    TASK_SUBMITTED = "task_submitted"
    TASK_COMPLETED = "task_completed"
    PLAN_GENERATED = "plan_generated"
    PLAN_EVALUATED = "plan_evaluated"
    PLAN_MODIFIED = "plan_modified"
    HUMAN_INTERVENTION = "human_intervention"
    AGENT_CREATED = "agent_created"
    AGENT_MODIFIED = "agent_modified"
    AGENT_PRUNED = "agent_pruned"
    TRUST_SCORE_UPDATED = "trust_score_updated"
    POLICY_VIOLATION = "policy_violation"
    SYSTEM_ERROR = "system_error"
    GENERATION_COMPLETED = "generation_completed"
    MEMORY_ACCESS = "memory_access"
    PERFORMANCE_METRICS = "performance_metrics"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    agent_id: Optional[str]
    task_id: str
    session_id: str
    description: str
    details: Dict[str, Any]
    severity: str = "info"  # debug, info, warning, error, critical
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class PlanDiff:
    """Represents differences between plan versions."""
    plan_id: str
    version_from: str
    version_to: str
    timestamp: datetime
    changes: List[Dict[str, Any]]
    similarity_score: float
    change_summary: str


class AuditLog:
    """Logs plan diffs, scores, hits/misses, and interventions with async support."""
    
    def __init__(self, db_path: str = "./memory/audit_log.db"):
        self.db_path = db_path
        self.current_session = None
        self.event_buffer = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        self.max_buffer_size = 1000  # Prevent memory issues
        self._initialized = False
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Start background flush task
        self._start_flush_task()
        
        logger.info("Audit log initialized", db_path=db_path)
    
    async def initialize(self):
        """Initialize the database asynchronously."""
        if not self._initialized:
            await self._initialize_database_async()
            self._initialized = True
    
    async def _initialize_database_async(self):
        """Initialize the audit log database asynchronously."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            async with aiosqlite.connect(self.db_path) as conn:
                # Create audit events table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        agent_id TEXT,
                        task_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        description TEXT NOT NULL,
                        details TEXT DEFAULT '{}',
                        severity TEXT DEFAULT 'info',
                        user_id TEXT,
                        ip_address TEXT
                    )
                ''')
                
                # Create plan diffs table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS plan_diffs (
                        diff_id TEXT PRIMARY KEY,
                        plan_id TEXT NOT NULL,
                        version_from TEXT NOT NULL,
                        version_to TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        changes TEXT DEFAULT '[]',
                        similarity_score REAL DEFAULT 0.0,
                        change_summary TEXT DEFAULT '',
                        task_id TEXT,
                        FOREIGN KEY (task_id) REFERENCES audit_events (task_id)
                    )
                ''')
                
                # Create performance metrics table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        metric_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        agent_id TEXT,
                        task_id TEXT,
                        session_id TEXT,
                        metadata TEXT DEFAULT '{}'
                    )
                ''')
                
                # Create indexes
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON audit_events(timestamp)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON audit_events(event_type)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_events_agent ON audit_events(agent_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_events_task ON audit_events(task_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_events_session ON audit_events(session_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_diffs_plan ON plan_diffs(plan_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_diffs_timestamp ON plan_diffs(timestamp)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name)')
                
                await conn.commit()
            
            logger.info("Audit log database initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize audit log database", error=str(e))
            raise
    
    def _start_flush_task(self):
        """Start background task to flush event buffer."""
        async def flush_buffer():
            while True:
                await asyncio.sleep(self.flush_interval)
                await self.flush_events()
        
        # Start the task
        asyncio.create_task(flush_buffer())
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new audit session."""
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.current_session = session_id
        
        # Log session start
        self.log_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            description=f"Session started: {session_id}",
            details={"session_type": "start"},
            severity="info"
        )
        
        logger.info("Audit session started", session_id=session_id)
        return session_id
    
    def end_session(self, session_id: str = None):
        """End the current audit session."""
        session_to_end = session_id or self.current_session
        if not session_to_end:
            return
        
        # Flush any remaining events
        asyncio.create_task(self.flush_events())
        
        # Log session end
        self.log_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            description=f"Session ended: {session_to_end}",
            details={"session_type": "end"},
            severity="info"
        )
        
        if session_to_end == self.current_session:
            self.current_session = None
        
        logger.info("Audit session ended", session_id=session_to_end)
    
    def log_event(self, event_type: AuditEventType, description: str, 
                 details: Dict[str, Any] = None, severity: str = "info",
                 agent_id: str = None, task_id: str = None, user_id: str = None,
                 ip_address: str = None) -> str:
        """Log an audit event."""
        try:
            # Generate event ID
            event_id = f"{event_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Use current session if no task_id provided
            if not task_id and self.current_session:
                task_id = self.current_session
            
            # Create event
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                agent_id=agent_id,
                task_id=task_id or "unknown",
                session_id=self.current_session or "no_session",
                description=description,
                details=details or {},
                severity=severity,
                user_id=user_id,
                ip_address=ip_address
            )
            
            # Add to buffer
            self.event_buffer.append(event)
            
            # Prevent buffer overflow
            if len(self.event_buffer) > self.max_buffer_size:
                logger.warning("Audit buffer overflow, forcing flush", 
                             buffer_size=len(self.event_buffer),
                             max_size=self.max_buffer_size)
                asyncio.create_task(self.flush_events())
            
            # Flush if buffer is full
            elif len(self.event_buffer) >= self.buffer_size:
                asyncio.create_task(self.flush_events())
            
            return event_id
            
        except Exception as e:
            logger.error("Failed to log audit event", event_type=event_type.value, error=str(e))
            return ""
    
    async def flush_events(self):
        """Flush buffered events to database asynchronously."""
        if not self.event_buffer:
            return
        
        try:
            events_to_flush = self.event_buffer.copy()
            self.event_buffer.clear()
            
            async with aiosqlite.connect(self.db_path) as conn:
                for event in events_to_flush:
                    await conn.execute('''
                        INSERT INTO audit_events 
                        (event_id, event_type, timestamp, agent_id, task_id, session_id, 
                         description, details, severity, user_id, ip_address)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event.event_id,
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        event.agent_id,
                        event.task_id,
                        event.session_id,
                        event.description,
                        json.dumps(event.details),
                        event.severity,
                        event.user_id,
                        event.ip_address
                    ))
                
                await conn.commit()
            
            logger.debug("Audit events flushed", count=len(events_to_flush))
            
        except Exception as e:
            logger.error("Failed to flush audit events", error=str(e))
            # Put events back in buffer
            self.event_buffer.extend(events_to_flush)
    
    async def log_plan_diff(self, plan_id: str, version_from: str, version_to: str,
                     old_content: str, new_content: str, task_id: str = None) -> str:
        """Log differences between plan versions asynchronously."""
        try:
            # Generate diff ID
            diff_id = f"diff_{plan_id}_{version_from}_{version_to}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Calculate differences in thread pool
            loop = asyncio.get_event_loop()
            changes = await loop.run_in_executor(
                self.thread_pool, 
                self._calculate_text_diff, 
                old_content, 
                new_content
            )
            
            # Calculate similarity score in thread pool
            similarity_score = await loop.run_in_executor(
                self.thread_pool,
                self._calculate_similarity,
                old_content,
                new_content
            )
            
            # Generate change summary
            change_summary = self._generate_change_summary(changes)
            
            # Create plan diff
            plan_diff = PlanDiff(
                plan_id=plan_id,
                version_from=version_from,
                version_to=version_to,
                timestamp=datetime.now(),
                changes=changes,
                similarity_score=similarity_score,
                change_summary=change_summary
            )
            
            # Save to database asynchronously
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('''
                    INSERT INTO plan_diffs 
                    (diff_id, plan_id, version_from, version_to, timestamp, changes, 
                     similarity_score, change_summary, task_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    diff_id,
                    plan_diff.plan_id,
                    plan_diff.version_from,
                    plan_diff.version_to,
                    plan_diff.timestamp.isoformat(),
                    json.dumps(plan_diff.changes),
                    plan_diff.similarity_score,
                    plan_diff.change_summary,
                    task_id
                ))
                
                await conn.commit()
            
            # Log as audit event
            self.log_event(
                event_type=AuditEventType.PLAN_MODIFIED,
                description=f"Plan modified: {plan_id} from {version_from} to {version_to}",
                details={
                    "plan_id": plan_id,
                    "version_from": version_from,
                    "version_to": version_to,
                    "similarity_score": similarity_score,
                    "changes_count": len(changes),
                    "change_summary": change_summary
                },
                severity="info"
            )
            
            return diff_id
            
        except Exception as e:
            logger.error("Failed to log plan diff", plan_id=plan_id, error=str(e))
            return ""
    
    def _calculate_text_diff(self, old_text: str, new_text: str) -> List[Dict[str, Any]]:
        """Calculate differences between two text versions."""
        try:
            # Split text into lines
            old_lines = old_text.splitlines()
            new_lines = new_text.splitlines()
            
            # Calculate diff
            differ = difflib.Differ()
            diff = list(differ.compare(old_lines, new_lines))
            
            changes = []
            current_change = None
            
            for line in diff:
                code = line[0]
                content = line[2:]
                
                if code == ' ':
                    # Unchanged line
                    if current_change:
                        changes.append(current_change)
                        current_change = None
                elif code == '-':
                    # Deleted line
                    if not current_change or current_change['type'] != 'deletion':
                        if current_change:
                            changes.append(current_change)
                        current_change = {
                            'type': 'deletion',
                            'lines': [content],
                            'line_number': old_lines.index(content) if content in old_lines else -1
                        }
                    else:
                        current_change['lines'].append(content)
                elif code == '+':
                    # Added line
                    if not current_change or current_change['type'] != 'addition':
                        if current_change:
                            changes.append(current_change)
                        current_change = {
                            'type': 'addition',
                            'lines': [content],
                            'line_number': new_lines.index(content) if content in new_lines else -1
                        }
                    else:
                        current_change['lines'].append(content)
                elif code == '?':
                    # Line diff info (ignore for now)
                    pass
            
            if current_change:
                changes.append(current_change)
            
            return changes
            
        except Exception as e:
            logger.error("Failed to calculate text diff", error=str(e))
            return []
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts."""
        try:
            # Use simple sequence matching for similarity
            matcher = difflib.SequenceMatcher(None, text1, text2)
            return matcher.ratio()
            
        except Exception as e:
            logger.error("Failed to calculate similarity", error=str(e))
            return 0.0
    
    def _generate_change_summary(self, changes: List[Dict[str, Any]]) -> str:
        """Generate a summary of changes."""
        try:
            if not changes:
                return "No changes detected"
            
            additions = len([c for c in changes if c['type'] == 'addition'])
            deletions = len([c for c in changes if c['type'] == 'deletion'])
            
            if additions > 0 and deletions > 0:
                return f"Modified: {additions} additions, {deletions} deletions"
            elif additions > 0:
                return f"Added: {additions} lines"
            elif deletions > 0:
                return f"Deleted: {deletions} lines"
            else:
                return "Minor changes detected"
                
        except Exception as e:
            logger.error("Failed to generate change summary", error=str(e))
            return "Change summary unavailable"
    
    async def log_metric(self, metric_name: str, metric_value: float, 
                  agent_id: str = None, task_id: str = None,
                  metadata: Dict[str, Any] = None) -> str:
        """Log a performance metric asynchronously."""
        try:
            # Generate metric ID
            metric_id = f"metric_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Save to database asynchronously
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('''
                    INSERT INTO performance_metrics 
                    (metric_id, timestamp, metric_name, metric_value, agent_id, task_id, session_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric_id,
                    datetime.now().isoformat(),
                    metric_name,
                    metric_value,
                    agent_id,
                    task_id,
                    self.current_session,
                    json.dumps(metadata or {})
                ))
                
                await conn.commit()
            
            return metric_id
            
        except Exception as e:
            logger.error("Failed to log metric", metric_name=metric_name, error=str(e))
            return ""
    
    async def get_events(self, event_type: AuditEventType = None, 
                       agent_id: str = None, task_id: str = None,
                       session_id: str = None, limit: int = 100,
                       start_time: datetime = None, end_time: datetime = None) -> List[AuditEvent]:
        """Get audit events with optional filtering."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Build query
                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type.value)
                
                if agent_id:
                    query += " AND agent_id = ?"
                    params.append(agent_id)
                
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()
            
            # Convert to AuditEvent objects
            events = []
            for row in rows:
                event = AuditEvent(
                    event_id=row[0],
                    event_type=AuditEventType(row[1]),
                    timestamp=datetime.fromisoformat(row[2]),
                    agent_id=row[3],
                    task_id=row[4],
                    session_id=row[5],
                    description=row[6],
                    details=json.loads(row[7]) if row[7] else {},
                    severity=row[8],
                    user_id=row[9],
                    ip_address=row[10]
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error("Failed to get audit events", error=str(e))
            return []
    
    async def get_plan_history(self, plan_id: str, limit: int = 50) -> List[PlanDiff]:
        """Get the history of changes for a plan."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute('''
                    SELECT diff_id, plan_id, version_from, version_to, timestamp, 
                           changes, similarity_score, change_summary, task_id
                    FROM plan_diffs
                    WHERE plan_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (plan_id, limit))
                
                rows = await cursor.fetchall()
            
            # Convert to PlanDiff objects
            diffs = []
            for row in rows:
                diff = PlanDiff(
                    plan_id=row[1],
                    version_from=row[2],
                    version_to=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    changes=json.loads(row[5]) if row[5] else [],
                    similarity_score=row[6],
                    change_summary=row[7]
                )
                diffs.append(diff)
            
            return diffs
            
        except Exception as e:
            logger.error("Failed to get plan history", plan_id=plan_id, error=str(e))
            return []
    
    async def get_metrics(self, metric_name: str = None, agent_id: str = None,
                        task_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance metrics."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Build query
                query = "SELECT * FROM performance_metrics WHERE 1=1"
                params = []
                
                if metric_name:
                    query += " AND metric_name = ?"
                    params.append(metric_name)
                
                if agent_id:
                    query += " AND agent_id = ?"
                    params.append(agent_id)
                
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()
            
            # Convert to dictionaries
            metrics = []
            for row in rows:
                metric = {
                    "metric_id": row[0],
                    "timestamp": row[1],
                    "metric_name": row[2],
                    "metric_value": row[3],
                    "agent_id": row[4],
                    "task_id": row[5],
                    "session_id": row[6],
                    "metadata": json.loads(row[7]) if row[7] else {}
                }
                metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to get metrics", error=str(e))
            return []
    
    async def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Execute all queries concurrently
                queries = [
                    ('event_counts', 'SELECT event_type, COUNT(*) FROM audit_events GROUP BY event_type'),
                    ('total_events', 'SELECT COUNT(*) FROM audit_events'),
                    ('total_diffs', 'SELECT COUNT(*) FROM plan_diffs'),
                    ('total_metrics', 'SELECT COUNT(*) FROM performance_metrics'),
                    ('severity_counts', 'SELECT severity, COUNT(*) FROM audit_events GROUP BY severity')
                ]
                
                results = {}
                for name, query in queries:
                    cursor = await conn.execute(query)
                    if name == 'event_counts':
                        results[name] = {row[0]: row[1] for row in await cursor.fetchall()}
                    elif name == 'severity_counts':
                        results[name] = {row[0]: row[1] for row in await cursor.fetchall()}
                    else:
                        results[name] = (await cursor.fetchone())[0]
            
            return {
                "total_events": results.get('total_events', 0),
                "total_plan_diffs": results.get('total_diffs', 0),
                "total_metrics": results.get('total_metrics', 0),
                "event_counts_by_type": results.get('event_counts', {}),
                "event_counts_by_severity": results.get('severity_counts', {}),
                "buffer_size": len(self.event_buffer),
                "max_buffer_size": self.max_buffer_size,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get audit statistics", error=str(e))
            return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old audit data."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            async with aiosqlite.connect(self.db_path) as conn:
                # Clean up old events
                await conn.execute('DELETE FROM audit_events WHERE timestamp < ?', (cutoff_str,))
                events_deleted = (await conn.execute('SELECT changes()')).rowcount
                
                # Clean up old plan diffs
                await conn.execute('DELETE FROM plan_diffs WHERE timestamp < ?', (cutoff_str,))
                diffs_deleted = (await conn.execute('SELECT changes()')).rowcount
                
                # Clean up old metrics
                await conn.execute('DELETE FROM performance_metrics WHERE timestamp < ?', (cutoff_str,))
                metrics_deleted = (await conn.execute('SELECT changes()')).rowcount
                
                await conn.commit()
            
            logger.info("Audit data cleanup completed", 
                       events_deleted=events_deleted,
                       diffs_deleted=diffs_deleted,
                       metrics_deleted=metrics_deleted,
                       days_kept=days_to_keep)
            
            return {
                "events_deleted": events_deleted,
                "diffs_deleted": diffs_deleted,
                "metrics_deleted": metrics_deleted
            }
            
        except Exception as e:
            logger.error("Failed to cleanup old audit data", error=str(e))
            return {"events_deleted": 0, "diffs_deleted": 0, "metrics_deleted": 0}
    
    async def export_audit_data(self, output_path: str, 
                              start_time: datetime = None, 
                              end_time: datetime = None) -> bool:
        """Export audit data to a JSON file."""
        try:
            # Get all data
            events = await self.get_events(start_time=start_time, end_time=end_time, limit=10000)
            
            # Get plan diffs
            async with aiosqlite.connect(self.db_path) as conn:
                query = "SELECT * FROM plan_diffs WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                cursor = await conn.execute(query, params)
                diff_rows = await cursor.fetchall()
            
            # Get metrics
            async with aiosqlite.connect(self.db_path) as conn:
                query = "SELECT * FROM performance_metrics WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                cursor = await conn.execute(query, params)
                metric_rows = await cursor.fetchall()
            
            # Convert to export format
            export_data = {
                "audit_events": [asdict(event) for event in events],
                "plan_diffs": [
                    {
                        "diff_id": row[0],
                        "plan_id": row[1],
                        "version_from": row[2],
                        "version_to": row[3],
                        "timestamp": row[4],
                        "changes": json.loads(row[5]) if row[5] else [],
                        "similarity_score": row[6],
                        "change_summary": row[7],
                        "task_id": row[8]
                    }
                    for row in diff_rows
                ],
                "performance_metrics": [
                    {
                        "metric_id": row[0],
                        "timestamp": row[1],
                        "metric_name": row[2],
                        "metric_value": row[3],
                        "agent_id": row[4],
                        "task_id": row[5],
                        "session_id": row[6],
                        "metadata": json.loads(row[7]) if row[7] else {}
                    }
                    for row in metric_rows
                ],
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None,
                    "events_count": len(events),
                    "diffs_count": len(diff_rows),
                    "metrics_count": len(metric_rows)
                }
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info("Audit data exported", output_path=output_path,
                       events=len(events), diffs=len(diff_rows), metrics=len(metric_rows))
            
            return True
            
        except Exception as e:
            logger.error("Failed to export audit data", error=str(e))
            return False