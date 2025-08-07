"""
Audit Manager Component

Handles audit logging and session management.
"""

from typing import Dict, Any
from datetime import datetime
import structlog

from governance.audit_log import AuditLog, AuditEventType
from ..models.orchestrator_config import OrchestratorConfig

logger = structlog.get_logger()


class AuditManager:
    """Handles audit logging and session management."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.audit_log = AuditLog()
        self.current_session_id = None
    
    def set_audit_log(self, audit_log: AuditLog):
        """Set the audit log dependency."""
        self.audit_log = audit_log
        logger.info("Audit log dependency set")
    
    async def start_session(self, session_id: str):
        """Start a new audit session."""
        try:
            self.current_session_id = session_id
            self.audit_log.start_session(session_id)
            logger.info("Audit session started", session_id=session_id)
        except Exception as e:
            logger.error("Failed to start audit session", error=str(e))
            raise
    
    async def end_session(self):
        """End the current audit session."""
        try:
            if self.current_session_id:
                self.audit_log.end_session()
                logger.info("Audit session ended", session_id=self.current_session_id)
                self.current_session_id = None
        except Exception as e:
            logger.error("Failed to end audit session", error=str(e))
    
    async def log_task_start(self, task: str, session_id: str):
        """Log task start event."""
        try:
            self.audit_log.log_event(
                event_type=AuditEventType.TASK_SUBMITTED,
                description=f"Task submitted: {task}",
                details={"task": task, "session_id": session_id},
                severity="info"
            )
            logger.info("Task start logged", task=task, session_id=session_id)
        except Exception as e:
            logger.error("Failed to log task start", error=str(e))
    
    async def log_task_completion(self, task: str, generations: int, execution_time: float):
        """Log task completion event."""
        try:
            self.audit_log.log_event(
                event_type=AuditEventType.TASK_COMPLETED,
                description=f"Task completed: {task}",
                details={
                    "task": task,
                    "total_generations": generations,
                    "execution_time": execution_time
                },
                severity="info"
            )
            logger.info("Task completion logged", task=task, generations=generations)
        except Exception as e:
            logger.error("Failed to log task completion", error=str(e))
    
    async def log_generation_completion(self, generation: int, best_score: float, 
                                      average_score: float, execution_time: float):
        """Log generation completion event."""
        try:
            self.audit_log.log_event(
                event_type=AuditEventType.GENERATION_COMPLETED,
                description=f"Generation {generation} completed",
                details={
                    "generation": generation,
                    "best_score": best_score,
                    "average_score": average_score,
                    "execution_time": execution_time
                },
                severity="info"
            )
            logger.info("Generation completion logged", generation=generation)
        except Exception as e:
            logger.error("Failed to log generation completion", error=str(e))
    
    async def log_agent_pruning(self, agent_id: str, reason: str, generation: int):
        """Log agent pruning event."""
        try:
            self.audit_log.log_event(
                event_type=AuditEventType.AGENT_PRUNED,
                description=f"Agent {agent_id} pruned",
                details={
                    "agent_id": agent_id,
                    "reason": reason,
                    "generation": generation
                },
                severity="info"
            )
            logger.info("Agent pruning logged", agent_id=agent_id, reason=reason)
        except Exception as e:
            logger.error("Failed to log agent pruning", error=str(e))
    
    async def log_error(self, error_message: str, details: Dict[str, Any] = None):
        """Log an error event."""
        try:
            self.audit_log.log_event(
                event_type=AuditEventType.SYSTEM_ERROR,
                description=f"System error: {error_message}",
                details=details or {},
                severity="error"
            )
            logger.error("Error logged", error_message=error_message)
        except Exception as e:
            logger.error("Failed to log error", error=str(e))
    
    async def log_event(self, event_type: AuditEventType, description: str, 
                       details: Dict[str, Any] = None, severity: str = "info"):
        """Log a custom event."""
        try:
            self.audit_log.log_event(
                event_type=event_type,
                description=description,
                details=details or {},
                severity=severity
            )
            logger.info("Custom event logged", event_type=event_type.value, description=description)
        except Exception as e:
            logger.error("Failed to log custom event", error=str(e))
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics."""
        try:
            return await self.audit_log.get_audit_statistics()
        except Exception as e:
            logger.error("Failed to get audit statistics", error=str(e))
            return {"error": str(e)} 