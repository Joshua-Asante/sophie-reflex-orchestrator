"""
Human Approval System

Event-driven human approval system with async notifications, approval persistence,
and multi-party governance for SOPHIE's autonomous execution.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import uuid

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalLevel(Enum):
    """Levels of approval required."""
    AUTONOMOUS = "autonomous"  # No approval needed
    NOTIFICATION = "notification"  # Inform human after execution
    APPROVAL = "approval"  # Require explicit approval
    SUPERVISION = "supervision"  # Human must be present
    MULTI_PARTY = "multi_party"  # Multiple parties must approve


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    id: str
    directive_id: str
    directive_description: str
    plan_summary: str
    risk_level: str
    estimated_duration: float
    required_approvers: List[str]
    optional_approvers: List[str]
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approvals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    denials: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalEvent:
    """An approval event for logging and auditing."""
    id: str
    request_id: str
    event_type: str  # "created", "approved", "denied", "expired"
    approver_id: Optional[str]
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class ApprovalQueue:
    """Queue for managing approval requests."""
    
    def __init__(self):
        self.pending_requests: deque = deque()
        self.approved_requests: List[ApprovalRequest] = []
        self.denied_requests: List[ApprovalRequest] = []
        self.expired_requests: List[ApprovalRequest] = []
        self.event_log: List[ApprovalEvent] = []
    
    def add_request(self, request: ApprovalRequest):
        """Add a new approval request to the queue."""
        self.pending_requests.append(request)
        
        # Log event
        event = ApprovalEvent(
            id=str(uuid.uuid4()),
            request_id=request.id,
            event_type="created",
            approver_id=None,
            details={"directive": request.directive_description}
        )
        self.event_log.append(event)
        
        logger.info(f"Added approval request: {request.directive_description}")
    
    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return list(self.pending_requests)
    
    def approve_request(self, request_id: str, approver_id: str, rationale: str = None) -> bool:
        """Approve a request."""
        for request in self.pending_requests:
            if request.id == request_id:
                request.approvals[approver_id] = {
                    "timestamp": time.time(),
                    "rationale": rationale
                }
                
                # Check if all required approvers have approved
                if len(request.approvals) >= len(request.required_approvers):
                    request.status = ApprovalStatus.APPROVED
                    self.approved_requests.append(request)
                    self.pending_requests.remove(request)
                    
                    # Log event
                    event = ApprovalEvent(
                        id=str(uuid.uuid4()),
                        request_id=request_id,
                        event_type="approved",
                        approver_id=approver_id,
                        details={"rationale": rationale}
                    )
                    self.event_log.append(event)
                    
                    logger.info(f"Request approved: {request.directive_description}")
                    return True
        
        return False
    
    def deny_request(self, request_id: str, denier_id: str, rationale: str = None) -> bool:
        """Deny a request."""
        for request in self.pending_requests:
            if request.id == request_id:
                request.denials[denier_id] = {
                    "timestamp": time.time(),
                    "rationale": rationale
                }
                request.status = ApprovalStatus.DENIED
                self.denied_requests.append(request)
                self.pending_requests.remove(request)
                
                # Log event
                event = ApprovalEvent(
                    id=str(uuid.uuid4()),
                    request_id=request_id,
                    event_type="denied",
                    approver_id=denier_id,
                    details={"rationale": rationale}
                )
                self.event_log.append(event)
                
                logger.info(f"Request denied: {request.directive_description}")
                return True
        
        return False
    
    def check_expired_requests(self):
        """Check and expire requests that have passed their expiration time."""
        current_time = time.time()
        expired_requests = []
        
        for request in self.pending_requests:
            if request.expires_at and current_time > request.expires_at:
                request.status = ApprovalStatus.EXPIRED
                expired_requests.append(request)
                
                # Log event
                event = ApprovalEvent(
                    id=str(uuid.uuid4()),
                    request_id=request.id,
                    event_type="expired",
                    approver_id=None,
                    details={"expired_at": request.expires_at}
                )
                self.event_log.append(event)
        
        for request in expired_requests:
            self.pending_requests.remove(request)
            self.expired_requests.append(request)
            logger.info(f"Request expired: {request.directive_description}")


class HumanApprovalSystem:
    """Main human approval system for SOPHIE."""
    
    def __init__(self):
        self.approval_queue = ApprovalQueue()
        self.notification_callbacks: List[Callable] = []
        self.approval_callbacks: List[Callable] = []
        self.expiration_check_interval = 60  # seconds
        self.default_expiration_time = 3600  # 1 hour
        self.expiration_task = None
    
    async def start(self):
        """Start the approval system."""
        if self.expiration_task is None:
            self.expiration_task = asyncio.create_task(self._expiration_checker())
    
    async def stop(self):
        """Stop the approval system."""
        if self.expiration_task:
            self.expiration_task.cancel()
            try:
                await self.expiration_task
            except asyncio.CancelledError:
                pass
            self.expiration_task = None
    
    async def request_approval(
        self,
        directive_id: str,
        directive_description: str,
        plan_summary: str,
        risk_level: str = "low",
        estimated_duration: float = 60.0,
        required_approvers: List[str] = None,
        optional_approvers: List[str] = None,
        expiration_time: Optional[float] = None
    ) -> ApprovalRequest:
        """Request human approval for a directive."""
        
        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            directive_id=directive_id,
            directive_description=directive_description,
            plan_summary=plan_summary,
            risk_level=risk_level,
            estimated_duration=estimated_duration,
            required_approvers=required_approvers or ["default_approver"],
            optional_approvers=optional_approvers or [],
            expires_at=expiration_time or (time.time() + self.default_expiration_time)
        )
        
        # Add to queue
        self.approval_queue.add_request(request)
        
        # Notify approvers
        await self._notify_approvers(request)
        
        return request
    
    async def approve_request(self, request_id: str, approver_id: str, rationale: str = None) -> bool:
        """Approve a request."""
        success = self.approval_queue.approve_request(request_id, approver_id, rationale)
        
        if success:
            # Trigger approval callbacks
            for callback in self.approval_callbacks:
                try:
                    await callback(request_id, "approved", approver_id)
                except Exception as e:
                    logger.error(f"Approval callback failed: {e}")
        
        return success
    
    async def deny_request(self, request_id: str, denier_id: str, rationale: str = None) -> bool:
        """Deny a request."""
        success = self.approval_queue.deny_request(request_id, denier_id, rationale)
        
        if success:
            # Trigger approval callbacks
            for callback in self.approval_callbacks:
                try:
                    await callback(request_id, "denied", denier_id)
                except Exception as e:
                    logger.error(f"Denial callback failed: {e}")
        
        return success
    
    async def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return self.approval_queue.get_pending_requests()
    
    async def get_approval_status(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get the status of a specific approval request."""
        all_requests = (
            self.approval_queue.get_pending_requests() +
            self.approval_queue.approved_requests +
            self.approval_queue.denied_requests +
            self.approval_queue.expired_requests
        )
        
        for request in all_requests:
            if request.id == request_id:
                return request
        
        return None
    
    async def get_approval_history(self) -> Dict[str, Any]:
        """Get approval system history."""
        return {
            "pending_count": len(self.approval_queue.get_pending_requests()),
            "approved_count": len(self.approval_queue.approved_requests),
            "denied_count": len(self.approval_queue.denied_requests),
            "expired_count": len(self.approval_queue.expired_requests),
            "total_events": len(self.approval_queue.event_log),
            "recent_events": [
                {
                    "event_type": event.event_type,
                    "request_id": event.request_id,
                    "timestamp": event.timestamp,
                    "approver_id": event.approver_id
                }
                for event in self.approval_queue.event_log[-10:]  # Last 10 events
            ]
        }
    
    def add_notification_callback(self, callback: Callable):
        """Add a notification callback."""
        self.notification_callbacks.append(callback)
    
    def add_approval_callback(self, callback: Callable):
        """Add an approval callback."""
        self.approval_callbacks.append(callback)
    
    async def _notify_approvers(self, request: ApprovalRequest):
        """Notify approvers of a new request."""
        notification = {
            "type": "approval_request",
            "request_id": request.id,
            "directive": request.directive_description,
            "risk_level": request.risk_level,
            "estimated_duration": request.estimated_duration,
            "required_approvers": request.required_approvers,
            "expires_at": request.expires_at
        }
        
        for callback in self.notification_callbacks:
            try:
                await callback(notification)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
    
    async def _expiration_checker(self):
        """Periodically check for expired requests."""
        while True:
            try:
                self.approval_queue.check_expired_requests()
                await asyncio.sleep(self.expiration_check_interval)
            except Exception as e:
                logger.error(f"Expiration checker failed: {e}")
                await asyncio.sleep(self.expiration_check_interval)


# Global instance
human_approval_system = HumanApprovalSystem()


# Convenience functions for easy integration
async def request_approval(
    directive_id: str,
    directive_description: str,
    plan_summary: str,
    risk_level: str = "low",
    estimated_duration: float = 60.0,
    required_approvers: List[str] = None
) -> ApprovalRequest:
    """Request human approval for a directive."""
    return await human_approval_system.request_approval(
        directive_id, directive_description, plan_summary,
        risk_level, estimated_duration, required_approvers
    )


async def approve_request(request_id: str, approver_id: str, rationale: str = None) -> bool:
    """Approve a request."""
    return await human_approval_system.approve_request(request_id, approver_id, rationale)


async def deny_request(request_id: str, denier_id: str, rationale: str = None) -> bool:
    """Deny a request."""
    return await human_approval_system.deny_request(request_id, denier_id, rationale)


async def get_pending_approvals() -> List[ApprovalRequest]:
    """Get all pending approval requests."""
    return await human_approval_system.get_pending_approvals()


async def get_approval_status(request_id: str) -> Optional[ApprovalRequest]:
    """Get the status of a specific approval request."""
    return await human_approval_system.get_approval_status(request_id)


async def get_approval_history() -> Dict[str, Any]:
    """Get approval system history."""
    return await human_approval_system.get_approval_history() 