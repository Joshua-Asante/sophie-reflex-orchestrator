"""
Sophie HITL Webhook Server - Optimized for Modular Architecture

Enhanced FastAPI application with improved performance, security, and monitoring.
Supports WebSocket connections for real-time updates and better integration
with the modular orchestrator architecture.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
import structlog
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import os
from pathlib import Path
import httpx
from contextlib import asynccontextmanager
import time
import secrets
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading

logger = structlog.get_logger()

# Pydantic models with validation
class PlanApprovalRequest(BaseModel):
    plan_id: str = Field(..., min_length=1, max_length=100)
    task_id: str = Field(..., min_length=1, max_length=200)
    agent_id: str = Field(..., min_length=1, max_length=100)
    plan_content: str = Field(..., min_length=1, max_length=10000)
    trust_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    evaluation_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('plan_id')
    def validate_plan_id(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Plan ID must be alphanumeric with hyphens/underscores only')
        return v

class PlanApprovalResponse(BaseModel):
    approved: bool
    plan_id: str
    decision_timestamp: datetime
    user_id: Optional[str] = None
    reason: Optional[str] = None
    modifications: Dict[str, Any] = Field(default_factory=dict)

class PlanDecisionRequest(BaseModel):
    reason: Optional[str] = None
    user_id: Optional[str] = None
    modifications: Dict[str, Any] = Field(default_factory=dict)

class PendingPlan(BaseModel):
    plan_id: str
    task_id: str
    agent_id: str
    plan_content: str
    trust_score: float
    confidence_score: float
    evaluation_score: Optional[float] = None
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: float = Field(default=0.0)
    
    def calculate_priority(self) -> float:
        """Calculate priority based on scores."""
        trust_weight = 0.4
        confidence_weight = 0.4
        eval_weight = 0.2
        
        priority = (self.trust_score * trust_weight) + \
                  (self.confidence_score * confidence_weight) + \
                  ((self.evaluation_score or 0.0) * eval_weight)
        
        self.priority = priority
        return priority

@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection."""
    websocket: WebSocket
    client_id: str
    connected_at: datetime
    last_activity: datetime
    subscriptions: List[str] = None
    
    def __post_init__(self):
        if self.subscriptions is None:
            self.subscriptions = []

class WebhookServer:
    """Enhanced FastAPI app for HITL approve/reject functionality with WebSocket support."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001, 
                 max_connections: int = 100, enable_websocket: bool = True):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.enable_websocket = enable_websocket
        
        # Enhanced storage with thread safety
        self.pending_plans: Dict[str, PendingPlan] = {}
        self.decision_history: List[PlanApprovalResponse] = []
        self.websocket_connections: Dict[str, WebSocketConnection] = {}
        self.connection_lock = threading.Lock()
        
        # Performance tracking
        self.metrics = {
            'total_requests': 0,
            'total_decisions': 0,
            'websocket_connections': 0,
            'average_response_time': 0.0,
            'start_time': datetime.now()
        }
        
        # Rate limiting
        self.rate_limit_store = defaultdict(list)
        self.rate_limit_lock = threading.Lock()
        
        # Background tasks
        self.cleanup_task = None
        self.metrics_task = None
        
        # Setup FastAPI app with enhanced configuration
        self.app = self._create_fastapi_app()
        
        # Setup static files and templates
        self._setup_static_files()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Webhook server initialized", 
                   host=host, port=port, max_connections=max_connections)
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI app with enhanced configuration."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.cleanup_task = asyncio.create_task(self._background_cleanup())
            self.metrics_task = asyncio.create_task(self._background_metrics())
            logger.info("Webhook server starting up")
            yield
            # Shutdown
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.metrics_task:
                self.metrics_task.cancel()
            logger.info("Webhook server shutting down")
        
        app = FastAPI(
            title="Sophie HITL Webhook Server",
            version="2.0.0",
            description="Enhanced Human-in-the-Loop Webhook Server with WebSocket Support",
            lifespan=lifespan
        )
        
        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        return app
    
    def _setup_static_files(self):
        """Setup static files and templates with error handling."""
        try:
            # Create static directory if it doesn't exist
            static_dir = Path(__file__).parent / "static"
            static_dir.mkdir(exist_ok=True)
            
            # Create templates directory if it doesn't exist
            templates_dir = Path(__file__).parent / "templates"
            templates_dir.mkdir(exist_ok=True)
            
            # Mount static files
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            
            # Setup templates
            self.templates = Jinja2Templates(directory=str(templates_dir))
            
            logger.info("Static files and templates setup completed")
            
        except Exception as e:
            logger.error("Failed to setup static files", error=str(e))
            raise
    
    def _setup_routes(self):
        """Setup API routes with enhanced functionality."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            """Main dashboard page with enhanced data."""
            try:
                # Sort plans by priority
                sorted_plans = sorted(
                    self.pending_plans.values(),
                    key=lambda p: p.calculate_priority(),
                    reverse=True
                )
                
                # Get recent decisions
                recent_decisions = self.decision_history[-10:]
                
                # Calculate additional stats
                stats = self._calculate_stats()
                
                return self.templates.TemplateResponse("dashboard.html", {
                    "request": request,
                    "pending_plans": sorted_plans,
                    "recent_decisions": recent_decisions,
                    "stats": stats,
                    "websocket_enabled": self.enable_websocket
                })
                
            except Exception as e:
                logger.error("Failed to render dashboard", error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/health")
        async def health_check():
            """Enhanced health check endpoint."""
            try:
                uptime = datetime.now() - self.metrics['start_time']
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "uptime_seconds": uptime.total_seconds(),
                    "pending_plans": len(self.pending_plans),
                    "total_decisions": len(self.decision_history),
                    "websocket_connections": len(self.websocket_connections),
                    "average_response_time": self.metrics['average_response_time']
                }
            except Exception as e:
                logger.error("Health check failed", error=str(e))
                raise HTTPException(status_code=500, detail="Health check failed")
        
        @self.app.post("/api/plans/submit")
        async def submit_plan(plan_request: PlanApprovalRequest, background_tasks: BackgroundTasks):
            """Submit a plan for human review with enhanced validation."""
            try:
                # Rate limiting
                if not self._check_rate_limit("submit_plan", 10, 60):  # 10 requests per minute
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                plan_id = plan_request.plan_id
                
                # Check if plan already exists
                if plan_id in self.pending_plans:
                    raise HTTPException(status_code=400, detail="Plan already submitted")
                
                # Create pending plan
                pending_plan = PendingPlan(
                    plan_id=plan_id,
                    task_id=plan_request.task_id,
                    agent_id=plan_request.agent_id,
                    plan_content=plan_request.plan_content,
                    trust_score=plan_request.trust_score,
                    confidence_score=plan_request.confidence_score,
                    evaluation_score=plan_request.evaluation_score,
                    created_at=datetime.now(),
                    metadata=plan_request.metadata
                )
                
                # Calculate priority
                pending_plan.calculate_priority()
                
                # Store pending plan
                self.pending_plans[plan_id] = pending_plan
                
                # Notify WebSocket clients
                background_tasks.add_task(self._notify_websocket_clients, {
                    "type": "plan_submitted",
                    "plan": asdict(pending_plan)
                })
                
                logger.info("Plan submitted for review", 
                           plan_id=plan_id, agent_id=plan_request.agent_id,
                           priority=pending_plan.priority)
                
                return {"status": "submitted", "plan_id": plan_id, "priority": pending_plan.priority}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to submit plan", error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/plans/pending")
        async def get_pending_plans(sort_by: str = "priority", limit: int = 50):
            """Get all pending plans with sorting options."""
            try:
                plans = list(self.pending_plans.values())
                
                # Sort plans
                if sort_by == "priority":
                    plans.sort(key=lambda p: p.priority, reverse=True)
                elif sort_by == "created_at":
                    plans.sort(key=lambda p: p.created_at, reverse=True)
                elif sort_by == "trust_score":
                    plans.sort(key=lambda p: p.trust_score, reverse=True)
                
                # Apply limit
                plans = plans[:limit]
                
                return {"plans": [asdict(plan) for plan in plans]}
                
            except Exception as e:
                logger.error("Failed to get pending plans", error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/plans/{plan_id}")
        async def get_plan(plan_id: str):
            """Get a specific pending plan."""
            try:
                if plan_id not in self.pending_plans:
                    raise HTTPException(status_code=404, detail="Plan not found")
                
                return asdict(self.pending_plans[plan_id])
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to get plan", plan_id=plan_id, error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/api/plans/{plan_id}/approve")
        async def approve_plan(plan_id: str, decision_request: PlanDecisionRequest, 
                             background_tasks: BackgroundTasks):
            """Approve a pending plan with enhanced logging."""
            try:
                if plan_id not in self.pending_plans:
                    raise HTTPException(status_code=404, detail="Plan not found")
                
                plan = self.pending_plans[plan_id]
                
                # Create approval response
                approval = PlanApprovalResponse(
                    approved=True,
                    plan_id=plan_id,
                    decision_timestamp=datetime.now(),
                    user_id=decision_request.user_id,
                    reason=decision_request.reason,
                    modifications=decision_request.modifications
                )
                
                # Add to decision history
                self.decision_history.append(approval)
                
                # Remove from pending plans
                del self.pending_plans[plan_id]
                
                # Update metrics
                self.metrics['total_decisions'] += 1
                
                # Notify WebSocket clients
                background_tasks.add_task(self._notify_websocket_clients, {
                    "type": "plan_decided",
                    "decision": asdict(approval)
                })
                
                logger.info("Plan approved", plan_id=plan_id, user_id=decision_request.user_id, 
                           reason=decision_request.reason)
                
                return approval
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to approve plan", plan_id=plan_id, error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/api/plans/{plan_id}/reject")
        async def reject_plan(plan_id: str, decision_request: PlanDecisionRequest,
                            background_tasks: BackgroundTasks):
            """Reject a pending plan with enhanced logging."""
            try:
                if plan_id not in self.pending_plans:
                    raise HTTPException(status_code=404, detail="Plan not found")
                
                plan = self.pending_plans[plan_id]
                
                # Create rejection response
                rejection = PlanApprovalResponse(
                    approved=False,
                    plan_id=plan_id,
                    decision_timestamp=datetime.now(),
                    user_id=decision_request.user_id,
                    reason=decision_request.reason,
                    modifications=decision_request.modifications
                )
                
                # Add to decision history
                self.decision_history.append(rejection)
                
                # Remove from pending plans
                del self.pending_plans[plan_id]
                
                # Update metrics
                self.metrics['total_decisions'] += 1
                
                # Notify WebSocket clients
                background_tasks.add_task(self._notify_websocket_clients, {
                    "type": "plan_decided",
                    "decision": asdict(rejection)
                })
                
                logger.info("Plan rejected", plan_id=plan_id, user_id=decision_request.user_id, 
                           reason=decision_request.reason)
                
                return rejection
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to reject plan", plan_id=plan_id, error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/api/plans/{plan_id}/fork")
        async def fork_plan(plan_id: str, decision_request: PlanDecisionRequest,
                           background_tasks: BackgroundTasks):
            """Fork a pending plan (request regeneration)."""
            try:
                if plan_id not in self.pending_plans:
                    raise HTTPException(status_code=404, detail="Plan not found")
                
                plan = self.pending_plans[plan_id]
                
                # Create fork response
                fork_response = PlanApprovalResponse(
                    approved=False,  # Fork is essentially a rejection with regeneration
                    plan_id=plan_id,
                    decision_timestamp=datetime.now(),
                    user_id=decision_request.user_id,
                    reason=decision_request.reason or "Plan forked for regeneration",
                    modifications=decision_request.modifications
                )
                
                # Add to decision history
                self.decision_history.append(fork_response)
                
                # Remove from pending plans
                del self.pending_plans[plan_id]
                
                # Update metrics
                self.metrics['total_decisions'] += 1
                
                # Notify WebSocket clients
                background_tasks.add_task(self._notify_websocket_clients, {
                    "type": "plan_decided",
                    "decision": asdict(fork_response)
                })
                
                logger.info("Plan forked", plan_id=plan_id, user_id=decision_request.user_id)
                
                return fork_response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to fork plan", plan_id=plan_id, error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/decisions")
        async def get_decision_history(limit: int = 50, offset: int = 0):
            """Get decision history with pagination."""
            try:
                start_idx = max(0, len(self.decision_history) - limit - offset)
                end_idx = max(0, len(self.decision_history) - offset)
                decisions = self.decision_history[start_idx:end_idx]
                
                return {
                    "decisions": [asdict(decision) for decision in decisions],
                    "total": len(self.decision_history),
                    "limit": limit,
                    "offset": offset
                }
                
            except Exception as e:
                logger.error("Failed to get decision history", error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/stats")
        async def get_stats():
            """Get enhanced server statistics."""
            try:
                stats = self._calculate_stats()
                stats.update({
                    "server_uptime": (datetime.now() - self.metrics['start_time']).total_seconds(),
                    "websocket_connections": len(self.websocket_connections),
                    "average_response_time": self.metrics['average_response_time']
                })
                return stats
                
            except Exception as e:
                logger.error("Failed to get stats", error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        # WebSocket endpoint
        if self.enable_websocket:
            @self.app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                """WebSocket endpoint for real-time updates."""
                await self._handle_websocket_connection(websocket)
        
        @self.app.post("/api/webhook/decision")
        async def decision_webhook(request: Request):
            """Webhook endpoint for external decision notifications."""
            try:
                data = await request.json()
                
                # Process webhook data
                plan_id = data.get("plan_id")
                decision = data.get("decision")
                reason = data.get("reason", "")
                
                if not plan_id or decision not in ["approve", "reject", "fork"]:
                    raise HTTPException(status_code=400, detail="Invalid webhook data")
                
                # Process the decision
                decision_request = PlanDecisionRequest(
                    reason=data.get("reason", ""),
                    user_id=data.get("user_id"),
                    modifications=data.get("modifications", {})
                )
                
                if decision == "approve":
                    return await approve_plan(plan_id, decision_request, BackgroundTasks())
                elif decision == "reject":
                    return await reject_plan(plan_id, decision_request, BackgroundTasks())
                elif decision == "fork":
                    return await fork_plan(plan_id, decision_request, BackgroundTasks())
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Webhook processing failed", error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.delete("/api/plans/{plan_id}")
        async def delete_plan(plan_id: str):
            """Delete a pending plan."""
            try:
                if plan_id not in self.pending_plans:
                    raise HTTPException(status_code=404, detail="Plan not found")
                
                del self.pending_plans[plan_id]
                
                logger.info("Plan deleted", plan_id=plan_id)
                
                return {"status": "deleted", "plan_id": plan_id}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to delete plan", plan_id=plan_id, error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/api/plans/cleanup")
        async def cleanup_old_plans(max_age_hours: int = 24):
            """Clean up old pending plans."""
            try:
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                plans_to_remove = []
                
                for plan_id, plan in self.pending_plans.items():
                    if plan.created_at < cutoff_time:
                        plans_to_remove.append(plan_id)
                
                for plan_id in plans_to_remove:
                    del self.pending_plans[plan_id]
                
                logger.info("Cleaned up old plans", 
                           count=len(plans_to_remove), max_age_hours=max_age_hours)
                
                return {"status": "cleaned", "removed_plans": plans_to_remove}
                
            except Exception as e:
                logger.error("Failed to cleanup plans", error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connection with enhanced features."""
        try:
            await websocket.accept()
            
            # Generate client ID
            client_id = secrets.token_urlsafe(16)
            
            # Create connection object
            connection = WebSocketConnection(
                websocket=websocket,
                client_id=client_id,
                connected_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Add to connections
            with self.connection_lock:
                if len(self.websocket_connections) >= self.max_connections:
                    await websocket.close(code=1008, reason="Too many connections")
                    return
                
                self.websocket_connections[client_id] = connection
                self.metrics['websocket_connections'] = len(self.websocket_connections)
            
            logger.info("WebSocket client connected", client_id=client_id)
            
            # Send initial data
            await websocket.send_json({
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Handle messages
            try:
                while True:
                    data = await websocket.receive_json()
                    connection.last_activity = datetime.now()
                    
                    # Handle subscription messages
                    if data.get("type") == "subscribe":
                        connection.subscriptions.extend(data.get("topics", []))
                        await websocket.send_json({
                            "type": "subscription_confirmed",
                            "topics": data.get("topics", [])
                        })
                    
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected", client_id=client_id)
            except Exception as e:
                logger.error("WebSocket error", client_id=client_id, error=str(e))
            finally:
                # Remove connection
                with self.connection_lock:
                    if client_id in self.websocket_connections:
                        del self.websocket_connections[client_id]
                        self.metrics['websocket_connections'] = len(self.websocket_connections)
                
        except Exception as e:
            logger.error("Failed to handle WebSocket connection", error=str(e))
    
    async def _notify_websocket_clients(self, message: Dict[str, Any]):
        """Notify all WebSocket clients of updates."""
        if not self.enable_websocket:
            return
        
        disconnected_clients = []
        
        for client_id, connection in self.websocket_connections.items():
            try:
                await connection.websocket.send_json(message)
            except Exception as e:
                logger.warn("Failed to send message to WebSocket client", 
                           client_id=client_id, error=str(e))
                disconnected_clients.append(client_id)
        
        # Remove disconnected clients
        if disconnected_clients:
            with self.connection_lock:
                for client_id in disconnected_clients:
                    if client_id in self.websocket_connections:
                        del self.websocket_connections[client_id]
                self.metrics['websocket_connections'] = len(self.websocket_connections)
    
    def _check_rate_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check rate limiting for a given key."""
        now = time.time()
        
        with self.rate_limit_lock:
            # Clean old entries
            self.rate_limit_store[key] = [
                timestamp for timestamp in self.rate_limit_store[key]
                if now - timestamp < window_seconds
            ]
            
            # Check if limit exceeded
            if len(self.rate_limit_store[key]) >= max_requests:
                return False
            
            # Add current request
            self.rate_limit_store[key].append(now)
            return True
    
    def _calculate_stats(self) -> Dict[str, Any]:
        """Calculate enhanced statistics."""
        try:
            recent_decisions = self.decision_history[-10:]
            recent_approvals = [d for d in recent_decisions if d.approved]
            recent_rejections = [d for d in recent_decisions if not d.approved]
            
            # Calculate average scores
            if self.pending_plans:
                avg_trust = sum(p.trust_score for p in self.pending_plans.values()) / len(self.pending_plans)
                avg_confidence = sum(p.confidence_score for p in self.pending_plans.values()) / len(self.pending_plans)
            else:
                avg_trust = avg_confidence = 0.0
            
            return {
                "pending_plans_count": len(self.pending_plans),
                "total_decisions": len(self.decision_history),
                "recent_approvals": len(recent_approvals),
                "recent_rejections": len(recent_rejections),
                "average_trust_score": round(avg_trust, 3),
                "average_confidence_score": round(avg_confidence, 3),
                "high_priority_plans": len([p for p in self.pending_plans.values() if p.priority >= 0.7]),
                "medium_priority_plans": len([p for p in self.pending_plans.values() if 0.4 <= p.priority < 0.7]),
                "low_priority_plans": len([p for p in self.pending_plans.values() if p.priority < 0.4])
            }
        except Exception as e:
            logger.error("Failed to calculate stats", error=str(e))
            return {}
    
    async def _background_cleanup(self):
        """Background task for cleanup operations."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up old decisions (keep last 1000)
                if len(self.decision_history) > 1000:
                    self.decision_history = self.decision_history[-1000:]
                
                # Clean up inactive WebSocket connections
                now = datetime.now()
                inactive_clients = []
                
                for client_id, connection in self.websocket_connections.items():
                    if (now - connection.last_activity).total_seconds() > 3600:  # 1 hour
                        inactive_clients.append(client_id)
                
                for client_id in inactive_clients:
                    try:
                        await self.websocket_connections[client_id].websocket.close()
                    except:
                        pass
                    del self.websocket_connections[client_id]
                
                if inactive_clients:
                    logger.info("Cleaned up inactive WebSocket connections", count=len(inactive_clients))
                
            except Exception as e:
                logger.error("Background cleanup failed", error=str(e))
    
    async def _background_metrics(self):
        """Background task for metrics collection."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Update average response time (simplified)
                if self.metrics['total_requests'] > 0:
                    # This would be updated with actual response times
                    pass
                
                logger.info("Server metrics", metrics=self.metrics)
                
            except Exception as e:
                logger.error("Background metrics failed", error=str(e))
    
    async def submit_plan_for_review(self, plan_request: PlanApprovalRequest) -> bool:
        """Submit a plan for human review (internal method)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://{self.host}:{self.port}/api/plans/submit",
                    json=plan_request.dict()
                )
                return response.status_code == 200
                
        except Exception as e:
            logger.error("Failed to submit plan for review", error=str(e))
            return False
    
    async def wait_for_decision(self, plan_id: str, timeout: int = 300) -> Optional[PlanApprovalResponse]:
        """Wait for a human decision on a plan."""
        try:
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                # Check if plan is still pending
                if plan_id not in self.pending_plans:
                    # Plan was decided, find the decision
                    for decision in reversed(self.decision_history):
                        if decision.plan_id == plan_id:
                            return decision
                    
                    # Decision not found
                    return None
                
                # Wait before checking again
                await asyncio.sleep(5)
            
            # Timeout reached
            return None
            
        except Exception as e:
            logger.error("Failed to wait for decision", plan_id=plan_id, error=str(e))
            return None
    
    def get_pending_plan(self, plan_id: str) -> Optional[PendingPlan]:
        """Get a pending plan by ID."""
        return self.pending_plans.get(plan_id)
    
    def get_all_pending_plans(self) -> List[PendingPlan]:
        """Get all pending plans."""
        return list(self.pending_plans.values())
    
    def get_decision_history(self, limit: int = 50) -> List[PlanApprovalResponse]:
        """Get decision history."""
        return self.decision_history[-limit:]
    
    def run(self):
        """Run the webhook server."""
        logger.info("Starting webhook server", host=self.host, port=self.port)
        uvicorn.run(self.app, host=self.host, port=self.port)
    
    async def run_async(self):
        """Run the webhook server asynchronously."""
        try:
            config = uvicorn.Config(self.app, host=self.host, port=self.port)
            server = uvicorn.Server(config)
            await server.serve()
        except OSError as e:
            if "Address already in use" in str(e) or "only one usage of each socket address" in str(e):
                logger.warning(f"Port {self.port} is already in use, skipping webhook server startup")
                return
            else:
                raise
        except SystemExit:
            logger.warning(f"Port {self.port} is already in use, skipping webhook server startup")
            return


# Create default HTML template
def create_default_templates():
    """Create default HTML templates for the UI."""
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Dashboard template with enhanced features
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sophie HITL Dashboard</title>
    <link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sophie HITL Dashboard</h1>
            <p>Human-in-the-Loop Decision Interface</p>
            <div class="connection-status">
                <span id="connection-status" class="status-indicator offline"></span>
                <span id="last-refresh">Never</span>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="pending-plans-count">{{ stats.pending_plans_count }}</div>
                <div class="stat-label">Pending Plans</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="total-decisions">{{ stats.total_decisions }}</div>
                <div class="stat-label">Total Decisions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="recent-approvals">{{ stats.recent_approvals }}</div>
                <div class="stat-label">Recent Approvals</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="recent-rejections">{{ stats.recent_rejections }}</div>
                <div class="stat-label">Recent Rejections</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Pending Plans</h2>
            <div class="controls">
                <input type="text" id="search-input" placeholder="Search plans..." class="search-input">
                <select class="filter-select" name="priority">
                    <option value="all">All Priorities</option>
                    <option value="high">High Priority</option>
                    <option value="medium">Medium Priority</option>
                    <option value="low">Low Priority</option>
                </select>
                <button id="refresh-btn" class="btn">🔄 Refresh</button>
                <button id="cleanup-btn" class="btn">🧹 Cleanup</button>
            </div>
            <div id="pending-plans-container">
                {% if pending_plans %}
                    {% for plan in pending_plans %}
                    <div class="plan-card" data-plan-id="{{ plan.plan_id }}" data-priority="{{ 'high' if plan.priority >= 0.7 else 'medium' if plan.priority >= 0.4 else 'low' }}">
                        <div class="plan-header">
                            <div class="plan-title">
                                <span class="priority-indicator priority-{{ 'high' if plan.priority >= 0.7 else 'medium' if plan.priority >= 0.4 else 'low' }}"></span>
                                Plan: {{ plan.plan_id }}
                            </div>
                            <div class="plan-scores">
                                <div class="score {{ 'high' if plan.trust_score >= 0.7 else 'medium' if plan.trust_score >= 0.4 else 'low' }}" title="Trust Score: {{ "%.2f"|format(plan.trust_score) }}">
                                    Trust: {{ "%.2f"|format(plan.trust_score) }}
                                </div>
                                <div class="score {{ 'high' if plan.confidence_score >= 0.7 else 'medium' if plan.confidence_score >= 0.4 else 'low' }}" title="Confidence Score: {{ "%.2f"|format(plan.confidence_score) }}">
                                    Confidence: {{ "%.2f"|format(plan.confidence_score) }}
                                </div>
                                {% if plan.evaluation_score %}
                                <div class="score {{ 'high' if plan.evaluation_score >= 0.7 else 'medium' if plan.evaluation_score >= 0.4 else 'low' }}" title="Evaluation Score: {{ "%.2f"|format(plan.evaluation_score) }}">
                                    Eval: {{ "%.2f"|format(plan.evaluation_score) }}
                                </div>
                                {% endif %}
                            </div>
                        </div>
                        <div class="plan-content">
                            <strong>Task:</strong> {{ plan.task_id }}<br>
                            <strong>Agent:</strong> {{ plan.agent_id }}<br>
                            <strong>Content:</strong><br>
                            <div class="plan-content-text" data-content="{{ plan.plan_content }}">
                                {{ plan.plan_content[:200] }}{% if plan.plan_content|length > 200 %}...{% endif %}
                            </div>
                            {% if plan.plan_content|length > 200 %}
                            <button class="btn btn-expand">Show More</button>
                            {% endif %}
                        </div>
                        <div class="plan-actions">
                            <button class="btn btn-approve" data-plan-id="{{ plan.plan_id }}" aria-label="Approve plan {{ plan.plan_id }}">
                                ✅ Approve
                            </button>
                            <button class="btn btn-reject" data-plan-id="{{ plan.plan_id }}" aria-label="Reject plan {{ plan.plan_id }}">
                                ❌ Reject
                            </button>
                            <button class="btn btn-fork" data-plan-id="{{ plan.plan_id }}" aria-label="Fork plan {{ plan.plan_id }}">
                                ♻️ Fork
                            </button>
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="empty-state">
                        <h3>No pending plans</h3>
                        <p>All plans have been processed or no plans require review.</p>
                    </div>
                {% endif %}
            </div>
        </div>
        
        <div class="section">
            <h2>Recent Decisions</h2>
            <div id="decisions-container">
                {% if recent_decisions %}
                    {% for decision in recent_decisions %}
                    <div class="decision-item" data-decision-id="{{ decision.plan_id }}">
                        <div class="decision-info">
                            <div class="decision-title">{{ decision.plan_id }}</div>
                            <div class="decision-meta">
                                {{ decision.decision_timestamp.strftime('%Y-%m-%d %H:%M:%S') }}
                                {% if decision.user_id %}
                                by {{ decision.user_id }}
                                {% endif %}
                            </div>
                            {% if decision.reason %}
                            <div class="decision-reason">{{ decision.reason }}</div>
                            {% endif %}
                        </div>
                        <div class="decision-status {{ 'approved' if decision.approved else 'rejected' }}">
                            {{ 'Approved' if decision.approved else 'Rejected' }}
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="empty-state">
                        <h3>No decisions yet</h3>
                        <p>No human decisions have been recorded.</p>
                    </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <div id="loading-overlay" style="display: none;">
        <div class="loading"></div>
    </div>
    
    <script src="/static/dashboard.js"></script>
</body>
</html>
    """
    
    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(dashboard_html)
    
    logger.info("Default templates created", templates_dir=str(templates_dir))


if __name__ == "__main__":
    # Create default templates
    create_default_templates()
    
    # Create and run the server
    server = WebhookServer()
    server.run() 