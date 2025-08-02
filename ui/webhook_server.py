from typing import Dict, Any, List, Optional, Tuple
import asyncio
import structlog
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path
import httpx

logger = structlog.get_logger()


# Pydantic models for request/response
class PlanApprovalRequest(BaseModel):
    plan_id: str
    task_id: str
    agent_id: str
    plan_content: str
    trust_score: float
    confidence_score: float
    evaluation_score: Optional[float] = None
    metadata: Dict[str, Any] = {}


class PlanApprovalResponse(BaseModel):
    approved: bool
    plan_id: str
    decision_timestamp: datetime
    user_id: Optional[str] = None
    reason: Optional[str] = None
    modifications: Dict[str, Any] = {}


class PendingPlan(BaseModel):
    plan_id: str
    task_id: str
    agent_id: str
    plan_content: str
    trust_score: float
    confidence_score: float
    evaluation_score: Optional[float] = None
    created_at: datetime
    metadata: Dict[str, Any] = {}


class WebhookServer:
    """Minimal FastAPI app for HITL approve/reject buttons."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Sophie HITL Webhook Server", version="1.0.0")
        
        # Storage for pending plans and decisions
        self.pending_plans: Dict[str, PendingPlan] = {}
        self.decision_history: List[PlanApprovalResponse] = []
        
        # Setup static files and templates
        self._setup_static_files()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Webhook server initialized", host=host, port=port)
    
    def _setup_static_files(self):
        """Setup static files and templates."""
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
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            """Main dashboard page."""
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "pending_plans": list(self.pending_plans.values()),
                "recent_decisions": self.decision_history[-10:]
            })
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.post("/api/plans/submit")
        async def submit_plan(plan_request: PlanApprovalRequest):
            """Submit a plan for human review."""
            try:
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
                
                # Store pending plan
                self.pending_plans[plan_id] = pending_plan
                
                logger.info("Plan submitted for review", plan_id=plan_id, agent_id=plan_request.agent_id)
                
                return {"status": "submitted", "plan_id": plan_id}
                
            except Exception as e:
                logger.error("Failed to submit plan", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/plans/pending")
        async def get_pending_plans():
            """Get all pending plans."""
            return {"plans": list(self.pending_plans.values())}
        
        @self.app.get("/api/plans/{plan_id}")
        async def get_plan(plan_id: str):
            """Get a specific pending plan."""
            if plan_id not in self.pending_plans:
                raise HTTPException(status_code=404, detail="Plan not found")
            
            return self.pending_plans[plan_id]
        
        @self.app.post("/api/plans/{plan_id}/approve")
        async def approve_plan(plan_id: str, reason: str = "", user_id: str = None):
            """Approve a pending plan."""
            try:
                if plan_id not in self.pending_plans:
                    raise HTTPException(status_code=404, detail="Plan not found")
                
                plan = self.pending_plans[plan_id]
                
                # Create approval response
                approval = PlanApprovalResponse(
                    approved=True,
                    plan_id=plan_id,
                    decision_timestamp=datetime.now(),
                    user_id=user_id,
                    reason=reason,
                    modifications={}
                )
                
                # Add to decision history
                self.decision_history.append(approval)
                
                # Remove from pending plans
                del self.pending_plans[plan_id]
                
                logger.info("Plan approved", plan_id=plan_id, user_id=user_id)
                
                return approval
                
            except Exception as e:
                logger.error("Failed to approve plan", plan_id=plan_id, error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/plans/{plan_id}/reject")
        async def reject_plan(plan_id: str, reason: str = "", user_id: str = None):
            """Reject a pending plan."""
            try:
                if plan_id not in self.pending_plans:
                    raise HTTPException(status_code=404, detail="Plan not found")
                
                plan = self.pending_plans[plan_id]
                
                # Create rejection response
                rejection = PlanApprovalResponse(
                    approved=False,
                    plan_id=plan_id,
                    decision_timestamp=datetime.now(),
                    user_id=user_id,
                    reason=reason,
                    modifications={}
                )
                
                # Add to decision history
                self.decision_history.append(rejection)
                
                # Remove from pending plans
                del self.pending_plans[plan_id]
                
                logger.info("Plan rejected", plan_id=plan_id, user_id=user_id, reason=reason)
                
                return rejection
                
            except Exception as e:
                logger.error("Failed to reject plan", plan_id=plan_id, error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/plans/{plan_id}/fork")
        async def fork_plan(plan_id: str, modifications: Dict[str, Any] = None, user_id: str = None):
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
                    user_id=user_id,
                    reason="Plan forked for regeneration",
                    modifications=modifications or {}
                )
                
                # Add to decision history
                self.decision_history.append(fork_response)
                
                # Remove from pending plans
                del self.pending_plans[plan_id]
                
                logger.info("Plan forked", plan_id=plan_id, user_id=user_id)
                
                return fork_response
                
            except Exception as e:
                logger.error("Failed to fork plan", plan_id=plan_id, error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/decisions")
        async def get_decision_history(limit: int = 50):
            """Get decision history."""
            return {"decisions": self.decision_history[-limit:]}
        
        @self.app.get("/api/stats")
        async def get_stats():
            """Get server statistics."""
            return {
                "pending_plans_count": len(self.pending_plans),
                "total_decisions": len(self.decision_history),
                "recent_approvals": len([d for d in self.decision_history[-10:] if d.approved]),
                "recent_rejections": len([d for d in self.decision_history[-10:] if not d.approved]),
                "server_uptime": "N/A"  # Could track actual uptime
            }
        
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
                if decision == "approve":
                    return await self.approve_plan(plan_id, reason)
                elif decision == "reject":
                    return await self.reject_plan(plan_id, reason)
                elif decision == "fork":
                    return await self.fork_plan(plan_id, data.get("modifications", {}))
                
            except Exception as e:
                logger.error("Webhook processing failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/plans/{plan_id}")
        async def delete_plan(plan_id: str):
            """Delete a pending plan."""
            try:
                if plan_id not in self.pending_plans:
                    raise HTTPException(status_code=404, detail="Plan not found")
                
                del self.pending_plans[plan_id]
                
                logger.info("Plan deleted", plan_id=plan_id)
                
                return {"status": "deleted", "plan_id": plan_id}
                
            except Exception as e:
                logger.error("Failed to delete plan", plan_id=plan_id, error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
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
                
                logger.info("Cleaned up old plans", count=len(plans_to_remove), max_age_hours=max_age_hours)
                
                return {"status": "cleaned", "removed_plans": plans_to_remove}
                
            except Exception as e:
                logger.error("Failed to cleanup plans", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
    
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
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()


# Create default HTML template
def create_default_templates():
    """Create default HTML templates for the UI."""
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Dashboard template
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sophie HITL Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .plans-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .plan-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background: #f9f9f9;
        }
        .plan-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .plan-title {
            font-weight: bold;
            color: #2c3e50;
        }
        .plan-scores {
            display: flex;
            gap: 15px;
            font-size: 0.9em;
        }
        .score {
            padding: 4px 8px;
            border-radius: 4px;
            background: #ecf0f1;
        }
        .score.high {
            background: #d5f4e6;
            color: #27ae60;
        }
        .score.medium {
            background: #fef9e7;
            color: #f39c12;
        }
        .score.low {
            background: #fadbd8;
            color: #e74c3c;
        }
        .plan-content {
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
            border-left: 4px solid #3498db;
        }
        .plan-actions {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .btn-approve {
            background: #27ae60;
            color: white;
        }
        .btn-approve:hover {
            background: #229954;
        }
        .btn-reject {
            background: #e74c3c;
            color: white;
        }
        .btn-reject:hover {
            background: #c0392b;
        }
        .btn-fork {
            background: #f39c12;
            color: white;
        }
        .btn-fork:hover {
            background: #e67e22;
        }
        .decisions-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .decision-item {
            padding: 10px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .decision-status {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .decision-status.approved {
            background: #d5f4e6;
            color: #27ae60;
        }
        .decision-status.rejected {
            background: #fadbd8;
            color: #e74c3c;
        }
        .decision-status.forked {
            background: #fef9e7;
            color: #f39c12;
        }
        .no-plans {
            text-align: center;
            color: #7f8c8d;
            padding: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sophie HITL Dashboard</h1>
            <p>Human-in-the-Loop Decision Interface</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{{ pending_plans|length }}</div>
                <div>Pending Plans</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ recent_decisions|length }}</div>
                <div>Recent Decisions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ recent_decisions|selectattr('approved')|list|length }}</div>
                <div>Approvals</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ recent_decisions|rejectattr('approved')|list|length }}</div>
                <div>Rejections</div>
            </div>
        </div>
        
        <div class="plans-section">
            <h2>Pending Plans</h2>
            {% if pending_plans %}
                {% for plan in pending_plans %}
                <div class="plan-card">
                    <div class="plan-header">
                        <div class="plan-title">Plan: {{ plan.plan_id }}</div>
                        <div class="plan-scores">
                            <div class="score {% if plan.trust_score >= 0.7 %}high{% elif plan.trust_score >= 0.4 %}medium{% else %}low{% endif %}">
                                Trust: {{ "%.2f"|format(plan.trust_score) }}
                            </div>
                            <div class="score {% if plan.confidence_score >= 0.7 %}high{% elif plan.confidence_score >= 0.4 %}medium{% else %}low{% endif %}">
                                Confidence: {{ "%.2f"|format(plan.confidence_score) }}
                            </div>
                            {% if plan.evaluation_score %}
                            <div class="score {% if plan.evaluation_score >= 0.7 %}high{% elif plan.evaluation_score >= 0.4 %}medium{% else %}low{% endif %}">
                                Eval: {{ "%.2f"|format(plan.evaluation_score) }}
                            </div>
                            {% endif %}
                        </div>
                    </div>
                    <div class="plan-content">
                        <strong>Task:</strong> {{ plan.task_id }}<br>
                        <strong>Agent:</strong> {{ plan.agent_id }}<br>
                        <strong>Content:</strong><br>
                        <div style="margin-top: 10px; white-space: pre-wrap;">{{ plan.plan_content }}</div>
                    </div>
                    <div class="plan-actions">
                        <button class="btn btn-approve" onclick="approvePlan('{{ plan.plan_id }}')">✅ Approve</button>
                        <button class="btn btn-reject" onclick="rejectPlan('{{ plan.plan_id }}')">❌ Reject</button>
                        <button class="btn btn-fork" onclick="forkPlan('{{ plan.plan_id }}')">♻️ Fork</button>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div class="no-plans">
                    <h3>No pending plans</h3>
                    <p>All plans have been processed or no plans require review.</p>
                </div>
            {% endif %}
        </div>
        
        <div class="decisions-section">
            <h2>Recent Decisions</h2>
            {% if recent_decisions %}
                {% for decision in recent_decisions %}
                <div class="decision-item">
                    <div>
                        <strong>{{ decision.plan_id }}</strong>
                        <div style="font-size: 0.9em; color: #7f8c8d;">
                            {{ decision.decision_timestamp.strftime('%Y-%m-%d %H:%M:%S') }}
                            {% if decision.user_id %}
                            by {{ decision.user_id }}
                            {% endif %}
                        </div>
                        {% if decision.reason %}
                        <div style="font-size: 0.9em; margin-top: 5px;">{{ decision.reason }}</div>
                        {% endif %}
                    </div>
                    <div class="decision-status {% if decision.approved %}approved{% else %}rejected{% endif %}">
                        {{ 'Approved' if decision.approved else 'Rejected' }}
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div class="no-plans">
                    <h3>No decisions yet</h3>
                    <p>No human decisions have been recorded.</p>
                </div>
            {% endif %}
        </div>
    </div>
    
    <script>
        async function approvePlan(planId) {
            try {
                const response = await fetch(`/api/plans/${planId}/approve`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        reason: prompt('Reason for approval (optional):') || ''
                    })
                });
                
                if (response.ok) {
                    alert('Plan approved successfully!');
                    location.reload();
                } else {
                    alert('Failed to approve plan');
                }
            } catch (error) {
                alert('Error approving plan: ' + error.message);
            }
        }
        
        async function rejectPlan(planId) {
            try {
                const reason = prompt('Reason for rejection:');
                if (reason === null) return; // User cancelled
                
                const response = await fetch(`/api/plans/${planId}/reject`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        reason: reason
                    })
                });
                
                if (response.ok) {
                    alert('Plan rejected successfully!');
                    location.reload();
                } else {
                    alert('Failed to reject plan');
                }
            } catch (error) {
                alert('Error rejecting plan: ' + error.message);
            }
        }
        
        async function forkPlan(planId) {
            try {
                const response = await fetch(`/api/plans/${planId}/fork`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        reason: 'Plan forked for regeneration'
                    })
                });
                
                if (response.ok) {
                    alert('Plan forked successfully!');
                    location.reload();
                } else {
                    alert('Failed to fork plan');
                }
            } catch (error) {
                alert('Error forking plan: ' + error.message);
            }
        }
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            location.reload();
        }, 30000);
    </script>
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