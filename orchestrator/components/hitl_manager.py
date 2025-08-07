"""
HITL Manager Component

Handles human-in-the-loop intervention and policy evaluation.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from governance.policy_engine import PolicyEngine, PolicyContext, PolicyDecision
from ui.webhook_server import WebhookServer, PlanApprovalRequest
from memory.trust_tracker import TrustTracker, TrustEventType
from governance.audit_log import AuditLog, AuditEventType
from ..models.orchestrator_config import OrchestratorConfig

logger = structlog.get_logger()


class HITLManager:
    """Handles human-in-the-loop intervention and policy evaluation."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.policy_engine = None
        self.hitl_server = None
        self.trust_tracker = None
        self.audit_log = None
    
    def set_dependencies(self, policy_engine: PolicyEngine, trust_tracker: TrustTracker, 
                        audit_log: AuditLog):
        """Set required dependencies."""
        self.policy_engine = policy_engine
        self.trust_tracker = trust_tracker
        self.audit_log = audit_log
    
    async def initialize(self):
        """Initialize HITL components."""
        try:
            if self.config.hitl_enabled:
                self.hitl_server = WebhookServer()
                # Start HITL server in background
                asyncio.create_task(self.hitl_server.run_async())
                logger.info("HITL server initialized")
            
        except Exception as e:
            logger.error("Failed to initialize HITL manager", error=str(e))
            raise
    
    async def check_interventions(self, evaluation_results: List[Dict[str, Any]], 
                                generation: int) -> List[Dict[str, Any]]:
        """Check for HITL intervention based on evaluation results."""
        try:
            if not self.config.hitl_enabled:
                return []
            
            logger.info("Checking HITL intervention")
            
            interventions = []
            
            for evaluation in evaluation_results:
                agent_id = evaluation.get("prover_result", {}).get("agent_id")
                overall_score = evaluation.get("overall_score", 0.0)
                
                # Create policy context
                context = PolicyContext(
                    agent_id=agent_id,
                    agent_type="prover",
                    action="generate_solution",
                    content=evaluation.get("prover_result", {}).get("result", {}).get("best_variant", {}).get("content", ""),
                    trust_score=await self._get_agent_trust_score(agent_id),
                    confidence_score=evaluation.get("prover_result", {}).get("confidence_score", 0.0),
                    iteration_count=generation,
                    timestamp=datetime.now(),
                    additional_context={
                        "overall_score": overall_score,
                        "generation": generation
                    }
                )
                
                # Evaluate policies
                policy_result = await self.policy_engine.evaluate_action(context)
                
                if policy_result.decision == PolicyDecision.REQUIRE_HUMAN_REVIEW:
                    # Submit for HITL review
                    hitl_result = await self._submit_for_hitl_review(evaluation, policy_result)
                    
                    if hitl_result:
                        interventions.append({
                            "agent_id": agent_id,
                            "type": "hitl_review",
                            "policy_result": policy_result.__dict__,
                            "hitl_result": hitl_result,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Apply HITL decision
                        await self._apply_hitl_decision(agent_id, hitl_result)
                
                elif policy_result.decision == PolicyDecision.BLOCK:
                    interventions.append({
                        "agent_id": agent_id,
                        "type": "policy_block",
                        "policy_result": policy_result.__dict__,
                        "timestamp": datetime.now().isoformat()
                    })
            
            logger.info("HITL intervention check completed", interventions=len(interventions))
            return interventions
            
        except Exception as e:
            logger.error("HITL intervention check failed", error=str(e))
            return []
    
    async def _submit_for_hitl_review(self, evaluation: Dict[str, Any], 
                                     policy_result: Any) -> Optional[Dict[str, Any]]:
        """Submit a solution for HITL review."""
        try:
            if not self.hitl_server:
                return None
            
            prover_result = evaluation.get("prover_result", {})
            best_variant = prover_result.get("result", {}).get("best_variant", {})
            
            # Create approval request
            approval_request = PlanApprovalRequest(
                plan_id=f"plan_{prover_result['agent_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                task_id=prover_result.get("task", ""),
                agent_id=prover_result["agent_id"],
                plan_content=best_variant.get("content", ""),
                trust_score=await self._get_agent_trust_score(prover_result["agent_id"]),
                confidence_score=prover_result.get("confidence_score", 0.0),
                evaluation_score=evaluation.get("overall_score", 0.0),
                metadata={
                    "policy_result": policy_result.__dict__
                }
            )
            
            # Submit for review
            success = await self.hitl_server.submit_plan_for_review(approval_request)
            
            if success:
                # Wait for decision
                decision = await self.hitl_server.wait_for_decision(
                    approval_request.plan_id,
                    timeout=self.config.hitl_timeout
                )
                
                if decision:
                    return decision.__dict__
            
            return None
            
        except Exception as e:
            logger.error("Failed to submit for HITL review", error=str(e))
            return None
    
    async def _apply_hitl_decision(self, agent_id: str, hitl_result: Dict[str, Any]):
        """Apply HITL decision to agent trust score."""
        try:
            approved = hitl_result.get("approved", False)
            reason = hitl_result.get("reason", "")
            
            if approved:
                # Positive trust adjustment
                await self.trust_tracker.record_event(
                    agent_id=agent_id,
                    event_type=TrustEventType.HUMAN_APPROVAL,
                    adjustment=0.15,
                    context={"reason": reason},
                    description=f"Human approval: {reason}"
                )
            else:
                # Negative trust adjustment
                await self.trust_tracker.record_event(
                    agent_id=agent_id,
                    event_type=TrustEventType.HUMAN_REJECTION,
                    adjustment=-0.25,
                    context={"reason": reason},
                    description=f"Human rejection: {reason}"
                )
            
            # Log HITL decision
            self.audit_log.log_event(
                event_type=AuditEventType.HUMAN_INTERVENTION,
                description=f"HITL decision for agent {agent_id}: {'Approved' if approved else 'Rejected'}",
                details={
                    "agent_id": agent_id,
                    "approved": approved,
                    "reason": reason
                },
                severity="info"
            )
            
        except Exception as e:
            logger.error("Failed to apply HITL decision", agent_id=agent_id, error=str(e))
    
    async def _get_agent_trust_score(self, agent_id: str) -> float:
        """Get the current trust score for an agent."""
        try:
            if not self.trust_tracker:
                return 0.5
            
            trust_data = await self.trust_tracker.get_trust_score(agent_id)
            return trust_data.score if trust_data else 0.5
        except Exception as e:
            logger.error("Failed to get agent trust score", agent_id=agent_id, error=str(e))
            return 0.5 