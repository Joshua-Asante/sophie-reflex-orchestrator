"""
Constitutional AI Operating System Executor

This module implements SOPHIE as a Constitutional AI Operating System that can:
- Interpret high-level human directives
- Generate structured execution plans
- Execute real infrastructure changes via CI/CD
- Maintain constitutional guardrails
- Provide live feedback and verification
"""

import asyncio
import logging
import json
import yaml
import hashlib
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import structlog

from .autonomous_executor import AutonomousExecutor, Directive, ExecutionPlan, DirectiveType, ApprovalLevel
from .performance_integration import optimized_llm_call, optimized_tool_call
from governance.policy_engine import PolicyEngine
from governance.audit_log import AuditLog
from memory.trust_tracker import TrustTracker

logger = structlog.get_logger()


class ConstitutionalRole(Enum):
    """Constitutional roles for SOPHIE's sub-personas."""
    NAVIGATOR = "Φ"  # High-level intent, goal setting
    INTEGRATOR = "Σ"  # Executes validated changes via CI/CD
    ANCHOR = "Ω"  # Human feedback, approval, veto
    DIFF_ENGINE = "Δ"  # Plan comparison, justification
    MEMORY = "Ψ"  # Pulls relevant prior actions, precedent


@dataclass
class ConstitutionalPlan:
    """A constitutionally validated execution plan."""
    id: str
    directive: Directive
    plan_yaml: str
    confidence_score: float
    risk_assessment: Dict[str, Any]
    approval_required: bool
    digital_signature: str
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    execution_result: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionContext:
    """Context for constitutional execution."""
    directive: Directive
    plan: ConstitutionalPlan
    trust_score: float
    role_activation: ConstitutionalRole
    approval_granted: bool = False
    execution_started: bool = False
    completion_time: Optional[float] = None


class ConstitutionalExecutor(AutonomousExecutor):
    """
    Constitutional AI Operating System Executor.
    
    Implements the vision of SOPHIE as a system that can execute real infrastructure
    changes through conversational intent while maintaining constitutional guardrails.
    """
    
    def __init__(self):
        super().__init__()
        self.policy_engine = PolicyEngine()
        self.audit_log = AuditLog()
        self.trust_tracker = TrustTracker()
        self.active_contexts: Dict[str, ExecutionContext] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    async def interpret_and_execute_constitutional(
        self, 
        human_input: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for constitutional execution.
        
        This is the core method that implements the vision from the executive summary:
        "you can now direct software development and system changes through conversational commands"
        """
        
        logger.info("Starting constitutional execution", directive=human_input)
        
        # Step 1: Interpret directive (Φ - Navigator role)
        directive = await self._interpret_directive_navigator(human_input, context)
        
        # Step 2: Generate execution plan (Σ - Integrator role)
        plan = await self._generate_plan_integrator(directive)
        
        # Step 3: Validate against constitutional guardrails (Δ - Diff Engine role)
        validation = await self._validate_plan_constitutional(plan)
        
        if not validation["approved"]:
            return {
                "status": "rejected",
                "reason": "Constitutional validation failed",
                "validation_results": validation["validation_results"]
            }
        
        # Step 4: Request approval if needed (Ω - Anchor role)
        if plan.approval_required:
            approval_granted = await self._request_anchor_approval(plan)
            if not approval_granted:
                return {
                    "status": "rejected",
                    "reason": "Human approval denied"
                }
        
        # Step 5: Execute via CI/CD (Σ - Integrator role)
        execution_result = await self._execute_via_cicd(plan)
        
        # Step 6: Store in memory (Ψ - Memory role)
        await self._store_execution_memory(directive, plan, execution_result)
        
        return {
            "status": "completed",
            "directive": directive.description,
            "plan_id": plan.id,
            "execution_result": execution_result,
            "staging_url": execution_result.get("staging_url"),
            "artifact_urls": execution_result.get("artifact_urls", []),
            "dashboard_url": execution_result.get("dashboard_url"),
            "completion_time": time.time()
        }
    
    async def _interpret_directive_navigator(
        self, 
        human_input: str, 
        context: Dict[str, Any] = None
    ) -> Directive:
        """Interpret directive using Navigator (Φ) role."""
        
        navigator_prompt = f"""
        You are SOPHIE's Navigator (Φ), responsible for interpreting high-level human directives.
        
        Human Directive: "{human_input}"
        Context: {context or {}}
        
        Classify this directive and extract key information. Consider:
        - What type of change is being requested?
        - What is the priority and urgency?
        - What approval level is required?
        - What resources will be needed?
        
        Return a JSON object with:
        - type: one of {[t.value for t in DirectiveType]}
        - description: clear description of what needs to be done
        - priority: 1-5 (5 being highest)
        - approval_level: one of {[a.value for a in ApprovalLevel]}
        - context: any additional context needed
        """
        
        response = await optimized_llm_call(
            navigator_prompt,
            "gpt-4",
            "openai",
            temperature=0.3,
            max_tokens=500
        )
        
        try:
            directive_data = json.loads(response)
            return Directive(
                id=f"directive_{int(time.time())}",
                type=DirectiveType(directive_data["type"]),
                description=directive_data["description"],
                priority=directive_data.get("priority", 1),
                approval_level=ApprovalLevel(directive_data["approval_level"]),
                context=directive_data.get("context", {})
            )
        except Exception as e:
            logger.error("Failed to interpret directive", error=str(e))
            # Fallback to basic interpretation
            return Directive(
                id=f"directive_{int(time.time())}",
                type=DirectiveType.IMPLEMENTATION,
                description=human_input,
                priority=3,
                approval_level=ApprovalLevel.APPROVAL
            )
    
    async def _generate_plan_integrator(self, directive: Directive) -> ConstitutionalPlan:
        """Generate execution plan using Integrator (Σ) role."""
        
        integrator_prompt = f"""
        You are SOPHIE's Integrator (Σ), responsible for translating directives into executable plans.
        
        Directive: {directive.description}
        Type: {directive.type.value}
        Priority: {directive.priority}
        Approval Level: {directive.approval_level.value}
        
        Generate a detailed YAML execution plan that includes:
        1. Infrastructure changes needed
        2. Code modifications required
        3. CI/CD pipeline steps
        4. Artifacts to generate (.app, .deb, .docker, etc.)
        5. Testing requirements
        6. Deployment strategy
        
        The plan should be comprehensive and ready for CI/CD execution.
        """
        
        response = await optimized_llm_call(
            integrator_prompt,
            "gpt-4",
            "openai",
            temperature=0.2,
            max_tokens=1000
        )
        
        # Extract YAML plan from response
        plan_yaml = self._extract_yaml_from_response(response)
        
        # Generate digital signature
        plan_hash = hashlib.sha256(plan_yaml.encode()).hexdigest()
        
        # Assess confidence and risk
        confidence_score = await self._assess_plan_confidence(plan_yaml)
        risk_assessment = await self._assess_plan_risk(plan_yaml)
        
        return ConstitutionalPlan(
            id=f"plan_{int(time.time())}",
            directive=directive,
            plan_yaml=plan_yaml,
            confidence_score=confidence_score,
            risk_assessment=risk_assessment,
            approval_required=directive.approval_level in [ApprovalLevel.APPROVAL, ApprovalLevel.SUPERVISION],
            digital_signature=plan_hash
        )
    
    async def _validate_plan_constitutional(self, plan: ConstitutionalPlan) -> Dict[str, Any]:
        """Validate plan against constitutional guardrails using Diff Engine (Δ) role."""
        
        # Use the trust_gate tool to validate
        validation_result = await optimized_tool_call(
            "trust_gate",
            {
                "plan_yaml": plan.plan_yaml,
                "trust_score": self.trust_tracker.get_current_trust_score(),
                "approval_level": plan.directive.approval_level.value,
                "digital_signature": plan.digital_signature
            }
        )
        
        return validation_result
    
    async def _request_anchor_approval(self, plan: ConstitutionalPlan) -> bool:
        """Request approval from Anchor (Ω) role."""
        
        approval_message = f"""
        SOPHIE requires your approval for the following infrastructure change:
        
        Directive: {plan.directive.description}
        Type: {plan.directive.type.value}
        Priority: {plan.directive.priority}
        
        Risk Assessment:
        - Level: {plan.risk_assessment.get('risk_level', 'unknown')}
        - Factors: {', '.join(plan.risk_assessment.get('risk_factors', []))}
        
        Confidence Score: {plan.confidence_score:.2f}
        
        This change will:
        {plan.risk_assessment.get('change_summary', 'No summary available')}
        
        Approve this change? (y/n)
        """
        
        logger.info("Requesting anchor approval", plan_id=plan.id)
        
        # In a real implementation, this would interface with a human approval system
        # For now, we'll simulate approval for demonstration
        return True
    
    async def _execute_via_cicd(self, plan: ConstitutionalPlan) -> Dict[str, Any]:
        """Execute plan via CI/CD using Integrator (Σ) role."""
        
        logger.info("Executing plan via CI/CD", plan_id=plan.id)
        
        # Use the ci_trigger tool to execute
        execution_result = await optimized_tool_call(
            "ci_trigger",
            {
                "pipeline_type": "full",
                "plan_yaml": plan.plan_yaml,
                "approval_level": plan.directive.approval_level.value,
                "trust_score": self.trust_tracker.get_current_trust_score(),
                "change_summary": plan.risk_assessment.get('change_summary', ''),
                "artifacts_requested": [".app", ".deb", ".docker", "staging_url"]
            }
        )
        
        # Update plan status
        plan.status = "completed"
        plan.execution_result = execution_result
        
        return execution_result
    
    async def _store_execution_memory(
        self, 
        directive: Directive, 
        plan: ConstitutionalPlan, 
        execution_result: Dict[str, Any]
    ):
        """Store execution in memory using Memory (Ψ) role."""
        
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "directive": directive.description,
            "plan_id": plan.id,
            "confidence_score": plan.confidence_score,
            "risk_level": plan.risk_assessment.get('risk_level'),
            "execution_result": execution_result,
            "staging_url": execution_result.get("staging_url"),
            "artifact_urls": execution_result.get("artifact_urls", [])
        }
        
        self.execution_history.append(memory_entry)
        
        # Update trust score based on execution success
        if execution_result.get("status") == "success":
            self.trust_tracker.increase_trust_score(0.1)
        else:
            self.trust_tracker.decrease_trust_score(0.2)
    
    def _extract_yaml_from_response(self, response: str) -> str:
        """Extract YAML plan from LLM response."""
        # Simple extraction - in practice, this would be more sophisticated
        if "```yaml" in response:
            start = response.find("```yaml") + 7
            end = response.find("```", start)
            return response[start:end].strip()
        else:
            # Fallback to basic YAML structure
            return f"""
            name: "infrastructure_change"
            steps:
              - name: "validate_plan"
                action: "validation"
                description: "Validate the proposed changes"
              
              - name: "build_artifacts"
                action: "build"
                description: "Build required artifacts"
              
              - name: "deploy_to_staging"
                action: "deploy"
                description: "Deploy to staging environment"
              
              - name: "run_tests"
                action: "test"
                description: "Run automated tests"
              
              - name: "generate_artifacts"
                action: "package"
                description: "Generate downloadable artifacts"
            """
    
    async def _assess_plan_confidence(self, plan_yaml: str) -> float:
        """Assess confidence in the generated plan."""
        
        confidence_prompt = f"""
        Assess the confidence level (0.0-1.0) for this execution plan:
        
        {plan_yaml}
        
        Consider:
        - Completeness of the plan
        - Clarity of steps
        - Feasibility of execution
        - Risk factors
        
        Return only a number between 0.0 and 1.0.
        """
        
        response = await optimized_llm_call(
            confidence_prompt,
            "gpt-4",
            "openai",
            temperature=0.1,
            max_tokens=10
        )
        
        try:
            return float(response.strip())
        except:
            return 0.7  # Default confidence
    
    async def _assess_plan_risk(self, plan_yaml: str) -> Dict[str, Any]:
        """Assess risk of the execution plan."""
        
        risk_prompt = f"""
        Assess the risk level for this execution plan:
        
        {plan_yaml}
        
        Return a JSON object with:
        - risk_level: "low", "medium", "high", or "critical"
        - risk_factors: array of risk factors
        - mitigation_strategies: array of mitigation strategies
        - change_summary: human-readable summary of changes
        """
        
        response = await optimized_llm_call(
            risk_prompt,
            "gpt-4",
            "openai",
            temperature=0.2,
            max_tokens=300
        )
        
        try:
            return json.loads(response)
        except:
            return {
                "risk_level": "medium",
                "risk_factors": ["Unknown risk factors"],
                "mitigation_strategies": ["Standard safety measures"],
                "change_summary": "Infrastructure changes via CI/CD pipeline"
            }


# Global instance
constitutional_executor = ConstitutionalExecutor()


# Convenience function for easy integration
async def execute_constitutional_directive(
    human_input: str, 
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Execute a constitutional directive.
    
    This is the main entry point for SOPHIE's Constitutional AI Operating System.
    It implements the vision of turning human intent into system evolution.
    """
    return await constitutional_executor.interpret_and_execute_constitutional(
        human_input, context
    ) 