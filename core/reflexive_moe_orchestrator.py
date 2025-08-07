"""
Reflexive Mixture of Experts (MoE) Orchestrator

SOPHIE as a sovereign MoE conductor that can adapt to nearly every tool in every environment
to execute according to the user's intent. Based on what's needed for the prompt, SOPHIE can
classify predetermined roles (Corporate, Creative, Council) and delegate them to the strongest models.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import structlog

from .constitutional_executor import ConstitutionalExecutor, ConstitutionalRole
from .performance_integration import optimized_llm_call, optimized_tool_call
from memory.trust_tracker import TrustTracker
from governance.audit_log import AuditLog

logger = structlog.get_logger()


class ExpertRole(Enum):
    """Expert roles for SOPHIE's MoE system."""
    CORPORATE = "corporate"  # Structured, goal-oriented workflows
    CREATIVE = "creative"    # Expressive, divergent, aesthetic outputs
    COUNCIL = "council"      # Reflection, critique, comparison


class IntentType(Enum):
    """Types of user intent that SOPHIE can classify."""
    EXECUTION = "execution"      # Task completion, workflow automation
    ANALYSIS = "analysis"        # Data analysis, research, investigation
    CREATION = "creation"        # Content generation, design, ideation
    PLANNING = "planning"        # Strategy, planning, decision-making
    COORDINATION = "coordination"  # Multi-agent coordination, orchestration
    INFRASTRUCTURE = "infrastructure"  # System changes, deployment, configuration


@dataclass
class ExpertAgent:
    """An expert agent in SOPHIE's MoE system."""
    id: str
    name: str
    role: ExpertRole
    model: str
    provider: str
    domain_strengths: List[str]
    trust_score: float
    performance_metrics: Dict[str, float]
    is_available: bool = True
    last_used: Optional[datetime] = None


@dataclass
class IntentClassification:
    """Classification of user intent."""
    primary_intent: IntentType
    confidence_score: float
    suggested_roles: List[ExpertRole]
    context_requirements: Dict[str, Any]
    risk_level: str
    estimated_complexity: float


@dataclass
class MoEExecutionPlan:
    """Execution plan for MoE orchestration."""
    intent: IntentClassification
    selected_experts: List[ExpertAgent]
    collaboration_strategy: str
    execution_steps: List[Dict[str, Any]]
    trust_thresholds: Dict[str, float]
    fallback_strategy: str
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)


class ReflexiveMoEOrchestrator:
    """
    Reflexive Mixture of Experts Orchestrator.
    
    SOPHIE as a sovereign MoE conductor that can adapt to nearly every tool in every environment
    to execute according to the user's intent.
    """
    
    def __init__(self):
        self.trust_tracker = TrustTracker()
        self.audit_log = AuditLog()
        self.constitutional_executor = ConstitutionalExecutor()
        
        # Initialize expert agents
        self.expert_agents = self._initialize_expert_agents()
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
        
        # Role-specific trust metrics
        self.role_trust_scores = {
            ExpertRole.CORPORATE: 0.8,
            ExpertRole.CREATIVE: 0.75,
            ExpertRole.COUNCIL: 0.85
        }
    
    def _initialize_expert_agents(self) -> List[ExpertAgent]:
        """Initialize the expert agents for each role."""
        
        return [
            # Corporate Experts
            ExpertAgent(
                id="corporate_gpt4",
                name="Corporate GPT-4",
                role=ExpertRole.CORPORATE,
                model="gpt-4",
                provider="openai",
                domain_strengths=["workflow_automation", "data_analysis", "project_management"],
                trust_score=0.85,
                performance_metrics={"accuracy": 0.92, "speed": 0.88, "reliability": 0.90}
            ),
            ExpertAgent(
                id="corporate_claude",
                name="Corporate Claude",
                role=ExpertRole.CORPORATE,
                model="claude-3-sonnet",
                provider="anthropic",
                domain_strengths=["strategy", "planning", "decision_making"],
                trust_score=0.82,
                performance_metrics={"accuracy": 0.89, "speed": 0.85, "reliability": 0.88}
            ),
            
            # Creative Experts
            ExpertAgent(
                id="creative_claude",
                name="Creative Claude",
                role=ExpertRole.CREATIVE,
                model="claude-3-opus",
                provider="anthropic",
                domain_strengths=["content_generation", "design", "storytelling"],
                trust_score=0.88,
                performance_metrics={"creativity": 0.95, "originality": 0.92, "aesthetics": 0.90}
            ),
            ExpertAgent(
                id="creative_gpt4",
                name="Creative GPT-4",
                role=ExpertRole.CREATIVE,
                model="gpt-4",
                provider="openai",
                domain_strengths=["ideation", "brainstorming", "creative_problem_solving"],
                trust_score=0.80,
                performance_metrics={"creativity": 0.88, "originality": 0.85, "aesthetics": 0.87}
            ),
            
            # Council Experts
            ExpertAgent(
                id="council_ensemble",
                name="Council Ensemble",
                role=ExpertRole.COUNCIL,
                model="ensemble",
                provider="sophie",
                domain_strengths=["critique", "comparison", "validation"],
                trust_score=0.90,
                performance_metrics={"judgment": 0.93, "fairness": 0.91, "consistency": 0.89}
            ),
            ExpertAgent(
                id="council_reflective",
                name="Council Reflective",
                role=ExpertRole.COUNCIL,
                model="claude-3-sonnet",
                provider="anthropic",
                domain_strengths=["reflection", "analysis", "evaluation"],
                trust_score=0.87,
                performance_metrics={"judgment": 0.89, "fairness": 0.88, "consistency": 0.90}
            )
        ]
    
    async def orchestrate_intent(
        self, 
        user_prompt: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration method for SOPHIE's reflexive MoE system.
        
        This is the core method that implements the vision of SOPHIE as a sovereign
        MoE conductor that can adapt to any tool in any environment.
        """
        
        logger.info("Starting reflexive MoE orchestration", prompt=user_prompt[:100])
        
        # Step 1: Parse and classify user intent
        intent = await self._parse_user_intent(user_prompt, context)
        
        # Step 2: Select optimal expert agents
        selected_experts = await self._select_expert_agents(intent)
        
        # Step 3: Create execution plan
        execution_plan = await self._create_execution_plan(intent, selected_experts)
        
        # Step 4: Execute with reflexive monitoring
        result = await self._execute_with_reflexive_monitoring(execution_plan, user_prompt)
        
        # Step 5: Update trust metrics and memory
        await self._update_trust_and_memory(execution_plan, result)
        
        return result
    
    async def _parse_user_intent(
        self, 
        user_prompt: str, 
        context: Dict[str, Any] = None
    ) -> IntentClassification:
        """Parse and classify user intent using LLM analysis."""
        
        intent_analysis_prompt = f"""
        Analyze this user prompt and classify the intent:
        
        Prompt: "{user_prompt}"
        Context: {context or {}}
        
        Classify the intent into one of these categories:
        - EXECUTION: Task completion, workflow automation, implementation
        - ANALYSIS: Data analysis, research, investigation, evaluation
        - CREATION: Content generation, design, ideation, creative work
        - PLANNING: Strategy, planning, decision-making, coordination
        - COORDINATION: Multi-agent coordination, orchestration, management
        - INFRASTRUCTURE: System changes, deployment, configuration, technical
        
        Also determine:
        - Confidence score (0.0-1.0)
        - Suggested expert roles (corporate, creative, council)
        - Risk level (low, medium, high)
        - Estimated complexity (0.0-1.0)
        
        Return a JSON object with this structure.
        """
        
        response = await optimized_llm_call(
            intent_analysis_prompt,
            "gpt-4",
            "openai",
            temperature=0.2,
            max_tokens=500
        )
        
        try:
            intent_data = json.loads(response)
            
            return IntentClassification(
                primary_intent=IntentType(intent_data["primary_intent"]),
                confidence_score=float(intent_data["confidence_score"]),
                suggested_roles=[ExpertRole(role) for role in intent_data["suggested_roles"]],
                context_requirements=intent_data.get("context_requirements", {}),
                risk_level=intent_data.get("risk_level", "low"),
                estimated_complexity=float(intent_data.get("estimated_complexity", 0.5))
            )
        except Exception as e:
            logger.error("Failed to parse intent", error=str(e))
            # Fallback classification
            return IntentClassification(
                primary_intent=IntentType.EXECUTION,
                confidence_score=0.7,
                suggested_roles=[ExpertRole.CORPORATE],
                context_requirements={},
                risk_level="low",
                estimated_complexity=0.5
            )
    
    async def _select_expert_agents(self, intent: IntentClassification) -> List[ExpertAgent]:
        """Select optimal expert agents based on intent and trust metrics."""
        
        selected_experts = []
        
        # Filter available experts by suggested roles
        available_experts = [
            expert for expert in self.expert_agents 
            if expert.role in intent.suggested_roles and expert.is_available
        ]
        
        # Score experts based on multiple factors
        expert_scores = []
        for expert in available_experts:
            # Base trust score
            base_score = expert.trust_score
            
            # Role-specific trust adjustment
            role_trust = self.role_trust_scores.get(expert.role, 0.5)
            role_adjustment = (role_trust - 0.5) * 0.2
            
            # Performance adjustment
            performance_score = sum(expert.performance_metrics.values()) / len(expert.performance_metrics)
            performance_adjustment = (performance_score - 0.5) * 0.3
            
            # Domain strength adjustment
            domain_match = 0.0
            if intent.primary_intent == IntentType.EXECUTION and "workflow_automation" in expert.domain_strengths:
                domain_match = 0.2
            elif intent.primary_intent == IntentType.CREATION and "content_generation" in expert.domain_strengths:
                domain_match = 0.2
            elif intent.primary_intent == IntentType.ANALYSIS and "data_analysis" in expert.domain_strengths:
                domain_match = 0.2
            
            final_score = base_score + role_adjustment + performance_adjustment + domain_match
            expert_scores.append((expert, final_score))
        
        # Sort by score and select top experts
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select experts based on complexity and intent
        if intent.estimated_complexity > 0.7:
            # High complexity: use multiple experts
            selected_experts = [expert for expert, score in expert_scores[:3]]
        elif intent.primary_intent == IntentType.COUNCIL:
            # Council intent: use council experts
            selected_experts = [expert for expert, score in expert_scores if expert.role == ExpertRole.COUNCIL][:2]
        else:
            # Standard: use top 1-2 experts
            selected_experts = [expert for expert, score in expert_scores[:2]]
        
        return selected_experts
    
    async def _create_execution_plan(
        self, 
        intent: IntentClassification, 
        selected_experts: List[ExpertAgent]
    ) -> MoEExecutionPlan:
        """Create an execution plan for the MoE orchestration."""
        
        # Determine collaboration strategy
        if len(selected_experts) == 1:
            collaboration_strategy = "single_expert"
        elif len(selected_experts) == 2:
            collaboration_strategy = "parallel_execution"
        else:
            collaboration_strategy = "sequential_refinement"
        
        # Create execution steps
        execution_steps = []
        
        if collaboration_strategy == "single_expert":
            execution_steps.append({
                "step": "expert_execution",
                "expert_id": selected_experts[0].id,
                "description": f"Execute with {selected_experts[0].name}",
                "trust_threshold": selected_experts[0].trust_score
            })
        elif collaboration_strategy == "parallel_execution":
            for expert in selected_experts:
                execution_steps.append({
                    "step": "parallel_execution",
                    "expert_id": expert.id,
                    "description": f"Execute with {expert.name}",
                    "trust_threshold": expert.trust_score
                })
            execution_steps.append({
                "step": "consensus_formation",
                "description": "Form consensus from parallel executions",
                "trust_threshold": 0.8
            })
        else:  # sequential_refinement
            for i, expert in enumerate(selected_experts):
                execution_steps.append({
                    "step": "sequential_refinement",
                    "expert_id": expert.id,
                    "description": f"Refinement {i+1} with {expert.name}",
                    "trust_threshold": expert.trust_score
                })
        
        return MoEExecutionPlan(
            intent=intent,
            selected_experts=selected_experts,
            collaboration_strategy=collaboration_strategy,
            execution_steps=execution_steps,
            trust_thresholds={expert.id: expert.trust_score for expert in selected_experts},
            fallback_strategy="constitutional_executor"
        )
    
    async def _execute_with_reflexive_monitoring(
        self, 
        execution_plan: MoEExecutionPlan, 
        user_prompt: str
    ) -> Dict[str, Any]:
        """Execute the plan with reflexive monitoring and adaptation."""
        
        logger.info("Executing MoE plan", strategy=execution_plan.collaboration_strategy)
        
        results = []
        audit_trail = []
        
        for step in execution_plan.execution_steps:
            step_start = time.time()
            
            # Execute step
            if step["step"] == "expert_execution":
                result = await self._execute_single_expert(
                    execution_plan.selected_experts[0], 
                    user_prompt
                )
                results.append(result)
                
            elif step["step"] == "parallel_execution":
                # Execute with multiple experts in parallel
                expert = next(e for e in execution_plan.selected_experts if e.id == step["expert_id"])
                result = await self._execute_single_expert(expert, user_prompt)
                results.append(result)
                
            elif step["step"] == "consensus_formation":
                # Form consensus from parallel results
                result = await self._form_consensus(results)
                results = [result]  # Replace with consensus result
                
            elif step["step"] == "sequential_refinement":
                # Refine based on previous results
                expert = next(e for e in execution_plan.selected_experts if e.id == step["expert_id"])
                result = await self._execute_refinement(expert, user_prompt, results[-1] if results else None)
                results.append(result)
            
            # Record audit trail
            step_duration = time.time() - step_start
            audit_trail.append({
                "step": step["step"],
                "expert_id": step.get("expert_id"),
                "duration": step_duration,
                "trust_threshold": step["trust_threshold"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if we need to fall back to constitutional executor
            if len(results) > 0 and results[-1].get("confidence_score", 0) < step["trust_threshold"]:
                logger.warning("Trust threshold not met, falling back to constitutional executor")
                constitutional_result = await self.constitutional_executor.interpret_and_execute_constitutional(
                    user_prompt
                )
                results.append(constitutional_result)
                break
        
        # Update execution plan with audit trail
        execution_plan.audit_trail = audit_trail
        
        # Return final result
        final_result = results[-1] if results else {"status": "failed", "reason": "No experts available"}
        final_result["moe_plan"] = {
            "intent": execution_plan.intent.primary_intent.value,
            "confidence_score": execution_plan.intent.confidence_score,
            "collaboration_strategy": execution_plan.collaboration_strategy,
            "selected_experts": [expert.name for expert in execution_plan.selected_experts],
            "audit_trail": audit_trail
        }
        
        return final_result
    
    async def _execute_single_expert(self, expert: ExpertAgent, user_prompt: str) -> Dict[str, Any]:
        """Execute a single expert agent."""
        
        expert_prompt = f"""
        You are {expert.name}, an expert in {expert.role.value} tasks.
        
        Your domain strengths: {', '.join(expert.domain_strengths)}
        Your trust score: {expert.trust_score}
        
        User request: "{user_prompt}"
        
        Provide your expert response based on your specialized knowledge and capabilities.
        """
        
        response = await optimized_llm_call(
            expert_prompt,
            expert.model,
            expert.provider,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Update expert usage
        expert.last_used = datetime.now()
        
        return {
            "expert_id": expert.id,
            "expert_name": expert.name,
            "role": expert.role.value,
            "response": response,
            "confidence_score": expert.trust_score,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _form_consensus(self, parallel_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Form consensus from parallel expert executions."""
        
        consensus_prompt = f"""
        Form a consensus from these expert responses:
        
        {json.dumps(parallel_results, indent=2)}
        
        Synthesize the best elements from each expert's response into a unified, high-quality result.
        """
        
        consensus_response = await optimized_llm_call(
            consensus_prompt,
            "gpt-4",
            "openai",
            temperature=0.2,
            max_tokens=1000
        )
        
        return {
            "type": "consensus",
            "response": consensus_response,
            "confidence_score": sum(r.get("confidence_score", 0) for r in parallel_results) / len(parallel_results),
            "source_experts": [r["expert_name"] for r in parallel_results],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_refinement(self, expert: ExpertAgent, user_prompt: str, previous_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute refinement based on previous result."""
        
        refinement_prompt = f"""
        You are {expert.name}, refining this previous result:
        
        Previous result: {previous_result.get('response', 'No previous result')}
        
        User request: "{user_prompt}"
        
        Refine and improve the previous result based on your expertise in {expert.role.value}.
        """
        
        response = await optimized_llm_call(
            refinement_prompt,
            expert.model,
            expert.provider,
            temperature=0.3,
            max_tokens=1000
        )
        
        return {
            "expert_id": expert.id,
            "expert_name": expert.name,
            "role": expert.role.value,
            "response": response,
            "confidence_score": expert.trust_score,
            "refinement_of": previous_result.get("expert_id"),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _update_trust_and_memory(self, execution_plan: MoEExecutionPlan, result: Dict[str, Any]):
        """Update trust metrics and memory based on execution results."""
        
        # Update expert trust scores based on performance
        for expert in execution_plan.selected_experts:
            if result.get("status") == "completed":
                # Increase trust score slightly for successful execution
                expert.trust_score = min(1.0, expert.trust_score + 0.01)
            else:
                # Decrease trust score for failed execution
                expert.trust_score = max(0.0, expert.trust_score - 0.02)
        
        # Update role trust scores
        for expert in execution_plan.selected_experts:
            role = expert.role
            current_role_trust = self.role_trust_scores.get(role, 0.5)
            if result.get("status") == "completed":
                self.role_trust_scores[role] = min(1.0, current_role_trust + 0.005)
            else:
                self.role_trust_scores[role] = max(0.0, current_role_trust - 0.01)
        
        # Store in execution history
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_prompt": result.get("moe_plan", {}).get("intent", "unknown"),
            "selected_experts": [expert.name for expert in execution_plan.selected_experts],
            "collaboration_strategy": execution_plan.collaboration_strategy,
            "result_status": result.get("status", "unknown"),
            "audit_trail": execution_plan.audit_trail
        })


# Global instance
reflexive_moe_orchestrator = ReflexiveMoEOrchestrator()


# Convenience function for easy integration
async def orchestrate_with_reflexive_moe(
    user_prompt: str, 
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Orchestrate user intent using SOPHIE's reflexive MoE system.
    
    This is the main entry point for SOPHIE as a sovereign MoE conductor
    that can adapt to any tool in any environment.
    """
    return await reflexive_moe_orchestrator.orchestrate_intent(user_prompt, context) 