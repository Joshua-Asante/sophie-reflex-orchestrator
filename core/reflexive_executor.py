"""
Reflexive Execution Engine

Enhanced autonomous execution with step-level reflection, dynamic plan adaptation,
and semantic memory integration for SOPHIE's advanced autonomous capabilities.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from .autonomous_executor import AutonomousExecutor, Directive, ExecutionPlan, DirectiveType
from .performance_integration import optimized_llm_call, optimized_tool_call
from .performance_monitor import performance_monitor

logger = logging.getLogger(__name__)


class ReflectionLevel(Enum):
    """Levels of reflection for execution steps."""
    NONE = "none"
    LIGHT = "light"  # Basic logging
    MODERATE = "moderate"  # Analysis and scoring
    DEEP = "deep"  # Full reasoning trace


class PlanStack:
    """Stack of active execution plans for multi-goal reasoning."""
    
    def __init__(self):
        self.active_plans: deque = deque()
        self.completed_plans: List[Dict[str, Any]] = []
        self.interrupted_plans: List[Dict[str, Any]] = []
    
    def push_plan(self, plan: ExecutionPlan, context: Dict[str, Any] = None):
        """Push a new plan onto the stack."""
        plan_entry = {
            "plan": plan,
            "context": context or {},
            "started_at": time.time(),
            "status": "active",
            "reflection_log": []
        }
        self.active_plans.append(plan_entry)
        logger.info(f"Pushed plan: {plan.directive.description}")
    
    def pop_plan(self) -> Optional[Dict[str, Any]]:
        """Pop the top plan from the stack."""
        if not self.active_plans:
            return None
        
        plan_entry = self.active_plans.pop()
        plan_entry["status"] = "completed"
        plan_entry["completed_at"] = time.time()
        self.completed_plans.append(plan_entry)
        
        logger.info(f"Popped plan: {plan_entry['plan'].directive.description}")
        return plan_entry
    
    def interrupt_current_plan(self, reason: str):
        """Interrupt the current plan and save for later resumption."""
        if not self.active_plans:
            return
        
        plan_entry = self.active_plans.pop()
        plan_entry["status"] = "interrupted"
        plan_entry["interrupted_at"] = time.time()
        plan_entry["interruption_reason"] = reason
        self.interrupted_plans.append(plan_entry)
        
        logger.info(f"Interrupted plan: {plan_entry['plan'].directive.description} - {reason}")
    
    def get_current_plan(self) -> Optional[Dict[str, Any]]:
        """Get the current active plan."""
        return self.active_plans[-1] if self.active_plans else None
    
    def get_plan_history(self) -> Dict[str, Any]:
        """Get the complete plan history."""
        return {
            "active_count": len(self.active_plans),
            "completed_count": len(self.completed_plans),
            "interrupted_count": len(self.interrupted_plans),
            "active_plans": [p["plan"].directive.description for p in self.active_plans],
            "recent_completed": [p["plan"].directive.description for p in self.completed_plans[-5:]]
        }


@dataclass
class ReasoningTrace:
    """Detailed reasoning trace for execution steps."""
    step_number: int
    step_description: str
    decision_reasoning: str
    alternatives_considered: List[str]
    confidence_score: float
    trust_metrics: Dict[str, float]
    reflection_insights: Optional[str] = None
    adaptation_triggered: bool = False
    timestamp: float = field(default_factory=time.time)


class ReflexiveExecutor(AutonomousExecutor):
    """Enhanced autonomous executor with reflexive capabilities."""
    
    def __init__(self):
        super().__init__()
        self.plan_stack = PlanStack()
        self.reasoning_traces: Dict[str, List[ReasoningTrace]] = {}
        self.reflection_level = ReflectionLevel.MODERATE
        self.adaptation_threshold = 0.7  # Confidence threshold for plan adaptation
        
    async def execute_directive_reflexive(
        self, 
        directive: Directive, 
        plan: ExecutionPlan,
        reflection_level: ReflectionLevel = ReflectionLevel.MODERATE
    ) -> Dict[str, Any]:
        """Execute a directive with reflexive capabilities."""
        
        logger.info(f"Starting reflexive execution: {directive.description}")
        
        # Push plan to stack
        self.plan_stack.push_plan(plan, {"directive": directive})
        
        # Initialize reasoning trace
        directive_id = directive.id
        self.reasoning_traces[directive_id] = []
        
        # Execute with step-level reflection
        results = []
        for i, step in enumerate(plan.steps):
            try:
                # Pre-step reflection
                pre_reflection = await self._reflect_on_step_start(step, i, directive)
                
                # Execute step
                logger.info(f"Executing step {i+1}/{len(plan.steps)}: {step.get('description', 'Unknown')}")
                result = await self._execute_step_with_reflection(step, i, directive)
                
                # Post-step reflection
                post_reflection = await self._reflect_on_step_completion(step, result, i, directive)
                
                # Create reasoning trace
                trace = ReasoningTrace(
                    step_number=i + 1,
                    step_description=step.get('description', 'Unknown'),
                    decision_reasoning=pre_reflection.get('reasoning', ''),
                    alternatives_considered=pre_reflection.get('alternatives', []),
                    confidence_score=post_reflection.get('confidence', 0.0),
                    trust_metrics=post_reflection.get('trust_metrics', {}),
                    reflection_insights=post_reflection.get('insights'),
                    adaptation_triggered=post_reflection.get('adaptation_triggered', False)
                )
                self.reasoning_traces[directive_id].append(trace)
                
                # Check if adaptation is needed
                if post_reflection.get('adaptation_triggered', False):
                    await self._adapt_plan_mid_execution(plan, step, result, directive)
                
                results.append({
                    "step": i + 1,
                    "description": step.get("description", "Unknown"),
                    "result": result,
                    "status": "success",
                    "reflection": post_reflection
                })
                
            except Exception as e:
                logger.error(f"Step {i+1} failed: {e}")
                
                # Reflect on failure
                failure_reflection = await self._reflect_on_step_failure(step, e, i, directive)
                
                trace = ReasoningTrace(
                    step_number=i + 1,
                    step_description=step.get('description', 'Unknown'),
                    decision_reasoning="Step failed",
                    alternatives_considered=[],
                    confidence_score=0.0,
                    trust_metrics={"reliability": 0.0},
                    reflection_insights=failure_reflection.get('insights'),
                    adaptation_triggered=True
                )
                self.reasoning_traces[directive_id].append(trace)
                
                results.append({
                    "step": i + 1,
                    "description": step.get("description", "Unknown"),
                    "result": None,
                    "status": "failed",
                    "error": str(e),
                    "reflection": failure_reflection
                })
                
                # Handle failure based on directive type
                if directive.type == DirectiveType.OPTIMIZATION:
                    continue
                else:
                    break
        
        # Pop plan from stack
        self.plan_stack.pop_plan()
        
        # Update directive status
        directive.status = "completed" if all(r["status"] == "success" for r in results) else "failed"
        directive.result = results
        
        # Store in history with reasoning traces
        self.execution_history.append({
            "directive": directive,
            "plan": plan,
            "results": results,
            "reasoning_traces": self.reasoning_traces.get(directive_id, []),
            "timestamp": time.time()
        })
        
        return {
            "directive_id": directive.id,
            "status": directive.status,
            "results": results,
            "reasoning_traces": self.reasoning_traces.get(directive_id, []),
            "duration": time.time() - directive.created_at,
            "plan_stack_status": self.plan_stack.get_plan_history()
        }
    
    async def _reflect_on_step_start(self, step: Dict[str, Any], step_index: int, directive: Directive) -> Dict[str, Any]:
        """Reflect on a step before execution."""
        
        reflection_prompt = f"""
        Analyze this execution step before running it:
        
        Step: {step.get('description', 'Unknown')}
        Step Index: {step_index + 1}
        Directive: {directive.description}
        Directive Type: {directive.type.value}
        
        Provide a JSON response with:
        - reasoning: Why this step was chosen
        - alternatives: Other approaches that could be taken
        - risk_assessment: Potential risks or issues
        - expected_outcome: What should happen
        """
        
        try:
            response = await optimized_llm_call(
                reflection_prompt,
                "gpt-4",
                "openai",
                temperature=0.3,
                max_tokens=400
            )
            
            reflection = json.loads(response)
            logger.info(f"Pre-step reflection: {reflection.get('reasoning', '')[:100]}...")
            return reflection
            
        except Exception as e:
            logger.warning(f"Pre-step reflection failed: {e}")
            return {
                "reasoning": "Fallback execution",
                "alternatives": [],
                "risk_assessment": "unknown",
                "expected_outcome": "step completion"
            }
    
    async def _execute_step_with_reflection(self, step: Dict[str, Any], step_index: int, directive: Directive) -> Any:
        """Execute a step with monitoring."""
        
        # Monitor performance
        performance_monitor.start_component_timer("reflexive_executor", f"step_{step_index}")
        
        try:
            # Execute the step based on its type
            if step.get("type") == "llm_call":
                result = await self._execute_llm_step(step)
            elif step.get("type") == "tool_call":
                result = await self._execute_tool_step(step)
            elif step.get("type") == "coordination":
                result = await self._execute_coordination_step(step)
            else:
                result = await self._execute_generic_step(step)
            
            return result
            
        finally:
            performance_monitor.end_component_timer("reflexive_executor", f"step_{step_index}")
    
    async def _reflect_on_step_completion(self, step: Dict[str, Any], result: Any, step_index: int, directive: Directive) -> Dict[str, Any]:
        """Reflect on a step after execution."""
        
        reflection_prompt = f"""
        Analyze this completed execution step:
        
        Step: {step.get('description', 'Unknown')}
        Result: {str(result)[:200]}
        Step Index: {step_index + 1}
        Directive: {directive.description}
        
        Provide a JSON response with:
        - confidence_score: 0.0-1.0 confidence in the result
        - trust_metrics: {{"reliability": 0.0-1.0, "accuracy": 0.0-1.0}}
        - insights: Key learnings or observations
        - adaptation_needed: Whether the plan should be modified
        - adaptation_reason: Why adaptation is needed (if any)
        """
        
        try:
            response = await optimized_llm_call(
                reflection_prompt,
                "gpt-4",
                "openai",
                temperature=0.3,
                max_tokens=400
            )
            
            reflection = json.loads(response)
            
            # Check if adaptation is needed
            adaptation_triggered = reflection.get('adaptation_needed', False)
            if adaptation_triggered:
                logger.info(f"Adaptation triggered: {reflection.get('adaptation_reason', 'Unknown')}")
            
            logger.info(f"Post-step reflection: confidence={reflection.get('confidence_score', 0.0)}")
            return reflection
            
        except Exception as e:
            logger.warning(f"Post-step reflection failed: {e}")
            return {
                "confidence_score": 0.5,
                "trust_metrics": {"reliability": 0.5, "accuracy": 0.5},
                "insights": "Reflection failed",
                "adaptation_triggered": False
            }
    
    async def _reflect_on_step_failure(self, step: Dict[str, Any], error: Exception, step_index: int, directive: Directive) -> Dict[str, Any]:
        """Reflect on a step failure."""
        
        reflection_prompt = f"""
        Analyze this failed execution step:
        
        Step: {step.get('description', 'Unknown')}
        Error: {str(error)}
        Step Index: {step_index + 1}
        Directive: {directive.description}
        
        Provide a JSON response with:
        - failure_analysis: Why the step failed
        - recovery_strategy: How to handle this failure
        - insights: Key learnings from the failure
        - adaptation_needed: Whether the plan should be modified
        """
        
        try:
            response = await optimized_llm_call(
                reflection_prompt,
                "gpt-4",
                "openai",
                temperature=0.3,
                max_tokens=400
            )
            
            reflection = json.loads(response)
            logger.info(f"Failure reflection: {reflection.get('failure_analysis', '')[:100]}...")
            return reflection
            
        except Exception as e:
            logger.warning(f"Failure reflection failed: {e}")
            return {
                "failure_analysis": "Unknown error",
                "recovery_strategy": "Continue with next step",
                "insights": "Reflection failed",
                "adaptation_needed": True
            }
    
    async def _adapt_plan_mid_execution(self, plan: ExecutionPlan, step: Dict[str, Any], result: Any, directive: Directive):
        """Adapt the execution plan mid-execution."""
        
        adaptation_prompt = f"""
        The current execution plan needs adaptation:
        
        Current Step: {step.get('description', 'Unknown')}
        Step Result: {str(result)[:200]}
        Directive: {directive.description}
        Remaining Steps: {len(plan.steps)} steps
        
        Suggest plan adaptations in JSON format:
        - modified_steps: Array of modified step descriptions
        - reasoning: Why these changes are needed
        - risk_assessment: Risks of the adaptation
        """
        
        try:
            response = await optimized_llm_call(
                adaptation_prompt,
                "gpt-4",
                "openai",
                temperature=0.3,
                max_tokens=600
            )
            
            adaptation = json.loads(response)
            
            # Apply adaptations to the plan
            if 'modified_steps' in adaptation:
                # Replace remaining steps with adapted ones
                current_step_index = plan.steps.index(step)
                plan.steps = plan.steps[:current_step_index + 1] + adaptation['modified_steps']
                
                logger.info(f"Plan adapted: {adaptation.get('reasoning', 'Unknown')}")
                
        except Exception as e:
            logger.warning(f"Plan adaptation failed: {e}")
    
    async def get_reasoning_trace(self, directive_id: str) -> List[ReasoningTrace]:
        """Get the reasoning trace for a directive."""
        return self.reasoning_traces.get(directive_id, [])
    
    async def get_reflexive_status(self) -> Dict[str, Any]:
        """Get current reflexive execution status."""
        base_status = await self.get_autonomous_status()
        
        return {
            **base_status,
            "plan_stack": self.plan_stack.get_plan_history(),
            "reflection_level": self.reflection_level.value,
            "adaptation_threshold": self.adaptation_threshold,
            "reasoning_traces_count": len(self.reasoning_traces)
        }


# Global instance
reflexive_executor = ReflexiveExecutor()


# Convenience functions for easy integration
async def execute_directive_reflexive(human_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute a directive with reflexive capabilities."""
    directive = await reflexive_executor.interpret_directive(human_input, context)
    plan = await reflexive_executor.create_execution_plan(directive)
    return await reflexive_executor.execute_directive_reflexive(directive, plan)


async def get_reasoning_trace(directive_id: str) -> List[ReasoningTrace]:
    """Get reasoning trace for a directive."""
    return await reflexive_executor.get_reasoning_trace(directive_id)


async def get_reflexive_status() -> Dict[str, Any]:
    """Get reflexive execution status."""
    return await reflexive_executor.get_reflexive_status() 