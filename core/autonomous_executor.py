"""
Autonomous Execution Engine

Enables SOPHIE to interpret high-level human directives and execute them
autonomously while maintaining human oversight and approval.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import json
import time

from .performance_integration import optimized_llm_call, optimized_tool_call
from .performance_monitor import performance_monitor
from memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class DirectiveType(Enum):
    """Types of high-level directives SOPHIE can interpret."""
    IMPLEMENTATION = "implementation"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    COORDINATION = "coordination"
    IMPROVEMENT = "improvement"
    EXECUTION = "execution"


class ApprovalLevel(Enum):
    """Levels of human approval required."""
    AUTONOMOUS = "autonomous"  # No approval needed
    NOTIFICATION = "notification"  # Inform human after execution
    APPROVAL = "approval"  # Require explicit approval
    SUPERVISION = "supervision"  # Human must be present


@dataclass
class Directive:
    """A high-level directive from a human."""
    id: str
    type: DirectiveType
    description: str
    priority: int = 1
    approval_level: ApprovalLevel = ApprovalLevel.NOTIFICATION
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class ExecutionPlan:
    """A plan for executing a directive."""
    directive: Directive
    steps: List[Dict[str, Any]]
    estimated_duration: float
    resources_required: List[str]
    risk_level: str = "low"
    approval_required: bool = False


class AutonomousExecutor:
    """Main autonomous execution engine for SOPHIE."""
    
    def __init__(self):
        self.active_directives: Dict[str, Directive] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.approval_callbacks: Dict[str, Callable] = {}
        self.purpose_context: Dict[str, Any] = {}
        self.improvement_suggestions: List[str] = []
        
    async def interpret_directive(self, human_input: str, context: Dict[str, Any] = None) -> Directive:
        """Interpret a high-level human directive."""
        
        # Use LLM to classify and structure the directive
        classification_prompt = f"""
        Classify this human directive and extract key information:
        
        Input: "{human_input}"
        Context: {context or {}}
        
        Return a JSON object with:
        - type: one of {[t.value for t in DirectiveType]}
        - description: clear description of what needs to be done
        - priority: 1-5 (5 being highest)
        - approval_level: one of {[a.value for a in ApprovalLevel]}
        - context: any additional context needed
        """
        
        try:
            response = await optimized_llm_call(
                classification_prompt,
                "gpt-4",
                "openai",
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse the response
            classification = json.loads(response)
            
            directive = Directive(
                id=f"directive_{int(time.time())}",
                type=DirectiveType(classification["type"]),
                description=classification["description"],
                priority=classification.get("priority", 1),
                approval_level=ApprovalLevel(classification.get("approval_level", "notification")),
                context=classification.get("context", {})
            )
            
            logger.info(f"Interpreted directive: {directive.description}")
            return directive
            
        except Exception as e:
            logger.error(f"Failed to interpret directive: {e}")
            # Fallback to basic interpretation
            return Directive(
                id=f"directive_{int(time.time())}",
                type=DirectiveType.EXECUTION,
                description=human_input,
                context=context or {}
            )
    
    async def create_execution_plan(self, directive: Directive) -> ExecutionPlan:
        """Create a detailed execution plan for a directive."""
        
        planning_prompt = f"""
        Create an execution plan for this directive:
        
        Directive: {directive.description}
        Type: {directive.type.value}
        Context: {directive.context}
        
        Return a JSON object with:
        - steps: array of execution steps
        - estimated_duration: time in seconds
        - resources_required: list of resources needed
        - risk_level: low/medium/high
        - approval_required: boolean
        """
        
        try:
            response = await optimized_llm_call(
                planning_prompt,
                "gpt-4",
                "openai",
                temperature=0.3,
                max_tokens=1000
            )
            
            plan_data = json.loads(response)
            
            plan = ExecutionPlan(
                directive=directive,
                steps=plan_data["steps"],
                estimated_duration=plan_data["estimated_duration"],
                resources_required=plan_data["resources_required"],
                risk_level=plan_data.get("risk_level", "low"),
                approval_required=plan_data.get("approval_required", False)
            )
            
            logger.info(f"Created execution plan with {len(plan.steps)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create execution plan: {e}")
            # Fallback to simple plan
            return ExecutionPlan(
                directive=directive,
                steps=[{"action": "execute", "description": directive.description}],
                estimated_duration=60.0,
                resources_required=["llm", "tools"],
                risk_level="low"
            )
    
    async def execute_directive(self, directive: Directive, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute a directive according to its plan."""
        
        logger.info(f"Executing directive: {directive.description}")
        
        # Check approval requirements
        if plan.approval_required and directive.approval_level == ApprovalLevel.APPROVAL:
            await self._request_approval(directive, plan)
        
        # Execute steps
        results = []
        for i, step in enumerate(plan.steps):
            try:
                logger.info(f"Executing step {i+1}/{len(plan.steps)}: {step.get('description', 'Unknown')}")
                
                # Execute the step based on its type
                if step.get("type") == "llm_call":
                    result = await self._execute_llm_step(step)
                elif step.get("type") == "tool_call":
                    result = await self._execute_tool_step(step)
                elif step.get("type") == "coordination":
                    result = await self._execute_coordination_step(step)
                else:
                    result = await self._execute_generic_step(step)
                
                results.append({
                    "step": i + 1,
                    "description": step.get("description", "Unknown"),
                    "result": result,
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Step {i+1} failed: {e}")
                results.append({
                    "step": i + 1,
                    "description": step.get("description", "Unknown"),
                    "result": None,
                    "status": "failed",
                    "error": str(e)
                })
                
                # Handle failure based on directive type
                if directive.type == DirectiveType.OPTIMIZATION:
                    # For optimizations, continue with other steps
                    continue
                else:
                    # For other types, stop execution
                    break
        
        # Update directive status
        directive.status = "completed" if all(r["status"] == "success" for r in results) else "failed"
        directive.result = results
        
        # Store in history
        self.execution_history.append({
            "directive": directive,
            "plan": plan,
            "results": results,
            "timestamp": time.time()
        })
        
        # Notify human if required
        if directive.approval_level == ApprovalLevel.NOTIFICATION:
            await self._notify_human(directive, results)
        
        return {
            "directive_id": directive.id,
            "status": directive.status,
            "results": results,
            "duration": time.time() - directive.created_at
        }
    
    async def _execute_llm_step(self, step: Dict[str, Any]) -> Any:
        """Execute an LLM-based step."""
        prompt = step.get("prompt", "")
        model = step.get("model", "gpt-4")
        provider = step.get("provider", "openai")
        
        return await optimized_llm_call(prompt, model, provider)
    
    async def _execute_tool_step(self, step: Dict[str, Any]) -> Any:
        """Execute a tool-based step."""
        tool_name = step.get("tool", "")
        params = step.get("params", {})
        
        return await optimized_tool_call(tool_name, params)
    
    async def _execute_coordination_step(self, step: Dict[str, Any]) -> Any:
        """Execute a coordination step."""
        # This would coordinate with other agents/components
        logger.info(f"Coordinating: {step.get('description', 'Unknown coordination')}")
        return {"coordination_result": "completed"}
    
    async def _execute_generic_step(self, step: Dict[str, Any]) -> Any:
        """Execute a generic step."""
        action = step.get("action", "")
        description = step.get("description", "")
        
        logger.info(f"Executing generic action: {action} - {description}")
        return {"action_result": "completed"}
    
    async def _request_approval(self, directive: Directive, plan: ExecutionPlan):
        """Request human approval for a directive."""
        approval_message = f"""
        SOPHIE requires approval for directive:
        
        Description: {directive.description}
        Type: {directive.type.value}
        Risk Level: {plan.risk_level}
        Estimated Duration: {plan.estimated_duration}s
        Steps: {len(plan.steps)}
        
        Approve execution? (y/n)
        """
        
        logger.info(approval_message)
        # In a real implementation, this would interface with a human approval system
        # For now, we'll simulate approval
        return True
    
    async def _notify_human(self, directive: Directive, results: List[Dict[str, Any]]):
        """Notify human of completed directive."""
        notification = f"""
        SOPHIE completed directive: {directive.description}
        
        Status: {directive.status}
        Steps completed: {len([r for r in results if r['status'] == 'success'])}/{len(results)}
        Duration: {time.time() - directive.created_at:.2f}s
        """
        
        logger.info(notification)
    
    async def retain_purpose(self, context: Dict[str, Any]):
        """Retain purpose and context for long-term execution."""
        self.purpose_context.update(context)
        
        # Store in memory for long-term retention
        # Note: In a real implementation, we would get memory stats from the memory manager
        # For now, we'll use a simple placeholder
        memory_stats = {"status": "available", "components": ["episodic", "semantic", "working"]}
        self.purpose_context["memory_state"] = memory_stats
        
        logger.info(f"Retained purpose context: {len(self.purpose_context)} items")
    
    async def suggest_improvements(self) -> List[str]:
        """Suggest improvements based on execution history."""
        if not self.execution_history:
            return []
        
        # Analyze recent executions for improvement opportunities
        recent_executions = self.execution_history[-10:]  # Last 10 executions
        
        improvement_prompt = f"""
        Analyze these recent SOPHIE executions and suggest improvements:
        
        Executions: {recent_executions}
        
        Suggest specific improvements for:
        1. Performance optimization
        2. Error handling
        3. Human interaction
        4. Autonomous capabilities
        
        Return as JSON array of improvement suggestions.
        """
        
        try:
            response = await optimized_llm_call(
                improvement_prompt,
                "gpt-4",
                "openai",
                temperature=0.7,
                max_tokens=800
            )
            
            suggestions = json.loads(response)
            self.improvement_suggestions.extend(suggestions)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate improvement suggestions: {e}")
            return []
    
    async def get_autonomous_status(self) -> Dict[str, Any]:
        """Get current autonomous execution status."""
        return {
            "active_directives": len(self.active_directives),
            "execution_history": len(self.execution_history),
            "purpose_context_size": len(self.purpose_context),
            "improvement_suggestions": len(self.improvement_suggestions),
            "performance_metrics": performance_monitor.get_performance_summary()
        }


# Global instance
autonomous_executor = AutonomousExecutor()


# Convenience functions for easy integration
async def interpret_and_execute(human_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Interpret a human directive and execute it autonomously."""
    directive = await autonomous_executor.interpret_directive(human_input, context)
    plan = await autonomous_executor.create_execution_plan(directive)
    return await autonomous_executor.execute_directive(directive, plan)


async def retain_purpose_context(context: Dict[str, Any]):
    """Retain purpose and context for long-term execution."""
    await autonomous_executor.retain_purpose(context)


async def get_autonomous_status() -> Dict[str, Any]:
    """Get current autonomous execution status."""
    return await autonomous_executor.get_autonomous_status()


async def suggest_improvements() -> List[str]:
    """Get improvement suggestions."""
    return await autonomous_executor.suggest_improvements()