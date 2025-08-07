"""
reflex_router.py

This is the central cognitive router for the SOPHIE v1 system, acting as its "brain."
It receives an initial user prompt, uses an LLM-based assessor to analyze its
complexity and nature, and then delegates the task to the most appropriate
specialized execution engine.
"""

from typing import Dict, Any, Optional

# --- Import the Four Execution Engines ---
# Mock imports for now - these would be implemented in production
# from reasoning.reasoning import Reasoner
# from core.plan_executor import PlanExecutor
# from orchestration.orchestrator import CouncilOrchestrator
# from core.code_task import CodeTaskManager

# --- Import Supporting Components ---
from .query_complexity_assessor import QueryComplexityAssessor
from .plan_generator import generate_plan
from memory.episodic_memory import EpisodicMemory


class ReflexRouter:
    """
    SOPHIE's central cognitive routing mechanism.

    It determines which of the four execution modes should handle the next task,
    based on a complexity and intent assessment of the user's prompt.
    """

    def __init__(self):
        """Initializes the router and all its subordinate systems."""
        # The assessor helps the router understand the user's request.
        self.assessor = QueryComplexityAssessor()

        # Instantiate each of the four execution engines.
        # self.reasoner = Reasoner()  # Mock for now
        # self.plan_executor = PlanExecutor(verbose=True)  # Mock for now
        from .council_orchestrator import CouncilOrchestrator
        self.council = CouncilOrchestrator()
        # self.project_manager = CodeTaskManager()  # Mock for now
        # self.plan_generator = PlanGenerator()  # Mock for now

        # The router is responsible for ensuring all actions are logged.
        self.episodic_memory = EpisodicMemory()

    async def route(self, user_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Routes the user input to the appropriate execution engine.

        This is the main entry point for any user request into the SOPHIE system.

        Args:
            user_prompt: The natural language prompt from the user.
            context: Optional additional context for the request.

        Returns:
            A structured dictionary containing the result of the execution.
        """
        print(f"🧠 [ReflexRouter] Received prompt: '{user_prompt}'")
        print(" assessing complexity and intent...")

        # 1. Assess the prompt to decide on a strategy.
        assessment = self.assessor.assess(user_prompt)
        print(f"📊 [ReflexRouter] Assessment complete: {assessment}")

        # 2. Route to the appropriate execution engine based on the assessment.
        result: Dict[str, Any] = {}
        execution_mode = "Unknown"

        # --- Routing Logic ---

        # Route to Project Mode if the goal is to create a complex, versioned artifact.
        if "develop" in user_prompt.lower() or "create a script" in user_prompt.lower() or "write a report" in user_prompt.lower():
            execution_mode = "Project Mode (CodeTaskManager)"
            # Mock implementation for now
            result = {"status": "Project mode not implemented yet", "prompt": user_prompt}

        # Route to Council Mode if the query is ambiguous, divergent, or high-stakes.
        elif assessment.get('has_ambiguity') or assessment.get('needs_divergence'):
            execution_mode = "Council Mode"
            result = await self.council.run_council(user_prompt)

        # Route to Planner Mode if the query implies multiple steps or tool use.
        elif assessment.get('num_implicit_subtasks', 0) > 1:
            execution_mode = "Planner Mode"
            print("📝 [ReflexRouter] Generating a new plan...")
            try:
                # Mock plan generation for now
                result = {"status": "Planner mode not fully implemented yet", "prompt": user_prompt}
            except Exception as e:
                result = {"error": f"Plan generation or execution failed: {e}"}

        # Default to Reasoner Mode for direct, single-shot cognitive tasks.
        else:
            execution_mode = "Reasoner Mode"
            # Mock reasoning for now
            result = {"status": "Reasoner mode not implemented yet", "prompt": user_prompt}

        print(f"✅ [ReflexRouter] Execution complete. Mode used: {execution_mode}")

        # 3. Log the entire interaction to episodic memory.
        self.episodic_memory.store_episode(
            user_input=user_prompt,
            system_output=str(result),  # Store the final result as a string
            context={
                "execution_mode": execution_mode,
                "initial_assessment": assessment,
                **(context or {})
            }
        )

        return {
            "execution_mode": execution_mode,
            "assessment": assessment,
            "result": result
        }
