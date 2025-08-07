"""
council_orchestrator.py

Core coordinator for Council Mode execution. This engine routes a single prompt
through a sequential pipeline of specialized LLM agents, each performing a
specific step in a proprietary workflow: Strategy, Tooling, Execution, Review,
and Tests. The final output is a summary of this collaborative chain of thought.
"""

import asyncio
from typing import Dict, Any, List

# --- Import Core Golden Path Components ---
from llm_registry import LLMRegistry
from llm.model_gateway import ModelGateway
from utils.debug_trace import DebugTrace
from governance.trust_manager import TrustManager


class CouncilOrchestrator:
    """
    Manages the sequential "Assembly Line" execution mode.
    """

    def __init__(self):
        """Initializes all necessary components for council operation."""
        self.registry = LLMRegistry()
        self.gateway = ModelGateway.get()
        self.trust_manager = TrustManager()
        self.logger = DebugTrace()

        # Define the roles for each stage in the sequential pipeline.
        # This maps a proprietary stage to a model role defined in the LLMRegistry.
        self.pipeline_stages = {
            "Strategy": "Navigator",
            "Tooling": "Specialist",
            "Execution": "Pragmatist",
            "Review": "Integrator",
            "Tests": "Adversarial Probe"
        }

    def _get_model_for_stage(self, stage_name: str) -> Any:
        """Finds the registered model assigned to a specific pipeline role."""
        role = self.pipeline_stages.get(stage_name)
        if not role:
            raise ValueError(f"No role defined for stage: {stage_name}")

        for model in self.registry.all_models().values():
            if model.role == role:
                return model

        raise ValueError(f"No model found in registry with role: {role}")

    async def run_council(self, user_prompt: str) -> Dict[str, Any]:
        """
        Executes the full sequential council workflow for a given prompt.

        Args:
            user_prompt: The natural language prompt to be sent to the council.

        Returns:
            A dictionary containing the final summary and the detailed
            chain of thought from the sequential execution.
        """
        session_id = f"council-seq-{asyncio.get_running_loop().time()}"
        print(f"🏭 [CouncilOrchestrator] Sequential Pipeline activated for session: {session_id}")

        chain_of_thought: List[Dict[str, Any]] = []
        current_input = user_prompt

        # Sequentially execute each stage of the pipeline
        for stage_name, role in self.pipeline_stages.items():
            print(f"➡️  Processing Stage: {stage_name} ({role})")

            try:
                model = self._get_model_for_stage(stage_name)

                # Frame the prompt for the current stage, including the previous output
                stage_prompt = f"You are the {role}. The original goal is: '{user_prompt}'.\n\n"
                if chain_of_thought:
                    previous_stage_output = chain_of_thought[-1]['output']
                    stage_prompt += f"The output from the previous stage ({chain_of_thought[-1]['stage']}) was:\n---\n{previous_stage_output}\n---\n\n"
                stage_prompt += f"Your task is to perform the '{stage_name}' step. Provide a concise output for the next stage."

                # Call the model for the current stage
                output = await self.gateway.chat(stage_prompt, provider=model.adapter_key)

                stage_result = {
                    "stage": stage_name,
                    "model": model.name,
                    "role": role,
                    "output": output
                }
                chain_of_thought.append(stage_result)
                current_input = output # The output of one stage is the input for the next

                self.logger.log("stage_complete", stage_result, session_id)

            except Exception as e:
                error_details = {"stage": stage_name, "error": str(e)}
                self.logger.log("stage_error", error_details, session_id)
                return {
                    "status": "error",
                    "error_message": f"Pipeline failed at stage '{stage_name}'.",
                    "details": str(e),
                    "chain_of_thought": chain_of_thought
                }

        # Summarize the entire sequence for the final output
        final_summary = self._summarize_chain(user_prompt, chain_of_thought)

        final_decision = {
            "status": "complete",
            "summary": final_summary,
            "chain_of_thought": chain_of_thought
        }
        self.logger.log("final_pipeline_summary", final_decision, session_id)

        return final_decision

    def _summarize_chain(self, original_prompt: str, chain: List[Dict[str, Any]]) -> str:
        """Creates a final summary of the entire sequential process."""
        summary = f"For the original prompt '{original_prompt}', the council performed the following steps:\n\n"
        for item in chain:
            summary += f"**Stage: {item['stage']} ({item['model']})**\n"
            summary += f"{item['output']}\n\n"
        return summary.strip()
