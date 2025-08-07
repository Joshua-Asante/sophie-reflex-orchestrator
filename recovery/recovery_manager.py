"""
recovery_manager.py

Manages the failure recovery process for the PlanExecutor. When a task fails,
it engages the RevisionEngine to generate a corrected step and logs the
entire recovery attempt for auditing and future learning.
"""

from typing import Optional, Dict, Any
import traceback

from models.failure_report import FailureReport
from recovery.revision_engine import generate_revised_step
from memory.episodic_memory import EpisodicMemory


class RecoveryManager:
    """
    Handles failures in plan execution by triggering LLM-based revisions
    and logging the episode for reflection and traceability.
    """

    def __init__(self, max_attempts: int = 2):
        """
        Initializes the recovery manager.

        Args:
            max_attempts (int): The maximum number of revision attempts for a single failed step.
        """
        self.max_attempts = max_attempts
        self.episodic_memory = EpisodicMemory()

    def handle_failure(
            self,
            failure_report: FailureReport,
            attempt: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Handles a failed step by triggering a revision attempt.

        This method coordinates the process of generating a failure report,
        querying the revision engine for a fix, logging the entire event,
        and returning the proposed new step to the PlanExecutor.

        Args:
            failure_report (FailureReport): A structured report of the failure.
            attempt (int): The current attempt number for this task.

        Returns:
            An optional dictionary containing the revised step, or None if
            recovery is not possible or max attempts have been exceeded.
        """

        if attempt >= self.max_attempts:
            print(f"🔴 [RecoveryManager] Max attempts reached for step: {failure_report.step_name}")
            self._log_recovery_episode(failure_report, None, attempt)
            return None

        # Generate a revised step using the revision engine
        print(
            f"🟠 [RecoveryManager] Attempting revision for step '{failure_report.step_name}' (Attempt {attempt + 1}/{self.max_attempts}).")
        revised_step = generate_revised_step(failure_report)

        # Log the full recovery episode
        self._log_recovery_episode(failure_report, revised_step, attempt)

        if revised_step:
            print(f"🟢 [RecoveryManager] Revision successful. Returning new step for '{failure_report.step_name}'.")
        else:
            print(
                f"🔴 [RecoveryManager] Revision engine failed to produce a valid step for '{failure_report.step_name}'.")

        return revised_step

    def _log_recovery_episode(
            self,
            failure_report: FailureReport,
            revised_step: Optional[Dict[str, Any]],
            attempt: int
    ) -> None:
        """
        Writes a reflexive failure-handling episode to episodic memory.
        """

        # Format a clear user input for the log, showing the failed step
        user_input = (
            f"A failure occurred while executing the plan.\n"
            f"Step Name: {failure_report.step_name}\n"
            f"Tool Used: {failure_report.agent}\n"  # Note: 'agent' in FailureReport is the tool name
            f"Parameters: {failure_report.args}"
        )

        # Format the system output, showing the error and the proposed solution
        system_output = (
            f"[ERROR TYPE]: {failure_report.error_type}\n"
            f"[ERROR MESSAGE]: {failure_report.error_message}\n\n"
            f"[TRACEBACK]:\n{failure_report.traceback}\n\n"
            f"[PROPOSED REVISION]:\n{revised_step or 'None. Max attempts reached or revision failed.'}"
        )

        # Create a rich context dictionary for detailed analysis
        context = {
            "event_type": "failure_recovery",
            "step_name": failure_report.step_name,
            "tool_used": failure_report.agent,
            "parameters": failure_report.args,
            "attempt_number": attempt + 1,
            "revision_successful": revised_step is not None,
            "revised_step": revised_step,
        }

        # Store the entire episode
        self.episodic_memory.store_episode(
            user_input=user_input,
            system_output=system_output,
            context=context,
            feedback="auto-recovery triggered"
        )