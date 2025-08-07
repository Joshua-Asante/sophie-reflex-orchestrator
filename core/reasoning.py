"""
The core cognitive engine for SOPHIE. It uses a ReAct (Reason-Act) loop to understand a goal, use tools to gather information, and formulate a final answer.
"""

import inspect
import asyncio
import ast # Refactored: Use ast.literal_eval instead of eval for security
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

from core.tool_registry import ToolRegistry
from core.reasoning_modes import ReasoningMode # Refactored: Import reasoning modes
from pydantic import BaseModel, Field


class ReasoningConfig(BaseModel):
    """Configuration for a reasoning cycle."""
    provider: str = Field(
        default="gemini",
        description="The LLM provider to use for the reasoning loop."
    )
    max_turns: int = Field(
        default=7,
        description="Maximum number of reason-act turns before halting."
    )
    mode: ReasoningMode = Field(
        default=ReasoningMode.FAST_PATH,
        description="The reasoning strategy to employ for the cycle."
    )


class ReasoningResult(BaseModel):
    """Structured output from a successful reasoning cycle."""
    success: bool
    final_answer: str
    turn_history: List[Dict[str, str]]
    error_message: Optional[str] = None


class Reasoner:
    """
    Implements a ReAct (Reason-Act) agent capable of using tools to achieve a goal.
    This is the primary engine for tasks requiring autonomous, multi-step thought.
    """

    def __init__(
        self,
        llm_provider: Callable[[List[Dict[str, str]], str], Coroutine[Any, Any, str]],
        config: Optional[ReasoningConfig] = None
    ):
        """
        Initializes the Reasoner.

        Args:
            llm_provider: An async function that takes messages and a provider string,
                         and returns the LLM's response. This decouples the Reasoner
                         from a specific ModelGateway implementation.
            config: An optional configuration object for the reasoning cycle.
        """
        self.config = config or ReasoningConfig()
        self.llm_provider = llm_provider
        self.tool_registry = ToolRegistry()

    async def execute(self, goal: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """
        Executes the full reasoning cycle for a given goal.

        Args:
            goal: The high-level goal for the agent to achieve.
            context: An optional dictionary of initial context.

        Returns:
            A ReasoningResult object containing the final answer and execution history.
        """
        available_tools = self.tool_registry.get_all_tools()
        system_prompt = self._build_system_prompt(available_tools)
        user_prompt = self._build_user_prompt(goal, context or {})

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # TODO: Implement logic paths based on self.config.mode
        # For now, a single ReAct loop is used for all modes.

        for turn in range(self.config.max_turns):
            try:
                # REASON: The LLM thinks about what to do next.
                llm_response = await self.llm_provider(messages, provider=self.config.provider)
                messages.append({"role": "assistant", "content": llm_response})

                # Check for final answer
                if "FINAL_ANSWER:" in llm_response:
                    final_answer = llm_response.split("FINAL_ANSWER:", 1)[1].strip()
                    return ReasoningResult(
                        success=True,
                        final_answer=final_answer,
                        turn_history=messages
                    )

                # ACT: Parse and execute a tool call
                tool_call = self._parse_tool_call(llm_response, available_tools)
                if tool_call:
                    tool_func, tool_args = tool_call
                    print(f"🛠️  [Reasoner] Using Tool: {tool_func.__name__} with args: {tool_args}")
                    try:
                        observation = str(await self._execute_tool(tool_func, tool_args))
                    except Exception as e:
                        observation = f"[Error] Tool '{tool_func.__name__}' failed: {e}"

                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    messages.append({
                        "role": "user",
                        "content": "Observation: No tool was called. Please proceed by calling a tool or providing a FINAL_ANSWER."
                    })

            except Exception as e:
                return ReasoningResult(
                    success=False,
                    final_answer="",
                    turn_history=messages,
                    error_message=f"An unexpected error occurred in turn {turn + 1}: {e}"
                )

        return ReasoningResult(
            success=False,
            final_answer="",
            turn_history=messages,
            error_message="Reached maximum turns without providing a final answer."
        )

    def _build_system_prompt(self, tools: Dict[str, Callable]) -> str:
        """Builds the system prompt with instructions and available tools."""
        tool_signatures = "\n".join(
            f"- {name}{inspect.signature(func)}" for name, func in tools.items()
        )
        return (
            "You are SOPHIE, a reasoning agent. Your goal is to solve the user's request by thinking step-by-step. "
            "You may invoke ONE tool at a time from the available tools list to gather information. "
            "When you have enough information to answer the user's request, you must respond with 'FINAL_ANSWER:'.\n\n"
            "Your response format is:\n"
            "Thought: <Your reasoning about what to do next>\n"
            "TOOL: <tool_name>(arg1='value1', arg2='value2')\n\n"
            "OR, when you are finished:\n\n"
            "Thought: <Your final thought process>\n"
            "FINAL_ANSWER: <Your final, comprehensive answer to the user's original goal>\n\n"
            f"Available tools:\n{tool_signatures}"
        )

    @staticmethod
    def _build_user_prompt(goal: str, context: Dict[str, Any]) -> str:
        """Builds the initial user prompt with the goal and context."""
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
        return f"The user's goal is: {goal}\n\nInitial Context:\n{context_str}".strip()

    @staticmethod
    def _parse_tool_call(response: str, tools: Dict[str, Callable]) -> Optional[Tuple[Callable, Dict]]:
        """
        Parses a tool call string like 'TOOL: my_tool(arg='value')' from the LLM response.
        Refactored to use ast.literal_eval for safer argument parsing.
        """
        if "TOOL:" not in response:
            return None

        try:
            tool_line = next(line for line in response.splitlines() if "TOOL:" in line)
            action_str = tool_line.split("TOOL:", 1)[1].strip()

            tool_name = action_str.split("(", 1)[0].strip()
            if tool_name not in tools:
                return None

            arg_str = action_str[len(tool_name):].strip().lstrip("(").rstrip(")")

            # Use literal_eval for safely evaluating Python literals
            if arg_str:
                args = ast.literal_eval(f"dict({arg_str})")
            else:
                args = {}

            return tools[tool_name], args
        except (StopIteration, SyntaxError, ValueError):
            # Handles cases where TOOL: line is missing, or args are malformed
            return None

    @staticmethod
    async def _execute_tool(tool_func: Callable, tool_args: Dict) -> Any:
        """Executes a tool function, handling both sync and async functions."""
        if inspect.iscoroutinefunction(tool_func):
            return await tool_func(**tool_args)
        else:
            # For synchronous functions, run them in a separate thread
            # to avoid blocking the asyncio event loop.
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: tool_func(**tool_args))