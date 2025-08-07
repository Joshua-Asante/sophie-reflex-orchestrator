# sophie/core/data_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

class AgentConfig(BaseModel):
    """Configuration for a single agent."""
    name: str = Field(..., description="The unique name for the agent.")
    goal: str = Field(..., description="The primary objective for the agent.")
    provider: Literal["gemini", "openai"] = Field("gemini", description="The LLM provider to use.")
    tools: List[str] = Field([], description="A list of tool names the agent is allowed to use.")

class Task(BaseModel):
    """Defines a single task to be executed by an agent."""
    name: str = Field(..., description="A descriptive name for the task.")
    goal: str = Field(..., description="The specific goal for this task, passed to the agent.")
    agent: str = Field(..., description="The name of the agent config to use for this task.")
    context: Dict[str, Any] = Field({}, description="Additional context to pass to the agent's user prompt.")

class Plan(BaseModel):
    """
    Defines a complete plan, consisting of agent configurations and a sequence of tasks.
    This model is designed to be deserialized from a YAML plan file.
    """
    agents: List[AgentConfig] = Field(..., description="A list of all available agent configurations for this plan.")
    tasks: List[Task] = Field(..., description="The sequence of tasks to execute to complete the plan.")

class TaskRequest(BaseModel):
    """Request model for task execution."""
    mode: str = Field(..., description="Task mode: engineering, research, general, security, creative")
    prompt: str = Field(..., description="User prompt or task description")
    tools: Optional[List[str]] = Field(default=[], description="List of tools to use")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

class TaskResponse(BaseModel):
    """Response model for task execution."""
    success: bool = Field(..., description="Whether the task was successful")
    result: str = Field(..., description="Task result or response")
    confidence: float = Field(..., description="Confidence score (0-1)")
    execution_time: float = Field(..., description="Execution time in seconds")
    model_used: Optional[str] = Field(default=None, description="AI model used")
    tools_used: Optional[List[str]] = Field(default=[], description="Tools used during execution")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")

class TaskStatus(BaseModel):
    """Status model for task execution."""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status: pending, running, completed, failed")
    progress: float = Field(..., description="Progress percentage (0-100)")
    current_step: Optional[str] = Field(default=None, description="Current execution step")
    estimated_time_remaining: Optional[float] = Field(default=None, description="Estimated time remaining in seconds")
    result: Optional[TaskResponse] = Field(default=None, description="Task result when completed")