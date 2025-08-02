from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import structlog
from datetime import datetime

logger = structlog.get_logger()


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class AgentConfig:
    name: str
    prompt: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1
    hyperparameters: Dict[str, Any] = None


@dataclass
class AgentResult:
    agent_id: str
    agent_name: str
    result: Any
    confidence_score: float
    execution_time: float
    status: AgentStatus
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class BaseAgent(ABC):
    """Abstract base class for all agents in the Sophie Reflex Orchestrator."""
    
    def __init__(self, config: AgentConfig, agent_id: str = None):
        self.config = config
        self.agent_id = agent_id or f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.status = AgentStatus.IDLE
        self.trust_score = 0.5
        self.execution_count = 0
        self.success_count = 0
        self.last_execution_time = None
        self.metadata = {}
        
        # Initialize hyperparameters
        self.hyperparameters = config.hyperparameters or {}
        
        logger.info("Agent initialized", agent_id=self.agent_id, agent_name=config.name)
    
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Execute the agent's primary function."""
        pass
    
    @abstractmethod
    async def generate_prompt(self, task: str, context: Dict[str, Any] = None) -> str:
        """Generate the prompt for the LLM based on the task and context."""
        pass
    
    async def call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the appropriate LLM based on the agent's configuration."""
        # This will be implemented by specific agent types
        # For now, it's a placeholder that will be overridden
        raise NotImplementedError("LLM calling must be implemented by subclasses")
    
    async def execute_with_retry(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Execute the agent with retry logic."""
        start_time = datetime.now()
        self.status = AgentStatus.RUNNING
        self.execution_count += 1
        
        for attempt in range(self.config.max_retries + 1):
            try:
                logger.info(
                    "Agent execution attempt",
                    agent_id=self.agent_id,
                    attempt=attempt + 1,
                    max_attempts=self.config.max_retries + 1
                )
                
                result = await asyncio.wait_for(
                    self.execute(task, context),
                    timeout=self.config.timeout
                )
                
                self.status = AgentStatus.COMPLETED
                self.success_count += 1
                self.last_execution_time = datetime.now()
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(
                    "Agent execution completed",
                    agent_id=self.agent_id,
                    execution_time=execution_time,
                    confidence_score=result.confidence_score
                )
                
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Execution timeout after {self.config.timeout} seconds"
                logger.warning(
                    "Agent execution timeout",
                    agent_id=self.agent_id,
                    attempt=attempt + 1,
                    error=error_msg
                )
                
                if attempt == self.config.max_retries:
                    self.status = AgentStatus.FAILED
                    return AgentResult(
                        agent_id=self.agent_id,
                        agent_name=self.config.name,
                        result=None,
                        confidence_score=0.0,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        status=AgentStatus.FAILED,
                        error_message=error_msg
                    )
                
                await asyncio.sleep(self.config.retry_delay)
                
            except Exception as e:
                error_msg = f"Execution failed: {str(e)}"
                logger.error(
                    "Agent execution error",
                    agent_id=self.agent_id,
                    attempt=attempt + 1,
                    error=error_msg,
                    exc_info=True
                )
                
                if attempt == self.config.max_retries:
                    self.status = AgentStatus.FAILED
                    return AgentResult(
                        agent_id=self.agent_id,
                        agent_name=self.config.name,
                        result=None,
                        confidence_score=0.0,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        status=AgentStatus.FAILED,
                        error_message=error_msg
                    )
                
                await asyncio.sleep(self.config.retry_delay)
    
    def update_trust_score(self, adjustment: float):
        """Update the agent's trust score."""
        self.trust_score = max(0.0, min(1.0, self.trust_score + adjustment))
        logger.info(
            "Trust score updated",
            agent_id=self.agent_id,
            new_trust_score=self.trust_score,
            adjustment=adjustment
        )
    
    def get_success_rate(self) -> float:
        """Calculate the agent's success rate."""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the agent."""
        return {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "status": self.status.value,
            "trust_score": self.trust_score,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "success_rate": self.get_success_rate(),
            "last_execution_time": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "hyperparameters": self.hyperparameters,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        return f"{self.config.name} ({self.agent_id})"
    
    def __repr__(self) -> str:
        return f"BaseAgent(name={self.config.name}, id={self.agent_id}, status={self.status})"