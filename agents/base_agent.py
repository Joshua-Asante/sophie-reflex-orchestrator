from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import structlog
import hashlib
import json
import time
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import aiohttp
import redis.asyncio as redis
from dataclasses_json import dataclass_json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

logger = structlog.get_logger()


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass_json
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
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Advanced configuration
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60  # 1 minute
    rate_limit_per_minute: int = 60
    adaptive_learning_enabled: bool = True
    context_window_size: int = 10
    memory_integration_enabled: bool = True


@dataclass_json
@dataclass
class AgentResult:
    agent_id: str
    agent_name: str
    result: Any
    confidence_score: float
    execution_time: float
    status: AgentStatus
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Advanced result fields
    cache_hit: bool = False
    circuit_state: CircuitState = CircuitState.CLOSED
    rate_limit_remaining: Optional[int] = None
    adaptive_learning_applied: bool = False
    context_used: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    cache_hit_rate: float = 0.0
    circuit_breaker_trips: int = 0
    rate_limit_hits: int = 0
    last_execution_time: Optional[datetime] = None
    trust_score_history: List[float] = field(default_factory=list)
    confidence_score_history: List[float] = field(default_factory=list)


class LLMClientManager:
    """Manages LLM client connections with pooling and circuit breaker."""

    def __init__(self):
        self.clients = {}
        self.circuit_breakers = {}
        self.rate_limiters = {}
        self.session = None
        self.redis_client = None

    async def get_client(self, model: str, config: Dict[str, Any]) -> Any:
        """Get or create LLM client with connection pooling."""
        if model not in self.clients:
            try:
                if model == "openai":
                    # Route OpenAI traffic through OpenRouter
                    from llm.openrouter_client import OpenRouterAsyncClient
                    self.clients[model] = OpenRouterAsyncClient()
                    logger.info("OpenRouter client (openai route) created successfully")

                elif model == "google":
                    # Route Google via OpenRouter
                    from llm.openrouter_client import OpenRouterAsyncClient
                    self.clients[model] = OpenRouterAsyncClient()
                    logger.info("OpenRouter client (google route) created successfully")

                elif model == "xai":
                    from llm.openrouter_client import OpenRouterAsyncClient
                    self.clients[model] = OpenRouterAsyncClient()
                    logger.info("OpenRouter client (xai route) created successfully")

                elif model == "mistral":
                    from llm.openrouter_client import OpenRouterAsyncClient
                    self.clients[model] = OpenRouterAsyncClient()
                    logger.info("OpenRouter client (mistral route) created successfully")

                elif model == "deepseek":
                    from llm.openrouter_client import OpenRouterAsyncClient
                    self.clients[model] = OpenRouterAsyncClient()
                    logger.info("OpenRouter client (deepseek route) created successfully")

                elif model == "kimi":
                    from llm.openrouter_client import OpenRouterAsyncClient
                    self.clients[model] = OpenRouterAsyncClient()
                    logger.info("OpenRouter client (kimi route) created successfully")

                elif model == "glm":
                    from llm.openrouter_client import OpenRouterAsyncClient
                    self.clients[model] = OpenRouterAsyncClient()
                    logger.info("OpenRouter client (glm route) created successfully")

                else:
                    raise ValueError(f"Unsupported model: {model}")

            except Exception as e:
                logger.error(f"Failed to create client for {model}", error=str(e), error_type=type(e).__name__)
                raise

        return self.clients[model]

    async def get_circuit_breaker(self, key: str) -> CircuitState:
        """Get circuit breaker state for a specific key."""
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "last_failure_time": None
            }
        return self.circuit_breakers[key]["state"]

    async def record_failure(self, key: str, threshold: int, timeout: int):
        """Record a failure and potentially open circuit breaker."""
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "last_failure_time": None
            }

        cb = self.circuit_breakers[key]
        cb["failure_count"] += 1
        cb["last_failure_time"] = datetime.now()

        if cb["failure_count"] >= threshold:
            cb["state"] = CircuitState.OPEN
            logger.warning("Circuit breaker opened", key=key, failure_count=cb["failure_count"])

        # Auto-close after timeout
        if cb["state"] == CircuitState.OPEN:
            asyncio.create_task(self._auto_close_circuit(key, timeout))

    async def _auto_close_circuit(self, key: str, timeout: int):
        """Automatically close circuit breaker after timeout."""
        await asyncio.sleep(timeout)
        if key in self.circuit_breakers:
            self.circuit_breakers[key]["state"] = CircuitState.HALF_OPEN
            logger.info("Circuit breaker half-open", key=key)

    async def record_success(self, key: str):
        """Record a success and close circuit breaker."""
        if key in self.circuit_breakers:
            self.circuit_breakers[key]["state"] = CircuitState.CLOSED
            self.circuit_breakers[key]["failure_count"] = 0


class CacheManager:
    """Manages response caching with Redis and in-memory fallback."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        self.in_memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}

        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info("Redis cache initialized", redis_url=redis_url)
            except Exception as e:
                logger.warning("Redis cache initialization failed, using in-memory", error=str(e))

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        try:
            if self.redis_client:
                cached = await self.redis_client.get(key)
                if cached:
                    self.cache_stats["hits"] += 1
                    return json.loads(cached)

            # Fallback to in-memory cache
            if key in self.in_memory_cache:
                self.cache_stats["hits"] += 1
                return self.in_memory_cache[key]

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Set cached response."""
        try:
            if self.redis_client:
                await self.redis_client.setex(key, ttl, json.dumps(value))

            # Also cache in memory
            self.in_memory_cache[key] = value

        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total if total > 0 else 0.0

        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "in_memory_size": len(self.in_memory_cache)
        }


class AdaptiveLearning:
    """Implements adaptive learning for agents based on performance history."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.performance_history = []
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.1

    def update_performance(self, result: AgentResult):
        """Update performance history and learn from results."""
        self.performance_history.append({
            "timestamp": datetime.now(),
            "confidence_score": result.confidence_score,
            "execution_time": result.execution_time,
            "success": result.status == AgentStatus.COMPLETED
        })

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]

    def get_adaptive_parameters(self) -> Dict[str, Any]:
        """Get adaptive parameters based on performance history."""
        if len(self.performance_history) < 5:
            return {}

        recent_performance = self.performance_history[-10:]
        avg_confidence = sum(p["confidence_score"] for p in recent_performance) / len(recent_performance)
        avg_execution_time = sum(p["execution_time"] for p in recent_performance) / len(recent_performance)
        success_rate = sum(1 for p in recent_performance if p["success"]) / len(recent_performance)

        adaptations = {}

        # Adapt temperature based on confidence
        if avg_confidence < 0.6:
            adaptations["temperature_adjustment"] = -0.1
        elif avg_confidence > 0.8:
            adaptations["temperature_adjustment"] = 0.1

        # Adapt max_tokens based on execution time
        if avg_execution_time > 25:  # Near timeout
            adaptations["max_tokens_adjustment"] = -200
        elif avg_execution_time < 5:  # Very fast
            adaptations["max_tokens_adjustment"] = 200

        # Adapt retry strategy based on success rate
        if success_rate < 0.7:
            adaptations["retry_delay_adjustment"] = 2  # Increase delay

        return adaptations


class BaseAgent(ABC):
    """Enhanced abstract base class for all agents in the Sophie Reflex Orchestrator."""

    # Class-level managers for connection pooling
    _llm_manager = LLMClientManager()
    _cache_manager = CacheManager()

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

        # Advanced features
        self.adaptive_learning = AdaptiveLearning(self.agent_id) if config.adaptive_learning_enabled else None
        self.performance_metrics = PerformanceMetrics()
        self.context_window = []
        self.rate_limit_tokens = config.rate_limit_per_minute
        self.last_rate_limit_reset = datetime.now()

        logger.info("Enhanced agent initialized",
                   agent_id=self.agent_id,
                   agent_name=config.name,
                   adaptive_learning=config.adaptive_learning_enabled,
                   cache_enabled=config.cache_enabled)

    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Execute the agent's primary function."""
        pass

    @abstractmethod
    async def generate_prompt(self, task: str, context: Dict[str, Any] = None) -> str:
        """Generate the prompt for the LLM based on the task and context."""
        pass

    async def call_llm(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced LLM calling with caching, circuit breaker, and rate limiting."""
        context = context or {}
        start_time = time.time()

        # Generate cache key
        cache_key = self._generate_cache_key(prompt, context)

        # Check cache first
        if self.config.cache_enabled:
            cached_response = await self._cache_manager.get(cache_key)
            if cached_response:
                logger.info("Cache hit", agent_id=self.agent_id, cache_key=cache_key)
                return {**cached_response, "cache_hit": True}

        # Check circuit breaker
        circuit_key = f"{self.agent_id}_{self.config.model}"
        circuit_state = await self._llm_manager.get_circuit_breaker(circuit_key)

        if circuit_state == CircuitState.OPEN:
            raise Exception("Circuit breaker is open")

        # Check rate limiting
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")

        try:
            # Call LLM
            response = await self._call_llm_implementation(prompt, context)

            # Record success
            await self._llm_manager.record_success(circuit_key)

            # Cache response
            if self.config.cache_enabled:
                await self._cache_manager.set(cache_key, response, self.config.cache_ttl)

            execution_time = time.time() - start_time
            logger.info("LLM call successful",
                       agent_id=self.agent_id,
                       execution_time=execution_time,
                       cache_key=cache_key)

            return response

        except Exception as e:
            # Record failure
            await self._llm_manager.record_failure(
                circuit_key,
                self.config.circuit_breaker_threshold,
                self.config.circuit_breaker_timeout
            )
            raise

    async def _call_llm_implementation(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of LLM calling - to be overridden by subclasses."""
        raise NotImplementedError("LLM calling must be implemented by subclasses")

    def _generate_cache_key(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate cache key for prompt and context."""
        content = f"{prompt}:{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _check_rate_limit(self) -> bool:
        """Check and update rate limiting."""
        now = datetime.now()

        # Reset tokens if minute has passed
        if (now - self.last_rate_limit_reset).seconds >= 60:
            self.rate_limit_tokens = self.config.rate_limit_per_minute
            self.last_rate_limit_reset = now

        if self.rate_limit_tokens <= 0:
            return False

        self.rate_limit_tokens -= 1
        return True

    async def execute_with_retry(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Enhanced execute with retry logic, adaptive learning, and comprehensive monitoring."""
        start_time = datetime.now()
        self.status = AgentStatus.RUNNING
        self.execution_count += 1

        # Apply adaptive learning if enabled
        adaptive_params = {}
        if self.adaptive_learning:
            adaptive_params = self.adaptive_learning.get_adaptive_parameters()
            if adaptive_params:
                logger.info("Applied adaptive parameters",
                           agent_id=self.agent_id,
                           adaptive_params=adaptive_params)

        # Update context with adaptive parameters
        context = context or {}
        context["adaptive_parameters"] = adaptive_params

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.info(
                    "Agent execution attempt",
                    agent_id=self.agent_id,
                    attempt=attempt + 1,
                    max_attempts=self.config.max_retries + 1,
                    adaptive_params=adaptive_params
                )

                result = await asyncio.wait_for(
                    self.execute(task, context),
                    timeout=self.config.timeout
                )

                # Update performance metrics
                execution_time = (datetime.now() - start_time).total_seconds()
                self._update_performance_metrics(result, execution_time)

                # Apply adaptive learning
                if self.adaptive_learning:
                    self.adaptive_learning.update_performance(result)

                self.status = AgentStatus.COMPLETED
                self.success_count += 1
                self.last_execution_time = datetime.now()

                logger.info(
                    "Agent execution completed",
                    agent_id=self.agent_id,
                    execution_time=execution_time,
                    confidence_score=result.confidence_score,
                    cache_hit=result.cache_hit
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
                    return self._create_error_result(error_msg, start_time)

                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff

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
                    return self._create_error_result(error_msg, start_time)

                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff

    def _update_performance_metrics(self, result: AgentResult, execution_time: float):
        """Update performance metrics."""
        self.performance_metrics.total_executions += 1

        if result.status == AgentStatus.COMPLETED:
            self.performance_metrics.successful_executions += 1
        else:
            self.performance_metrics.failed_executions += 1

        # Update average execution time
        total_time = self.performance_metrics.average_execution_time * (self.performance_metrics.total_executions - 1)
        self.performance_metrics.average_execution_time = (total_time + execution_time) / self.performance_metrics.total_executions

        # Update cache hit rate
        cache_stats = self._cache_manager.get_stats()
        self.performance_metrics.cache_hit_rate = cache_stats["hit_rate"]

        # Update history
        self.performance_metrics.trust_score_history.append(self.trust_score)
        self.performance_metrics.confidence_score_history.append(result.confidence_score)

        # Keep history manageable
        if len(self.performance_metrics.trust_score_history) > 100:
            self.performance_metrics.trust_score_history = self.performance_metrics.trust_score_history[-50:]
            self.performance_metrics.confidence_score_history = self.performance_metrics.confidence_score_history[-50:]

        self.performance_metrics.last_execution_time = datetime.now()

    def _create_error_result(self, error_msg: str, start_time: datetime) -> AgentResult:
        """Create error result with comprehensive metadata."""
        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.config.name,
            result=None,
            confidence_score=0.0,
            execution_time=(datetime.now() - start_time).total_seconds(),
            status=AgentStatus.FAILED,
            error_message=error_msg,
            metadata={
                "attempts": self.config.max_retries + 1,
                "circuit_state": "unknown",
                "rate_limit_remaining": self.rate_limit_tokens
            }
        )

    def update_trust_score(self, adjustment: float):
        """Update the agent's trust score with bounds checking."""
        old_trust_score = self.trust_score
        self.trust_score = max(0.0, min(1.0, self.trust_score + adjustment))

        logger.info(
            "Trust score updated",
            agent_id=self.agent_id,
            old_trust_score=old_trust_score,
            new_trust_score=self.trust_score,
            adjustment=adjustment
        )

    def get_success_rate(self) -> float:
        """Calculate the agent's success rate."""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
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
            "metadata": self.metadata,
            "performance_metrics": {
                "total_executions": self.performance_metrics.total_executions,
                "successful_executions": self.performance_metrics.successful_executions,
                "failed_executions": self.performance_metrics.failed_executions,
                "average_execution_time": self.performance_metrics.average_execution_time,
                "cache_hit_rate": self.performance_metrics.cache_hit_rate,
                "circuit_breaker_trips": self.performance_metrics.circuit_breaker_trips,
                "rate_limit_hits": self.performance_metrics.rate_limit_hits
            },
            "adaptive_learning": {
                "enabled": self.adaptive_learning is not None,
                "history_size": len(self.adaptive_learning.performance_history) if self.adaptive_learning else 0
            },
            "cache_stats": self._cache_manager.get_stats()
        }

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the agent (backward compatibility)."""
        return self.get_performance_metrics()

    def __str__(self) -> str:
        return f"{self.config.name} ({self.agent_id})"

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.config.name}, id={self.agent_id}, status={self.status})"
