"""
Enhanced Error Recovery System

Provides intelligent error recovery with circuit breakers, exponential backoff,
and adaptive retry strategies for improved system stability.
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors for different recovery strategies."""
    NETWORK_ERROR = "network_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorInfo:
    """Information about an error for recovery decisions."""
    error_type: ErrorType
    error_message: str
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def record_success(self):
        """Record a success and potentially close the circuit."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker closed after successful request")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moved to half-open state")
                return True
            return False
        
        if self.state == "HALF_OPEN":
            return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "time_since_last_failure": time.time() - self.last_failure_time
        }


class ExponentialBackoff:
    """Exponential backoff with jitter for retry strategies."""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, max_retries: int = 5):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
    
    def get_delay(self, retry_count: int) -> float:
        """Calculate delay for a given retry count."""
        if retry_count >= self.max_retries:
            return self.max_delay
        
        # Exponential backoff: base_delay * 2^retry_count
        delay = self.base_delay * (2 ** retry_count)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, 0.1 * delay)
        delay += jitter
        
        return min(delay, self.max_delay)
    
    def should_retry(self, retry_count: int, error_type: ErrorType) -> bool:
        """Determine if retry should be attempted."""
        if retry_count >= self.max_retries:
            return False
        
        # Don't retry certain error types
        if error_type in [ErrorType.AUTHENTICATION_ERROR, ErrorType.VALIDATION_ERROR]:
            return False
        
        return True


class ErrorRecoveryManager:
    """Manages error recovery strategies for different components."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.backoff_strategies: Dict[str, ExponentialBackoff] = {}
        self.error_history: Dict[str, List[ErrorInfo]] = {}
    
    def get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get or create circuit breaker for a component."""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
        return self.circuit_breakers[component]
    
    def get_backoff_strategy(self, component: str) -> ExponentialBackoff:
        """Get or create backoff strategy for a component."""
        if component not in self.backoff_strategies:
            self.backoff_strategies[component] = ExponentialBackoff()
        return self.backoff_strategies[component]
    
    def record_error(self, component: str, error_info: ErrorInfo):
        """Record an error for a component."""
        if component not in self.error_history:
            self.error_history[component] = []
        
        self.error_history[component].append(error_info)
        
        # Update circuit breaker
        circuit_breaker = self.get_circuit_breaker(component)
        circuit_breaker.record_failure()
        
        # Keep only recent errors (last 100)
        if len(self.error_history[component]) > 100:
            self.error_history[component] = self.error_history[component][-100:]
    
    def record_success(self, component: str):
        """Record a success for a component."""
        circuit_breaker = self.get_circuit_breaker(component)
        circuit_breaker.record_success()
    
    def can_execute(self, component: str) -> bool:
        """Check if execution is allowed for a component."""
        circuit_breaker = self.get_circuit_breaker(component)
        return circuit_breaker.can_execute()
    
    def get_retry_delay(self, component: str, retry_count: int) -> float:
        """Get retry delay for a component."""
        backoff = self.get_backoff_strategy(component)
        return backoff.get_delay(retry_count)
    
    def should_retry(self, component: str, retry_count: int, error_type: ErrorType) -> bool:
        """Determine if retry should be attempted."""
        backoff = self.get_backoff_strategy(component)
        return backoff.should_retry(retry_count, error_type)
    
    def get_component_status(self, component: str) -> Dict[str, Any]:
        """Get status for a specific component."""
        circuit_breaker = self.get_circuit_breaker(component)
        backoff = self.get_backoff_strategy(component)
        
        recent_errors = self.error_history.get(component, [])
        error_types = [error.error_type.value for error in recent_errors[-10:]]
        
        return {
            "circuit_breaker": circuit_breaker.get_status(),
            "recent_error_types": error_types,
            "total_errors": len(recent_errors),
            "can_execute": circuit_breaker.can_execute()
        }
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global error recovery status."""
        component_statuses = {}
        for component in self.circuit_breakers.keys():
            component_statuses[component] = self.get_component_status(component)
        
        return {
            "components": component_statuses,
            "total_components": len(self.circuit_breakers)
        }


class RetryableOperation:
    """Wrapper for operations that can be retried with error recovery."""
    
    def __init__(self, recovery_manager: ErrorRecoveryManager, component: str):
        self.recovery_manager = recovery_manager
        self.component = component
    
    async def execute(self, operation: Callable[[], Awaitable[Any]], 
                     context: Dict[str, Any] = None) -> Any:
        """Execute an operation with retry logic."""
        retry_count = 0
        context = context or {}
        
        while True:
            # Check circuit breaker
            if not self.recovery_manager.can_execute(self.component):
                raise Exception(f"Circuit breaker is open for {self.component}")
            
            try:
                # Execute the operation
                result = await operation()
                
                # Record success
                self.recovery_manager.record_success(self.component)
                
                return result
                
            except Exception as e:
                # Determine error type
                error_type = self._classify_error(e)
                error_info = ErrorInfo(
                    error_type=error_type,
                    error_message=str(e),
                    retry_count=retry_count,
                    context=context
                )
                
                # Record error
                self.recovery_manager.record_error(self.component, error_info)
                
                # Check if we should retry
                if not self.recovery_manager.should_retry(self.component, retry_count, error_type):
                    logger.error(f"Max retries reached for {self.component}: {e}")
                    raise e
                
                # Calculate delay
                delay = self.recovery_manager.get_retry_delay(self.component, retry_count)
                
                logger.warning(f"Retrying {self.component} in {delay:.2f}s (attempt {retry_count + 1}): {e}")
                
                # Wait before retry
                await asyncio.sleep(delay)
                retry_count += 1
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify an error for appropriate recovery strategy."""
        error_str = str(error).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT_ERROR
        elif "authentication" in error_str or "401" in error_str or "403" in error_str:
            return ErrorType.AUTHENTICATION_ERROR
        elif "validation" in error_str or "400" in error_str:
            return ErrorType.VALIDATION_ERROR
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR


class AdaptiveRetryStrategy:
    """Adaptive retry strategy that learns from error patterns."""
    
    def __init__(self, recovery_manager: ErrorRecoveryManager):
        self.recovery_manager = recovery_manager
        self.error_patterns: Dict[str, Dict[str, int]] = {}
    
    def analyze_error_patterns(self, component: str) -> Dict[str, Any]:
        """Analyze error patterns for a component."""
        errors = self.recovery_manager.error_history.get(component, [])
        
        if not errors:
            return {"total_errors": 0, "patterns": {}}
        
        # Count error types
        error_counts = {}
        for error in errors:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Calculate error rate
        total_errors = len(errors)
        error_rate = total_errors / max(1, len(errors))
        
        return {
            "total_errors": total_errors,
            "error_rate": error_rate,
            "patterns": error_counts,
            "recent_errors": [error.error_type.value for error in errors[-5:]]
        }
    
    def should_adapt_strategy(self, component: str) -> bool:
        """Determine if retry strategy should be adapted."""
        patterns = self.analyze_error_patterns(component)
        
        # Adapt if error rate is high
        if patterns["error_rate"] > 0.5:
            return True
        
        # Adapt if there are many rate limit errors
        if patterns["patterns"].get("rate_limit_error", 0) > 3:
            return True
        
        return False
    
    def adapt_strategy(self, component: str):
        """Adapt retry strategy based on error patterns."""
        patterns = self.analyze_error_patterns(component)
        
        if patterns["patterns"].get("rate_limit_error", 0) > 3:
            # Increase delays for rate limit errors
            backoff = self.recovery_manager.get_backoff_strategy(component)
            backoff.base_delay = min(backoff.base_delay * 2, backoff.max_delay)
            logger.info(f"Adapted strategy for {component}: increased base delay")
        
        if patterns["error_rate"] > 0.7:
            # Lower failure threshold for high error rates
            circuit_breaker = self.recovery_manager.get_circuit_breaker(component)
            circuit_breaker.failure_threshold = max(2, circuit_breaker.failure_threshold - 1)
            logger.info(f"Adapted strategy for {component}: lowered failure threshold")


# Global instance
error_recovery_manager = ErrorRecoveryManager()
adaptive_strategy = AdaptiveRetryStrategy(error_recovery_manager) 