"""
Performance Integration Module

Integrates all performance optimization components into a unified system
for the Sophie Reflex Orchestrator.
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable, Awaitable
import logging

from .connection_pool import connection_manager, ProviderType
from .batch_processor import batch_processor
from .smart_cache import smart_cache
from .error_recovery import error_recovery_manager, RetryableOperation
from .performance_monitor import performance_monitor, monitor_performance

logger = logging.getLogger(__name__)


class OptimizedLLMClient:
    """Optimized LLM client that uses all performance components."""
    
    def __init__(self):
        self.retryable_operations = {}
    
    @monitor_performance("llm_client", "generate")
    async def generate_text(
        self,
        prompt: str,
        model: str,
        provider: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: str = None
    ) -> str:
        """Generate text using optimized LLM client."""
        
        # Check cache first
        cached_response = await smart_cache.get_llm_response(
            prompt, model, provider, temperature, max_tokens
        )
        if cached_response:
            logger.info("Cache hit for LLM request")
            return cached_response
        
        # Get retryable operation for this provider
        if provider not in self.retryable_operations:
            self.retryable_operations[provider] = RetryableOperation(
                error_recovery_manager, f"llm_{provider}"
            )
        
        retryable_op = self.retryable_operations[provider]
        
        # Execute with retry logic
        result = await retryable_op.execute(
            lambda: self._make_llm_request(prompt, model, provider, temperature, max_tokens, api_key),
            {"prompt_length": len(prompt), "model": model}
        )
        
        # Cache the result
        smart_cache.cache_llm_response(prompt, result, model, provider, temperature, max_tokens)
        
        return result
    
    async def _make_llm_request(
        self,
        prompt: str,
        model: str,
        provider: str,
        temperature: float,
        max_tokens: int,
        api_key: str
    ) -> str:
        """Make the actual LLM request."""
        try:
            # Get provider type
            provider_type = ProviderType(provider)
            
            # Get client from connection pool
            client = await connection_manager.get_client(provider_type, api_key)
            
            # Prepare request payload
            payload = self._prepare_payload(provider_type, model, prompt, temperature, max_tokens)
            
            # Make request
            response = await self._send_request(client, provider_type, payload)
            
            # Extract response text
            result = self._extract_response_text(provider_type, response)
            
            # Return client to pool
            await connection_manager.return_client(provider_type, client)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def _prepare_payload(self, provider_type: ProviderType, model: str, prompt: str, 
                        temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Prepare request payload for different providers."""
        if provider_type == ProviderType.OPENAI:
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        elif provider_type == ProviderType.GOOGLE:
            return {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }
        else:
            # Generic payload for other providers
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
    
    async def _send_request(self, client, provider_type: ProviderType, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to the appropriate endpoint."""
        if provider_type == ProviderType.OPENAI:
            response = await client.post("/chat/completions", json=payload)
        elif provider_type == ProviderType.GOOGLE:
            response = await client.post(f"/v1beta/models/{payload.get('model', 'gemini-pro')}:generateContent", json=payload)
        else:
            response = await client.post("/chat/completions", json=payload)
        
        response.raise_for_status()
        return response.json()
    
    def _extract_response_text(self, provider_type: ProviderType, response: Dict[str, Any]) -> str:
        """Extract response text from provider-specific response format."""
        if provider_type == ProviderType.OPENAI:
            return response["choices"][0]["message"]["content"]
        elif provider_type == ProviderType.GOOGLE:
            return response["candidates"][0]["content"]["parts"][0]["text"]
        else:
            # Generic extraction
            if "choices" in response:
                return response["choices"][0]["message"]["content"]
            elif "candidates" in response:
                return response["candidates"][0]["content"]["parts"][0]["text"]
            else:
                raise ValueError("Unknown response format")


class OptimizedToolExecutor:
    """Optimized tool executor with caching and error recovery."""
    
    def __init__(self):
        self.retryable_operations = {}
    
    @monitor_performance("tool_executor", "execute")
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool with optimization."""
        
        # Check cache first
        cached_result = await smart_cache.get_tool_result(tool_name, params)
        if cached_result:
            logger.info(f"Cache hit for tool: {tool_name}")
            return cached_result
        
        # Get retryable operation for this tool
        if tool_name not in self.retryable_operations:
            self.retryable_operations[tool_name] = RetryableOperation(
                error_recovery_manager, f"tool_{tool_name}"
            )
        
        retryable_op = self.retryable_operations[tool_name]
        
        # Execute with retry logic
        result = await retryable_op.execute(
            lambda: self._execute_tool_internal(tool_name, params),
            {"tool_name": tool_name, "params": str(params)}
        )
        
        # Cache the result
        smart_cache.cache_tool_result(tool_name, params, result)
        
        return result
    
    async def _execute_tool_internal(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute the actual tool."""
        # This would integrate with your existing tool registry
        # For now, we'll simulate tool execution
        logger.info(f"Executing tool: {tool_name} with params: {params}")
        
        # Simulate tool execution
        await asyncio.sleep(0.1)
        
        return f"Result from {tool_name}: {params}"


class PerformanceOptimizer:
    """Main performance optimizer that coordinates all components."""
    
    def __init__(self):
        self.llm_client = OptimizedLLMClient()
        self.tool_executor = OptimizedToolExecutor()
        self.optimization_enabled = True
    
    async def optimize_system(self):
        """Run system optimization."""
        if not self.optimization_enabled:
            return
        
        logger.info("Starting system optimization...")
        
        # Clean up expired cache entries
        await smart_cache.cleanup_expired()
        
        # Adapt error recovery strategies
        for component in error_recovery_manager.circuit_breakers.keys():
            # Note: adaptive_strategy would be imported from error_recovery module
            # For now, we'll skip this to avoid import issues
            pass
        
        # Log performance summary
        summary = performance_monitor.get_performance_summary()
        logger.info(f"Performance summary: {summary}")
        
        # Log cache statistics
        cache_stats = smart_cache.get_statistics()
        logger.info(f"Cache statistics: {cache_stats}")
        
        # Log batch statistics
        batch_stats = batch_processor.get_batch_statistics()
        logger.info(f"Batch statistics: {batch_stats}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "performance": performance_monitor.get_performance_summary(),
            "cache": smart_cache.get_statistics(),
            "batch_processing": batch_processor.get_batch_statistics(),
            "error_recovery": error_recovery_manager.get_global_status(),
            "connection_pools": {
                "total_providers": len(connection_manager.pool.configs),
                "active_connections": sum(
                    len(pool._queue) if hasattr(pool, '_queue') else 0 
                    for pool in connection_manager.pool.pools.values()
                )
            }
        }
    
    async def shutdown(self):
        """Shutdown all performance components."""
        logger.info("Shutting down performance optimization system...")
        
        # Shutdown batch processor
        await batch_processor.shutdown()
        
        # Shutdown cache
        await smart_cache.shutdown()
        
        # Close connection pools
        await connection_manager.close()
        
        logger.info("Performance optimization system shutdown complete")


# Global instance
performance_optimizer = PerformanceOptimizer()


# Convenience functions for easy integration
async def optimized_llm_call(prompt: str, model: str, provider: str, 
                           temperature: float = 0.7, max_tokens: int = 2000,
                           api_key: str = None) -> str:
    """Make an optimized LLM call."""
    return await performance_optimizer.llm_client.generate_text(
        prompt, model, provider, temperature, max_tokens, api_key
    )


async def optimized_tool_call(tool_name: str, params: Dict[str, Any]) -> Any:
    """Make an optimized tool call."""
    return await performance_optimizer.tool_executor.execute_tool(tool_name, params)


async def get_performance_status() -> Dict[str, Any]:
    """Get current performance status."""
    return await performance_optimizer.get_system_status()


async def run_performance_optimization():
    """Run performance optimization."""
    await performance_optimizer.optimize_system()


async def shutdown_performance_system():
    """Shutdown the performance optimization system."""
    await performance_optimizer.shutdown() 