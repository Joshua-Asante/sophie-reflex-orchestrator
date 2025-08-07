"""
Batch Processor for LLM Requests

Groups similar requests together to improve performance and reduce API costs
through intelligent batching and parallel processing.
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BatchType(Enum):
    """Types of batches for different request patterns."""
    SIMILAR_PROMPTS = "similar_prompts"
    SAME_MODEL = "same_model"
    SAME_PROVIDER = "same_provider"
    PARALLEL_TASKS = "parallel_tasks"


@dataclass
class BatchRequest:
    """Represents a request that can be batched."""
    id: str
    prompt: str
    model: str
    provider: str
    temperature: float
    max_tokens: int
    created_at: float = field(default_factory=time.time)
    priority: int = 1  # Higher number = higher priority
    callback: Optional[Callable[[Any], Awaitable[None]]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the request."""
        content = f"{self.prompt[:50]}{self.model}{self.provider}{self.temperature}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def similarity_score(self, other: 'BatchRequest') -> float:
        """Calculate similarity score with another request."""
        # Simple similarity based on prompt length and model
        if self.model != other.model or self.provider != other.provider:
            return 0.0
        
        # Calculate prompt similarity (simple length-based for now)
        len_diff = abs(len(self.prompt) - len(other.prompt))
        max_len = max(len(self.prompt), len(other.prompt))
        if max_len == 0:
            return 1.0
        
        length_similarity = 1.0 - (len_diff / max_len)
        
        # Temperature similarity
        temp_diff = abs(self.temperature - other.temperature)
        temp_similarity = 1.0 - min(temp_diff, 1.0)
        
        return (length_similarity + temp_similarity) / 2


@dataclass
class Batch:
    """A batch of similar requests."""
    id: str
    requests: List[BatchRequest]
    batch_type: BatchType
    created_at: float = field(default_factory=time.time)
    max_wait_time: float = 5.0  # seconds
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the batch."""
        if not self.requests:
            return hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        first_req = self.requests[0]
        content = f"{first_req.model}{first_req.provider}{self.batch_type.value}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def should_process(self) -> bool:
        """Check if batch should be processed based on time and size."""
        time_elapsed = time.time() - self.created_at
        return time_elapsed >= self.max_wait_time or len(self.requests) >= 10
    
    def get_combined_prompt(self) -> str:
        """Get a combined prompt for all requests in the batch."""
        if not self.requests:
            return ""
        
        # For similar prompts, we can combine them
        if self.batch_type == BatchType.SIMILAR_PROMPTS:
            prompts = [req.prompt for req in self.requests]
            return "\n\n---\n\n".join(prompts)
        
        # For other types, use the first request's prompt
        return self.requests[0].prompt
    
    def get_representative_request(self) -> BatchRequest:
        """Get a representative request for the batch."""
        if not self.requests:
            raise ValueError("Batch has no requests")
        
        # Return the highest priority request, or the first one
        return max(self.requests, key=lambda r: r.priority)


class BatchProcessor:
    """Manages batching of LLM requests for improved performance."""
    
    def __init__(self, max_batch_size: int = 10, max_wait_time: float = 5.0):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batches: Dict[str, Batch] = {}
        self.processing = False
        self._processing_task: Optional[asyncio.Task] = None
    
    async def add_request(self, request: BatchRequest) -> str:
        """Add a request to the appropriate batch."""
        batch_key = self._get_batch_key(request)
        
        if batch_key not in self.batches:
            # Create new batch
            batch_type = self._determine_batch_type(request)
            self.batches[batch_key] = Batch(
                id="",  # Will be auto-generated in __post_init__
                requests=[request],
                batch_type=batch_type,
                max_wait_time=self.max_wait_time
            )
            logger.debug(f"Created new batch {batch_key} for request {request.id}")
        else:
            # Add to existing batch
            self.batches[batch_key].requests.append(request)
            logger.debug(f"Added request {request.id} to batch {batch_key}")
        
        # Start processing if not already running
        if not self.processing:
            self._processing_task = asyncio.create_task(self._process_batches())
        
        return batch_key
    
    def _get_batch_key(self, request: BatchRequest) -> str:
        """Generate a key for grouping requests into batches."""
        # Group by model and provider for now
        return f"{request.provider}:{request.model}"
    
    def _determine_batch_type(self, request: BatchRequest) -> BatchType:
        """Determine the type of batch for a request."""
        # For now, use same provider as default
        return BatchType.SAME_PROVIDER
    
    async def _process_batches(self):
        """Process batches that are ready."""
        self.processing = True
        
        while self.batches:
            ready_batches = []
            
            # Find batches that are ready to process
            for batch_key, batch in self.batches.items():
                if batch.should_process():
                    ready_batches.append((batch_key, batch))
            
            # Process ready batches
            for batch_key, batch in ready_batches:
                await self._process_batch(batch_key, batch)
                del self.batches[batch_key]
            
            # Wait a bit before checking again
            if self.batches:
                await asyncio.sleep(0.1)
        
        self.processing = False
    
    async def _process_batch(self, batch_key: str, batch: Batch):
        """Process a single batch."""
        logger.info(f"Processing batch {batch_key} with {len(batch.requests)} requests")
        
        try:
            # Get representative request for API call
            rep_request = batch.get_representative_request()
            
            # Make the API call (this would integrate with your LLM client)
            result = await self._make_batch_api_call(batch, rep_request)
            
            # Distribute results to individual requests
            await self._distribute_results(batch, result)
            
            logger.info(f"Successfully processed batch {batch_key}")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_key}: {e}")
            # Handle individual request failures
            await self._handle_batch_failure(batch, e)
    
    async def _make_batch_api_call(self, batch: Batch, rep_request: BatchRequest) -> Any:
        """Make the actual API call for the batch."""
        # This is a placeholder - integrate with your LLM client
        # For now, we'll simulate the API call
        
        combined_prompt = batch.get_combined_prompt()
        
        # Simulate API call
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Return mock result
        return {
            "response": f"Batch response for {len(batch.requests)} requests",
            "batch_id": batch.id,
            "requests_processed": len(batch.requests)
        }
    
    async def _distribute_results(self, batch: Batch, result: Any):
        """Distribute batch results to individual requests."""
        for request in batch.requests:
            if request.callback:
                try:
                    await request.callback(result)
                except Exception as e:
                    logger.error(f"Error in request callback {request.id}: {e}")
    
    async def _handle_batch_failure(self, batch: Batch, error: Exception):
        """Handle batch processing failure."""
        for request in batch.requests:
            if request.callback:
                try:
                    await request.callback({"error": str(error)})
                except Exception as e:
                    logger.error(f"Error in failure callback {request.id}: {e}")
    
    async def shutdown(self):
        """Shutdown the batch processor."""
        # Process remaining batches
        for batch_key, batch in list(self.batches.items()):
            await self._process_batch(batch_key, batch)
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass


class SmartBatchProcessor:
    """High-level batch processor with intelligent request grouping."""
    
    def __init__(self):
        self.processor = BatchProcessor()
        self.request_history: List[BatchRequest] = []
    
    async def submit_request(
        self,
        prompt: str,
        model: str,
        provider: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        priority: int = 1,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> str:
        """Submit a request for batching."""
        request = BatchRequest(
            id="",  # Will be auto-generated in __post_init__
            prompt=prompt,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            priority=priority,
            callback=callback
        )
        
        # Store in history for learning
        self.request_history.append(request)
        
        # Add to batch processor
        batch_key = await self.processor.add_request(request)
        return batch_key
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get statistics about current batches."""
        stats = {
            "total_batches": len(self.processor.batches),
            "total_requests": sum(len(batch.requests) for batch in self.processor.batches.values()),
            "batch_types": {},
            "providers": {}
        }
        
        for batch in self.processor.batches.values():
            batch_type = batch.batch_type.value
            stats["batch_types"][batch_type] = stats["batch_types"].get(batch_type, 0) + 1
            
            if batch.requests:
                provider = batch.requests[0].provider
                stats["providers"][provider] = stats["providers"].get(provider, 0) + len(batch.requests)
        
        return stats
    
    async def shutdown(self):
        """Shutdown the smart batch processor."""
        await self.processor.shutdown()


# Global instance
batch_processor = SmartBatchProcessor() 