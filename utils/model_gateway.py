"""
Model Gateway for Sophie Reflex Orchestrator
Provides unified interface for calling different AI models
"""

import asyncio
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def call_model(
    model_name: str, 
    prompt: str, 
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Call an AI model with the given prompt and context.
    
    Args:
        model_name: Name of the model to call (e.g., 'chatgpt-4o', 'openai-embedding-3-small')
        prompt: The prompt to send to the model
        context: Optional context dictionary
        
    Returns:
        Model response as string
    """
    # This is a placeholder implementation
    # In a real system, this would integrate with actual model APIs
    
    logger.info(f"Calling model {model_name} with prompt: {prompt[:100]}...")
    
    # For now, return a mock response
    # TODO: Implement actual model integration
    if "embedding" in model_name.lower():
        # Mock embedding response
        return "[0.1, 0.2, 0.3, ...]"  # Mock embedding vector
    else:
        # Mock text response
        return f"Mock response from {model_name}: {prompt[:50]}..."


async def call_model_async(
    model_name: str, 
    prompt: str, 
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Async version of call_model.
    
    Args:
        model_name: Name of the model to call
        prompt: The prompt to send to the model
        context: Optional context dictionary
        
    Returns:
        Model response as string
    """
    # Run the sync version in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        call_model, 
        model_name, 
        prompt, 
        context
    ) 