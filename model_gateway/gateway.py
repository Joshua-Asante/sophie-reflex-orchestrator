"""
Model Gateway

Provides unified interface for dispatching requests to different LLM models.
"""

import asyncio
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def dispatch_to_model(model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch a request to a specific model.
    
    Args:
        model_name: Name of the model to use
        payload: Request payload with messages, temperature, etc.
        
    Returns:
        Model response or error information
    """
    try:
        # This is a placeholder implementation
        # In a real system, this would integrate with actual model APIs
        
        logger.info(f"Dispatching to model {model_name}")
        
        # Mock response for testing
        if "error" in payload.get("messages", [{}])[0].get("content", ""):
            return {"error": "Mock error response"}
        
        # Return mock successful response
        return {
            "choices": [
                {
                    "message": {
                        "content": """
name: revised_step
agent: web_scrape
args:
  url: "https://example.com"
  timeout: 30
                        """.strip()
                    }
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"Model dispatch failed: {e}")
        return {"error": str(e)}


async def dispatch_to_model_async(model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async version of dispatch_to_model.
    
    Args:
        model_name: Name of the model to use
        payload: Request payload
        
    Returns:
        Model response or error information
    """
    # Run the sync version in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        dispatch_to_model,
        model_name,
        payload
    ) 