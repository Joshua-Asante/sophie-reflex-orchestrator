"""
Debug Trace Component

Provides tracing and logging for service operations.
"""

import logging
from typing import Dict, Any
from datetime import datetime


class DebugTrace:
    """Provides debug tracing for service operations."""
    
    def __init__(self):
        self.logger = logging.getLogger("service.debug_trace")
    
    def log(self, event: str, data: Dict[str, Any], session_id: str = None) -> None:
        """
        Log a debug event with associated data.
        
        Args:
            event: Event name
            data: Event data
            session_id: Optional session identifier
        """
        timestamp = datetime.utcnow().isoformat()
        log_data = {
            "event": event,
            "timestamp": timestamp,
            "data": data
        }
        
        if session_id:
            log_data["session_id"] = session_id
        
        self.logger.debug(
            f"Service Event: {event}",
            extra=log_data
        )
    
    def trace(self, operation: str, **kwargs) -> None:
        """
        Trace an operation with keyword arguments.
        
        Args:
            operation: Operation name
            **kwargs: Operation parameters
        """
        self.log(operation, kwargs) 