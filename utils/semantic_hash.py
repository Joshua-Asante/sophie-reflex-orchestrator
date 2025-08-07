"""
Semantic hash utilities for Sophie Reflex Orchestrator
"""

import hashlib
from typing import Any


def semantic_hash(text: str) -> str:
    """
    Generate a semantic hash from text content.
    
    Args:
        text: Text to hash
        
    Returns:
        Hex digest of the text (first 16 characters)
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def semantic_hash_object(obj: Any) -> str:
    """
    Generate a semantic hash from any object by converting to string.
    
    Args:
        obj: Object to hash
        
    Returns:
        Hex digest of the object representation
    """
    text = str(obj)
    return semantic_hash(text) 