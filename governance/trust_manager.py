"""
Governance Trust Manager

Provides trust management for governance operations.
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TrustManager:
    """Trust manager for governance operations."""
    
    def __init__(self):
        self._trust_scores: Dict[str, float] = {}
        logger.info("Governance TrustManager initialized")
    
    def get_trust(self, source: str) -> float:
        """
        Get trust score for a source.
        
        Args:
            source: Source identifier
            
        Returns:
            Trust score (0.0 to 1.0)
        """
        return self._trust_scores.get(source, 0.5)
    
    def set_trust(self, source: str, score: float) -> None:
        """
        Set trust score for a source.
        
        Args:
            source: Source identifier
            score: Trust score (0.0 to 1.0)
        """
        self._trust_scores[source] = max(0.0, min(1.0, score))
        logger.debug(f"Set trust for {source}: {score}")
    
    def update_trust(self, source: str, change: float) -> float:
        """
        Update trust score for a source.
        
        Args:
            source: Source identifier
            change: Change to apply
            
        Returns:
            New trust score
        """
        current = self.get_trust(source)
        new_score = max(0.0, min(1.0, current + change))
        self._trust_scores[source] = new_score
        logger.debug(f"Updated trust for {source}: {current} -> {new_score}")
        return new_score
    
    def get_all_scores(self) -> Dict[str, float]:
        """Get all trust scores."""
        return self._trust_scores.copy() 