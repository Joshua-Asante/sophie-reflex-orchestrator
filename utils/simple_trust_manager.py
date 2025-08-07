"""
Simple Trust Manager

Lightweight trust management for specific use cases where the full orchestrator
trust manager is not needed.
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SimpleTrustManager:
    """Lightweight trust manager for specific use cases."""

    def __init__(self, default_score: float = 0.5):
        """
        Initialize the simple trust manager.
        
        Args:
            default_score: Default trust score for new sources
        """
        self._trust_scores: Dict[str, float] = {}
        self.default_score = default_score
        logger.info("SimpleTrustManager initialized")

    def set_trust(self, source: str, score: float) -> None:
        """
        Sets an initial trust score for a source.
        
        Args:
            source: Source identifier
            score: Trust score (0.0 to 1.0)
        """
        self._trust_scores[source] = self._clamp(score)
        logger.debug(f"Set trust for {source}: {score}")

    def get_trust(self, source: str) -> float:
        """
        Retrieves the trust score for a given source.
        
        Args:
            source: Source identifier
            
        Returns:
            Trust score (0.0 to 1.0)
        """
        return self._trust_scores.get(source, self.default_score)

    def update_trust(self, source: str, change_factor: float) -> float:
        """
        Updates the trust for a source by a given factor and returns the new score.
        
        Args:
            source: Source identifier
            change_factor: Change to apply (positive increases, negative decreases)
            
        Returns:
            New trust score
        """
        current_score = self.get_trust(source)
        new_score = self._clamp(current_score + change_factor)
        self._trust_scores[source] = new_score
        
        logger.debug(f"Updated trust for {source}: {current_score} -> {new_score} (change: {change_factor})")
        return new_score

    def get_all_scores(self) -> Dict[str, float]:
        """
        Returns a copy of the current trust scores.
        
        Returns:
            Dictionary of source -> trust score
        """
        return self._trust_scores.copy()

    def get_sources(self) -> list[str]:
        """
        Returns list of all tracked sources.
        
        Returns:
            List of source identifiers
        """
        return list(self._trust_scores.keys())

    def remove_source(self, source: str) -> bool:
        """
        Removes a source from tracking.
        
        Args:
            source: Source identifier
            
        Returns:
            True if source was removed, False if not found
        """
        if source in self._trust_scores:
            del self._trust_scores[source]
            logger.debug(f"Removed source: {source}")
            return True
        return False

    def reset_scores(self) -> None:
        """Resets all trust scores to default."""
        self._trust_scores.clear()
        logger.info("Reset all trust scores")

    @staticmethod
    def _clamp(score: float) -> float:
        """
        Ensures a score stays within the valid range [0.0, 1.0].
        
        Args:
            score: Raw score
            
        Returns:
            Clamped score
        """
        return max(0.0, min(1.0, score))

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the trust scores.
        
        Returns:
            Dictionary with statistics
        """
        if not self._trust_scores:
            return {
                "total_sources": 0,
                "average_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0
            }

        scores = list(self._trust_scores.values())
        return {
            "total_sources": len(scores),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "sources": self._trust_scores.copy()
        } 