"""
Smart Cache for LLM Responses

Provides intelligent caching with semantic similarity matching and exact matching
to reduce API costs and improve response times.
"""

import asyncio
import time
import hashlib
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached response."""
    prompt_hash: str
    semantic_hash: str
    response: str
    model: str
    provider: str
    temperature: float
    max_tokens: int
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.last_accessed:
            self.last_accessed = time.time()
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def is_expired(self, ttl: float) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > ttl
    
    def get_similarity_score(self, other_prompt: str, other_model: str, other_provider: str) -> float:
        """Calculate similarity score with another prompt."""
        # Basic similarity check - in production, use proper NLP similarity
        if self.model != other_model or self.provider != other_provider:
            return 0.0
        
        # Simple length-based similarity
        len_diff = abs(len(self.prompt_hash) - len(other_prompt))
        max_len = max(len(self.prompt_hash), len(other_prompt))
        if max_len == 0:
            return 1.0
        
        return 1.0 - (len_diff / max_len)


class SemanticCache:
    """Cache with semantic similarity matching."""
    
    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.8):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.entries: Dict[str, CacheEntry] = {}
        self.semantic_index: Dict[str, List[str]] = {}  # semantic_hash -> entry_ids
    
    def _generate_prompt_hash(self, prompt: str) -> str:
        """Generate hash for exact matching."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _generate_semantic_hash(self, prompt: str) -> str:
        """Generate semantic hash for similarity matching."""
        # Simple semantic hash based on key words
        # In production, use proper embeddings
        words = prompt.lower().split()
        key_words = [w for w in words if len(w) > 3]  # Filter short words
        key_words.sort()
        semantic_text = " ".join(key_words)
        return hashlib.md5(semantic_text.encode()).hexdigest()[:16]
    
    def add_entry(self, prompt: str, response: str, model: str, provider: str, 
                  temperature: float, max_tokens: int) -> str:
        """Add a new entry to the cache."""
        prompt_hash = self._generate_prompt_hash(prompt)
        semantic_hash = self._generate_semantic_hash(prompt)
        
        entry = CacheEntry(
            prompt_hash=prompt_hash,
            semantic_hash=semantic_hash,
            response=response,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Add to exact match cache
        self.entries[prompt_hash] = entry
        
        # Add to semantic index
        if semantic_hash not in self.semantic_index:
            self.semantic_index[semantic_hash] = []
        self.semantic_index[semantic_hash].append(prompt_hash)
        
        # Evict if cache is full
        if len(self.entries) > self.max_size:
            self._evict_least_used()
        
        logger.debug(f"Cached response for prompt hash: {prompt_hash[:8]}")
        return prompt_hash
    
    def get_exact_match(self, prompt: str) -> Optional[CacheEntry]:
        """Get exact match for a prompt."""
        prompt_hash = self._generate_prompt_hash(prompt)
        entry = self.entries.get(prompt_hash)
        
        if entry:
            entry.update_access()
            logger.debug(f"Exact cache hit for prompt hash: {prompt_hash[:8]}")
            return entry
        
        return None
    
    def get_semantic_match(self, prompt: str, model: str, provider: str) -> Optional[CacheEntry]:
        """Get semantic match for a prompt."""
        semantic_hash = self._generate_semantic_hash(prompt)
        
        if semantic_hash not in self.semantic_index:
            return None
        
        # Check all entries with similar semantic hash
        best_match = None
        best_score = 0.0
        
        for prompt_hash in self.semantic_index[semantic_hash]:
            entry = self.entries.get(prompt_hash)
            if not entry:
                continue
            
            # Check if model and provider match
            if entry.model != model or entry.provider != provider:
                continue
            
            # Calculate similarity score
            score = entry.get_similarity_score(prompt, model, provider)
            
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = entry
        
        if best_match:
            best_match.update_access()
            logger.debug(f"Semantic cache hit (score: {best_score:.2f}) for prompt")
            return best_match
        
        return None
    
    def _evict_least_used(self):
        """Evict least used entries when cache is full."""
        if not self.entries:
            return
        
        # Find least used entry
        least_used = min(self.entries.values(), 
                        key=lambda e: (e.access_count, -e.last_accessed))
        
        # Remove from entries
        del self.entries[least_used.prompt_hash]
        
        # Remove from semantic index
        for entry_ids in self.semantic_index.values():
            if least_used.prompt_hash in entry_ids:
                entry_ids.remove(least_used.prompt_hash)
        
        logger.debug(f"Evicted least used entry: {least_used.prompt_hash[:8]}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.entries)
        total_accesses = sum(entry.access_count for entry in self.entries.values())
        
        # Calculate hit rates (this would need to be tracked separately in production)
        return {
            "total_entries": total_entries,
            "total_accesses": total_accesses,
            "avg_accesses_per_entry": total_accesses / total_entries if total_entries > 0 else 0,
            "semantic_index_size": len(self.semantic_index)
        }


class ResultCache:
    """Cache for tool execution results."""
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.entries: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
    
    def _generate_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key for tool execution."""
        # Create a deterministic key from tool name and parameters
        param_str = json.dumps(params, sort_keys=True)
        content = f"{tool_name}:{param_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for tool execution."""
        key = self._generate_key(tool_name, params)
        
        if key in self.entries:
            self.access_times[key] = time.time()
            logger.debug(f"Tool cache hit for: {tool_name}")
            return self.entries[key]
        
        return None
    
    def set(self, tool_name: str, params: Dict[str, Any], result: Any):
        """Cache result for tool execution."""
        key = self._generate_key(tool_name, params)
        
        self.entries[key] = result
        self.access_times[key] = time.time()
        
        # Evict if cache is full
        if len(self.entries) > self.max_size:
            self._evict_least_recent()
        
        logger.debug(f"Cached tool result for: {tool_name}")
    
    def _evict_least_recent(self):
        """Evict least recently used entry."""
        if not self.access_times:
            return
        
        least_recent_key = min(self.access_times.keys(), 
                              key=lambda k: self.access_times[k])
        
        del self.entries[least_recent_key]
        del self.access_times[least_recent_key]
        
        logger.debug(f"Evicted least recent tool cache entry")


class SmartCache:
    """High-level cache manager combining semantic and result caching."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.semantic_cache = SemanticCache()
        self.result_cache = ResultCache()
        
        # Load cached data if available
        self._load_cache()
    
    def _load_cache(self):
        """Load cached data from disk."""
        try:
            semantic_cache_file = self.cache_dir / "semantic_cache.pkl"
            if semantic_cache_file.exists():
                with open(semantic_cache_file, 'rb') as f:
                    self.semantic_cache.entries = pickle.load(f)
                logger.info("Loaded semantic cache from disk")
            
            result_cache_file = self.cache_dir / "result_cache.pkl"
            if result_cache_file.exists():
                with open(result_cache_file, 'rb') as f:
                    self.result_cache.entries = pickle.load(f)
                logger.info("Loaded result cache from disk")
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cache data to disk."""
        try:
            semantic_cache_file = self.cache_dir / "semantic_cache.pkl"
            with open(semantic_cache_file, 'wb') as f:
                pickle.dump(self.semantic_cache.entries, f)
            
            result_cache_file = self.cache_dir / "result_cache.pkl"
            with open(result_cache_file, 'wb') as f:
                pickle.dump(self.result_cache.entries, f)
            
            logger.info("Saved cache to disk")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    async def get_llm_response(self, prompt: str, model: str, provider: str,
                              temperature: float, max_tokens: int) -> Optional[str]:
        """Get cached LLM response if available."""
        # Try exact match first
        entry = self.semantic_cache.get_exact_match(prompt)
        if entry and entry.model == model and entry.provider == provider:
            return entry.response
        
        # Try semantic match
        entry = self.semantic_cache.get_semantic_match(prompt, model, provider)
        if entry:
            return entry.response
        
        return None
    
    def cache_llm_response(self, prompt: str, response: str, model: str, provider: str,
                          temperature: float, max_tokens: int):
        """Cache LLM response."""
        self.semantic_cache.add_entry(prompt, response, model, provider, 
                                     temperature, max_tokens)
    
    async def get_tool_result(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached tool result if available."""
        return self.result_cache.get(tool_name, params)
    
    def cache_tool_result(self, tool_name: str, params: Dict[str, Any], result: Any):
        """Cache tool result."""
        self.result_cache.set(tool_name, params, result)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        semantic_stats = self.semantic_cache.get_statistics()
        result_stats = {
            "total_entries": len(self.result_cache.entries),
            "total_accesses": len(self.result_cache.access_times)
        }
        
        return {
            "semantic_cache": semantic_stats,
            "result_cache": result_stats,
            "cache_dir": str(self.cache_dir)
        }
    
    async def clear_cache(self):
        """Clear all cached data."""
        self.semantic_cache.entries.clear()
        self.semantic_cache.semantic_index.clear()
        self.result_cache.entries.clear()
        self.result_cache.access_times.clear()
        
        # Remove cache files
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        logger.info("Cleared all cache data")
    
    async def cleanup_expired(self, ttl: float = 3600):  # 1 hour default
        """Remove expired cache entries."""
        # Clean semantic cache
        expired_entries = [
            entry_id for entry_id, entry in self.semantic_cache.entries.items()
            if entry.is_expired(ttl)
        ]
        
        for entry_id in expired_entries:
            entry = self.semantic_cache.entries[entry_id]
            del self.semantic_cache.entries[entry_id]
            
            # Remove from semantic index
            for entry_ids in self.semantic_cache.semantic_index.values():
                if entry_id in entry_ids:
                    entry_ids.remove(entry_id)
        
        if expired_entries:
            logger.info(f"Cleaned up {len(expired_entries)} expired semantic cache entries")
    
    async def shutdown(self):
        """Shutdown the cache manager."""
        await self.cleanup_expired()
        self._save_cache()
        logger.info("Cache manager shutdown complete")


# Global instance
smart_cache = SmartCache() 