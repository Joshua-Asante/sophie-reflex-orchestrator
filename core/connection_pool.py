"""
Connection Pool for LLM APIs

Provides efficient connection pooling for multiple LLM providers to reduce
connection overhead and improve performance.
"""

import asyncio
import time
import httpx
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GOOGLE = "google"
    XAI = "xai"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    KIMI = "kimi"


@dataclass
class ConnectionConfig:
    """Configuration for a connection."""
    base_url: str
    headers: Dict[str, str]
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit_per_minute: int = 60


class ConnectionPool:
    """Manages connections to LLM APIs with pooling and rate limiting."""
    
    def __init__(self, max_connections: int = 10, max_connections_per_provider: int = 3):
        self.max_connections = max_connections
        self.max_connections_per_provider = max_connections_per_provider
        self.pools: Dict[ProviderType, asyncio.Queue] = {}
        self.configs: Dict[ProviderType, ConnectionConfig] = {}
        self.rate_limiters: Dict[ProviderType, Dict[str, List[float]]] = {}
        self._init_pools()
    
    def _init_pools(self):
        """Initialize connection pools for each provider."""
        for provider in ProviderType:
            self.pools[provider] = asyncio.Queue(maxsize=self.max_connections_per_provider)
            self.rate_limiters[provider] = {"requests": []}
    
    def register_provider(self, provider: ProviderType, config: ConnectionConfig):
        """Register a provider with its configuration."""
        self.configs[provider] = config
        logger.info(f"Registered provider: {provider.value}")
    
    async def get_connection(self, provider: ProviderType) -> httpx.AsyncClient:
        """Get a connection from the pool or create a new one."""
        try:
            # Try to get from pool first
            if not self.pools[provider].empty():
                return self.pools[provider].get_nowait()
        except asyncio.QueueEmpty:
            pass
        
        # Create new connection if pool is empty
        config = self.configs.get(provider)
        if not config:
            raise ValueError(f"No configuration found for provider: {provider}")
        
        client = httpx.AsyncClient(
            base_url=config.base_url,
            headers=config.headers,
            timeout=httpx.Timeout(config.timeout),
            limits=httpx.Limits(max_connections=1)
        )
        
        logger.debug(f"Created new connection for {provider.value}")
        return client
    
    async def return_connection(self, provider: ProviderType, client: httpx.AsyncClient):
        """Return a connection to the pool."""
        try:
            if not self.pools[provider].full():
                self.pools[provider].put_nowait(client)
                logger.debug(f"Returned connection to pool for {provider.value}")
            else:
                await client.aclose()
                logger.debug(f"Closed connection for {provider.value} (pool full)")
        except Exception as e:
            logger.warning(f"Error returning connection: {e}")
            await client.aclose()
    
    async def check_rate_limit(self, provider: ProviderType) -> bool:
        """Check if we're within rate limits for the provider."""
        now = time.time()
        requests = self.rate_limiters[provider]["requests"]
        
        # Remove old requests (older than 1 minute)
        requests = [req_time for req_time in requests if now - req_time < 60]
        self.rate_limiters[provider]["requests"] = requests
        
        config = self.configs.get(provider)
        if not config:
            return True
        
        if len(requests) >= config.rate_limit_per_minute:
            logger.warning(f"Rate limit exceeded for {provider.value}")
            return False
        
        return True
    
    async def record_request(self, provider: ProviderType):
        """Record a request for rate limiting."""
        self.rate_limiters[provider]["requests"].append(time.time())
    
    async def close_all(self):
        """Close all connections in all pools."""
        for provider, pool in self.pools.items():
            while not pool.empty():
                try:
                    client = pool.get_nowait()
                    await client.aclose()
                except asyncio.QueueEmpty:
                    break
        
        logger.info("Closed all connection pools")


class ConnectionPoolManager:
    """High-level manager for connection pooling."""
    
    def __init__(self):
        self.pool = ConnectionPool()
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default configurations for all providers."""
        configs = {
            ProviderType.OPENAI: ConnectionConfig(
                base_url="https://api.openai.com/v1",
                headers={"Content-Type": "application/json"},
                timeout=30.0,
                rate_limit_per_minute=60
            ),
            ProviderType.GOOGLE: ConnectionConfig(
                base_url="https://generativelanguage.googleapis.com",
                headers={"Content-Type": "application/json"},
                timeout=30.0,
                rate_limit_per_minute=60
            ),
            ProviderType.XAI: ConnectionConfig(
                base_url="https://api.x.ai/v1",
                headers={"Content-Type": "application/json"},
                timeout=30.0,
                rate_limit_per_minute=60
            ),
            ProviderType.MISTRAL: ConnectionConfig(
                base_url="https://api.mistral.ai/v1",
                headers={"Content-Type": "application/json"},
                timeout=30.0,
                rate_limit_per_minute=60
            ),
            ProviderType.DEEPSEEK: ConnectionConfig(
                base_url="https://api.deepseek.com/v1",
                headers={"Content-Type": "application/json"},
                timeout=30.0,
                rate_limit_per_minute=60
            ),
            ProviderType.KIMI: ConnectionConfig(
                base_url="https://api.moonshot.cn/v1",
                headers={"Content-Type": "application/json"},
                timeout=30.0,
                rate_limit_per_minute=60
            )
        }
        
        for provider, config in configs.items():
            self.pool.register_provider(provider, config)
    
    async def get_client(self, provider: ProviderType, api_key: str) -> httpx.AsyncClient:
        """Get a client with proper authentication."""
        # Check rate limits
        if not await self.pool.check_rate_limit(provider):
            raise Exception(f"Rate limit exceeded for {provider.value}")
        
        client = await self.pool.get_connection(provider)
        
        # Add API key to headers
        if provider == ProviderType.OPENAI:
            client.headers["Authorization"] = f"Bearer {api_key}"
        elif provider == ProviderType.GOOGLE:
            client.headers["x-goog-api-key"] = api_key
        else:
            client.headers["Authorization"] = f"Bearer {api_key}"
        
        await self.pool.record_request(provider)
        return client
    
    async def return_client(self, provider: ProviderType, client: httpx.AsyncClient):
        """Return a client to the pool."""
        await self.pool.return_connection(provider, client)
    
    async def close(self):
        """Close all connections."""
        await self.pool.close_all()


# Global instance
connection_manager = ConnectionPoolManager() 