"""
Resource Manager for Sophie Reflex Orchestrator

Handles async file I/O operations and proper resource cleanup.
"""

import asyncio
import aiofiles
import json
import os
from typing import Dict, Any, Optional, List, AsyncGenerator
from contextlib import asynccontextmanager
import structlog

logger = structlog.get_logger()


class AsyncFileManager:
    """Manages async file I/O operations."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
    
    async def read_file_async(self, file_path: str) -> str:
        """Read a file asynchronously."""
        full_path = os.path.join(self.base_dir, file_path)
        
        try:
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            logger.debug("File read successfully", file_path=file_path)
            return content
            
        except FileNotFoundError:
            logger.error("File not found", file_path=file_path)
            raise
        except Exception as e:
            logger.error("Failed to read file", file_path=file_path, error=str(e))
            raise
    
    async def write_file_async(self, file_path: str, content: str) -> None:
        """Write content to a file asynchronously."""
        full_path = os.path.join(self.base_dir, file_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        try:
            async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            logger.debug("File written successfully", file_path=file_path)
            
        except Exception as e:
            logger.error("Failed to write file", file_path=file_path, error=str(e))
            raise
    
    async def write_json_async(self, file_path: str, data: Dict[str, Any]) -> None:
        """Write JSON data to a file asynchronously."""
        content = json.dumps(data, indent=2, default=str)
        await self.write_file_async(file_path, content)
    
    async def read_json_async(self, file_path: str) -> Dict[str, Any]:
        """Read JSON data from a file asynchronously."""
        content = await self.read_file_async(file_path)
        return json.loads(content)
    
    async def append_file_async(self, file_path: str, content: str) -> None:
        """Append content to a file asynchronously."""
        full_path = os.path.join(self.base_dir, file_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        try:
            async with aiofiles.open(full_path, 'a', encoding='utf-8') as f:
                await f.write(content)
            
            logger.debug("Content appended successfully", file_path=file_path)
            
        except Exception as e:
            logger.error("Failed to append to file", file_path=file_path, error=str(e))
            raise
    
    async def file_exists_async(self, file_path: str) -> bool:
        """Check if a file exists asynchronously."""
        full_path = os.path.join(self.base_dir, file_path)
        return os.path.exists(full_path)
    
    async def list_files_async(self, directory: str = ".") -> List[str]:
        """List files in a directory asynchronously."""
        full_path = os.path.join(self.base_dir, directory)
        
        if not os.path.exists(full_path):
            return []
        
        files = []
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            if os.path.isfile(item_path):
                files.append(item)
        
        return files


class ResourceManager:
    """Manages resource cleanup and lifecycle."""
    
    def __init__(self):
        self._resources: List[Any] = []
        self._cleanup_tasks: List[asyncio.Task] = []
        self._file_manager = AsyncFileManager()
    
    def register_resource(self, resource: Any) -> None:
        """Register a resource for cleanup."""
        self._resources.append(resource)
        logger.debug("Resource registered for cleanup", resource_type=type(resource).__name__)
    
    def register_cleanup_task(self, task: asyncio.Task) -> None:
        """Register a cleanup task."""
        self._cleanup_tasks.append(task)
        logger.debug("Cleanup task registered")
    
    async def cleanup_resources(self) -> None:
        """Clean up all registered resources."""
        logger.info("Starting resource cleanup")
        
        # Cancel cleanup tasks
        for task in self._cleanup_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clean up resources
        for resource in self._resources:
            try:
                if hasattr(resource, 'close'):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
                elif hasattr(resource, 'stop'):
                    if asyncio.iscoroutinefunction(resource.stop):
                        await resource.stop()
                    else:
                        resource.stop()
                elif hasattr(resource, 'cleanup'):
                    if asyncio.iscoroutinefunction(resource.cleanup):
                        await resource.cleanup()
                    else:
                        resource.cleanup()
                
                logger.debug("Resource cleaned up", resource_type=type(resource).__name__)
                
            except Exception as e:
                logger.warning("Failed to cleanup resource", 
                             resource_type=type(resource).__name__, error=str(e))
        
        self._resources.clear()
        self._cleanup_tasks.clear()
        logger.info("Resource cleanup completed")
    
    @asynccontextmanager
    async def managed_resource(self, resource: Any):
        """Context manager for automatic resource cleanup."""
        try:
            self.register_resource(resource)
            yield resource
        finally:
            # Note: This doesn't immediately cleanup, but registers for later cleanup
            pass
    
    async def save_results_async(self, results: Dict[str, Any], output_path: str) -> None:
        """Save results to a file asynchronously."""
        try:
            await self._file_manager.write_json_async(output_path, results)
            logger.info("Results saved successfully", output_path=output_path)
        except Exception as e:
            logger.error("Failed to save results", output_path=output_path, error=str(e))
            raise
    
    async def load_config_async(self, config_path: str) -> Dict[str, Any]:
        """Load configuration asynchronously."""
        try:
            config = await self._file_manager.read_json_async(config_path)
            logger.info("Configuration loaded successfully", config_path=config_path)
            return config
        except Exception as e:
            logger.error("Failed to load configuration", config_path=config_path, error=str(e))
            raise
    
    async def append_log_async(self, log_file: str, message: str) -> None:
        """Append a message to a log file asynchronously."""
        timestamp = asyncio.get_event_loop().time()
        log_entry = f"[{timestamp:.3f}] {message}\n"
        await self._file_manager.append_file_async(log_file, log_entry)


class AsyncConfigLoader:
    """Async configuration loader with caching."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self._cache: Dict[str, Any] = {}
        self._file_manager = AsyncFileManager(config_dir)
    
    async def load_config_async(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """Load a configuration file asynchronously."""
        if use_cache and config_name in self._cache:
            logger.debug("Using cached configuration", config_name=config_name)
            return self._cache[config_name]
        
        config_path = f"{config_name}.yaml"
        
        try:
            content = await self._file_manager.read_file_async(config_path)
            
            # Parse YAML
            import yaml
            config = yaml.safe_load(content)
            
            if use_cache:
                self._cache[config_name] = config
            
            logger.info("Configuration loaded asynchronously", config_name=config_name)
            return config
            
        except Exception as e:
            logger.error("Failed to load configuration asynchronously", 
                        config_name=config_name, error=str(e))
            raise
    
    async def load_all_configs_async(self) -> Dict[str, Any]:
        """Load all configuration files asynchronously."""
        configs_to_load = ['system', 'agents', 'rubric', 'policies']
        configs = {}
        
        # Load all configs concurrently
        tasks = [self.load_config_async(config_name) for config_name in configs_to_load]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for config_name, result in zip(configs_to_load, results):
            if isinstance(result, Exception):
                logger.error("Failed to load config", config_name=config_name, error=str(result))
                raise result
            configs[config_name] = result
        
        logger.info("All configurations loaded asynchronously")
        return configs
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
        logger.debug("Configuration cache cleared") 