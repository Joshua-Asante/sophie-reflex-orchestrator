"""
Agent Registry - Centralized management and discovery system for all agents.
Provides dynamic agent loading, health monitoring, and capability matching.
"""

from typing import Dict, Any, List, Optional, Type, Callable, Set
import asyncio
import structlog
import importlib
import inspect
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
from concurrent.futures import ThreadPoolExecutor

from .base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus

logger = structlog.get_logger()


class AgentCapability(Enum):
    """Defines different agent capabilities for matching and discovery."""
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_WRITING = "creative_writing" 
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    EVALUATION = "evaluation"
    REFINEMENT = "refinement"
    PLANNING = "planning"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    OPTIMIZATION = "optimization"
    DEBUGGING = "debugging"
    TESTING = "testing"


@dataclass
class AgentMetadata:
    """Comprehensive metadata for registered agents."""
    agent_class: Type[BaseAgent]
    capabilities: Set[AgentCapability]
    specializations: List[str]
    performance_tier: str  # "high", "medium", "low"
    resource_requirements: Dict[str, Any]
    compatibility_matrix: Dict[str, float]  # Compatibility with other agents
    health_status: AgentStatus = AgentStatus.IDLE
    last_health_check: Optional[datetime] = None
    registration_time: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0


@dataclass
class AgentPool:
    """Manages a pool of agent instances for load balancing."""
    agent_type: str
    instances: List[BaseAgent] = field(default_factory=list)
    max_instances: int = 5
    current_load: int = 0
    load_balancer_strategy: str = "round_robin"  # "round_robin", "least_loaded", "performance_based"
    health_check_interval: int = 300  # seconds


class AgentRegistry:
    """
    Centralized registry for agent management, discovery, and orchestration.
    Provides advanced features like health monitoring, load balancing, and capability matching.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.registered_agents: Dict[str, AgentMetadata] = {}
        self.agent_pools: Dict[str, AgentPool] = {}
        self.capability_index: Dict[AgentCapability, List[str]] = {}
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.config_path = config_path
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.registry_stats = {
            "total_registrations": 0,
            "active_agents": 0,
            "health_checks_performed": 0,
            "capability_matches": 0,
            "load_balancing_decisions": 0
        }
        
        # Load configuration if provided
        if config_path and Path(config_path).exists():
            self._load_registry_config(config_path)
            
        logger.info("Agent registry initialized", config_path=config_path)
    
    def register_agent(self, 
                      agent_name: str,
                      agent_class: Type[BaseAgent],
                      capabilities: Set[AgentCapability],
                      specializations: List[str] = None,
                      performance_tier: str = "medium",
                      resource_requirements: Dict[str, Any] = None,
                      compatibility_matrix: Dict[str, float] = None) -> bool:
        """Register a new agent type with comprehensive metadata."""
        try:
            metadata = AgentMetadata(
                agent_class=agent_class,
                capabilities=capabilities,
                specializations=specializations or [],
                performance_tier=performance_tier,
                resource_requirements=resource_requirements or {},
                compatibility_matrix=compatibility_matrix or {}
            )
            
            self.registered_agents[agent_name] = metadata
            
            # Update capability index
            for capability in capabilities:
                if capability not in self.capability_index:
                    self.capability_index[capability] = []
                self.capability_index[capability].append(agent_name)
            
            # Create agent pool
            self.agent_pools[agent_name] = AgentPool(agent_type=agent_name)
            
            self.registry_stats["total_registrations"] += 1
            
            logger.info("Agent registered successfully", 
                       agent_name=agent_name,
                       capabilities=[c.value for c in capabilities],
                       specializations=specializations)
            return True
            
        except Exception as e:
            logger.error("Failed to register agent", 
                        agent_name=agent_name, 
                        error=str(e))
            return False
    
    def discover_agents_by_capability(self, 
                                    required_capabilities: Set[AgentCapability],
                                    performance_tier: Optional[str] = None,
                                    exclude_agents: Set[str] = None) -> List[str]:
        """Discover agents that match required capabilities and criteria."""
        exclude_agents = exclude_agents or set()
        matching_agents = []
        
        for agent_name, metadata in self.registered_agents.items():
            if agent_name in exclude_agents:
                continue
                
            # Check capability match
            if not required_capabilities.issubset(metadata.capabilities):
                continue
                
            # Check performance tier if specified
            if performance_tier and metadata.performance_tier != performance_tier:
                continue
                
            # Check health status
            if metadata.health_status not in [AgentStatus.IDLE, AgentStatus.COMPLETED]:
                continue
                
            matching_agents.append(agent_name)
        
        # Sort by success rate and performance
        matching_agents.sort(key=lambda name: (
            self.registered_agents[name].success_rate,
            -self.registered_agents[name].average_response_time
        ), reverse=True)
        
        self.registry_stats["capability_matches"] += 1
        
        logger.info("Agent discovery completed", 
                   required_capabilities=[c.value for c in required_capabilities],
                   matching_agents=matching_agents)
        
        return matching_agents
    
    def get_optimal_agent_combination(self, 
                                    task_requirements: Dict[str, Any],
                                    max_agents: int = 3) -> List[Dict[str, Any]]:
        """Get optimal combination of agents for complex tasks."""
        required_capabilities = set(task_requirements.get("capabilities", []))
        task_complexity = task_requirements.get("complexity", "medium")
        resource_constraints = task_requirements.get("resource_constraints", {})
        
        # Find agents for each capability
        agent_combinations = []
        
        for capability in required_capabilities:
            capable_agents = self.capability_index.get(capability, [])
            
            # Filter by resource constraints and health
            filtered_agents = []
            for agent_name in capable_agents:
                metadata = self.registered_agents[agent_name]
                
                if metadata.health_status not in [AgentStatus.IDLE, AgentStatus.COMPLETED]:
                    continue
                    
                # Check resource compatibility
                if self._check_resource_compatibility(metadata.resource_requirements, 
                                                   resource_constraints):
                    filtered_agents.append({
                        "agent_name": agent_name,
                        "capability": capability,
                        "metadata": metadata,
                        "compatibility_score": self._calculate_compatibility_score(
                            metadata, task_requirements)
                    })
            
            # Sort by compatibility score
            filtered_agents.sort(key=lambda x: x["compatibility_score"], reverse=True)
            agent_combinations.extend(filtered_agents[:max_agents])
        
        # Remove duplicates and optimize combination
        unique_agents = {}
        for agent_info in agent_combinations:
            agent_name = agent_info["agent_name"]
            if agent_name not in unique_agents:
                unique_agents[agent_name] = agent_info
            else:
                # Merge capabilities
                existing = unique_agents[agent_name]
                existing["capabilities"] = existing.get("capabilities", set()) | {agent_info["capability"]}
        
        # Sort final combination by overall score
        final_combination = list(unique_agents.values())
        final_combination.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        return final_combination[:max_agents]
    
    def create_agent_instance(self, 
                            agent_name: str, 
                            config_overrides: Dict[str, Any] = None) -> Optional[BaseAgent]:
        """Create a new instance of a registered agent with load balancing."""
        if agent_name not in self.registered_agents:
            logger.error("Agent not registered", agent_name=agent_name)
            return None
        
        try:
            metadata = self.registered_agents[agent_name]
            pool = self.agent_pools[agent_name]
            
            # Check if we can reuse an existing instance
            available_instance = self._get_available_instance(pool)
            if available_instance:
                self.registry_stats["load_balancing_decisions"] += 1
                return available_instance
            
            # Create new instance if pool not full
            if len(pool.instances) < pool.max_instances:
                # Create default config
                default_config = AgentConfig(
                    name=agent_name,
                    prompt=f"You are a {agent_name} agent.",
                    model="gpt-4",
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Apply overrides
                if config_overrides:
                    for key, value in config_overrides.items():
                        if hasattr(default_config, key):
                            setattr(default_config, key, value)
                
                # Create instance
                agent_instance = metadata.agent_class(default_config)
                pool.instances.append(agent_instance)
                pool.current_load += 1
                
                metadata.usage_count += 1
                self.registry_stats["active_agents"] += 1
                self.registry_stats["load_balancing_decisions"] += 1
                
                logger.info("New agent instance created", 
                           agent_name=agent_name,
                           pool_size=len(pool.instances))
                
                return agent_instance
            
            logger.warning("Agent pool at capacity", 
                          agent_name=agent_name,
                          max_instances=pool.max_instances)
            return None
            
        except Exception as e:
            logger.error("Failed to create agent instance", 
                        agent_name=agent_name, 
                        error=str(e))
            return None
    
    def start_health_monitoring(self, check_interval: int = 300):
        """Start background health monitoring for all registered agents."""
        if self.health_monitor_task and not self.health_monitor_task.done():
            logger.warning("Health monitoring already running")
            return
        
        self.health_monitor_task = asyncio.create_task(
            self._health_monitor_loop(check_interval)
        )
        logger.info("Health monitoring started", check_interval=check_interval)
    
    async def _health_monitor_loop(self, check_interval: int):
        """Background loop for health monitoring."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all agent pools."""
        for agent_name, pool in self.agent_pools.items():
            try:
                metadata = self.registered_agents[agent_name]
                
                # Check each instance in the pool
                healthy_instances = []
                for instance in pool.instances:
                    if await self._check_agent_health(instance):
                        healthy_instances.append(instance)
                        metadata.health_status = AgentStatus.IDLE
                    else:
                        metadata.health_status = AgentStatus.FAILED
                        logger.warning("Unhealthy agent instance detected", 
                                     agent_name=agent_name,
                                     agent_id=instance.agent_id)
                
                # Update pool with healthy instances
                pool.instances = healthy_instances
                pool.current_load = len(healthy_instances)
                
                metadata.last_health_check = datetime.now()
                self.registry_stats["health_checks_performed"] += 1
                
            except Exception as e:
                logger.error("Health check failed for agent", 
                           agent_name=agent_name, 
                           error=str(e))
    
    async def _check_agent_health(self, agent: BaseAgent) -> bool:
        """Check if an individual agent instance is healthy."""
        try:
            # Simple health check - ensure agent can respond
            test_result = await asyncio.wait_for(
                agent.execute("health_check", {"test": True}),
                timeout=10.0
            )
            return test_result.status != AgentStatus.FAILED
        except Exception:
            return False
    
    def _get_available_instance(self, pool: AgentPool) -> Optional[BaseAgent]:
        """Get an available agent instance from the pool using load balancing strategy."""
        if not pool.instances:
            return None
        
        if pool.load_balancer_strategy == "round_robin":
            # Simple round-robin selection
            return pool.instances[pool.current_load % len(pool.instances)]
        
        elif pool.load_balancer_strategy == "least_loaded":
            # Find instance with lowest current load (if trackable)
            return min(pool.instances, key=lambda x: getattr(x, 'current_tasks', 0))
        
        elif pool.load_balancer_strategy == "performance_based":
            # Select based on performance metrics
            return max(pool.instances, key=lambda x: x.get_success_rate())
        
        return pool.instances[0]  # Fallback
    
    def _check_resource_compatibility(self, 
                                    agent_requirements: Dict[str, Any],
                                    task_constraints: Dict[str, Any]) -> bool:
        """Check if agent resource requirements are compatible with task constraints."""
        for resource, requirement in agent_requirements.items():
            constraint = task_constraints.get(resource)
            if constraint is not None:
                if isinstance(requirement, (int, float)) and isinstance(constraint, (int, float)):
                    if requirement > constraint:
                        return False
        return True
    
    def _calculate_compatibility_score(self, 
                                     metadata: AgentMetadata,
                                     task_requirements: Dict[str, Any]) -> float:
        """Calculate compatibility score between agent and task."""
        score = 0.0
        
        # Base score from success rate
        score += metadata.success_rate * 0.4
        
        # Performance tier bonus
        tier_scores = {"high": 0.3, "medium": 0.2, "low": 0.1}
        score += tier_scores.get(metadata.performance_tier, 0.1)
        
        # Specialization match bonus
        task_domain = task_requirements.get("domain", "")
        if task_domain in metadata.specializations:
            score += 0.2
        
        # Response time penalty (lower is better)
        if metadata.average_response_time > 0:
            score -= min(metadata.average_response_time / 1000, 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _load_registry_config(self, config_path: str):
        """Load registry configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Process configuration
            for agent_name, agent_config in config.get("agents", {}).items():
                # Auto-register agents from config
                # This would require dynamic importing based on config
                pass
                
        except Exception as e:
            logger.error("Failed to load registry config", 
                        config_path=config_path, 
                        error=str(e))
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        return {
            **self.registry_stats,
            "registered_agents_count": len(self.registered_agents),
            "active_pools": len(self.agent_pools),
            "capability_coverage": len(self.capability_index),
            "total_instances": sum(len(pool.instances) for pool in self.agent_pools.values())
        }
    
    def export_registry_state(self, output_path: str):
        """Export current registry state for backup/analysis."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "registered_agents": {
                name: {
                    "capabilities": [c.value for c in meta.capabilities],
                    "specializations": meta.specializations,
                    "performance_tier": meta.performance_tier,
                    "usage_count": meta.usage_count,
                    "success_rate": meta.success_rate,
                    "health_status": meta.health_status.value
                }
                for name, meta in self.registered_agents.items()
            },
            "stats": self.get_registry_stats()
        }
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info("Registry state exported", output_path=output_path)


# Global registry instance
_global_registry = None

def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry

def register_agent_type(agent_name: str, 
                       agent_class: Type[BaseAgent],
                       capabilities: Set[AgentCapability],
                       **kwargs) -> bool:
    """Convenience function to register an agent type."""
    registry = get_agent_registry()
    return registry.register_agent(agent_name, agent_class, capabilities, **kwargs)
