"""
Orchestrator Configuration Models
"""

import yaml
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_generations: int = 50
    max_agents: int = 10
    population_size: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    elite_count: int = 2
    trust_threshold: float = 0.7
    hitl_enabled: bool = True
    hitl_timeout: int = 300
    convergence_threshold: float = 0.95
    max_execution_time: int = 3600  # 1 hour
    
    @classmethod
    def from_file(cls, config_path: str) -> 'OrchestratorConfig':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            ga_config = config_data.get("genetic_algorithm", {})
            # Filter out parameters that don't exist in the dataclass
            valid_params = {
                k: v for k, v in ga_config.items() 
                if k in cls.__dataclass_fields__
            }
            return cls(**valid_params)
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "max_generations": self.max_generations,
            "max_agents": self.max_agents,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elite_count": self.elite_count,
            "trust_threshold": self.trust_threshold,
            "hitl_enabled": self.hitl_enabled,
            "hitl_timeout": self.hitl_timeout,
            "convergence_threshold": self.convergence_threshold,
            "max_execution_time": self.max_execution_time
        } 