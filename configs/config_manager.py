"""
Configuration Manager for Sophie Reflex Orchestrator

Handles configuration loading, validation, and schema checking.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class SystemConfig:
    """System configuration schema."""
    genetic_algorithm: Dict[str, Any]
    memory: Dict[str, Any]
    logging: Dict[str, Any]
    hitl: Dict[str, Any]


@dataclass
class AgentConfig:
    """Agent configuration schema."""
    base_agent: Dict[str, Any]
    provers: List[Dict[str, Any]]
    evaluators: List[Dict[str, Any]]
    refiners: List[Dict[str, Any]]
    agent_monitoring: Dict[str, Any]


@dataclass
class RubricConfig:
    """Rubric configuration schema."""
    categories: Dict[str, Any]
    scoring: Dict[str, Any]
    task_types: Dict[str, Any]
    evaluation_process: Dict[str, Any]
    performance_metrics: Dict[str, Any]


@dataclass
class PolicyConfig:
    """Policy configuration schema."""
    hitl: Dict[str, Any]
    agent_lifecycle: Dict[str, Any]
    trust: Dict[str, Any]
    resource_limits: Dict[str, Any]
    security: Dict[str, Any]
    performance: Dict[str, Any]
    quality_assurance: Dict[str, Any]
    adaptive_learning: Dict[str, Any]


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self._configs: Dict[str, Any] = {}
        self._validated_configs: Dict[str, Any] = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file."""
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info("Configuration loaded", config_name=config_name, path=config_path)
            self._configs[config_name] = config
            return config
            
        except yaml.YAMLError as e:
            logger.error("Failed to parse YAML configuration", config_name=config_name, error=str(e))
            raise ValueError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            logger.error("Failed to load configuration", config_name=config_name, error=str(e))
            raise
    
    def validate_system_config(self, config: Dict[str, Any]) -> SystemConfig:
        """Validate system configuration schema."""
        required_sections = ['genetic_algorithm', 'memory', 'logging', 'hitl']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in system config: {section}")
        
        # Validate genetic algorithm settings
        ga_config = config['genetic_algorithm']
        required_ga_fields = ['max_generations', 'population_size', 'mutation_rate']
        for field in required_ga_fields:
            if field not in ga_config:
                raise ValueError(f"Missing required genetic algorithm field: {field}")
        
        # Validate numeric fields
        if not isinstance(ga_config.get('max_generations'), int) or ga_config['max_generations'] <= 0:
            raise ValueError("max_generations must be a positive integer")
        
        if not isinstance(ga_config.get('population_size'), int) or ga_config['population_size'] <= 0:
            raise ValueError("population_size must be a positive integer")
        
        mutation_rate = ga_config.get('mutation_rate')
        if not isinstance(mutation_rate, (int, float)) or not (0 <= mutation_rate <= 1):
            raise ValueError("mutation_rate must be a number between 0 and 1")
        
        # Only pass the fields that SystemConfig expects
        system_config_data = {
            'genetic_algorithm': config['genetic_algorithm'],
            'memory': config['memory'],
            'logging': config.get('logging', {}),
            'hitl': config['hitl']
        }
        return SystemConfig(**system_config_data)
    
    def validate_agent_config(self, config: Dict[str, Any]) -> AgentConfig:
        """Validate agent configuration schema."""
        required_sections = ['base_agent', 'provers', 'evaluators', 'refiners', 'agent_monitoring']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in agent config: {section}")
        
        # Validate provers list
        provers = config.get('provers', [])
        if not isinstance(provers, list):
            raise ValueError("provers must be a list")
        
        for i, prover in enumerate(provers):
            if not isinstance(prover, dict):
                raise ValueError(f"Prover {i} must be a dictionary")
            
            required_prover_fields = ['name', 'description', 'prompt', 'model']
            for field in required_prover_fields:
                if field not in prover:
                    raise ValueError(f"Prover {i} missing required field: {field}")
        
        # Validate evaluators list
        evaluators = config.get('evaluators', [])
        if not isinstance(evaluators, list):
            raise ValueError("evaluators must be a list")
        
        for i, evaluator in enumerate(evaluators):
            if not isinstance(evaluator, dict):
                raise ValueError(f"Evaluator {i} must be a dictionary")
            
            required_evaluator_fields = ['name', 'description', 'prompt', 'model']
            for field in required_evaluator_fields:
                if field not in evaluator:
                    raise ValueError(f"Evaluator {i} missing required field: {field}")
        
        # Validate refiners list
        refiners = config.get('refiners', [])
        if not isinstance(refiners, list):
            raise ValueError("refiners must be a list")
        
        for i, refiner in enumerate(refiners):
            if not isinstance(refiner, dict):
                raise ValueError(f"Refiner {i} must be a dictionary")
            
            required_refiner_fields = ['name', 'description', 'prompt', 'model']
            for field in required_refiner_fields:
                if field not in refiner:
                    raise ValueError(f"Refiner {i} missing required field: {field}")
        
        return AgentConfig(**config)
    
    def validate_rubric_config(self, config: Dict[str, Any]) -> RubricConfig:
        """Validate rubric configuration schema."""
        required_sections = ['categories', 'scoring', 'task_types', 'evaluation_process', 'performance_metrics']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in rubric config: {section}")
        
        # Validate categories
        categories = config.get('categories', {})
        if not isinstance(categories, dict):
            raise ValueError("categories must be a dictionary")
        
        # Validate scoring
        scoring = config.get('scoring', {})
        if not isinstance(scoring, dict):
            raise ValueError("scoring must be a dictionary")
        
        # Validate task_types
        task_types = config.get('task_types', {})
        if not isinstance(task_types, dict):
            raise ValueError("task_types must be a dictionary")
        
        # Validate evaluation_process
        evaluation_process = config.get('evaluation_process', {})
        if not isinstance(evaluation_process, dict):
            raise ValueError("evaluation_process must be a dictionary")
        
        # Validate performance_metrics
        performance_metrics = config.get('performance_metrics', {})
        if not isinstance(performance_metrics, dict):
            raise ValueError("performance_metrics must be a dictionary")
        
        return RubricConfig(**config)
    
    def validate_policy_config(self, config: Dict[str, Any]) -> PolicyConfig:
        """Validate policy configuration schema."""
        required_sections = ['hitl', 'agent_lifecycle', 'trust', 'resource_limits', 'security', 'performance', 'quality_assurance', 'adaptive_learning']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in policy config: {section}")
        
        # Validate all sections are dictionaries
        for section in required_sections:
            section_data = config.get(section, {})
            if not isinstance(section_data, dict):
                raise ValueError(f"{section} must be a dictionary")
        
        return PolicyConfig(**config)
    
    def load_and_validate_all(self) -> Dict[str, Any]:
        """Load and validate all configuration files."""
        configs_to_load = ['system', 'agents', 'rubric', 'policies']
        
        for config_name in configs_to_load:
            try:
                config = self.load_config(config_name)
                
                # Validate based on config type
                if config_name == 'system':
                    validated_config = self.validate_system_config(config)
                elif config_name == 'agents':
                    validated_config = self.validate_agent_config(config)
                elif config_name == 'rubric':
                    validated_config = self.validate_rubric_config(config)
                elif config_name == 'policies':
                    validated_config = self.validate_policy_config(config)
                else:
                    validated_config = config
                
                self._validated_configs[config_name] = validated_config
                logger.info("Configuration validated successfully", config_name=config_name)
                
            except Exception as e:
                logger.error("Configuration validation failed", config_name=config_name, error=str(e))
                raise
        
        return self._validated_configs
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a validated configuration."""
        if config_name not in self._validated_configs:
            raise ValueError(f"Configuration '{config_name}' not loaded or validated")
        
        return self._validated_configs[config_name]
    
    def get_system_config(self) -> SystemConfig:
        """Get the validated system configuration."""
        return self.get_config('system')
    
    def get_agent_config(self) -> AgentConfig:
        """Get the validated agent configuration."""
        return self.get_config('agents')
    
    def get_rubric_config(self) -> RubricConfig:
        """Get the validated rubric configuration."""
        return self.get_config('rubric')
    
    def get_policy_config(self) -> PolicyConfig:
        """Get the validated policy configuration."""
        return self.get_config('policies')
    
    def validate_config_path(self, config_path: str) -> bool:
        """Validate that a configuration path exists and is readable."""
        if not os.path.exists(config_path):
            return False
        
        if not os.path.isfile(config_path):
            return False
        
        if not os.access(config_path, os.R_OK):
            return False
        
        return True
    
    def get_default_config_path(self, config_name: str) -> str:
        """Get the default path for a configuration file."""
        return os.path.join(self.config_dir, f"{config_name}.yaml")
    
    def list_available_configs(self) -> List[str]:
        """List all available configuration files."""
        configs = []
        if os.path.exists(self.config_dir):
            for file in os.listdir(self.config_dir):
                if file.endswith('.yaml') or file.endswith('.yml'):
                    configs.append(file[:-4] if file.endswith('.yaml') else file[:-4])
        return configs 