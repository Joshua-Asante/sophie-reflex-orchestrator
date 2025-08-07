"""
Agent Manager Component

Handles agent initialization, execution, and management.
"""

import asyncio
from typing import Dict, Any, List
import structlog

from agents.base_agent import AgentConfig, AgentStatus
from agents.prover import ProverAgent
from agents.evaluator import EvaluatorAgent
from agents.refiner import RefinerAgent
from ..models.orchestrator_config import OrchestratorConfig

logger = structlog.get_logger()


class AgentManager:
    """Manages agent initialization and execution."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.agents: List[ProverAgent] = []
        self.evaluators: List[EvaluatorAgent] = []
        self.refiners: List[RefinerAgent] = []
        self.agents_config = self._load_agents_config()
    
    def _load_agents_config(self) -> Dict[str, Any]:
        """Load agents configuration."""
        try:
            import yaml
            with open("configs/agents.yaml", 'r') as f:
                config = yaml.safe_load(f)
                # Ensure base_agent config is available
                if 'base_agent' not in config:
                    config['base_agent'] = {
                        'max_tokens': 4000,
                        'temperature': 0.7,
                        'timeout': 30,
                        'max_retries': 3,
                        'retry_delay': 1
                    }
                return config
        except Exception as e:
            logger.error("Failed to load agents configuration", error=str(e))
            return {
                'base_agent': {
                    'max_tokens': 4000,
                    'temperature': 0.7,
                    'timeout': 30,
                    'max_retries': 3,
                    'retry_delay': 1
                }
            }
    
    async def initialize_population(self):
        """Initialize the agent population."""
        try:
            self.agents = []
            
            # Create initial provers from configuration
            prover_configs = self.agents_config.get("provers", [])
            
            for config_data in prover_configs:
                for i in range(self.config.population_size):
                    # Use base agent config for missing fields
                    base_config = self.agents_config.get('base_agent', {})
                    agent_config = AgentConfig(
                        name=f"{config_data['name']}_{i}",
                        prompt=config_data['prompt'],
                        model=config_data['model'],
                        temperature=config_data.get('temperature', base_config.get('temperature', 0.7)),
                        max_tokens=config_data.get('max_tokens', base_config.get('max_tokens', 4000)),
                        timeout=config_data.get('timeout', base_config.get('timeout', 30)),
                        max_retries=config_data.get('max_retries', base_config.get('max_retries', 3)),
                        retry_delay=config_data.get('retry_delay', base_config.get('retry_delay', 1)),
                        hyperparameters=config_data.get('hyperparameters', {})
                    )
                    
                    agent = ProverAgent(config=agent_config)
                    self.agents.append(agent)
                    
                    logger.info("Agent created", agent_id=agent.agent_id, name=agent.config.name)
            
            # Create evaluators
            evaluator_configs = self.agents_config.get("evaluators", [])
            self.evaluators = []
            
            for config_data in evaluator_configs:
                # Use base agent config for missing fields
                base_config = self.agents_config.get('base_agent', {})
                agent_config = AgentConfig(
                    name=config_data['name'],
                    prompt=config_data['prompt'],
                    model=config_data['model'],
                    temperature=config_data.get('temperature', base_config.get('temperature', 0.7)),
                    max_tokens=config_data.get('max_tokens', base_config.get('max_tokens', 4000)),
                    hyperparameters=config_data.get('hyperparameters', {})
                )
                
                evaluator = EvaluatorAgent(
                    config=agent_config,
                    rubric_config=self._load_rubric_config()
                )
                self.evaluators.append(evaluator)
            
            # Create refiners
            refiner_configs = self.agents_config.get("refiners", [])
            self.refiners = []
            
            for config_data in refiner_configs:
                # Use base agent config for missing fields
                base_config = self.agents_config.get('base_agent', {})
                agent_config = AgentConfig(
                    name=config_data['name'],
                    prompt=config_data['prompt'],
                    model=config_data['model'],
                    temperature=config_data.get('temperature', base_config.get('temperature', 0.7)),
                    max_tokens=config_data.get('max_tokens', base_config.get('max_tokens', 4000)),
                    hyperparameters=config_data.get('hyperparameters', {})
                )
                
                refiner = RefinerAgent(config=agent_config)
                self.refiners.append(refiner)
            
            logger.info("Population initialized", agents=len(self.agents), 
                        evaluators=len(self.evaluators), refiners=len(self.refiners))
            
        except Exception as e:
            logger.error("Failed to initialize population", error=str(e))
            raise
    
    def _load_rubric_config(self) -> Dict[str, Any]:
        """Load rubric configuration."""
        try:
            import yaml
            with open("configs/rubric.yaml", 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load rubric configuration", error=str(e))
            return {}
    
    async def execute_provers(self, task: str) -> List[Dict[str, Any]]:
        """Execute all prover agents concurrently."""
        try:
            logger.info("Starting prover execution")
            
            prover_results = []
            
            # Execute all provers concurrently
            tasks = []
            for agent in self.agents:
                if isinstance(agent, ProverAgent):
                    agent_task = agent.execute_with_retry(task)
                    tasks.append(agent_task)
            
            if not tasks:
                logger.warning("No prover agents found")
                return prover_results
            
            # Wait for all provers to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("Prover execution failed", agent_id=self.agents[i].agent_id, error=str(result))
                    continue
                
                if result.status == AgentStatus.COMPLETED:
                    prover_results.append({
                        "agent_id": result.agent_id,
                        "agent_name": result.agent_name,
                        "result": result.result,
                        "confidence_score": result.confidence_score,
                        "execution_time": result.execution_time,
                        "metadata": result.metadata
                    })
            
            logger.info("Prover execution completed", results=len(prover_results))
            return prover_results
            
        except Exception as e:
            logger.error("Prover execution failed", error=str(e))
            raise
    
    async def get_agents(self) -> List[Any]:
        """Get all agents."""
        return self.agents
    
    async def get_agent_info(self) -> List[Dict[str, Any]]:
        """Get information about all agents."""
        return [agent.get_info() for agent in self.agents]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_agents": len(self.agents),
            "total_evaluators": len(self.evaluators),
            "total_refiners": len(self.refiners),
            "agent_types": {
                "provers": len([a for a in self.agents if isinstance(a, ProverAgent)]),
                "evaluators": len(self.evaluators),
                "refiners": len(self.refiners)
            }
        }
    
    def register_agent(self, agent):
        """Register an agent with the manager."""
        if isinstance(agent, ProverAgent):
            self.agents.append(agent)
            logger.info("Prover agent registered", agent_id=agent.agent_id)
        elif isinstance(agent, EvaluatorAgent):
            self.evaluators.append(agent)
            logger.info("Evaluator agent registered", agent_id=agent.agent_id)
        elif isinstance(agent, RefinerAgent):
            self.refiners.append(agent)
            logger.info("Refiner agent registered", agent_id=agent.agent_id)
        else:
            logger.warning("Unknown agent type registered", agent_type=type(agent))
    
    def get_registered_agents(self):
        """Get all registered agents."""
        return self.agents + self.evaluators + self.refiners
    
    def get_agent(self, agent_id: str):
        """Get a specific agent by ID."""
        for agent in self.get_registered_agents():
            if agent.agent_id == agent_id:
                return agent
        return None 