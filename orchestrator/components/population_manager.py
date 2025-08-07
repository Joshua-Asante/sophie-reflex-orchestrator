"""
Population Manager Component

Handles population updates, agent pruning, and creation.
"""

from typing import Dict, Any, List
import structlog

from agents.base_agent import AgentConfig
from agents.prover import ProverAgent
from agents.refiner import RefinerAgent
from ..models.orchestrator_config import OrchestratorConfig

logger = structlog.get_logger()


class PopulationManager:
    """Handles population updates, agent pruning, and creation."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.refiners = []
        self.agents = []
    
    def set_agents(self, agents: List[ProverAgent]):
        """Set the current agent population."""
        self.agents = agents
    
    def set_refiners(self, refiners: List[RefinerAgent]):
        """Set the refiner agents."""
        self.refiners = refiners
    
    async def update_population(self, evaluation_results: List[Dict[str, Any]], 
                              interventions: List[Dict[str, Any]], 
                              generation: int) -> Dict[str, Any]:
        """Update the agent population based on evaluation results."""
        try:
            if not self.refiners:
                logger.warning("No refiner agents found")
                return {"refinements": [], "new_agents": [], "pruned_agents": []}
            
            # Use the first refiner (could be enhanced to use multiple)
            refiner = self.refiners[0]
            
            # Prepare context for refiner
            context = {
                "evaluation_results": evaluation_results,
                "current_agents": self.agents,
                "interventions": interventions,
                "generation_info": {
                    "generation": generation,
                    "population_size": len(self.agents),
                    "best_score": max(eval_result.get("overall_score", 0.0) for eval_result in evaluation_results) if evaluation_results else 0.0
                }
            }
            
            # Execute refiner
            refiner_result = await refiner.execute_with_retry("Update population", context)
            
            if refiner_result.status == "completed":
                result_data = refiner_result.result
                
                # Process refinements
                new_agents = []
                pruned_agents = []
                
                # Add new agents from refiner result
                for agent_info in result_data.get("refinement_results", {}).get("new_agents", []):
                    if agent_info.get("agent_info"):
                        # Create new agent from info
                        agent_data = agent_info["agent_info"]
                        agent_config = AgentConfig(
                            name=agent_data["name"],
                            prompt=agent_data.get("prompt", "You are a helpful assistant."),
                            model=agent_data.get("model", "openai"),
                            temperature=agent_data.get("temperature", 0.7),
                            max_tokens=agent_data.get("max_tokens", 1000),
                            hyperparameters=agent_data.get("hyperparameters", {})
                        )
                        
                        new_agent = ProverAgent(
                            config=agent_config,
                            agent_id=agent_data["agent_id"]
                        )
                        new_agents.append(new_agent)
                
                # Process pruned agents
                for pruned_info in result_data.get("refinement_results", {}).get("pruned_agents", []):
                    pruned_agents.append(pruned_info)
                
                logger.info("Population updated", new_agents=len(new_agents), 
                           pruned_agents=len(pruned_agents))
                return {
                    "refinements": result_data.get("refinement_results", {}),
                    "new_agents": new_agents,
                    "pruned_agents": pruned_agents
                }
            else:
                logger.warning("Refiner execution failed")
                return {"refinements": [], "new_agents": [], "pruned_agents": []}
            
        except Exception as e:
            logger.error("Failed to update population", error=str(e))
            return {"refinements": [], "new_agents": [], "pruned_agents": []}
    
    async def prune_agents(self, agent_ids: List[str], reason: str = "Performance"):
        """Prune agents from the population."""
        try:
            original_count = len(self.agents)
            self.agents = [agent for agent in self.agents if agent.agent_id not in agent_ids]
            pruned_count = original_count - len(self.agents)
            
            logger.info("Agents pruned", pruned_count=pruned_count, reason=reason)
            return pruned_count
            
        except Exception as e:
            logger.error("Failed to prune agents", error=str(e))
            return 0
    
    async def add_agents(self, new_agents: List[ProverAgent]):
        """Add new agents to the population."""
        try:
            original_count = len(self.agents)
            self.agents.extend(new_agents)
            
            # Limit population size
            if len(self.agents) > self.config.max_agents:
                self.agents = self.agents[:self.config.max_agents]
            
            added_count = len(self.agents) - original_count
            logger.info("Agents added", added_count=added_count, total_agents=len(self.agents))
            return added_count
            
        except Exception as e:
            logger.error("Failed to add agents", error=str(e))
            return 0
    
    async def get_population_info(self) -> Dict[str, Any]:
        """Get information about the current population."""
        try:
            return {
                "total_agents": len(self.agents),
                "max_agents": self.config.max_agents,
                "agent_ids": [agent.agent_id for agent in self.agents],
                "agent_names": [agent.config.name for agent in self.agents]
            }
        except Exception as e:
            logger.error("Failed to get population info", error=str(e))
            return {"error": str(e)}
    
    async def get_best_agents(self, count: int = 5) -> List[ProverAgent]:
        """Get the best performing agents based on trust scores."""
        try:
            # This would need to be enhanced with actual trust score lookup
            # For now, return the first N agents
            return self.agents[:count]
        except Exception as e:
            logger.error("Failed to get best agents", error=str(e))
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        try:
            return {
                "total_agents": len(self.agents),
                "max_agents": self.config.max_agents,
                "population_utilization": len(self.agents) / self.config.max_agents,
                "refiner_count": len(self.refiners)
            }
        except Exception as e:
            logger.error("Failed to get population statistics", error=str(e))
            return {"error": str(e)} 