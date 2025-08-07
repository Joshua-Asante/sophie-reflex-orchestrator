"""
Mock Orchestrator for Testing

Provides a mock implementation of the orchestrator for testing CLI components
without requiring the full orchestrator system.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import structlog

from orchestrator import GenerationResult, OrchestratorStatus

logger = structlog.get_logger()


@dataclass
class MockGenerationResult:
    """Mock generation result for testing."""
    generation: int
    task: str
    agents: List[Any]
    best_solution: Optional[Dict[str, Any]]
    best_score: float
    average_score: float
    execution_time: float
    interventions: List[Dict[str, Any]]
    trust_scores: Dict[str, float]


class MockOrchestrator:
    """Mock orchestrator for testing CLI components."""
    
    def __init__(self, config_path: str = "configs/system.yaml"):
        self.config_path = config_path
        self.status = OrchestratorStatus.IDLE
        self.current_task = None
        self.current_generation = 0
        self.agents = []
        self.generation_results = []
        self.start_time = None
        self.session_id = None
        self.max_generations = 10
        self.population_size = 3
        
        # Mock agent data
        self._mock_agents = [
            {"name": "Mock Prover", "agent_id": "prover_1", "status": "active", 
             "trust_score": 0.85, "success_rate": 0.78, "model": "gpt-4"},
            {"name": "Mock Evaluator", "agent_id": "evaluator_1", "status": "active", 
             "trust_score": 0.92, "success_rate": 0.85, "model": "gpt-4"},
            {"name": "Mock Refiner", "agent_id": "refiner_1", "status": "active", 
             "trust_score": 0.88, "success_rate": 0.82, "model": "gpt-4"},
        ]
    
    async def start_task(self, task: str, session_id: str = None) -> str:
        """Start a mock task."""
        self.session_id = session_id or str(uuid.uuid4())
        self.current_task = task
        self.status = OrchestratorStatus.RUNNING
        self.start_time = datetime.now()
        self.current_generation = 0
        self.generation_results = []
        
        # Initialize mock agents
        self.agents = []
        for agent_data in self._mock_agents:
            mock_agent = MockAgent(agent_data)
            self.agents.append(mock_agent)
        
        logger.info("Mock task started", task=task, session_id=self.session_id)
        return self.session_id
    
    async def run_generation(self) -> GenerationResult:
        """Run a mock generation."""
        if self.current_generation >= self.max_generations:
            self.status = OrchestratorStatus.COMPLETED
            raise StopIteration("Max generations reached")
        
        self.current_generation += 1
        
        # Simulate generation time
        await asyncio.sleep(0.5)
        
        # Generate mock results
        best_score = 0.7 + (self.current_generation * 0.05) + (self.current_generation % 3 * 0.02)
        average_score = best_score - 0.1
        execution_time = 2.5 + (self.current_generation * 0.3)
        
        # Randomly add interventions
        interventions = []
        if self.current_generation % 3 == 0:
            interventions.append({
                "type": "human_review",
                "agent_id": "prover_1",
                "reason": "Quality check required",
                "timestamp": datetime.now().isoformat()
            })
        
        # Mock trust scores
        trust_scores = {
            "prover_1": 0.85 + (self.current_generation * 0.01),
            "evaluator_1": 0.92 + (self.current_generation * 0.005),
            "refiner_1": 0.88 + (self.current_generation * 0.008)
        }
        
        result = GenerationResult(
            generation=self.current_generation,
            task=self.current_task,
            agents=self.agents,
            best_solution={"overall_feedback": f"Mock solution for generation {self.current_generation}"},
            best_score=best_score,
            average_score=average_score,
            execution_time=execution_time,
            interventions=interventions,
            trust_scores=trust_scores
        )
        
        self.generation_results.append(result)
        logger.info("Mock generation completed", generation=self.current_generation, best_score=best_score)
        
        return result
    
    async def run_complete_task(self, task: str) -> List[GenerationResult]:
        """Run a complete mock task."""
        await self.start_task(task)
        
        results = []
        while self.should_continue():
            try:
                result = await self.run_generation()
                results.append(result)
            except StopIteration:
                break
        
        return results
    
    def should_continue(self) -> bool:
        """Check if the orchestrator should continue."""
        return (self.status == OrchestratorStatus.RUNNING and 
                self.current_generation < self.max_generations)
    
    async def finalize_task(self):
        """Finalize the mock task."""
        self.status = OrchestratorStatus.COMPLETED
        logger.info("Mock task finalized", session_id=self.session_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get mock orchestrator status."""
        elapsed_time = 0
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": self.status.value,
            "current_task": self.current_task,
            "current_generation": self.current_generation,
            "total_agents": len(self.agents),
            "best_score": max([r.best_score for r in self.generation_results]) if self.generation_results else 0.0,
            "average_score": sum([r.average_score for r in self.generation_results]) / len(self.generation_results) if self.generation_results else 0.0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "elapsed_time": elapsed_time
        }
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get mock results."""
        return [result.to_dict() for result in self.generation_results]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get mock statistics."""
        total_execution_time = sum([r.execution_time for r in self.generation_results])
        total_interventions = sum([len(r.interventions) for r in self.generation_results])
        
        # Calculate average trust score safely
        average_trust_score = 0.0
        if self.agents:
            average_trust_score = sum([agent.get_info()["trust_score"] for agent in self.agents]) / len(self.agents)
        
        return {
            "orchestrator": {
                "status": self.status.value,
                "total_generations": len(self.generation_results),
                "total_agents": len(self.agents),
                "total_execution_time": total_execution_time,
                "hitl_enabled": True
            },
            "trust": {
                "total_agents": len(self.agents),
                "total_events": len(self.generation_results) * len(self.agents),
                "average_trust_score": average_trust_score
            },
            "audit": {
                "total_events": len(self.generation_results),
                "total_plan_diffs": len(self.generation_results) * 2,
                "total_metrics": len(self.generation_results) * 3
            },
            "vector_store": {
                "backend": "mock",
                "total_entries": len(self.generation_results) * 5
            }
        }
    
    def pause(self):
        """Pause the mock orchestrator."""
        if self.status == OrchestratorStatus.RUNNING:
            self.status = OrchestratorStatus.PAUSED
            logger.info("Mock orchestrator paused")
    
    def resume(self):
        """Resume the mock orchestrator."""
        if self.status == OrchestratorStatus.PAUSED:
            self.status = OrchestratorStatus.RUNNING
            logger.info("Mock orchestrator resumed")
    
    def stop(self):
        """Stop the mock orchestrator."""
        self.status = OrchestratorStatus.IDLE
        logger.info("Mock orchestrator stopped")


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, agent_data: Dict[str, Any]):
        self.agent_data = agent_data
    
    def get_info(self) -> Dict[str, Any]:
        """Get mock agent info."""
        return self.agent_data


class MockOrchestratorFactory:
    """Factory for creating mock orchestrators."""
    
    @staticmethod
    def create_mock_orchestrator(config_path: str = "configs/system.yaml") -> MockOrchestrator:
        """Create a mock orchestrator instance."""
        return MockOrchestrator(config_path)
    
    @staticmethod
    def create_mock_orchestrator_with_results(config_path: str = "configs/system.yaml", 
                                            num_generations: int = 5) -> MockOrchestrator:
        """Create a mock orchestrator with pre-generated results."""
        orchestrator = MockOrchestrator(config_path)
        orchestrator.max_generations = num_generations
        return orchestrator


class MockConfigManager:
    """Mock configuration manager for testing."""
    
    def __init__(self):
        self.mock_configs = {
            "system": {
                "genetic_algorithm": {
                    "max_generations": 10,
                    "population_size": 3,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.3,
                    "elite_count": 1
                },
                "memory": {
                    "backend": "mock",
                    "max_entries": 1000
                },
                "logging": {
                    "level": "INFO",
                    "format": "json"
                },
                "hitl": {
                    "enabled": True,
                    "timeout": 300
                }
            },
            "agents": {
                "agents": [
                    {"name": "Mock Prover", "type": "prover", "model": "gpt-4"},
                    {"name": "Mock Evaluator", "type": "evaluator", "model": "gpt-4"},
                    {"name": "Mock Refiner", "type": "refiner", "model": "gpt-4"}
                ],
                "models": {
                    "gpt-4": {"provider": "openai", "max_tokens": 4000}
                },
                "trust": {
                    "initial_score": 0.8,
                    "decay_rate": 0.95
                }
            },
            "rubric": {
                "criteria": [
                    {"name": "quality", "description": "Solution quality", "weight": 0.4},
                    {"name": "creativity", "description": "Creative approach", "weight": 0.3},
                    {"name": "feasibility", "description": "Implementation feasibility", "weight": 0.3}
                ],
                "weights": {
                    "quality": 0.4,
                    "creativity": 0.3,
                    "feasibility": 0.3
                },
                "thresholds": {
                    "minimum_score": 0.5,
                    "excellent_score": 0.9
                }
            },
            "policies": {
                "policies": [
                    {"name": "quality_check", "type": "evaluation", "conditions": ["score < 0.7"]},
                    {"name": "human_review", "type": "intervention", "conditions": ["score < 0.5"]}
                ],
                "rules": {
                    "max_generations": 50,
                    "convergence_threshold": 0.95
                },
                "enforcement": {
                    "strict": True,
                    "auto_approve": False
                }
            }
        }
    
    def load_and_validate_all(self) -> Dict[str, Any]:
        """Load and validate all mock configurations."""
        return self.mock_configs
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a mock configuration."""
        if config_name not in self.mock_configs:
            raise ValueError(f"Mock configuration '{config_name}' not found")
        return self.mock_configs[config_name] 