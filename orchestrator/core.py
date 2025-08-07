"""
Core Orchestrator Module

The main coordinator that orchestrates the genetic algorithm loop and delegates
specific responsibilities to specialized modules.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import structlog
import yaml

from .components.agent_manager import AgentManager
from .components.evaluation_engine import EvaluationEngine
from .components.hitl_manager import HITLManager
from .components.trust_manager import TrustManager
from .components.audit_manager import AuditManager
from .components.memory_manager import MemoryManager
from .components.population_manager import PopulationManager
from .models.generation_result import GenerationResult
from .models.orchestrator_config import OrchestratorConfig
from .models.orchestrator_status import OrchestratorStatus
from governance.policy_engine import PolicyEngine
from memory.trust_tracker import TrustTracker
from governance.audit_log import AuditLog

logger = structlog.get_logger()


@dataclass
class OrchestratorState:
    """Represents the current state of the orchestrator."""
    status: OrchestratorStatus
    current_task: Optional[str]
    current_generation: int
    start_time: Optional[datetime]
    session_id: Optional[str]
    generation_results: List[GenerationResult]


class SophieReflexOrchestrator:
    """Core GA loop coordinator with modular components."""
    
    def __init__(self, config_path: str = "configs/system.yaml"):
        self.config_path = config_path
        self.state = OrchestratorState(
            status=OrchestratorStatus.IDLE,
            current_task=None,
            current_generation=0,
            start_time=None,
            session_id=None,
            generation_results=[]
        )
        
        # Load configuration
        self.config = OrchestratorConfig.from_file(config_path)
        
        # Load policies configuration
        self.policies_config = self._load_policies_config()
        
        # Initialize governance components
        self.policy_engine = PolicyEngine(self.policies_config)
        self.trust_tracker = TrustTracker({"database_path": "trust_tracker.db"})
        self.audit_log = AuditLog()
        
        # Initialize modular components
        self.agent_manager = AgentManager(self.config)
        self.evaluation_engine = EvaluationEngine(self.config)
        self.hitl_manager = HITLManager(self.config)
        self.trust_manager = TrustManager(self.config)
        self.audit_manager = AuditManager(self.config)
        self.memory_manager = MemoryManager(self.config)
        self.population_manager = PopulationManager(self.config)
        
        # Set up dependency injection
        self._setup_dependencies()
        
        logger.info("Sophie Reflex Orchestrator initialized", config_path=config_path)
    
    def _load_policies_config(self) -> Dict[str, Any]:
        """Load policies configuration from policies.yaml."""
        try:
            policies_path = "configs/policies.yaml"
            with open(policies_path, 'r', encoding='utf-8') as f:
                policies_config = yaml.safe_load(f)
            logger.info("Policies configuration loaded", policies_path=policies_path)
            return policies_config
        except Exception as e:
            logger.warning("Failed to load policies configuration, using defaults", error=str(e))
            return self._get_default_policies()
    
    def _get_default_policies(self) -> Dict[str, Any]:
        """Get default policies configuration."""
        return {
            "hitl": {
                "enabled": True,
                "approval_threshold": 0.7,
                "rejection_threshold": 0.4,
                "timeout_seconds": 300,
                "require_human_review": [
                    "trust_score < 0.6",
                    "confidence_score < 0.5",
                    "contains_sensitive_content",
                    "high_risk_task"
                ],
                "auto_approve": [
                    "trust_score >= 0.8",
                    "confidence_score >= 0.7"
                ]
            },
            "agent_lifecycle": {
                "prune_agents": ["trust_score < 0.3"],
                "fork_agents": ["trust_score > 0.8"],
                "mutate_agents": ["trust_score < 0.6"]
            },
            "trust": {
                "min_score": 0.0,
                "max_score": 1.0,
                "decay": {"enabled": True, "rate": 0.01, "max_decay": 0.3}
            },
            "resource_limits": {
                "max_concurrent_agents": 10,
                "max_execution_time": 300,
                "max_total_iterations": 100
            },
            "security": {
                "content_filtering": {"enabled": True, "block_categories": []},
                "access_control": {"require_authentication": False}
            },
            "performance": {
                "alerts": {
                    "low_trust_score": 0.3,
                    "high_failure_rate": 0.5,
                    "resource_exhaustion": 0.8
                }
            }
        }
    
    def _setup_dependencies(self):
        """Set up dependency injection for all components."""
        # Set HITL manager dependencies
        self.hitl_manager.set_dependencies(
            policy_engine=self.policy_engine,
            trust_tracker=self.trust_tracker,
            audit_log=self.audit_log
        )
        
        # Set evaluation engine dependencies
        self.evaluation_engine.set_memory_store(self.memory_manager)
        
        # Set trust manager dependencies
        self.trust_manager.set_trust_tracker(self.trust_tracker)
        
        # Set audit manager dependencies
        self.audit_manager.set_audit_log(self.audit_log)
        
        logger.info("Component dependencies configured")
    
    async def start_task(self, task: str, session_id: str = None) -> str:
        """Start a new task with the orchestrator."""
        try:
            if self.state.status == OrchestratorStatus.RUNNING:
                raise RuntimeError("Orchestrator is already running")
            
            # Initialize session
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Update state
            self.state.session_id = session_id
            self.state.current_task = task
            self.state.current_generation = 0
            self.state.generation_results = []
            self.state.start_time = datetime.now()
            self.state.status = OrchestratorStatus.RUNNING
            
            # Initialize components
            await self.audit_manager.start_session(session_id)
            await self.agent_manager.initialize_population()
            await self.hitl_manager.initialize()
            
            # Log task start
            await self.audit_manager.log_task_start(task, session_id)
            
            logger.info("Task started", task=task, session_id=session_id)
            return session_id
            
        except Exception as e:
            logger.error("Failed to start task", error=str(e))
            self.state.status = OrchestratorStatus.ERROR
            raise
    
    async def run_generation(self) -> GenerationResult:
        """Run a single generation of the GA loop."""
        try:
            if self.state.status != OrchestratorStatus.RUNNING:
                raise RuntimeError("Orchestrator is not running")
            
            generation_start = datetime.now()
            self.state.current_generation += 1
            
            logger.info("Starting generation", generation=self.state.current_generation)
            
            # Step 1: Generate solutions using agents
            prover_results = await self.agent_manager.execute_provers(self.state.current_task)
            
            # Step 2: Evaluate solutions
            evaluation_results = await self.evaluation_engine.evaluate_solutions(
                prover_results, self.state.current_task, self.state.current_generation
            )
            
            # Step 3: Check for HITL intervention
            interventions = await self.hitl_manager.check_interventions(
                evaluation_results, self.state.current_generation
            )
            
            # Step 4: Update population
            population_changes = await self.population_manager.update_population(
                evaluation_results, interventions, self.state.current_generation
            )
            
            # Step 5: Update trust scores
            trust_scores = await self.trust_manager.update_trust_scores(
                evaluation_results, self.state.current_generation
            )
            
            # Step 6: Store in memory
            await self.memory_manager.store_generation_data(
                prover_results, evaluation_results, self.state.current_generation
            )
            
            # Calculate generation statistics
            best_solution = self.evaluation_engine.find_best_solution(evaluation_results)
            best_score = best_solution.get("overall_score", 0.0) if best_solution else 0.0
            average_score = self.evaluation_engine.calculate_average_score(evaluation_results)
            execution_time = (datetime.now() - generation_start).total_seconds()
            
            # Create generation result
            generation_result = GenerationResult(
                generation=self.state.current_generation,
                task=self.state.current_task,
                agents=await self.agent_manager.get_agent_info(),
                best_solution=best_solution,
                best_score=best_score,
                average_score=average_score,
                execution_time=execution_time,
                interventions=interventions,
                trust_scores=trust_scores
            )
            
            self.state.generation_results.append(generation_result)
            
            # Log generation completion
            await self.audit_manager.log_generation_completion(
                self.state.current_generation, best_score, average_score, execution_time
            )
            
            logger.info(
                "Generation completed",
                generation=self.state.current_generation,
                best_score=best_score,
                average_score=average_score,
                execution_time=execution_time
            )
            
            return generation_result
            
        except Exception as e:
            logger.error("Generation failed", generation=self.state.current_generation, error=str(e))
            self.state.status = OrchestratorStatus.ERROR
            raise
    
    async def run_complete_task(self, task: str) -> List[GenerationResult]:
        """Run a complete task from start to finish."""
        try:
            session_id = await self.start_task(task)
            
            results = []
            while self.should_continue() and self.state.status == OrchestratorStatus.RUNNING:
                result = await self.run_generation()
                results.append(result)
                
                # Check if we should pause
                if self.state.status == OrchestratorStatus.PAUSED:
                    break
            
            # Finalize task
            await self.finalize_task()
            
            return results
            
        except Exception as e:
            logger.error("Failed to run complete task", error=str(e))
            self.state.status = OrchestratorStatus.ERROR
            raise
    
    def should_continue(self) -> bool:
        """Check if the orchestrator should continue running."""
        try:
            # Check max generations
            if self.state.current_generation >= self.config.max_generations:
                logger.info("Max generations reached")
                return False
            
            # Check max execution time
            if self.state.start_time:
                elapsed_time = (datetime.now() - self.state.start_time).total_seconds()
                if elapsed_time >= self.config.max_execution_time:
                    logger.info("Max execution time reached")
                    return False
            
            # Check convergence
            if len(self.state.generation_results) >= 5:
                recent_scores = [result.best_score for result in self.state.generation_results[-5:]]
                if max(recent_scores) - min(recent_scores) < 0.05:
                    logger.info("Convergence reached")
                    return False
            
            # Check if we have a good enough solution
            if self.state.generation_results:
                best_score = max(result.best_score for result in self.state.generation_results)
                if best_score >= self.config.convergence_threshold:
                    logger.info("Solution quality threshold reached")
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Failed to check continuation criteria", error=str(e))
            return False
    
    async def finalize_task(self):
        """Finalize the current task."""
        try:
            self.state.status = OrchestratorStatus.COMPLETED
            
            # Calculate execution time
            execution_time = 0
            if self.state.start_time:
                execution_time = (datetime.now() - self.state.start_time).total_seconds()
            
            # Log task completion
            await self.audit_manager.log_task_completion(
                self.state.current_task,
                self.state.current_generation,
                execution_time
            )
            
            # End audit session
            await self.audit_manager.end_session()
            
            logger.info("Task finalized", task=self.state.current_task, 
                       generations=self.state.current_generation, 
                       execution_time=execution_time)
            
        except Exception as e:
            logger.error("Failed to finalize task", error=str(e))
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the current status of the orchestrator."""
        return {
            "status": self.state.status.value,
            "current_task": self.state.current_task,
            "current_generation": self.state.current_generation,
            "total_agents": len(await self.agent_manager.get_agents()),
            "start_time": self.state.start_time.isoformat() if self.state.start_time else None,
            "generation_count": len(self.state.generation_results),
            "best_score": max(result.best_score for result in self.state.generation_results) if self.state.generation_results else 0.0,
            "average_score": sum(result.average_score for result in self.state.generation_results) / len(self.state.generation_results) if self.state.generation_results else 0.0
        }
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all generation results."""
        return [result.to_dict() for result in self.state.generation_results]
    
    def pause(self):
        """Pause the orchestrator."""
        if self.state.status == OrchestratorStatus.RUNNING:
            self.state.status = OrchestratorStatus.PAUSED
            logger.info("Orchestrator paused")
    
    def resume(self):
        """Resume the orchestrator."""
        if self.state.status == OrchestratorStatus.PAUSED:
            self.state.status = OrchestratorStatus.RUNNING
            logger.info("Orchestrator resumed")
    
    def stop(self):
        """Stop the orchestrator."""
        self.state.status = OrchestratorStatus.IDLE
        logger.info("Orchestrator stopped")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the orchestrator."""
        try:
            # Get statistics from all components
            trust_stats = await self.trust_manager.get_statistics()
            audit_stats = await self.audit_manager.get_statistics()
            memory_stats = await self.memory_manager.get_statistics()
            agent_stats = await self.agent_manager.get_statistics()
            
            # Calculate orchestrator-specific statistics
            total_execution_time = 0
            if self.state.start_time and self.state.status == OrchestratorStatus.COMPLETED:
                total_execution_time = (datetime.now() - self.state.start_time).total_seconds()
            
            return {
                "orchestrator": {
                    "status": self.state.status.value,
                    "current_generation": self.state.current_generation,
                    "total_generations": len(self.state.generation_results),
                    "total_agents": len(await self.agent_manager.get_agents()),
                    "total_execution_time": total_execution_time,
                    "hitl_enabled": self.config.hitl_enabled
                },
                "trust": trust_stats,
                "audit": audit_stats,
                "memory": memory_stats,
                "agents": agent_stats,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get statistics", error=str(e))
            return {"error": str(e)} 

# Backward compatibility alias
Orchestrator = SophieReflexOrchestrator 