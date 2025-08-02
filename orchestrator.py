from typing import Dict, Any, List, Optional, Tuple
import asyncio
import structlog
import yaml
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import random
import uuid

from agents.base_agent import AgentConfig, AgentStatus
from agents.prover import ProverAgent
from agents.evaluator import EvaluatorAgent
from agents.refiner import RefinerAgent
from memory.vector_store import VectorStore, VectorBackend, MemoryEntry
from memory.trust_tracker import TrustTracker, TrustEventType
from governance.policy_engine import PolicyEngine, PolicyContext, PolicyDecision
from governance.audit_log import AuditLog, AuditEventType
from ui.webhook_server import WebhookServer, PlanApprovalRequest

logger = structlog.get_logger()


class OrchestratorStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class GenerationResult:
    """Result of a single generation in the GA loop."""
    generation: int
    task: str
    agents: List[Any]
    best_solution: Optional[Dict[str, Any]]
    best_score: float
    average_score: float
    execution_time: float
    interventions: List[Dict[str, Any]]
    trust_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


class SophieReflexOrchestrator:
    """Core GA loop logic: task -> prove -> evaluate -> refine -> prune/fork."""
    
    def __init__(self, config_path: str = "configs/system.yaml"):
        self.config_path = config_path
        self.status = OrchestratorStatus.IDLE
        self.current_task = None
        self.current_generation = 0
        self.agents = []
        self.generation_results = []
        self.start_time = None
        
        # Load configurations
        self.system_config = self._load_config(config_path)
        self.agents_config = self._load_config("configs/agents.yaml")
        self.rubric_config = self._load_config("configs/rubric.yaml")
        self.policies_config = self._load_config("configs/policies.yaml")
        
        # Initialize orchestrator config
        self.orchestrator_config = OrchestratorConfig(**self.system_config.get("genetic_algorithm", {}))
        
        # Initialize components
        self.vector_store = VectorStore(
            backend=VectorBackend(self.system_config.get("memory", {}).get("backend", "chromadb")),
            config=self.system_config.get("memory", {})
        )
        
        self.trust_tracker = TrustTracker()
        self.policy_engine = PolicyEngine(self.policies_config)
        self.audit_log = AuditLog()
        
        # Initialize HITL server if enabled
        self.hitl_server = None
        if self.orchestrator_config.hitl_enabled:
            self.hitl_server = WebhookServer()
        
        logger.info("Sophie Reflex Orchestrator initialized", config_path=config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load configuration", config_path=config_path, error=str(e))
            raise
    
    async def start_task(self, task: str, session_id: str = None) -> str:
        """Start a new task with the orchestrator."""
        try:
            if self.status == OrchestratorStatus.RUNNING:
                raise RuntimeError("Orchestrator is already running")
            
            # Start audit session
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            self.audit_log.start_session(session_id)
            
            # Set up task
            self.current_task = task
            self.current_generation = 0
            self.generation_results = []
            self.start_time = datetime.now()
            self.status = OrchestratorStatus.RUNNING
            
            # Log task start
            self.audit_log.log_event(
                event_type=AuditEventType.TASK_SUBMITTED,
                description=f"Task submitted: {task}",
                details={"task": task, "session_id": session_id},
                severity="info"
            )
            
            # Initialize agent population
            await self._initialize_population()
            
            logger.info("Task started", task=task, session_id=session_id)
            
            # Start HITL server if enabled
            if self.hitl_server:
                asyncio.create_task(self.hitl_server.run_async())
            
            return session_id
            
        except Exception as e:
            logger.error("Failed to start task", error=str(e))
            self.status = OrchestratorStatus.ERROR
            raise
    
    async def _initialize_population(self):
        """Initialize the agent population."""
        try:
            self.agents = []
            
            # Create initial provers from configuration
            prover_configs = self.agents_config.get("provers", [])
            
            for config_data in prover_configs:
                for i in range(self.orchestrator_config.population_size):
                    agent_config = AgentConfig(
                        name=f"{config_data['name']}_{i}",
                        prompt=config_data['prompt'],
                        model=config_data['model'],
                        temperature=config_data['temperature'],
                        max_tokens=config_data['max_tokens'],
                        timeout=config_data.get('timeout', 30),
                        max_retries=config_data.get('max_retries', 3),
                        retry_delay=config_data.get('retry_delay', 1),
                        hyperparameters=config_data.get('hyperparameters', {})
                    )
                    
                    agent = ProverAgent(config=agent_config)
                    self.agents.append(agent)
                    
                    logger.info("Agent created", agent_id=agent.agent_id, name=agent.config.name)
            
            # Create evaluators
            evaluator_configs = self.agents_config.get("evaluators", [])
            self.evaluators = []
            
            for config_data in evaluator_configs:
                agent_config = AgentConfig(
                    name=config_data['name'],
                    prompt=config_data['prompt'],
                    model=config_data['model'],
                    temperature=config_data['temperature'],
                    max_tokens=config_data['max_tokens'],
                    hyperparameters=config_data.get('hyperparameters', {})
                )
                
                evaluator = EvaluatorAgent(
                    config=agent_config,
                    rubric_config=self.rubric_config
                )
                self.evaluators.append(evaluator)
            
            # Create refiners
            refiner_configs = self.agents_config.get("refiners", [])
            self.refiners = []
            
            for config_data in refiner_configs:
                agent_config = AgentConfig(
                    name=config_data['name'],
                    prompt=config_data['prompt'],
                    model=config_data['model'],
                    temperature=config_data['temperature'],
                    max_tokens=config_data['max_tokens'],
                    hyperparameters=config_data.get('hyperparameters', {})
                )
                
                refiner = RefinerAgent(config=agent_config)
                self.refiners.append(refiner)
            
            logger.info("Population initialized", agents=len(self.agents), 
                        evaluators=len(self.evaluators), refiners=len(self.refiners))
            
        except Exception as e:
            logger.error("Failed to initialize population", error=str(e))
            raise
    
    async def run_generation(self) -> GenerationResult:
        """Run a single generation of the GA loop."""
        try:
            if self.status != OrchestratorStatus.RUNNING:
                raise RuntimeError("Orchestrator is not running")
            
            generation_start = datetime.now()
            self.current_generation += 1
            
            logger.info("Starting generation", generation=self.current_generation)
            
            # Step 1: Prove - Generate solutions
            prover_results = await self._prove_step()
            
            # Step 2: Evaluate - Score solutions
            evaluation_results = await self._evaluate_step(prover_results)
            
            # Step 3: Check for HITL intervention
            interventions = await self._check_hitl_intervention(evaluation_results)
            
            # Step 4: Refine - Improve population
            refinement_result = await self._refine_step(evaluation_results, interventions)
            
            # Step 5: Update population
            await self._update_population(refinement_result)
            
            # Step 6: Track trust scores
            trust_scores = await self._update_trust_scores(evaluation_results)
            
            # Calculate generation statistics
            best_solution = self._find_best_solution(evaluation_results)
            best_score = best_solution.get("overall_score", 0.0) if best_solution else 0.0
            average_score = self._calculate_average_score(evaluation_results)
            execution_time = (datetime.now() - generation_start).total_seconds()
            
            # Create generation result
            generation_result = GenerationResult(
                generation=self.current_generation,
                task=self.current_task,
                agents=[agent.get_info() for agent in self.agents],
                best_solution=best_solution,
                best_score=best_score,
                average_score=average_score,
                execution_time=execution_time,
                interventions=interventions,
                trust_scores=trust_scores
            )
            
            self.generation_results.append(generation_result)
            
            # Log generation completion
            self.audit_log.log_event(
                event_type=AuditEventType.GENERATION_COMPLETED,
                description=f"Generation {self.current_generation} completed",
                details={
                    "generation": self.current_generation,
                    "best_score": best_score,
                    "average_score": average_score,
                    "execution_time": execution_time,
                    "interventions": len(interventions)
                },
                severity="info"
            )
            
            logger.info(
                "Generation completed",
                generation=self.current_generation,
                best_score=best_score,
                average_score=average_score,
                execution_time=execution_time
            )
            
            return generation_result
            
        except Exception as e:
            logger.error("Generation failed", generation=self.current_generation, error=str(e))
            self.status = OrchestratorStatus.ERROR
            raise
    
    async def _prove_step(self) -> List[Dict[str, Any]]:
        """Step 1: Generate solutions using provers."""
        try:
            logger.info("Starting prove step")
            
            prover_results = []
            
            # Execute all provers concurrently
            tasks = []
            for agent in self.agents:
                if isinstance(agent, ProverAgent):
                    task = agent.execute_with_retry(self.current_task)
                    tasks.append(task)
            
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
                    
                    # Store in memory
                    memory_entry = MemoryEntry(
                        id=f"prover_{result.agent_id}_{self.current_generation}",
                        task=self.current_task,
                        content=result.result.get("best_variant", {}).get("content", ""),
                        embedding=[],  # Will be generated by vector store
                        metadata={
                            "agent_id": result.agent_id,
                            "generation": self.current_generation,
                            "confidence_score": result.confidence_score,
                            "execution_time": result.execution_time
                        },
                        timestamp=datetime.now(),
                        agent_id=result.agent_id,
                        score=result.confidence_score,
                        tags=["prover", f"generation_{self.current_generation}"]
                    )
                    
                    await self.vector_store.add_entry(memory_entry)
            
            logger.info("Prove step completed", results=len(prover_results))
            return prover_results
            
        except Exception as e:
            logger.error("Prove step failed", error=str(e))
            raise
    
    async def _evaluate_step(self, prover_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 2: Score solutions using evaluators."""
        try:
            logger.info("Starting evaluate step")
            
            evaluation_results = []
            
            # Evaluate each prover result
            for prover_result in prover_results:
                # Use all evaluators for comprehensive evaluation
                evaluator_tasks = []
                for evaluator in self.evaluators:
                    context = {
                        "prover_output": prover_result["result"],
                        "original_task": self.current_task,
                        "generation": self.current_generation
                    }
                    task = evaluator.execute_with_retry(self.current_task, context)
                    evaluator_tasks.append(task)
                
                # Wait for all evaluators to complete
                evaluator_results = await asyncio.gather(*evaluator_tasks, return_exceptions=True)
                
                # Process evaluator results
                valid_evaluations = []
                for i, result in enumerate(evaluator_results):
                    if isinstance(result, Exception):
                        logger.error("Evaluator execution failed", 
                                   evaluator_id=self.evaluators[i].agent_id, error=str(result))
                        continue
                    
                    if result.status == AgentStatus.COMPLETED:
                        valid_evaluations.append(result.result)
                
                if valid_evaluations:
                    # Aggregate evaluation results
                    aggregated_result = self._aggregate_evaluations(valid_evaluations)
                    aggregated_result["prover_result"] = prover_result
                    
                    evaluation_results.append(aggregated_result)
                    
                    # Store in memory
                    memory_entry = MemoryEntry(
                        id=f"eval_{prover_result['agent_id']}_{self.current_generation}",
                        task=self.current_task,
                        content=json.dumps(aggregated_result),
                        embedding=[],
                        metadata={
                            "agent_id": prover_result["agent_id"],
                            "generation": self.current_generation,
                            "overall_score": aggregated_result.get("overall_score", 0.0),
                            "evaluator_count": len(valid_evaluations)
                        },
                        timestamp=datetime.now(),
                        agent_id=prover_result["agent_id"],
                        score=aggregated_result.get("overall_score", 0.0),
                        tags=["evaluation", f"generation_{self.current_generation}"]
                    )
                    
                    await self.vector_store.add_entry(memory_entry)
            
            logger.info("Evaluate step completed", evaluations=len(evaluation_results))
            return evaluation_results
            
        except Exception as e:
            logger.error("Evaluate step failed", error=str(e))
            raise
    
    def _aggregate_evaluations(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple evaluation results."""
        try:
            if not evaluations:
                return {"overall_score": 0.0, "error": "No valid evaluations"}
            
            # Calculate average score
            total_score = sum(eval_result.get("overall_score", 0.0) for eval_result in evaluations)
            average_score = total_score / len(evaluations)
            
            # Aggregate category scores
            all_categories = set()
            for eval_result in evaluations:
                all_categories.update(eval_result.get("category_scores", {}).keys())
            
            aggregated_categories = {}
            for category in all_categories:
                category_scores = []
                for eval_result in evaluations:
                    category_data = eval_result.get("category_scores", {}).get(category, {})
                    category_scores.append(category_data.get("score", 0.0))
                
                if category_scores:
                    aggregated_categories[category] = {
                        "score": sum(category_scores) / len(category_scores),
                        "weight": evaluations[0].get("category_scores", {}).get(category, {}).get("weight", 0.2)
                    }
            
            return {
                "overall_score": average_score,
                "category_scores": aggregated_categories,
                "evaluation_count": len(evaluations),
                "individual_evaluations": evaluations
            }
            
        except Exception as e:
            logger.error("Failed to aggregate evaluations", error=str(e))
            return {"overall_score": 0.0, "error": str(e)}
    
    async def _check_hitl_intervention(self, evaluation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 3: Check for HITL intervention."""
        try:
            if not self.orchestrator_config.hitl_enabled:
                return []
            
            logger.info("Checking HITL intervention")
            
            interventions = []
            
            for evaluation in evaluation_results:
                agent_id = evaluation.get("prover_result", {}).get("agent_id")
                overall_score = evaluation.get("overall_score", 0.0)
                
                # Create policy context
                context = PolicyContext(
                    agent_id=agent_id,
                    agent_type="prover",
                    action="generate_solution",
                    content=evaluation.get("prover_result", {}).get("result", {}).get("best_variant", {}).get("content", ""),
                    trust_score=await self._get_agent_trust_score(agent_id),
                    confidence_score=evaluation.get("prover_result", {}).get("confidence_score", 0.0),
                    iteration_count=self.current_generation,
                    timestamp=datetime.now(),
                    additional_context={
                        "overall_score": overall_score,
                        "generation": self.current_generation
                    }
                )
                
                # Evaluate policies
                policy_result = await self.policy_engine.evaluate_action(context)
                
                if policy_result.decision == PolicyDecision.REQUIRE_HUMAN_REVIEW:
                    # Submit for HITL review
                    hitl_result = await self._submit_for_hitl_review(evaluation, policy_result)
                    
                    if hitl_result:
                        interventions.append({
                            "agent_id": agent_id,
                            "type": "hitl_review",
                            "policy_result": policy_result.__dict__,
                            "hitl_result": hitl_result,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Apply HITL decision
                        await self._apply_hitl_decision(agent_id, hitl_result)
                
                elif policy_result.decision == PolicyDecision.BLOCK:
                    interventions.append({
                        "agent_id": agent_id,
                        "type": "policy_block",
                        "policy_result": policy_result.__dict__,
                        "timestamp": datetime.now().isoformat()
                    })
            
            logger.info("HITL intervention check completed", interventions=len(interventions))
            return interventions
            
        except Exception as e:
            logger.error("HITL intervention check failed", error=str(e))
            return []
    
    async def _submit_for_hitl_review(self, evaluation: Dict[str, Any], policy_result: Any) -> Optional[Dict[str, Any]]:
        """Submit a solution for HITL review."""
        try:
            if not self.hitl_server:
                return None
            
            prover_result = evaluation.get("prover_result", {})
            best_variant = prover_result.get("result", {}).get("best_variant", {})
            
            # Create approval request
            approval_request = PlanApprovalRequest(
                plan_id=f"plan_{prover_result['agent_id']}_{self.current_generation}",
                task_id=self.current_task,
                agent_id=prover_result["agent_id"],
                plan_content=best_variant.get("content", ""),
                trust_score=await self._get_agent_trust_score(prover_result["agent_id"]),
                confidence_score=prover_result.get("confidence_score", 0.0),
                evaluation_score=evaluation.get("overall_score", 0.0),
                metadata={
                    "generation": self.current_generation,
                    "policy_result": policy_result.__dict__
                }
            )
            
            # Submit for review
            success = await self.hitl_server.submit_plan_for_review(approval_request)
            
            if success:
                # Wait for decision
                decision = await self.hitl_server.wait_for_decision(
                    approval_request.plan_id,
                    timeout=self.orchestrator_config.hitl_timeout
                )
                
                if decision:
                    return decision.__dict__
            
            return None
            
        except Exception as e:
            logger.error("Failed to submit for HITL review", error=str(e))
            return None
    
    async def _apply_hitl_decision(self, agent_id: str, hitl_result: Dict[str, Any]):
        """Apply HITL decision to agent trust score."""
        try:
            approved = hitl_result.get("approved", False)
            reason = hitl_result.get("reason", "")
            
            if approved:
                # Positive trust adjustment
                await self.trust_tracker.record_event(
                    agent_id=agent_id,
                    event_type=TrustEventType.HUMAN_APPROVAL,
                    adjustment=0.15,
                    context={"reason": reason, "generation": self.current_generation},
                    description=f"Human approval: {reason}"
                )
            else:
                # Negative trust adjustment
                await self.trust_tracker.record_event(
                    agent_id=agent_id,
                    event_type=TrustEventType.HUMAN_REJECTION,
                    adjustment=-0.25,
                    context={"reason": reason, "generation": self.current_generation},
                    description=f"Human rejection: {reason}"
                )
            
            # Log HITL decision
            self.audit_log.log_event(
                event_type=AuditEventType.HUMAN_INTERVENTION,
                description=f"HITL decision for agent {agent_id}: {'Approved' if approved else 'Rejected'}",
                details={
                    "agent_id": agent_id,
                    "approved": approved,
                    "reason": reason,
                    "generation": self.current_generation
                },
                severity="info"
            )
            
        except Exception as e:
            logger.error("Failed to apply HITL decision", agent_id=agent_id, error=str(e))
    
    async def _refine_step(self, evaluation_results: List[Dict[str, Any]], 
                          interventions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 4: Improve population using refiners."""
        try:
            logger.info("Starting refine step")
            
            if not self.refiners:
                logger.warning("No refiner agents found")
                return {"refinements": [], "new_agents": []}
            
            # Use the first refiner (could be enhanced to use multiple)
            refiner = self.refiners[0]
            
            # Prepare context for refiner
            context = {
                "evaluation_results": evaluation_results,
                "current_agents": self.agents,
                "interventions": interventions,
                "generation_info": {
                    "generation": self.current_generation,
                    "population_size": len(self.agents),
                    "best_score": max(eval_result.get("overall_score", 0.0) for eval_result in evaluation_results) if evaluation_results else 0.0
                }
            }
            
            # Execute refiner
            refiner_result = await refiner.execute_with_retry(self.current_task, context)
            
            if refiner_result.status == AgentStatus.COMPLETED:
                result_data = refiner_result.result
                
                # Process refinements
                new_agents = []
                
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
                
                logger.info("Refine step completed", new_agents=len(new_agents))
                return {
                    "refinements": result_data.get("refinement_results", {}),
                    "new_agents": new_agents
                }
            else:
                logger.warning("Refiner execution failed")
                return {"refinements": [], "new_agents": []}
            
        except Exception as e:
            logger.error("Refine step failed", error=str(e))
            return {"refinements": [], "new_agents": []}
    
    async def _update_population(self, refinement_result: Dict[str, Any]):
        """Step 5: Update agent population based on refinement results."""
        try:
            # Add new agents
            new_agents = refinement_result.get("new_agents", [])
            self.agents.extend(new_agents)
            
            # Remove pruned agents
            pruned_agents = refinement_result.get("refinements", {}).get("pruned_agents", [])
            for pruned_info in pruned_agents:
                agent_id = pruned_info.get("agent_id")
                self.agents = [agent for agent in self.agents if agent.agent_id != agent_id]
                
                # Log agent pruning
                self.audit_log.log_event(
                    event_type=AuditEventType.AGENT_PRUNED,
                    description=f"Agent {agent_id} pruned",
                    details={
                        "agent_id": agent_id,
                        "reason": pruned_info.get("reason", "Unknown"),
                        "generation": self.current_generation
                    },
                    severity="info"
                )
            
            # Limit population size
            if len(self.agents) > self.orchestrator_config.max_agents:
                # Keep best performing agents
                self.agents = self.agents[:self.orchestrator_config.max_agents]
            
            logger.info("Population updated", total_agents=len(self.agents), 
                       new_agents=len(new_agents), pruned_agents=len(pruned_agents))
            
        except Exception as e:
            logger.error("Failed to update population", error=str(e))
    
    async def _update_trust_scores(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Step 6: Update trust scores based on evaluation results."""
        try:
            trust_scores = {}
            
            for evaluation in evaluation_results:
                agent_id = evaluation.get("prover_result", {}).get("agent_id")
                overall_score = evaluation.get("overall_score", 0.0)
                
                if agent_id:
                    # Get current trust score
                    trust_score_data = await self.trust_tracker.get_trust_score(agent_id)
                    current_trust = trust_score_data.score if trust_score_data else 0.5
                    
                    # Determine trust adjustment based on performance
                    if overall_score >= 0.8:
                        adjustment = 0.1
                        event_type = TrustEventType.HIGH_QUALITY_OUTPUT
                    elif overall_score >= 0.6:
                        adjustment = 0.05
                        event_type = TrustEventType.EXECUTION_SUCCESS
                    elif overall_score >= 0.4:
                        adjustment = 0.0
                        event_type = TrustEventType.EXECUTION_SUCCESS
                    else:
                        adjustment = -0.1
                        event_type = TrustEventType.LOW_QUALITY_OUTPUT
                    
                    # Record trust event
                    await self.trust_tracker.record_event(
                        agent_id=agent_id,
                        event_type=event_type,
                        adjustment=adjustment,
                        context={
                            "performance_score": overall_score,
                            "generation": self.current_generation,
                            "evaluation_result": evaluation
                        },
                        description=f"Performance-based adjustment: {adjustment:+.2f}"
                    )
                    
                    # Get updated trust score
                    updated_trust_data = await self.trust_tracker.get_trust_score(agent_id)
                    trust_scores[agent_id] = updated_trust_data.score if updated_trust_data else current_trust
            
            logger.info("Trust scores updated", agents_updated=len(trust_scores))
            return trust_scores
            
        except Exception as e:
            logger.error("Failed to update trust scores", error=str(e))
            return {}
    
    async def _get_agent_trust_score(self, agent_id: str) -> float:
        """Get the current trust score for an agent."""
        try:
            trust_data = await self.trust_tracker.get_trust_score(agent_id)
            return trust_data.score if trust_data else 0.5
        except Exception as e:
            logger.error("Failed to get agent trust score", agent_id=agent_id, error=str(e))
            return 0.5
    
    def _find_best_solution(self, evaluation_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best solution from evaluation results."""
        try:
            if not evaluation_results:
                return None
            
            best_result = max(evaluation_results, key=lambda x: x.get("overall_score", 0.0))
            return best_result
            
        except Exception as e:
            logger.error("Failed to find best solution", error=str(e))
            return None
    
    def _calculate_average_score(self, evaluation_results: List[Dict[str, Any]]) -> float:
        """Calculate the average score from evaluation results."""
        try:
            if not evaluation_results:
                return 0.0
            
            total_score = sum(result.get("overall_score", 0.0) for result in evaluation_results)
            return total_score / len(evaluation_results)
            
        except Exception as e:
            logger.error("Failed to calculate average score", error=str(e))
            return 0.0
    
    def should_continue(self) -> bool:
        """Check if the orchestrator should continue running."""
        try:
            # Check max generations
            if self.current_generation >= self.orchestrator_config.max_generations:
                logger.info("Max generations reached")
                return False
            
            # Check max execution time
            if self.start_time:
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
                if elapsed_time >= self.orchestrator_config.max_execution_time:
                    logger.info("Max execution time reached")
                    return False
            
            # Check convergence
            if len(self.generation_results) >= 5:
                recent_scores = [result.best_score for result in self.generation_results[-5:]]
                if max(recent_scores) - min(recent_scores) < 0.05:
                    logger.info("Convergence reached")
                    return False
            
            # Check if we have a good enough solution
            if self.generation_results:
                best_score = max(result.best_score for result in self.generation_results)
                if best_score >= self.orchestrator_config.convergence_threshold:
                    logger.info("Solution quality threshold reached")
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Failed to check continuation criteria", error=str(e))
            return False
    
    async def run_complete_task(self, task: str) -> List[GenerationResult]:
        """Run a complete task from start to finish."""
        try:
            session_id = await self.start_task(task)
            
            results = []
            while self.should_continue() and self.status == OrchestratorStatus.RUNNING:
                result = await self.run_generation()
                results.append(result)
                
                # Check if we should pause
                if self.status == OrchestratorStatus.PAUSED:
                    break
            
            # Finalize task
            await self.finalize_task()
            
            return results
            
        except Exception as e:
            logger.error("Failed to run complete task", error=str(e))
            self.status = OrchestratorStatus.ERROR
            raise
    
    async def finalize_task(self):
        """Finalize the current task."""
        try:
            self.status = OrchestratorStatus.COMPLETED
            
            # Log task completion
            if self.start_time:
                execution_time = (datetime.now() - self.start_time).total_seconds()
            else:
                execution_time = 0
            
            self.audit_log.log_event(
                event_type=AuditEventType.SYSTEM_ERROR,
                description=f"Task completed: {self.current_task}",
                details={
                    "task": self.current_task,
                    "total_generations": self.current_generation,
                    "execution_time": execution_time,
                    "final_status": self.status.value
                },
                severity="info"
            )
            
            # End audit session
            self.audit_log.end_session()
            
            logger.info("Task finalized", task=self.current_task, 
                       generations=self.current_generation, 
                       execution_time=execution_time)
            
        except Exception as e:
            logger.error("Failed to finalize task", error=str(e))
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the orchestrator."""
        return {
            "status": self.status.value,
            "current_task": self.current_task,
            "current_generation": self.current_generation,
            "total_agents": len(self.agents),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "generation_count": len(self.generation_results),
            "best_score": max(result.best_score for result in self.generation_results) if self.generation_results else 0.0,
            "average_score": sum(result.average_score for result in self.generation_results) / len(self.generation_results) if self.generation_results else 0.0
        }
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all generation results."""
        return [result.to_dict() for result in self.generation_results]
    
    def pause(self):
        """Pause the orchestrator."""
        if self.status == OrchestratorStatus.RUNNING:
            self.status = OrchestratorStatus.PAUSED
            logger.info("Orchestrator paused")
    
    def resume(self):
        """Resume the orchestrator."""
        if self.status == OrchestratorStatus.PAUSED:
            self.status = OrchestratorStatus.RUNNING
            logger.info("Orchestrator resumed")
    
    def stop(self):
        """Stop the orchestrator."""
        self.status = OrchestratorStatus.IDLE
        logger.info("Orchestrator stopped")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the orchestrator."""
        try:
            # Get trust statistics
            trust_stats = await self.trust_tracker.get_trust_statistics()
            
            # Get audit statistics
            audit_stats = await self.audit_log.get_audit_statistics()
            
            # Get vector store statistics
            vector_stats = self.vector_store.get_stats()
            
            # Get policy engine statistics
            policy_stats = self.policy_engine.get_policy_stats()
            
            # Calculate orchestrator-specific statistics
            total_execution_time = 0
            if self.start_time and self.status == OrchestratorStatus.COMPLETED:
                total_execution_time = (datetime.now() - self.start_time).total_seconds()
            
            return {
                "orchestrator": {
                    "status": self.status.value,
                    "current_generation": self.current_generation,
                    "total_generations": len(self.generation_results),
                    "total_agents": len(self.agents),
                    "total_execution_time": total_execution_time,
                    "hitl_enabled": self.orchestrator_config.hitl_enabled
                },
                "trust": trust_stats,
                "audit": audit_stats,
                "vector_store": vector_stats,
                "policy_engine": policy_stats,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get statistics", error=str(e))
            return {"error": str(e)}