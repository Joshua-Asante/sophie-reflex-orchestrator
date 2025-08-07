"""
Agent Factory - Dynamic agent creation and specialization system.
Creates optimized agents based on task requirements and performance history.
"""

from typing import Dict, Any, List, Optional, Type, Set
import asyncio
import structlog
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import random
import numpy as np

from .base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus
from .agent_registry import AgentCapability, AgentRegistry, get_agent_registry
from .prover import ProverAgent
from .evaluator import EvaluatorAgent
from .refiner import RefinerAgent

logger = structlog.get_logger()


class AgentArchetype(Enum):
    """Predefined agent archetypes with specific characteristics."""
    CREATIVE_INNOVATOR = "creative_innovator"
    ANALYTICAL_THINKER = "analytical_thinker"
    PRACTICAL_EXECUTOR = "practical_executor"
    QUALITY_GUARDIAN = "quality_guardian"
    EFFICIENCY_OPTIMIZER = "efficiency_optimizer"
    COLLABORATIVE_FACILITATOR = "collaborative_facilitator"
    DOMAIN_SPECIALIST = "domain_specialist"
    GENERALIST_ADAPTER = "generalist_adapter"


@dataclass
class AgentBlueprint:
    """Blueprint for creating specialized agents."""
    archetype: AgentArchetype
    base_config: AgentConfig
    specialization_parameters: Dict[str, Any]
    performance_targets: Dict[str, float]
    adaptation_rules: List[Dict[str, Any]]
    compatibility_matrix: Dict[str, float] = field(default_factory=dict)


@dataclass
class CreationRequest:
    """Request for creating a new agent."""
    task_domain: str
    required_capabilities: Set[AgentCapability]
    performance_requirements: Dict[str, float]
    resource_constraints: Dict[str, Any]
    collaboration_context: Optional[Dict[str, Any]] = None
    specialization_hints: List[str] = field(default_factory=list)


class AgentFactory:
    """
    Intelligent factory for creating optimized agents based on requirements.
    Uses machine learning and heuristics to design optimal agent configurations.
    """
    
    def __init__(self, agent_registry: Optional[AgentRegistry] = None):
        self.agent_registry = agent_registry or get_agent_registry()
        self.blueprints: Dict[AgentArchetype, AgentBlueprint] = {}
        self.creation_history: List[Dict[str, Any]] = []
        self.performance_database: Dict[str, List[float]] = {}
        
        # Initialize predefined blueprints
        self._initialize_blueprints()
        
        # Factory statistics
        self.factory_stats = {
            "agents_created": 0,
            "successful_creations": 0,
            "average_performance": 0.0,
            "specialization_success_rate": 0.0
        }
        
        logger.info("Agent factory initialized")
    
    def create_agent(self, request: CreationRequest) -> Optional[BaseAgent]:
        """Create an optimized agent based on the request."""
        try:
            # Analyze request and determine optimal archetype
            optimal_archetype = self._determine_optimal_archetype(request)
            
            # Get or create blueprint
            blueprint = self._get_or_create_blueprint(optimal_archetype, request)
            
            # Generate specialized configuration
            specialized_config = self._generate_specialized_config(blueprint, request)
            
            # Create agent instance
            agent = self._instantiate_agent(specialized_config, request)
            
            if agent:
                # Register creation in history
                self._record_creation(request, agent, optimal_archetype)
                
                # Update factory stats
                self.factory_stats["agents_created"] += 1
                
                logger.info("Agent created successfully", 
                           agent_id=agent.agent_id,
                           archetype=optimal_archetype.value,
                           capabilities=[c.value for c in request.required_capabilities])
                
                return agent
            
        except Exception as e:
            logger.error("Agent creation failed", 
                        task_domain=request.task_domain,
                        error=str(e))
        
        return None
    
    def create_agent_team(self, 
                         team_request: Dict[str, Any]) -> List[BaseAgent]:
        """Create a team of complementary agents for complex tasks."""
        team_size = team_request.get("team_size", 3)
        task_complexity = team_request.get("complexity", "medium")
        required_capabilities = set(team_request.get("capabilities", []))
        
        # Determine team composition
        team_composition = self._design_team_composition(
            required_capabilities, team_size, task_complexity
        )
        
        team_agents = []
        
        for role_spec in team_composition:
            request = CreationRequest(
                task_domain=team_request.get("domain", "general"),
                required_capabilities=role_spec["capabilities"],
                performance_requirements=role_spec["performance_targets"],
                resource_constraints=team_request.get("resource_constraints", {}),
                specialization_hints=role_spec["specialization_hints"]
            )
            
            agent = self.create_agent(request)
            if agent:
                team_agents.append(agent)
        
        logger.info("Agent team created", 
                   team_size=len(team_agents),
                   requested_size=team_size)
        
        return team_agents
    
    def evolve_agent(self, 
                    existing_agent: BaseAgent,
                    performance_feedback: Dict[str, Any]) -> BaseAgent:
        """Evolve an existing agent based on performance feedback."""
        try:
            # Analyze performance feedback
            evolution_strategy = self._analyze_evolution_needs(
                existing_agent, performance_feedback
            )
            
            # Create evolved configuration
            evolved_config = self._apply_evolution_strategy(
                existing_agent.config, evolution_strategy
            )
            
            # Create evolved agent
            evolved_agent = self._instantiate_agent_from_config(evolved_config)
            
            if evolved_agent:
                # Transfer learning from original agent
                self._transfer_learning(existing_agent, evolved_agent)
                
                logger.info("Agent evolved successfully",
                           original_id=existing_agent.agent_id,
                           evolved_id=evolved_agent.agent_id,
                           evolution_strategy=evolution_strategy["type"])
                
                return evolved_agent
            
        except Exception as e:
            logger.error("Agent evolution failed", 
                        agent_id=existing_agent.agent_id,
                        error=str(e))
        
        return existing_agent
    
    def _determine_optimal_archetype(self, request: CreationRequest) -> AgentArchetype:
        """Determine the optimal agent archetype for the request."""
        # Score each archetype based on request characteristics
        archetype_scores = {}
        
        for archetype in AgentArchetype:
            score = self._calculate_archetype_fitness(archetype, request)
            archetype_scores[archetype] = score
        
        # Return highest scoring archetype
        optimal_archetype = max(archetype_scores, key=archetype_scores.get)
        
        logger.debug("Archetype selection completed",
                    optimal_archetype=optimal_archetype.value,
                    scores=[(a.value, s) for a, s in archetype_scores.items()])
        
        return optimal_archetype
    
    def _calculate_archetype_fitness(self, 
                                   archetype: AgentArchetype,
                                   request: CreationRequest) -> float:
        """Calculate fitness score for an archetype given the request."""
        score = 0.0
        
        # Capability matching
        if archetype == AgentArchetype.CREATIVE_INNOVATOR:
            if AgentCapability.CREATIVE_WRITING in request.required_capabilities:
                score += 0.8
            if AgentCapability.PROBLEM_SOLVING in request.required_capabilities:
                score += 0.6
        
        elif archetype == AgentArchetype.ANALYTICAL_THINKER:
            if AgentCapability.DATA_ANALYSIS in request.required_capabilities:
                score += 0.9
            if AgentCapability.EVALUATION in request.required_capabilities:
                score += 0.7
        
        elif archetype == AgentArchetype.PRACTICAL_EXECUTOR:
            if AgentCapability.CODE_GENERATION in request.required_capabilities:
                score += 0.8
            if AgentCapability.OPTIMIZATION in request.required_capabilities:
                score += 0.7
        
        elif archetype == AgentArchetype.QUALITY_GUARDIAN:
            if AgentCapability.EVALUATION in request.required_capabilities:
                score += 0.9
            if AgentCapability.TESTING in request.required_capabilities:
                score += 0.8
        
        elif archetype == AgentArchetype.EFFICIENCY_OPTIMIZER:
            if AgentCapability.OPTIMIZATION in request.required_capabilities:
                score += 0.9
            if "efficiency" in request.specialization_hints:
                score += 0.3
        
        elif archetype == AgentArchetype.COLLABORATIVE_FACILITATOR:
            if request.collaboration_context:
                score += 0.7
            if AgentCapability.SYNTHESIS in request.required_capabilities:
                score += 0.6
        
        elif archetype == AgentArchetype.DOMAIN_SPECIALIST:
            # Check for domain-specific hints
            domain_keywords = ["medical", "legal", "financial", "technical", "scientific"]
            if any(keyword in request.task_domain.lower() for keyword in domain_keywords):
                score += 0.8
        
        elif archetype == AgentArchetype.GENERALIST_ADAPTER:
            # Bonus for diverse capability requirements
            if len(request.required_capabilities) > 3:
                score += 0.6
        
        # Performance requirements matching
        perf_requirements = request.performance_requirements
        if "accuracy" in perf_requirements and perf_requirements["accuracy"] > 0.8:
            if archetype in [AgentArchetype.ANALYTICAL_THINKER, AgentArchetype.QUALITY_GUARDIAN]:
                score += 0.2
        
        if "speed" in perf_requirements and perf_requirements["speed"] > 0.8:
            if archetype == AgentArchetype.EFFICIENCY_OPTIMIZER:
                score += 0.2
        
        # Historical performance bonus
        if archetype in self.performance_database:
            avg_performance = np.mean(self.performance_database[archetype])
            score += avg_performance * 0.1
        
        return score
    
    def _get_or_create_blueprint(self, 
                               archetype: AgentArchetype,
                               request: CreationRequest) -> AgentBlueprint:
        """Get existing blueprint or create new one for the archetype."""
        if archetype in self.blueprints:
            return self.blueprints[archetype]
        
        # Create new blueprint
        blueprint = self._create_blueprint_for_archetype(archetype, request)
        self.blueprints[archetype] = blueprint
        
        return blueprint
    
    def _create_blueprint_for_archetype(self, 
                                      archetype: AgentArchetype,
                                      request: CreationRequest) -> AgentBlueprint:
        """Create a new blueprint for the specified archetype."""
        base_config = AgentConfig(
            name=f"{archetype.value}_agent",
            prompt=self._generate_archetype_prompt(archetype),
            model="gpt-4",
            temperature=self._get_archetype_temperature(archetype),
            max_tokens=2000,
            hyperparameters=self._get_archetype_hyperparameters(archetype)
        )
        
        specialization_params = self._get_specialization_parameters(archetype, request)
        performance_targets = self._get_performance_targets(archetype)
        adaptation_rules = self._get_adaptation_rules(archetype)
        
        return AgentBlueprint(
            archetype=archetype,
            base_config=base_config,
            specialization_parameters=specialization_params,
            performance_targets=performance_targets,
            adaptation_rules=adaptation_rules
        )
    
    def _generate_specialized_config(self, 
                                   blueprint: AgentBlueprint,
                                   request: CreationRequest) -> AgentConfig:
        """Generate specialized configuration based on blueprint and request."""
        config = blueprint.base_config
        
        # Apply specialization parameters
        specialized_config = AgentConfig(
            name=f"{request.task_domain}_{blueprint.archetype.value}",
            prompt=self._customize_prompt(config.prompt, request),
            model=config.model,
            temperature=self._adjust_temperature(config.temperature, request),
            max_tokens=config.max_tokens,
            hyperparameters={
                **config.hyperparameters,
                **self._generate_request_specific_hyperparameters(request)
            }
        )
        
        return specialized_config
    
    def _instantiate_agent(self, 
                         config: AgentConfig,
                         request: CreationRequest) -> Optional[BaseAgent]:
        """Instantiate the appropriate agent type based on capabilities."""
        # Determine agent type based on primary capability
        primary_capabilities = request.required_capabilities
        
        if AgentCapability.EVALUATION in primary_capabilities:
            return EvaluatorAgent(config)
        elif AgentCapability.REFINEMENT in primary_capabilities:
            return RefinerAgent(config)
        else:
            # Default to ProverAgent for most other capabilities
            return ProverAgent(config)
    
    def _design_team_composition(self, 
                               required_capabilities: Set[AgentCapability],
                               team_size: int,
                               complexity: str) -> List[Dict[str, Any]]:
        """Design optimal team composition for the given requirements."""
        team_roles = []
        
        # Essential roles based on capabilities
        if AgentCapability.PROBLEM_SOLVING in required_capabilities:
            team_roles.append({
                "role": "problem_solver",
                "capabilities": {AgentCapability.PROBLEM_SOLVING},
                "performance_targets": {"accuracy": 0.8, "creativity": 0.7},
                "specialization_hints": ["analytical", "systematic"]
            })
        
        if AgentCapability.EVALUATION in required_capabilities:
            team_roles.append({
                "role": "evaluator",
                "capabilities": {AgentCapability.EVALUATION},
                "performance_targets": {"accuracy": 0.9, "consistency": 0.8},
                "specialization_hints": ["critical", "thorough"]
            })
        
        if AgentCapability.REFINEMENT in required_capabilities:
            team_roles.append({
                "role": "refiner",
                "capabilities": {AgentCapability.REFINEMENT},
                "performance_targets": {"improvement": 0.7, "efficiency": 0.6},
                "specialization_hints": ["iterative", "optimization"]
            })
        
        # Add complementary roles if team size allows
        remaining_slots = team_size - len(team_roles)
        
        if remaining_slots > 0 and AgentCapability.CREATIVE_WRITING in required_capabilities:
            team_roles.append({
                "role": "creative_writer",
                "capabilities": {AgentCapability.CREATIVE_WRITING},
                "performance_targets": {"creativity": 0.8, "engagement": 0.7},
                "specialization_hints": ["imaginative", "expressive"]
            })
            remaining_slots -= 1
        
        # Fill remaining slots with generalists
        for i in range(remaining_slots):
            team_roles.append({
                "role": f"generalist_{i+1}",
                "capabilities": required_capabilities,
                "performance_targets": {"versatility": 0.7, "adaptability": 0.8},
                "specialization_hints": ["flexible", "adaptive"]
            })
        
        return team_roles
    
    def _initialize_blueprints(self):
        """Initialize predefined agent blueprints."""
        # This would load from configuration files or database
        # For now, we'll create basic blueprints programmatically
        
        for archetype in AgentArchetype:
            base_config = AgentConfig(
                name=f"{archetype.value}_base",
                prompt=self._generate_archetype_prompt(archetype),
                model="gpt-4",
                temperature=self._get_archetype_temperature(archetype),
                max_tokens=2000
            )
            
            blueprint = AgentBlueprint(
                archetype=archetype,
                base_config=base_config,
                specialization_parameters={},
                performance_targets=self._get_performance_targets(archetype),
                adaptation_rules=[]
            )
            
            self.blueprints[archetype] = blueprint
    
    def _generate_archetype_prompt(self, archetype: AgentArchetype) -> str:
        """Generate base prompt for the archetype."""
        prompts = {
            AgentArchetype.CREATIVE_INNOVATOR: "You are a creative innovator who thinks outside the box and generates novel solutions.",
            AgentArchetype.ANALYTICAL_THINKER: "You are an analytical thinker who approaches problems systematically and logically.",
            AgentArchetype.PRACTICAL_EXECUTOR: "You are a practical executor who focuses on implementable and efficient solutions.",
            AgentArchetype.QUALITY_GUARDIAN: "You are a quality guardian who ensures high standards and thorough evaluation.",
            AgentArchetype.EFFICIENCY_OPTIMIZER: "You are an efficiency optimizer who seeks the most effective and streamlined approaches.",
            AgentArchetype.COLLABORATIVE_FACILITATOR: "You are a collaborative facilitator who excels at synthesis and team coordination.",
            AgentArchetype.DOMAIN_SPECIALIST: "You are a domain specialist with deep expertise in specific areas.",
            AgentArchetype.GENERALIST_ADAPTER: "You are a versatile generalist who adapts to diverse challenges and requirements."
        }
        
        return prompts.get(archetype, "You are a helpful AI assistant.")
    
    def _get_archetype_temperature(self, archetype: AgentArchetype) -> float:
        """Get optimal temperature setting for the archetype."""
        temperatures = {
            AgentArchetype.CREATIVE_INNOVATOR: 0.8,
            AgentArchetype.ANALYTICAL_THINKER: 0.3,
            AgentArchetype.PRACTICAL_EXECUTOR: 0.5,
            AgentArchetype.QUALITY_GUARDIAN: 0.2,
            AgentArchetype.EFFICIENCY_OPTIMIZER: 0.4,
            AgentArchetype.COLLABORATIVE_FACILITATOR: 0.6,
            AgentArchetype.DOMAIN_SPECIALIST: 0.4,
            AgentArchetype.GENERALIST_ADAPTER: 0.7
        }
        
        return temperatures.get(archetype, 0.5)
    
    def _get_archetype_hyperparameters(self, archetype: AgentArchetype) -> Dict[str, Any]:
        """Get archetype-specific hyperparameters."""
        hyperparams = {
            AgentArchetype.CREATIVE_INNOVATOR: {
                "creativity": 0.8,
                "max_variants": 5,
                "exploration_factor": 0.7
            },
            AgentArchetype.ANALYTICAL_THINKER: {
                "precision": 0.9,
                "detail_level": 0.8,
                "systematic_approach": True
            },
            AgentArchetype.PRACTICAL_EXECUTOR: {
                "efficiency": 0.8,
                "implementation_focus": True,
                "resource_awareness": 0.7
            },
            AgentArchetype.QUALITY_GUARDIAN: {
                "thoroughness": 0.9,
                "quality_threshold": 0.8,
                "consistency_check": True
            }
        }
        
        return hyperparams.get(archetype, {})
    
    def _get_performance_targets(self, archetype: AgentArchetype) -> Dict[str, float]:
        """Get performance targets for the archetype."""
        targets = {
            AgentArchetype.CREATIVE_INNOVATOR: {"creativity": 0.8, "novelty": 0.7},
            AgentArchetype.ANALYTICAL_THINKER: {"accuracy": 0.9, "logic": 0.8},
            AgentArchetype.PRACTICAL_EXECUTOR: {"efficiency": 0.8, "implementability": 0.9},
            AgentArchetype.QUALITY_GUARDIAN: {"quality": 0.9, "consistency": 0.8}
        }
        
        return targets.get(archetype, {"performance": 0.7})
    
    def _get_adaptation_rules(self, archetype: AgentArchetype) -> List[Dict[str, Any]]:
        """Get adaptation rules for the archetype."""
        # This would define how the agent should adapt based on feedback
        return []
    
    def _customize_prompt(self, base_prompt: str, request: CreationRequest) -> str:
        """Customize prompt based on request specifics."""
        customized = base_prompt
        
        # Add domain-specific context
        if request.task_domain:
            customized += f"\n\nYou specialize in {request.task_domain} tasks."
        
        # Add capability-specific instructions
        if request.required_capabilities:
            capabilities_text = ", ".join([cap.value for cap in request.required_capabilities])
            customized += f"\n\nYour primary capabilities include: {capabilities_text}."
        
        # Add specialization hints
        if request.specialization_hints:
            hints_text = ", ".join(request.specialization_hints)
            customized += f"\n\nFocus on being: {hints_text}."
        
        return customized
    
    def _adjust_temperature(self, base_temperature: float, request: CreationRequest) -> float:
        """Adjust temperature based on request requirements."""
        adjusted = base_temperature
        
        # Adjust based on performance requirements
        if "creativity" in request.performance_requirements:
            creativity_req = request.performance_requirements["creativity"]
            if creativity_req > 0.7:
                adjusted = min(0.9, adjusted + 0.2)
        
        if "accuracy" in request.performance_requirements:
            accuracy_req = request.performance_requirements["accuracy"]
            if accuracy_req > 0.8:
                adjusted = max(0.1, adjusted - 0.2)
        
        return adjusted
    
    def _generate_request_specific_hyperparameters(self, request: CreationRequest) -> Dict[str, Any]:
        """Generate hyperparameters specific to the request."""
        hyperparams = {}
        
        # Map performance requirements to hyperparameters
        for req, value in request.performance_requirements.items():
            if req == "speed":
                hyperparams["timeout"] = max(30, int(300 * (1 - value)))
            elif req == "thoroughness":
                hyperparams["detail_level"] = value
            elif req == "creativity":
                hyperparams["creativity"] = value
        
        # Add resource constraints
        if request.resource_constraints:
            hyperparams.update(request.resource_constraints)
        
        return hyperparams
    
    def _record_creation(self, 
                        request: CreationRequest,
                        agent: BaseAgent,
                        archetype: AgentArchetype):
        """Record agent creation for learning and analytics."""
        creation_record = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent.agent_id,
            "archetype": archetype.value,
            "task_domain": request.task_domain,
            "capabilities": [cap.value for cap in request.required_capabilities],
            "performance_requirements": request.performance_requirements,
            "specialization_hints": request.specialization_hints
        }
        
        self.creation_history.append(creation_record)
        
        # Limit history size
        if len(self.creation_history) > 1000:
            self.creation_history = self.creation_history[-500:]
    
    def _analyze_evolution_needs(self, 
                               agent: BaseAgent,
                               feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what evolution strategy to apply based on feedback."""
        strategy = {
            "type": "parameter_adjustment",
            "adjustments": {}
        }
        
        # Analyze performance gaps
        current_performance = feedback.get("performance_metrics", {})
        
        if "accuracy" in current_performance:
            accuracy = current_performance["accuracy"]
            if accuracy < 0.7:
                strategy["adjustments"]["temperature"] = -0.1
                strategy["adjustments"]["detail_level"] = 0.2
        
        if "speed" in current_performance:
            speed = current_performance["speed"]
            if speed < 0.6:
                strategy["adjustments"]["timeout"] = -30
                strategy["adjustments"]["max_tokens"] = -200
        
        return strategy
    
    def _apply_evolution_strategy(self, 
                                config: AgentConfig,
                                strategy: Dict[str, Any]) -> AgentConfig:
        """Apply evolution strategy to create new configuration."""
        new_config = AgentConfig(
            name=f"evolved_{config.name}",
            prompt=config.prompt,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            hyperparameters=config.hyperparameters.copy() if config.hyperparameters else {}
        )
        
        # Apply adjustments
        adjustments = strategy.get("adjustments", {})
        
        for param, adjustment in adjustments.items():
            if param == "temperature":
                new_config.temperature = max(0.0, min(1.0, new_config.temperature + adjustment))
            elif param == "max_tokens":
                new_config.max_tokens = max(100, new_config.max_tokens + adjustment)
            elif param in new_config.hyperparameters:
                current_value = new_config.hyperparameters[param]
                if isinstance(current_value, (int, float)):
                    new_config.hyperparameters[param] = current_value + adjustment
        
        return new_config
    
    def _instantiate_agent_from_config(self, config: AgentConfig) -> Optional[BaseAgent]:
        """Create agent instance from configuration."""
        # Default to ProverAgent for evolved agents
        return ProverAgent(config)
    
    def _transfer_learning(self, source_agent: BaseAgent, target_agent: BaseAgent):
        """Transfer learning from source to target agent."""
        # Transfer performance metrics and trust score
        if hasattr(source_agent, 'trust_score'):
            target_agent.trust_score = source_agent.trust_score * 0.8  # Slight reduction for evolution
        
        if hasattr(source_agent, 'performance_metrics'):
            target_agent.performance_metrics = source_agent.performance_metrics
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get comprehensive factory statistics."""
        return {
            **self.factory_stats,
            "blueprints_available": len(self.blueprints),
            "creation_history_size": len(self.creation_history),
            "performance_database_size": len(self.performance_database)
        }


# Global factory instance
_global_factory = None

def get_agent_factory() -> AgentFactory:
    """Get the global agent factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = AgentFactory()
    return _global_factory
