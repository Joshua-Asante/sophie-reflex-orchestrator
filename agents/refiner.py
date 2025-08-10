from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
import structlog
import random
import json
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import openai
import anthropic
from .base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus, CircuitState
from .prover import ProverAgent

logger = structlog.get_logger()


class PopulationManager:
    """Manages agent population with advanced genetic algorithm features."""

    def __init__(self, population_size: int = 10, diversity_threshold: float = 0.3):
        self.population_size = population_size
        self.diversity_threshold = diversity_threshold
        self.generation_history = []
        self.evolution_stats = {
            "total_generations": 0,
            "successful_mutations": 0,
            "successful_crossovers": 0,
            "population_improvements": 0
        }

    def calculate_population_diversity(self, agents: List[BaseAgent]) -> float:
        """Calculate population diversity based on agent characteristics."""
        if len(agents) < 2:
            return 0.0

        # Extract agent characteristics
        characteristics = []
        for agent in agents:
            char_vector = [
                agent.config.temperature,
                agent.trust_score,
                agent.get_success_rate(),
                len(agent.hyperparameters) if agent.hyperparameters else 0
            ]
            characteristics.append(char_vector)

        # Calculate pairwise distances
        distances = []
        for i in range(len(characteristics)):
            for j in range(i + 1, len(characteristics)):
                distance = np.linalg.norm(np.array(characteristics[i]) - np.array(characteristics[j]))
                distances.append(distance)

        # Normalize diversity score
        avg_distance = np.mean(distances) if distances else 0.0
        diversity_score = min(1.0, avg_distance / 2.0)  # Normalize to 0-1

        return diversity_score

    def select_parents(self, agents: List[BaseAgent], evaluation_results: List[Dict[str, Any]]) -> List[Tuple[BaseAgent, BaseAgent]]:
        """Select parent pairs for crossover using tournament selection."""
        if len(agents) < 2:
            return []

        # Create agent-score mapping
        agent_scores = {}
        for eval_result in evaluation_results:
            agent_id = eval_result.get("agent_id")
            score = eval_result.get("overall_score", 0.0)
            if agent_id:
                agent_scores[agent_id] = score

        # Tournament selection
        parent_pairs = []
        tournament_size = min(3, len(agents))

        for _ in range(len(agents) // 2):
            # Select first parent
            tournament1 = random.sample(agents, tournament_size)
            parent1 = max(tournament1, key=lambda a: agent_scores.get(a.agent_id, 0.0))

            # Select second parent (different from first)
            remaining_agents = [a for a in agents if a.agent_id != parent1.agent_id]
            if remaining_agents:
                tournament2 = random.sample(remaining_agents, min(tournament_size, len(remaining_agents)))
                parent2 = max(tournament2, key=lambda a: agent_scores.get(a.agent_id, 0.0))
                parent_pairs.append((parent1, parent2))

        return parent_pairs

    def update_evolution_stats(self, generation: int, mutations: int, crossovers: int,
                             avg_score_before: float, avg_score_after: float):
        """Update evolution statistics."""
        self.evolution_stats["total_generations"] += 1
        self.evolution_stats["successful_mutations"] += mutations
        self.evolution_stats["successful_crossovers"] += crossovers

        if avg_score_after > avg_score_before:
            self.evolution_stats["population_improvements"] += 1

        self.generation_history.append({
            "generation": generation,
            "mutations": mutations,
            "crossovers": crossovers,
            "avg_score_before": avg_score_before,
            "avg_score_after": avg_score_after,
            "improvement": avg_score_after - avg_score_before
        })


class AdaptiveMutationStrategy:
    """Implements adaptive mutation strategies based on population performance."""

    def __init__(self, base_mutation_strength: float = 0.3):
        self.base_mutation_strength = base_mutation_strength
        self.mutation_history = []
        self.adaptation_rate = 0.1

    def get_adaptive_mutation_strength(self, population_performance: Dict[str, Any]) -> float:
        """Get adaptive mutation strength based on population performance."""
        avg_score = population_performance.get("average_score", 0.5)
        diversity = population_performance.get("diversity_score", 0.5)
        convergence = population_performance.get("convergence_score", 0.5)

        # Adjust mutation strength based on performance
        if avg_score < 0.4:
            # Low performance: increase mutation for exploration
            mutation_strength = self.base_mutation_strength * 1.5
        elif avg_score > 0.8:
            # High performance: decrease mutation for exploitation
            mutation_strength = self.base_mutation_strength * 0.7
        else:
            mutation_strength = self.base_mutation_strength

        # Adjust based on diversity
        if diversity < 0.3:
            # Low diversity: increase mutation
            mutation_strength *= 1.3

        # Adjust based on convergence
        if convergence > 0.8:
            # High convergence: increase mutation to escape local optima
            mutation_strength *= 1.2

        return min(1.0, max(0.1, mutation_strength))

    def get_mutation_focus_areas(self, agent_performance: Dict[str, Any]) -> List[str]:
        """Get mutation focus areas based on agent performance."""
        focus_areas = ["clarity", "efficiency", "completeness"]

        # Add performance-specific focus areas
        if agent_performance.get("trust_score", 0.5) < 0.4:
            focus_areas.append("reliability")

        if agent_performance.get("success_rate", 0.5) < 0.6:
            focus_areas.append("effectiveness")

        if agent_performance.get("execution_time", 0.0) > 25:
            focus_areas.append("performance")

        return focus_areas


class RefinerAgent(BaseAgent):
    """Enhanced agent that uses advanced genetic algorithms to improve agent population."""

    def __init__(self, config: AgentConfig, agent_id: str = None):
        super().__init__(config, agent_id)
        self.mutation_strength = config.hyperparameters.get('mutation_strength', 0.3)
        self.focus_areas = config.hyperparameters.get('focus_areas', ['clarity', 'efficiency', 'completeness'])
        self.crossover_points = config.hyperparameters.get('crossover_points', 3)
        self.balance_ratio = config.hyperparameters.get('balance_ratio', 0.6)

        # Enhanced refiner features
        self.population_manager = PopulationManager()
        self.adaptive_mutation = AdaptiveMutationStrategy(self.mutation_strength)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.evolution_enabled = config.hyperparameters.get('evolution_enabled', True)
        self.quality_threshold = config.hyperparameters.get('quality_threshold', 0.6)

        # Track agent performance history
        self.agent_history = {}
        self.generation_count = 0
        self.evolution_metrics = {
            "total_refinements": 0,
            "successful_evolutions": 0,
            "population_improvements": 0,
            "diversity_maintained": 0
        }

        logger.info("Enhanced refiner initialized",
                   agent_id=self.agent_id,
                   evolution_enabled=self.evolution_enabled)

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Execute the refiner agent with advanced genetic algorithm features."""
        try:
            context = context or {}
            evaluation_results = context.get("evaluation_results", [])
            current_agents = context.get("current_agents", [])
            generation_info = context.get("generation_info", {})

            self.generation_count = generation_info.get("generation", 0)

            # Analyze current population with enhanced metrics
            population_analysis = await self._analyze_population_enhanced(current_agents, evaluation_results)

            # Get adaptive parameters
            adaptive_params = self._get_adaptive_parameters(population_analysis)

            # Determine refinement actions with advanced strategies
            refinement_actions = await self._determine_refinement_actions_enhanced(
                population_analysis, adaptive_params
            )

            # Execute refinement actions with parallel processing
            refinement_results = await self._execute_refinement_actions_enhanced(
                refinement_actions, current_agents, task, context, adaptive_params
            )

            # Update evolution metrics
            self._update_evolution_metrics(population_analysis, refinement_results)

            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.name,
                result={
                    "population_analysis": population_analysis,
                    "refinement_actions": refinement_actions,
                    "refinement_results": refinement_results,
                    "adaptive_parameters": adaptive_params,
                    "generation": self.generation_count,
                    "refiner_type": self.config.name
                },
                confidence_score=refinement_results.get("confidence", 0.7),
                execution_time=0.0,  # Will be set by execute_with_retry
                status=AgentStatus.COMPLETED,
                metadata={
                    "mutation_strength": adaptive_params.get("mutation_strength", self.mutation_strength),
                    "focus_areas": adaptive_params.get("focus_areas", self.focus_areas),
                    "agents_pruned": len(refinement_results.get("pruned_agents", [])),
                    "agents_created": len(refinement_results.get("new_agents", [])),
                    "agents_mutated": len(refinement_results.get("mutated_agents", [])),
                    "agents_crossed": len(refinement_results.get("crossover_agents", [])),
                    "evolution_metrics": self.evolution_metrics
                }
            )

        except Exception as e:
            logger.error("Refiner execution failed", agent_id=self.agent_id, error=str(e))
            raise

    async def _analyze_population_enhanced(self, agents: List[BaseAgent],
                                         evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced population analysis with advanced metrics."""
        if not agents or not evaluation_results:
            return {
                "total_agents": len(agents),
                "average_score": 0.0,
                "best_performers": [],
                "worst_performers": [],
                "diversity_score": 0.0,
                "convergence_score": 0.0,
                "population_health": 0.0,
                "evolution_potential": 0.0
            }

        # Calculate scores for each agent
        agent_scores = {}
        agent_performances = {}

        for eval_result in evaluation_results:
            agent_id = eval_result.get("agent_id")
            score = eval_result.get("overall_score", 0.0)
            if agent_id:
                agent_scores[agent_id] = score

                # Find corresponding agent for detailed analysis
                for agent in agents:
                    if agent.agent_id == agent_id:
                        agent_performances[agent_id] = {
                            "score": score,
                            "trust_score": agent.trust_score,
                            "success_rate": agent.get_success_rate(),
                            "execution_count": agent.execution_count,
                            "last_execution_time": agent.last_execution_time
                        }
                        break

        # Find best and worst performers
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)

        best_performers = [
            {"agent_id": agent_id, "score": score, "performance": agent_performances.get(agent_id, {})}
            for agent_id, score in sorted_agents[:3]
        ]

        worst_performers = [
            {"agent_id": agent_id, "score": score, "performance": agent_performances.get(agent_id, {})}
            for agent_id, score in sorted_agents[-3:]
        ]

        # Calculate advanced metrics
        average_score = sum(agent_scores.values()) / len(agent_scores) if agent_scores else 0.0

        # Calculate diversity using population manager
        diversity_score = self.population_manager.calculate_population_diversity(agents)

        # Calculate convergence score
        if len(agent_scores) > 1:
            score_range = max(agent_scores.values()) - min(agent_scores.values())
            convergence_score = max(0.0, 1.0 - score_range)
        else:
            convergence_score = 1.0

        # Calculate population health
        health_factors = [
            average_score,
            diversity_score,
            1.0 - convergence_score,  # Some diversity is good
            min(1.0, len(agents) / 10.0)  # Population size factor
        ]
        population_health = sum(health_factors) / len(health_factors)

        # Calculate evolution potential
        evolution_potential = self._calculate_evolution_potential(
            agent_scores, diversity_score, convergence_score
        )

        return {
            "total_agents": len(agents),
            "average_score": average_score,
            "best_performers": best_performers,
            "worst_performers": worst_performers,
            "diversity_score": diversity_score,
            "convergence_score": convergence_score,
            "population_health": population_health,
            "evolution_potential": evolution_potential,
            "agent_scores": agent_scores,
            "agent_performances": agent_performances
        }

    def _calculate_evolution_potential(self, agent_scores: Dict[str, float],
                                     diversity: float, convergence: float) -> float:
        """Calculate the potential for successful evolution."""
        if not agent_scores:
            return 0.0

        # Factors that indicate good evolution potential
        score_variance = np.var(list(agent_scores.values())) if len(agent_scores) > 1 else 0.0
        avg_score = np.mean(list(agent_scores.values()))

        # High variance + moderate diversity + not too converged = good potential
        evolution_factors = [
            min(1.0, score_variance * 5),  # Some variance is good
            diversity,  # Diversity is good
            1.0 - convergence,  # Not too converged
            avg_score  # Decent baseline performance
        ]

        return sum(evolution_factors) / len(evolution_factors)

    def _get_adaptive_parameters(self, population_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get adaptive parameters based on population analysis."""
        # Get adaptive mutation strength
        mutation_strength = self.adaptive_mutation.get_adaptive_mutation_strength(population_analysis)

        # Get adaptive focus areas
        focus_areas = self.focus_areas.copy()

        # Add performance-specific focus areas
        if population_analysis.get("average_score", 0.5) < 0.4:
            focus_areas.extend(["reliability", "effectiveness"])

        if population_analysis.get("diversity_score", 0.5) < 0.3:
            focus_areas.extend(["innovation", "creativity"])

        # Remove duplicates
        focus_areas = list(set(focus_areas))

        return {
            "mutation_strength": mutation_strength,
            "focus_areas": focus_areas,
            "crossover_rate": min(1.0, 0.3 + (population_analysis.get("evolution_potential", 0.0) * 0.4)),
            "selection_pressure": 1.0 + (population_analysis.get("average_score", 0.5) * 0.5)
        }

    async def _determine_refinement_actions_enhanced(self, population_analysis: Dict[str, Any],
                                                   adaptive_params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine refinement actions using advanced strategies."""
        actions = {
            "prune_agents": [],
            "mutate_agents": [],
            "create_agents": [],
            "crossover_pairs": [],
            "elite_preservation": []
        }

        avg_score = population_analysis.get("average_score", 0.0)
        diversity_score = population_analysis.get("diversity_score", 0.0)
        convergence_score = population_analysis.get("convergence_score", 0.0)
        evolution_potential = population_analysis.get("evolution_potential", 0.0)
        worst_performers = population_analysis.get("worst_performers", [])
        best_performers = population_analysis.get("best_performers", [])

        # Adaptive pruning based on performance and diversity
        if avg_score < 0.4 or diversity_score < 0.2:
            # Aggressive pruning for poor performance or low diversity
            actions["prune_agents"] = [agent["agent_id"] for agent in worst_performers[:2]]
        elif avg_score < 0.6:
            # Moderate pruning
            actions["prune_agents"] = [agent["agent_id"] for agent in worst_performers[:1]]

        # Adaptive mutation based on evolution potential
        mutation_candidates = []
        for agent in worst_performers:
            if 0.3 <= agent["score"] <= 0.7:  # Moderate performers
                mutation_candidates.append(agent["agent_id"])

        # Adjust mutation rate based on evolution potential
        mutation_rate = min(1.0, len(mutation_candidates) * adaptive_params["mutation_strength"])
        num_mutations = int(len(mutation_candidates) * mutation_rate)
        actions["mutate_agents"] = mutation_candidates[:num_mutations]

        # Adaptive creation based on diversity and performance
        creation_needed = False
        if diversity_score < 0.3:
            creation_needed = True
        elif avg_score < 0.5 and evolution_potential > 0.6:
            creation_needed = True

        if creation_needed:
            actions["create_agents"] = ["new_agent_" + str(i) for i in range(2)]

        # Adaptive crossover based on convergence and quality
        if convergence_score > 0.6 and len(best_performers) >= 2 and evolution_potential > 0.5:
            # High convergence with good performers: encourage crossover
            crossover_rate = adaptive_params.get("crossover_rate", 0.3)
            num_crossovers = max(1, int(len(best_performers) * crossover_rate))

            for i in range(min(num_crossovers, len(best_performers) - 1)):
                actions["crossover_pairs"].append((
                    best_performers[i]["agent_id"],
                    best_performers[i + 1]["agent_id"]
                ))

        # Elite preservation for top performers
        if best_performers:
            actions["elite_preservation"] = [agent["agent_id"] for agent in best_performers[:2]]

        return actions

    async def _execute_refinement_actions_enhanced(self, actions: Dict[str, Any],
                                                 current_agents: List[BaseAgent],
                                                 task: str, context: Dict[str, Any],
                                                 adaptive_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute refinement actions with parallel processing and enhanced strategies."""
        results = {
            "pruned_agents": [],
            "mutated_agents": [],
            "new_agents": [],
            "crossover_agents": [],
            "elite_agents": [],
            "confidence": 0.7
        }

        # Execute actions in parallel where possible
        tasks = []

        # Prune agents (synchronous)
        for agent_id in actions.get("prune_agents", []):
            results["pruned_agents"].append({
                "agent_id": agent_id,
                "reason": "Poor performance or low diversity",
                "timestamp": datetime.now().isoformat()
            })

        # Mutate agents (parallel)
        for agent_id in actions.get("mutate_agents", []):
            task = asyncio.create_task(
                self._mutate_agent_enhanced(agent_id, current_agents, task, context, adaptive_params)
            )
            tasks.append(("mutation", task))

        # Create new agents (parallel)
        for new_agent_id in actions.get("create_agents", []):
            task = asyncio.create_task(
                self._create_new_agent_enhanced(new_agent_id, task, context, adaptive_params)
            )
            tasks.append(("creation", task))

        # Perform crossover (parallel)
        for parent1_id, parent2_id in actions.get("crossover_pairs", []):
            task = asyncio.create_task(
                self._crossover_agents_enhanced(parent1_id, parent2_id, current_agents, task, context, adaptive_params)
            )
            tasks.append(("crossover", task))

        # Execute all tasks in parallel
        if tasks:
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            for i, (action_type, _) in enumerate(tasks):
                result = task_results[i]
                if isinstance(result, Exception):
                    logger.warning(f"{action_type} action failed", error=str(result))
                elif result:
                    if action_type == "mutation":
                        results["mutated_agents"].append(result)
                    elif action_type == "creation":
                        results["new_agents"].append(result)
                    elif action_type == "crossover":
                        results["crossover_agents"].append(result)

        # Elite preservation
        for agent_id in actions.get("elite_preservation", []):
            results["elite_agents"].append({
                "agent_id": agent_id,
                "preservation_reason": "Top performer",
                "timestamp": datetime.now().isoformat()
            })

        # Calculate confidence based on actions taken and their success
        total_actions = sum(len(results[key]) for key in ["pruned_agents", "mutated_agents", "new_agents", "crossover_agents"])
        if total_actions > 0:
            results["confidence"] = min(1.0, 0.5 + (total_actions * 0.1))

        return results

    async def _mutate_agent_enhanced(self, agent_id: str, current_agents: List[BaseAgent],
                                    task: str, context: Dict[str, Any],
                                    adaptive_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhanced agent mutation with adaptive strategies."""
        try:
            # Find the agent to mutate
            agent_to_mutate = None
            for agent in current_agents:
                if agent.agent_id == agent_id:
                    agent_to_mutate = agent
                    break

            if not agent_to_mutate:
                logger.warning("Agent not found for mutation", agent_id=agent_id)
                return None

            # Get adaptive focus areas
            focus_areas = adaptive_params.get("focus_areas", self.focus_areas)
            mutation_strength = adaptive_params.get("mutation_strength", self.mutation_strength)

            # Generate enhanced mutation prompt
            prompt = await self.generate_prompt_enhanced(task, {
                **context,
                "mutation_target": agent_to_mutate.get_info(),
                "focus_areas": focus_areas,
                "mutation_strength": mutation_strength,
                "adaptive_params": adaptive_params
            })

            # Get mutation suggestions
            llm_response = await self._call_llm_implementation(prompt, {
                "mutation_type": "adaptive",
                "agent_id": agent_id,
                "focus_areas": focus_areas
            })

            mutation_suggestions = llm_response.get("content", "")

            # Create mutated agent configuration with enhanced parameters
            mutated_config = self._create_mutated_config_enhanced(
                agent_to_mutate.config, mutation_suggestions, adaptive_params
            )

            # Create new mutated agent
            mutated_agent = ProverAgent(
                config=mutated_config,
                agent_id=f"{agent_id}_mutated_{self.generation_count}"
            )

            return {
                "original_agent_id": agent_id,
                "mutated_agent_id": mutated_agent.agent_id,
                "mutation_type": "adaptive_enhancement",
                "suggestions": mutation_suggestions,
                "focus_areas": focus_areas,
                "mutation_strength": mutation_strength,
                "new_config": mutated_agent.get_info(),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Enhanced agent mutation failed", agent_id=agent_id, error=str(e))
            return None

    async def _create_new_agent_enhanced(self, agent_id: str, task: str, context: Dict[str, Any],
                                       adaptive_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhanced new agent creation with adaptive strategies."""
        try:
            # Generate new agent configuration based on task and current population
            prompt = f"""
Create a new agent configuration for the task: "{task}"

Current generation: {self.generation_count}
Focus areas: {', '.join(adaptive_params.get('focus_areas', self.focus_areas))}
Mutation strength: {adaptive_params.get('mutation_strength', self.mutation_strength)}

Generate a new agent that:
1. Addresses gaps in the current population
2. Focuses on the specified focus areas
3. Uses adaptive parameters for optimal performance
4. Complements existing agents rather than duplicating them

The agent should be innovative yet practical, with clear specialization.
"""

            llm_response = await self._call_llm_implementation(prompt, {
                "creation_type": "adaptive",
                "focus_areas": adaptive_params.get('focus_areas', self.focus_areas)
            })

            config_text = llm_response.get("content", "")

            # Parse the configuration with enhanced parameters
            new_config = self._parse_agent_config_enhanced(config_text, agent_id, adaptive_params)

            # Create new agent
            new_agent = ProverAgent(
                config=new_config,
                agent_id=agent_id
            )

            return {
                "agent_id": new_agent.agent_id,
                "creation_type": "adaptive_generation",
                "config_source": "llm_generated_enhanced",
                "focus_areas": adaptive_params.get('focus_areas', self.focus_areas),
                "agent_info": new_agent.get_info(),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Enhanced new agent creation failed", agent_id=agent_id, error=str(e))
            return None

    async def _crossover_agents_enhanced(self, parent1_id: str, parent2_id: str,
                                       current_agents: List[BaseAgent], task: str,
                                       context: Dict[str, Any], adaptive_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhanced crossover between two parent agents."""
        try:
            # Find parent agents
            parent1 = None
            parent2 = None

            for agent in current_agents:
                if agent.agent_id == parent1_id:
                    parent1 = agent
                elif agent.agent_id == parent2_id:
                    parent2 = agent

            if not parent1 or not parent2:
                logger.warning("Parent agents not found for crossover", parent1=parent1_id, parent2=parent2_id)
                return None

            # Generate enhanced crossover prompt
            prompt = f"""
Perform advanced genetic crossover between two parent agents for the task: "{task}"

Parent 1: {parent1.config.name}
- Prompt: {parent1.config.prompt}
- Temperature: {parent1.config.temperature}
- Hyperparameters: {parent1.hyperparameters}
- Performance: Trust Score = {parent1.trust_score}, Success Rate = {parent1.get_success_rate()}

Parent 2: {parent2.config.name}
- Prompt: {parent2.config.prompt}
- Temperature: {parent2.config.temperature}
- Hyperparameters: {parent2.hyperparameters}
- Performance: Trust Score = {parent2.trust_score}, Success Rate = {parent2.get_success_rate()}

Focus Areas: {', '.join(adaptive_params.get('focus_areas', self.focus_areas))}
Balance Ratio: {adaptive_params.get('balance_ratio', self.balance_ratio)}

Create a new agent that:
1. Inherits the best traits from both parents
2. Addresses the specified focus areas
3. Uses adaptive parameters for optimal performance
4. Shows innovation beyond simple combination
"""

            llm_response = await self._call_llm_implementation(prompt, {
                "crossover_type": "enhanced",
                "parent1": parent1_id,
                "parent2": parent2_id,
                "focus_areas": adaptive_params.get('focus_areas', self.focus_areas)
            })

            crossover_text = llm_response.get("content", "")

            # Create enhanced crossover configuration
            crossover_config = self._create_crossover_config_enhanced(
                parent1.config, parent2.config, crossover_text, adaptive_params
            )

            # Create crossover agent
            crossover_agent = ProverAgent(
                config=crossover_config,
                agent_id=f"crossover_{parent1_id}_{parent2_id}_{self.generation_count}"
            )

            return {
                "parent1_id": parent1_id,
                "parent2_id": parent2_id,
                "crossover_agent_id": crossover_agent.agent_id,
                "crossover_type": "enhanced_genetic_crossover",
                "balance_ratio": adaptive_params.get('balance_ratio', self.balance_ratio),
                "focus_areas": adaptive_params.get('focus_areas', self.focus_areas),
                "agent_info": crossover_agent.get_info(),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Enhanced agent crossover failed", parent1=parent1_id, parent2=parent2_id, error=str(e))
            return None

    def _create_mutated_config_enhanced(self, original_config: AgentConfig,
                                      mutation_suggestions: str,
                                      adaptive_params: Dict[str, Any]) -> AgentConfig:
        """Create enhanced mutated configuration with adaptive parameters."""
        # Adjust hyperparameters with enhanced strategy
        new_hyperparameters = original_config.hyperparameters.copy() if original_config.hyperparameters else {}

        # Apply adaptive mutation strength
        mutation_strength = adaptive_params.get("mutation_strength", self.mutation_strength)

        for key in new_hyperparameters:
            if isinstance(new_hyperparameters[key], (int, float)):
                # Use adaptive mutation range
                mutation_range = mutation_strength * 2  # ±mutation_strength
                mutation = random.uniform(-mutation_range, mutation_range)
                new_hyperparameters[key] = max(0.0, min(1.0, new_hyperparameters[key] + mutation))

        # Adjust temperature with adaptive strategy
        base_temperature = original_config.temperature
        temperature_mutation = random.uniform(-0.3, 0.3) * mutation_strength
        new_temperature = max(0.1, min(1.0, base_temperature + temperature_mutation))

        # Add adaptive focus areas to hyperparameters
        focus_areas = adaptive_params.get("focus_areas", self.focus_areas)
        new_hyperparameters["focus_areas"] = focus_areas
        new_hyperparameters["mutation_strength"] = mutation_strength

        return AgentConfig(
            name=f"{original_config.name}_enhanced",
            prompt=original_config.prompt,  # Keep original prompt
            model=original_config.model,
            temperature=new_temperature,
            max_tokens=original_config.max_tokens,
            timeout=original_config.timeout,
            max_retries=original_config.max_retries,
            retry_delay=original_config.retry_delay,
            hyperparameters=new_hyperparameters
        )

    def _parse_agent_config_enhanced(self, config_text: str, agent_id: str,
                                   adaptive_params: Dict[str, Any]) -> AgentConfig:
        """Parse enhanced agent configuration from LLM response."""
        # Simple parsing - in production, this would be more sophisticated
        lines = config_text.split('\n')

        name = f"enhanced_agent_{agent_id}"
        prompt = "You are an enhanced AI assistant with adaptive capabilities."
        model = "capability:general_agentic"
        temperature = 0.7
        max_tokens = 1000

        for line in lines:
            line = line.strip().lower()
            if "name:" in line:
                name = line.split("name:")[1].strip()
            elif "prompt:" in line:
                prompt = line.split("prompt:")[1].strip()
            elif "model:" in line:
                model = line.split("model:")[1].strip()
            elif "temperature:" in line:
                try:
                    temperature = float(line.split("temperature:")[1].strip())
                except:
                    pass

        # Create enhanced hyperparameters
        enhanced_hyperparameters = {
            "creativity": 0.7,
            "detail_level": 0.7,
            "focus_areas": adaptive_params.get("focus_areas", self.focus_areas),
            "mutation_strength": adaptive_params.get("mutation_strength", self.mutation_strength),
            "enhanced": True
        }

        return AgentConfig(
            name=name,
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            hyperparameters=enhanced_hyperparameters
        )

    def _create_crossover_config_enhanced(self, config1: AgentConfig, config2: AgentConfig,
                                        crossover_text: str, adaptive_params: Dict[str, Any]) -> AgentConfig:
        """Create enhanced crossover configuration from two parent configurations."""
        # Blend hyperparameters with adaptive strategy
        hyperparameters1 = config1.hyperparameters or {}
        hyperparameters2 = config2.hyperparameters or {}

        blended_hyperparameters = {}
        all_keys = set(hyperparameters1.keys()) | set(hyperparameters2.keys())

        balance_ratio = adaptive_params.get("balance_ratio", self.balance_ratio)

        for key in all_keys:
            val1 = hyperparameters1.get(key, 0.5)
            val2 = hyperparameters2.get(key, 0.5)
            blended_hyperparameters[key] = val1 * balance_ratio + val2 * (1 - balance_ratio)

        # Blend temperatures with adaptive strategy
        blended_temperature = config1.temperature * balance_ratio + config2.temperature * (1 - balance_ratio)

        # Enhanced prompt blending
        blended_prompt = f"{config1.prompt}\n\nEnhanced crossover elements: {config2.prompt[:200]}..."

        # Add adaptive parameters
        blended_hyperparameters["focus_areas"] = adaptive_params.get("focus_areas", self.focus_areas)
        blended_hyperparameters["crossover_generated"] = True
        blended_hyperparameters["balance_ratio"] = balance_ratio

        return AgentConfig(
            name=f"enhanced_crossover_{config1.name}_{config2.name}",
            prompt=blended_prompt,
            model=config1.model,  # Use parent1's model
            temperature=blended_temperature,
            max_tokens=max(config1.max_tokens, config2.max_tokens),
            hyperparameters=blended_hyperparameters
        )

    def _update_evolution_metrics(self, population_analysis: Dict[str, Any],
                                refinement_results: Dict[str, Any]):
        """Update evolution metrics and statistics."""
        self.evolution_metrics["total_refinements"] += 1

        # Track successful evolutions
        successful_actions = sum(len(refinement_results.get(key, []))
                               for key in ["mutated_agents", "new_agents", "crossover_agents"])

        if successful_actions > 0:
            self.evolution_metrics["successful_evolutions"] += 1

        # Track population improvements
        if population_analysis.get("average_score", 0.0) > 0.6:
            self.evolution_metrics["population_improvements"] += 1

        # Track diversity maintenance
        if population_analysis.get("diversity_score", 0.0) > 0.4:
            self.evolution_metrics["diversity_maintained"] += 1

    async def generate_prompt_enhanced(self, task: str, context: Dict[str, Any]) -> str:
        """Generate enhanced prompt for the refiner."""
        context = context or {}

        if "mutation_target" in context:
            # Enhanced mutation prompt
            return f"""
{self.config.prompt}

Target Agent: {context['mutation_target']['name']}
Current Performance: Trust Score = {context['mutation_target']['trust_score']}

Focus Areas: {', '.join(context['focus_areas'])}
Mutation Strength: {context['mutation_strength']}
Adaptive Parameters: {context.get('adaptive_params', {})}

Suggest specific improvements to this agent's configuration and approach.
Focus on enhancing performance in the weak areas while maintaining strengths.
Consider the adaptive parameters for optimal evolution.
"""

        elif "crossover_target" in context:
            # Enhanced crossover prompt
            return f"""
{self.config.prompt}

Perform enhanced genetic crossover between the provided parent agents.
Create a new agent that inherits the best traits from both parents.
Use the specified balance ratio and focus areas for optimal combination.
"""

        else:
            # Enhanced general refinement prompt
            return f"""
{self.config.prompt}

Task: {task}

Generation: {self.generation_count}

Analyze the current agent population and suggest improvements using advanced genetic algorithms.
Consider mutation, crossover, and new agent creation strategies with adaptive parameters.
Focus on improving overall population performance, diversity, and evolution potential.
"""

    async def _call_llm_implementation(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced LLM calling implementation for refiners."""
        try:
            if self.config.model == "openai":
                return await self._call_openai_enhanced(prompt, context)
            elif self.config.model == "anthropic":
                return await self._call_anthropic_enhanced(prompt, context)
            else:
                raise ValueError(f"Unsupported model: {self.config.model}")

        except Exception as e:
            logger.error("LLM call failed", agent_id=self.agent_id, model=self.config.model, error=str(e))
            raise

    async def _call_openai_enhanced(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced OpenAI calling for refiners."""
        # This method is not used in the new_code, but kept for compatibility if other parts of the system rely on it.
        # The actual LLM calls are now handled by _call_llm_implementation.
        # For OpenAI, we'd typically use an async client here.
        # For simplicity, we'll simulate a call.
        logger.debug("Simulating OpenAI API call", prompt=prompt, model=self.config.model)
        await asyncio.sleep(0.5) # Simulate network delay
        return {
            "content": "Simulated OpenAI response for prompt: " + prompt,
            "confidence": 0.8,
            "reasoning": "Simulated OpenAI reasoning",
            "model": "gpt-4",
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 100, "total_tokens": 100}
        }

    async def _call_anthropic_enhanced(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Anthropic calling for refiners."""
        # This method is not used in the new_code, but kept for compatibility if other parts of the system rely on it.
        # The actual LLM calls are now handled by _call_llm_implementation.
        # For Anthropic, we'd typically use an async client here.
        # For simplicity, we'll simulate a call.
        logger.debug("Simulating Anthropic API call", prompt=prompt, model=self.config.model)
        await asyncio.sleep(0.5) # Simulate network delay
        return {
            "content": "Simulated Anthropic response for prompt: " + prompt,
            "confidence": 0.8,
            "reasoning": "Simulated Anthropic reasoning",
            "model": "claude-3-sonnet-20240229",
            "usage": {"input_tokens": len(prompt.split()), "output_tokens": 100, "total_tokens": 100}
        }

    def get_refinement_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics about the refiner's activities."""
        return {
            "generation_count": self.generation_count,
            "mutation_strength": self.mutation_strength,
            "focus_areas": self.focus_areas,
            "crossover_points": self.crossover_points,
            "balance_ratio": self.balance_ratio,
            "agents_tracked": len(self.agent_history),
            "evolution_metrics": self.evolution_metrics,
            "population_manager_stats": self.population_manager.evolution_stats,
            "adaptive_mutation_stats": {
                "base_strength": self.adaptive_mutation.base_mutation_strength,
                "adaptation_rate": self.adaptive_mutation.adaptation_rate
            }
        }

    async def generate_prompt(self, task: str, context: Dict[str, Any] = None) -> str:
        """Generate the prompt for the LLM based on the task and context."""
        return await self.generate_prompt_enhanced(task, context)
