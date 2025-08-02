from typing import Dict, Any, List, Optional, Tuple
import asyncio
import structlog
import random
import json
from datetime import datetime
import openai
import anthropic
from .base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus
from .prover import ProverAgent

logger = structlog.get_logger()


class RefinerAgent(BaseAgent):
    """Agent that uses score history to prune poor agents and create new ones."""
    
    def __init__(self, config: AgentConfig, agent_id: str = None):
        super().__init__(config, agent_id)
        self.mutation_strength = config.hyperparameters.get('mutation_strength', 0.3)
        self.focus_areas = config.hyperparameters.get('focus_areas', ['clarity', 'efficiency', 'completeness'])
        self.crossover_points = config.hyperparameters.get('crossover_points', 3)
        self.balance_ratio = config.hyperparameters.get('balance_ratio', 0.6)
        
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        
        # Track agent performance history
        self.agent_history = {}
        self.generation_count = 0
        
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Execute the refiner agent to improve agent population."""
        try:
            context = context or {}
            evaluation_results = context.get("evaluation_results", [])
            current_agents = context.get("current_agents", [])
            generation_info = context.get("generation_info", {})
            
            self.generation_count = generation_info.get("generation", 0)
            
            # Analyze current population performance
            population_analysis = await self._analyze_population(current_agents, evaluation_results)
            
            # Determine refinement actions
            refinement_actions = await self._determine_refinement_actions(population_analysis)
            
            # Execute refinement actions
            refinement_results = await self._execute_refinement_actions(
                refinement_actions, current_agents, task, context
            )
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.name,
                result={
                    "population_analysis": population_analysis,
                    "refinement_actions": refinement_actions,
                    "refinement_results": refinement_results,
                    "generation": self.generation_count,
                    "refiner_type": self.config.name
                },
                confidence_score=refinement_results.get("confidence", 0.7),
                execution_time=0.0,  # Will be set by execute_with_retry
                status=AgentStatus.COMPLETED,
                metadata={
                    "mutation_strength": self.mutation_strength,
                    "focus_areas": self.focus_areas,
                    "agents_pruned": len(refinement_results.get("pruned_agents", [])),
                    "agents_created": len(refinement_results.get("new_agents", [])),
                    "agents_mutated": len(refinement_results.get("mutated_agents", []))
                }
            )
            
        except Exception as e:
            logger.error("Refiner execution failed", agent_id=self.agent_id, error=str(e))
            raise
    
    async def _analyze_population(self, agents: List[BaseAgent], evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the current agent population performance."""
        if not agents or not evaluation_results:
            return {
                "total_agents": len(agents),
                "average_score": 0.0,
                "best_performers": [],
                "worst_performers": [],
                "diversity_score": 0.0,
                "convergence_score": 0.0
            }
        
        # Calculate scores for each agent
        agent_scores = {}
        for eval_result in evaluation_results:
            agent_id = eval_result.get("agent_id")
            score = eval_result.get("overall_score", 0.0)
            if agent_id:
                agent_scores[agent_id] = score
        
        # Find best and worst performers
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_performers = [
            {"agent_id": agent_id, "score": score}
            for agent_id, score in sorted_agents[:3]  # Top 3
        ]
        
        worst_performers = [
            {"agent_id": agent_id, "score": score}
            for agent_id, score in sorted_agents[-3:]  # Bottom 3
        ]
        
        # Calculate average score
        average_score = sum(agent_scores.values()) / len(agent_scores) if agent_scores else 0.0
        
        # Calculate diversity score (based on score distribution)
        if len(agent_scores) > 1:
            score_variance = sum((score - average_score) ** 2 for score in agent_scores.values()) / len(agent_scores)
            diversity_score = min(1.0, score_variance * 2)  # Normalize to 0-1
        else:
            diversity_score = 0.0
        
        # Calculate convergence score (how close scores are to each other)
        if len(agent_scores) > 1:
            score_range = max(agent_scores.values()) - min(agent_scores.values())
            convergence_score = max(0.0, 1.0 - score_range)  # Higher when scores are similar
        else:
            convergence_score = 1.0
        
        return {
            "total_agents": len(agents),
            "average_score": average_score,
            "best_performers": best_performers,
            "worst_performers": worst_performers,
            "diversity_score": diversity_score,
            "convergence_score": convergence_score,
            "agent_scores": agent_scores
        }
    
    async def _determine_refinement_actions(self, population_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine what refinement actions to take based on population analysis."""
        actions = {
            "prune_agents": [],
            "mutate_agents": [],
            "create_agents": [],
            "crossover_pairs": []
        }
        
        avg_score = population_analysis.get("average_score", 0.0)
        diversity_score = population_analysis.get("diversity_score", 0.0)
        convergence_score = population_analysis.get("convergence_score", 0.0)
        worst_performers = population_analysis.get("worst_performers", [])
        best_performers = population_analysis.get("best_performers", [])
        
        # Prune worst performers if score is too low
        if avg_score < 0.5:
            actions["prune_agents"] = [agent["agent_id"] for agent in worst_performers]
        
        # Mutate agents with moderate performance
        moderate_performers = [
            agent for agent in worst_performers
            if 0.3 <= agent["score"] <= 0.6
        ]
        actions["mutate_agents"] = [agent["agent_id"] for agent in moderate_performers]
        
        # Create new agents if diversity is low or average score is low
        if diversity_score < 0.3 or avg_score < 0.4:
            actions["create_agents"] = ["new_agent_" + str(i) for i in range(2)]
        
        # Perform crossover between best performers if convergence is high
        if convergence_score > 0.7 and len(best_performers) >= 2:
            actions["crossover_pairs"] = [
                (best_performers[0]["agent_id"], best_performers[1]["agent_id"])
            ]
        
        return actions
    
    async def _execute_refinement_actions(self, actions: Dict[str, Any], current_agents: List[BaseAgent], task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the determined refinement actions."""
        results = {
            "pruned_agents": [],
            "mutated_agents": [],
            "new_agents": [],
            "crossover_agents": [],
            "confidence": 0.7
        }
        
        # Prune agents
        for agent_id in actions.get("prune_agents", []):
            results["pruned_agents"].append({
                "agent_id": agent_id,
                "reason": "Poor performance",
                "timestamp": datetime.now().isoformat()
            })
        
        # Mutate agents
        for agent_id in actions.get("mutate_agents", []):
            mutated_agent = await self._mutate_agent(agent_id, current_agents, task, context)
            if mutated_agent:
                results["mutated_agents"].append(mutated_agent)
        
        # Create new agents
        for new_agent_id in actions.get("create_agents", []):
            new_agent = await self._create_new_agent(new_agent_id, task, context)
            if new_agent:
                results["new_agents"].append(new_agent)
        
        # Perform crossover
        for parent1_id, parent2_id in actions.get("crossover_pairs", []):
            crossover_agent = await self._crossover_agents(parent1_id, parent2_id, current_agents, task, context)
            if crossover_agent:
                results["crossover_agents"].append(crossover_agent)
        
        # Calculate confidence based on actions taken
        total_actions = sum(len(results[key]) for key in ["pruned_agents", "mutated_agents", "new_agents", "crossover_agents"])
        if total_actions > 0:
            results["confidence"] = min(1.0, 0.5 + (total_actions * 0.1))
        
        return results
    
    async def _mutate_agent(self, agent_id: str, current_agents: List[BaseAgent], task: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Mutate an existing agent to create a variant."""
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
            
            # Generate mutation prompt
            prompt = await self.generate_prompt(task, {
                **context,
                "mutation_target": agent_to_mutate.get_info(),
                "focus_areas": self.focus_areas,
                "mutation_strength": self.mutation_strength
            })
            
            # Get mutation suggestions
            llm_response = await self.call_llm(prompt)
            mutation_suggestions = llm_response.get("content", "")
            
            # Create mutated agent configuration
            mutated_config = self._create_mutated_config(agent_to_mutate.config, mutation_suggestions)
            
            # Create new mutated agent
            mutated_agent = ProverAgent(
                config=mutated_config,
                agent_id=f"{agent_id}_mutated_{self.generation_count}"
            )
            
            return {
                "original_agent_id": agent_id,
                "mutated_agent_id": mutated_agent.agent_id,
                "mutation_type": "hyperparameter_adjustment",
                "suggestions": mutation_suggestions,
                "new_config": mutated_agent.get_info(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Agent mutation failed", agent_id=agent_id, error=str(e))
            return None
    
    async def _create_new_agent(self, agent_id: str, task: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a completely new agent."""
        try:
            # Generate new agent configuration based on task and current population
            prompt = f"""
Create a new agent configuration for the task: "{task}"

Current generation: {self.generation_count}
Focus areas: {', '.join(self.focus_areas)}

Generate a new agent with:
1. A descriptive name
2. A specialized prompt
3. Appropriate hyperparameters
4. Model selection (openai or anthropic)

The agent should complement the existing population and address any gaps in capabilities.
"""
            
            llm_response = await self.call_llm(prompt)
            config_text = llm_response.get("content", "")
            
            # Parse the configuration
            new_config = self._parse_agent_config(config_text, agent_id)
            
            # Create new agent
            new_agent = ProverAgent(
                config=new_config,
                agent_id=agent_id
            )
            
            return {
                "agent_id": new_agent.agent_id,
                "creation_type": "new_generation",
                "config_source": "llm_generated",
                "agent_info": new_agent.get_info(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("New agent creation failed", agent_id=agent_id, error=str(e))
            return None
    
    async def _crossover_agents(self, parent1_id: str, parent2_id: str, current_agents: List[BaseAgent], task: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform crossover between two parent agents."""
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
            
            # Generate crossover prompt
            prompt = f"""
Perform crossover between two parent agents for the task: "{task}"

Parent 1: {parent1.config.name}
- Prompt: {parent1.config.prompt}
- Temperature: {parent1.config.temperature}
- Hyperparameters: {parent1.hyperparameters}

Parent 2: {parent2.config.name}
- Prompt: {parent2.config.prompt}
- Temperature: {parent2.config.temperature}
- Hyperparameters: {parent2.hyperparameters}

Create a new agent that combines the best traits of both parents.
Use a balance ratio of {self.balance_ratio} (favor parent 1).
"""
            
            llm_response = await self.call_llm(prompt)
            crossover_text = llm_response.get("content", "")
            
            # Create crossover configuration
            crossover_config = self._create_crossover_config(parent1.config, parent2.config, crossover_text)
            
            # Create crossover agent
            crossover_agent = ProverAgent(
                config=crossover_config,
                agent_id=f"crossover_{parent1_id}_{parent2_id}_{self.generation_count}"
            )
            
            return {
                "parent1_id": parent1_id,
                "parent2_id": parent2_id,
                "crossover_agent_id": crossover_agent.agent_id,
                "crossover_type": "genetic_crossover",
                "balance_ratio": self.balance_ratio,
                "agent_info": crossover_agent.get_info(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Agent crossover failed", parent1=parent1_id, parent2=parent2_id, error=str(e))
            return None
    
    def _create_mutated_config(self, original_config: AgentConfig, mutation_suggestions: str) -> AgentConfig:
        """Create a mutated configuration based on suggestions."""
        # Adjust hyperparameters
        new_hyperparameters = original_config.hyperparameters.copy() if original_config.hyperparameters else {}
        
        # Apply mutation strength
        for key in new_hyperparameters:
            if isinstance(new_hyperparameters[key], (int, float)):
                mutation = random.uniform(-self.mutation_strength, self.mutation_strength)
                new_hyperparameters[key] = max(0.0, min(1.0, new_hyperparameters[key] + mutation))
        
        # Adjust temperature
        new_temperature = max(0.1, min(1.0, original_config.temperature + random.uniform(-0.2, 0.2)))
        
        return AgentConfig(
            name=f"{original_config.name}_mutated",
            prompt=original_config.prompt,  # Keep original prompt
            model=original_config.model,
            temperature=new_temperature,
            max_tokens=original_config.max_tokens,
            timeout=original_config.timeout,
            max_retries=original_config.max_retries,
            retry_delay=original_config.retry_delay,
            hyperparameters=new_hyperparameters
        )
    
    def _parse_agent_config(self, config_text: str, agent_id: str) -> AgentConfig:
        """Parse agent configuration from LLM response."""
        # Simple parsing - in production, this would be more sophisticated
        lines = config_text.split('\n')
        
        name = f"new_agent_{agent_id}"
        prompt = "You are a helpful assistant."
        model = "openai"
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
        
        return AgentConfig(
            name=name,
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            hyperparameters={"creativity": 0.7, "detail_level": 0.7}
        )
    
    def _create_crossover_config(self, config1: AgentConfig, config2: AgentConfig, crossover_text: str) -> AgentConfig:
        """Create crossover configuration from two parent configurations."""
        # Blend hyperparameters
        hyperparameters1 = config1.hyperparameters or {}
        hyperparameters2 = config2.hyperparameters or {}
        
        blended_hyperparameters = {}
        all_keys = set(hyperparameters1.keys()) | set(hyperparameters2.keys())
        
        for key in all_keys:
            val1 = hyperparameters1.get(key, 0.5)
            val2 = hyperparameters2.get(key, 0.5)
            blended_hyperparameters[key] = val1 * self.balance_ratio + val2 * (1 - self.balance_ratio)
        
        # Blend temperatures
        blended_temperature = config1.temperature * self.balance_ratio + config2.temperature * (1 - self.balance_ratio)
        
        # Blend prompts (simple approach)
        blended_prompt = f"{config1.prompt}\n\nAdditional context from crossover: {config2.prompt[:200]}..."
        
        return AgentConfig(
            name=f"crossover_{config1.name}_{config2.name}",
            prompt=blended_prompt,
            model=config1.model,  # Use parent1's model
            temperature=blended_temperature,
            max_tokens=max(config1.max_tokens, config2.max_tokens),
            hyperparameters=blended_hyperparameters
        )
    
    async def generate_prompt(self, task: str, context: Dict[str, Any] = None) -> str:
        """Generate the prompt for the LLM based on the task and context."""
        context = context or {}
        
        if "mutation_target" in context:
            # Mutation prompt
            return f"""
{self.config.prompt}

Target Agent: {context['mutation_target']['name']}
Current Performance: Trust Score = {context['mutation_target']['trust_score']}

Focus Areas: {', '.join(context['focus_areas'])}
Mutation Strength: {context['mutation_strength']}

Suggest specific improvements to this agent's configuration and approach.
Focus on enhancing performance in the weak areas while maintaining strengths.
"""
        
        elif "crossover_target" in context:
            # Crossover prompt
            return f"""
{self.config.prompt}

Perform genetic crossover between the provided parent agents.
Create a new agent that inherits the best traits from both parents.
Use the specified balance ratio to determine the inheritance weights.
"""
        
        else:
            # General refinement prompt
            return f"""
{self.config.prompt}

Task: {task}

Generation: {self.generation_count}

Analyze the current agent population and suggest improvements.
Consider mutation, crossover, and new agent creation strategies.
Focus on improving overall population performance and diversity.
"""
    
    async def call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the appropriate LLM based on the agent's configuration."""
        try:
            if self.config.model == "openai":
                return await self._call_openai(prompt)
            elif self.config.model == "anthropic":
                return await self._call_anthropic(prompt)
            else:
                raise ValueError(f"Unsupported model: {self.config.model}")
                
        except Exception as e:
            logger.error("LLM call failed", agent_id=self.agent_id, model=self.config.model, error=str(e))
            raise
    
    async def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI's GPT model."""
        if not self.openai_client:
            self.openai_client = openai.AsyncOpenAI()
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "confidence": 0.75,
                "reasoning": "Refinement analysis using OpenAI GPT-4",
                "model": "gpt-4"
            }
            
        except Exception as e:
            logger.error("OpenAI API call failed", error=str(e))
            raise
    
    async def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic's Claude model."""
        if not self.anthropic_client:
            self.anthropic_client = anthropic.AsyncAnthropic()
        
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": f"{self.config.prompt}\n\n{prompt}"}
                ]
            )
            
            content = response.content[0].text
            
            return {
                "content": content,
                "confidence": 0.75,
                "reasoning": "Refinement analysis using Anthropic Claude 3",
                "model": "claude-3-sonnet-20240229"
            }
            
        except Exception as e:
            logger.error("Anthropic API call failed", error=str(e))
            raise
    
    def get_refinement_stats(self) -> Dict[str, Any]:
        """Get statistics about the refiner's activities."""
        return {
            "generation_count": self.generation_count,
            "mutation_strength": self.mutation_strength,
            "focus_areas": self.focus_areas,
            "crossover_points": self.crossover_points,
            "balance_ratio": self.balance_ratio,
            "agents_tracked": len(self.agent_history)
        }