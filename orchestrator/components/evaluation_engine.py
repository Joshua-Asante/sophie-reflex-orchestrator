"""
Evaluation Engine Component

Handles solution evaluation using evaluator agents.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import logging

logger = structlog.get_logger()


class TaskType(Enum):
    ENGINEERING = "engineering"
    RESEARCH = "research"
    GENERAL = "general"
    SECURITY = "security"
    CREATIVE = "creative"

@dataclass
class ExpertProfile:
    """Enhanced expert profile with detailed capabilities."""
    id: str
    name: str
    task_type: TaskType
    capabilities: List[str]
    performance_history: List[float]
    trust_score: float
    response_time_avg: float
    success_rate: float
    specializations: List[str]
    availability: bool = True

class EnhancedGeneticAlgorithm:
    """Enhanced genetic algorithm for expert selection with sophisticated fitness function."""
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # Enhanced fitness weights for different task types
        self.fitness_weights = {
            TaskType.ENGINEERING: {
                'technical_expertise': 0.3,
                'code_quality': 0.25,
                'problem_solving': 0.2,
                'response_time': 0.15,
                'trust_score': 0.1
            },
            TaskType.RESEARCH: {
                'research_depth': 0.3,
                'source_accuracy': 0.25,
                'analysis_quality': 0.2,
                'citation_quality': 0.15,
                'trust_score': 0.1
            },
            TaskType.SECURITY: {
                'security_expertise': 0.35,
                'compliance_knowledge': 0.25,
                'audit_quality': 0.2,
                'response_time': 0.1,
                'trust_score': 0.1
            },
            TaskType.CREATIVE: {
                'creativity_score': 0.3,
                'innovation_ability': 0.25,
                'quality_output': 0.2,
                'diversity': 0.15,
                'trust_score': 0.1
            },
            TaskType.GENERAL: {
                'versatility': 0.25,
                'communication': 0.25,
                'problem_solving': 0.2,
                'response_time': 0.15,
                'trust_score': 0.15
            }
        }
    
    def calculate_fitness(self, expert: ExpertProfile, task_type: TaskType, 
                         task_requirements: Dict[str, Any]) -> float:
        """
        Enhanced fitness function with task-specific optimization.
        
        Args:
            expert: Expert profile to evaluate
            task_type: Type of task being performed
            task_requirements: Specific requirements for the task
            
        Returns:
            Fitness score between 0 and 1
        """
        weights = self.fitness_weights[task_type]
        fitness_score = 0.0
        
        # Base fitness from expert profile
        if task_type == TaskType.ENGINEERING:
            fitness_score += self._calculate_engineering_fitness(expert, weights, task_requirements)
        elif task_type == TaskType.RESEARCH:
            fitness_score += self._calculate_research_fitness(expert, weights, task_requirements)
        elif task_type == TaskType.SECURITY:
            fitness_score += self._calculate_security_fitness(expert, weights, task_requirements)
        elif task_type == TaskType.CREATIVE:
            fitness_score += self._calculate_creative_fitness(expert, weights, task_requirements)
        else:  # GENERAL
            fitness_score += self._calculate_general_fitness(expert, weights, task_requirements)
        
        # Apply availability penalty
        if not expert.availability:
            fitness_score *= 0.5
        
        # Apply trust score bonus/penalty
        trust_multiplier = 1.0 + (expert.trust_score - 0.5) * 0.4
        fitness_score *= trust_multiplier
        
        return max(0.0, min(1.0, fitness_score))
    
    def _calculate_engineering_fitness(self, expert: ExpertProfile, weights: Dict[str, float], 
                                     requirements: Dict[str, Any]) -> float:
        """Calculate fitness for engineering tasks."""
        fitness = 0.0
        
        # Technical expertise based on capabilities
        tech_capabilities = ['code_generation', 'system_design', 'architecture', 'optimization']
        tech_score = sum(1 for cap in tech_capabilities if cap in expert.capabilities) / len(tech_capabilities)
        fitness += tech_score * weights['technical_expertise']
        
        # Code quality based on performance history
        if expert.performance_history:
            quality_score = np.mean(expert.performance_history[-10:])  # Last 10 performances
            fitness += quality_score * weights['code_quality']
        
        # Problem solving based on success rate
        fitness += expert.success_rate * weights['problem_solving']
        
        # Response time (inverse relationship)
        response_score = max(0, 1 - (expert.response_time_avg / 30))  # 30s baseline
        fitness += response_score * weights['response_time']
        
        # Trust score
        fitness += expert.trust_score * weights['trust_score']
        
        return fitness
    
    def _calculate_research_fitness(self, expert: ExpertProfile, weights: Dict[str, float], 
                                  requirements: Dict[str, Any]) -> float:
        """Calculate fitness for research tasks."""
        fitness = 0.0
        
        # Research depth based on specializations
        research_capabilities = ['deep_research', 'source_analysis', 'citation_generation', 'fact_checking']
        depth_score = sum(1 for cap in research_capabilities if cap in expert.capabilities) / len(research_capabilities)
        fitness += depth_score * weights['research_depth']
        
        # Source accuracy based on performance history
        if expert.performance_history:
            accuracy_score = np.mean(expert.performance_history[-5:])
            fitness += accuracy_score * weights['source_accuracy']
        
        # Analysis quality
        analysis_score = expert.success_rate
        fitness += analysis_score * weights['analysis_quality']
        
        # Citation quality
        citation_score = 0.8 if 'citation_generation' in expert.capabilities else 0.4
        fitness += citation_score * weights['citation_quality']
        
        # Trust score
        fitness += expert.trust_score * weights['trust_score']
        
        return fitness
    
    def _calculate_security_fitness(self, expert: ExpertProfile, weights: Dict[str, float], 
                                  requirements: Dict[str, Any]) -> float:
        """Calculate fitness for security tasks."""
        fitness = 0.0
        
        # Security expertise
        security_capabilities = ['vulnerability_assessment', 'compliance_audit', 'encryption', 'access_control']
        security_score = sum(1 for cap in security_capabilities if cap in expert.capabilities) / len(security_capabilities)
        fitness += security_score * weights['security_expertise']
        
        # Compliance knowledge
        compliance_score = 0.9 if 'compliance_audit' in expert.capabilities else 0.5
        fitness += compliance_score * weights['compliance_knowledge']
        
        # Audit quality
        audit_score = expert.success_rate
        fitness += audit_score * weights['audit_quality']
        
        # Response time (critical for security)
        response_score = max(0, 1 - (expert.response_time_avg / 15))  # 15s baseline for security
        fitness += response_score * weights['response_time']
        
        # Trust score (critical for security)
        fitness += expert.trust_score * weights['trust_score']
        
        return fitness
    
    def _calculate_creative_fitness(self, expert: ExpertProfile, weights: Dict[str, float], 
                                  requirements: Dict[str, Any]) -> float:
        """Calculate fitness for creative tasks."""
        fitness = 0.0
        
        # Creativity score based on specializations
        creative_capabilities = ['creative_writing', 'design_thinking', 'innovation', 'artistic_expression']
        creativity_score = sum(1 for cap in creative_capabilities if cap in expert.capabilities) / len(creative_capabilities)
        fitness += creativity_score * weights['creativity_score']
        
        # Innovation ability
        innovation_score = 0.8 if 'innovation' in expert.capabilities else 0.4
        fitness += innovation_score * weights['innovation_ability']
        
        # Quality output
        quality_score = expert.success_rate
        fitness += quality_score * weights['quality_output']
        
        # Diversity (encourages different approaches)
        diversity_score = len(expert.specializations) / 10  # Normalize to 0-1
        fitness += diversity_score * weights['diversity']
        
        # Trust score
        fitness += expert.trust_score * weights['trust_score']
        
        return fitness
    
    def _calculate_general_fitness(self, expert: ExpertProfile, weights: Dict[str, float], 
                                 requirements: Dict[str, Any]) -> float:
        """Calculate fitness for general tasks."""
        fitness = 0.0
        
        # Versatility based on number of capabilities
        versatility_score = min(1.0, len(expert.capabilities) / 10)
        fitness += versatility_score * weights['versatility']
        
        # Communication based on success rate
        communication_score = expert.success_rate
        fitness += communication_score * weights['communication']
        
        # Problem solving
        problem_solving_score = expert.success_rate
        fitness += problem_solving_score * weights['problem_solving']
        
        # Response time
        response_score = max(0, 1 - (expert.response_time_avg / 45))  # 45s baseline for general
        fitness += response_score * weights['response_time']
        
        # Trust score
        fitness += expert.trust_score * weights['trust_score']
        
        return fitness
    
    def optimize_expert_selection(self, experts: List[ExpertProfile], task_type: TaskType,
                                task_requirements: Dict[str, Any], num_experts: int = 3) -> List[ExpertProfile]:
        """
        Optimize expert selection using enhanced genetic algorithm.
        
        Args:
            experts: List of available experts
            task_type: Type of task to perform
            task_requirements: Specific requirements for the task
            num_experts: Number of experts to select
            
        Returns:
            List of selected experts ordered by fitness
        """
        if len(experts) <= num_experts:
            return sorted(experts, key=lambda e: self.calculate_fitness(e, task_type, task_requirements), reverse=True)
        
        # Initialize population with expert combinations
        population = self._initialize_population(experts, num_experts)
        
        # Evolve population
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = sum(self.calculate_fitness(expert, task_type, task_requirements) 
                            for expert in individual)
                fitness_scores.append(fitness)
            
            # Sort by fitness
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
            
            # Select elite
            elite = sorted_population[:self.elite_size]
            
            # Generate new population
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(sorted_population, fitness_scores)
                parent2 = self._tournament_selection(sorted_population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2, experts, num_experts)
                else:
                    child = parent1.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, experts)
                
                new_population.append(child)
            
            population = new_population
            
            # Log progress
            if generation % 20 == 0:
                best_fitness = max(fitness_scores)
                logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Return best individual
        best_individual = max(population, key=lambda ind: sum(
            self.calculate_fitness(expert, task_type, task_requirements) for expert in ind))
        
        return best_individual
    
    def _initialize_population(self, experts: List[ExpertProfile], num_experts: int) -> List[List[ExpertProfile]]:
        """Initialize population with random expert combinations."""
        population = []
        for _ in range(self.population_size):
            individual = random.sample(experts, min(num_experts, len(experts)))
            population.append(individual)
        return population
    
    def _tournament_selection(self, population: List[List[ExpertProfile]], 
                            fitness_scores: List[float], tournament_size: int = 3) -> List[ExpertProfile]:
        """Tournament selection for genetic algorithm."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_index]
    
    def _crossover(self, parent1: List[ExpertProfile], parent2: List[ExpertProfile], 
                  all_experts: List[ExpertProfile], num_experts: int) -> List[ExpertProfile]:
        """Crossover operation for genetic algorithm."""
        # Uniform crossover
        child = []
        for i in range(num_experts):
            if random.random() < 0.5 and i < len(parent1):
                child.append(parent1[i])
            elif i < len(parent2):
                child.append(parent2[i])
            else:
                # Fill with random expert if needed
                available_experts = [e for e in all_experts if e not in child]
                if available_experts:
                    child.append(random.choice(available_experts))
        
        return child
    
    def _mutate(self, individual: List[ExpertProfile], all_experts: List[ExpertProfile]) -> List[ExpertProfile]:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        
        # Randomly replace one expert
        if random.random() < 0.3:  # 30% chance of mutation
            mutation_index = random.randint(0, len(mutated) - 1)
            available_experts = [e for e in all_experts if e not in mutated]
            if available_experts:
                mutated[mutation_index] = random.choice(available_experts)
        
        return mutated

# Enhanced evaluation engine that uses the genetic algorithm
class EnhancedEvaluationEngine:
    """Enhanced evaluation engine with genetic algorithm optimization."""
    
    def __init__(self):
        self.genetic_algorithm = EnhancedGeneticAlgorithm()
        self.expert_profiles = self._initialize_expert_profiles()
    
    def _initialize_expert_profiles(self) -> List[ExpertProfile]:
        """Initialize expert profiles with realistic data."""
        profiles = [
            ExpertProfile(
                id="glm_4_5",
                name="GLM 4.5",
                task_type=TaskType.ENGINEERING,
                capabilities=["code_generation", "system_design", "architecture", "optimization", "creative_solutions"],
                performance_history=[0.92, 0.89, 0.94, 0.91, 0.93],
                trust_score=0.95,
                response_time_avg=2.5,
                success_rate=0.93,
                specializations=["React", "TypeScript", "Docker", "Microservices"]
            ),
            ExpertProfile(
                id="claude_3_5_sonnet",
                name="Claude 3.5 Sonnet",
                task_type=TaskType.SECURITY,
                capabilities=["vulnerability_assessment", "compliance_audit", "encryption", "access_control", "audit_logging"],
                performance_history=[0.88, 0.91, 0.89, 0.92, 0.90],
                trust_score=0.98,
                response_time_avg=3.2,
                success_rate=0.90,
                specializations=["GDPR", "HIPAA", "SOX", "Security"]
            ),
            ExpertProfile(
                id="gemini_pro",
                name="Gemini Pro",
                task_type=TaskType.RESEARCH,
                capabilities=["deep_research", "source_analysis", "citation_generation", "fact_checking", "long_context"],
                performance_history=[0.85, 0.87, 0.89, 0.86, 0.88],
                trust_score=0.87,
                response_time_avg=4.1,
                success_rate=0.88,
                specializations=["Research", "Citations", "Analysis", "Fact-checking"]
            )
        ]
        return profiles
    
    def select_optimal_experts(self, task_type: TaskType, task_requirements: Dict[str, Any], 
                              num_experts: int = 3) -> List[ExpertProfile]:
        """
        Select optimal experts using genetic algorithm optimization.
        
        Args:
            task_type: Type of task to perform
            task_requirements: Specific requirements for the task
            num_experts: Number of experts to select
            
        Returns:
            List of selected experts ordered by fitness
        """
        # Filter experts by task type and availability
        available_experts = [e for e in self.expert_profiles 
                           if e.task_type == task_type and e.availability]
        
        if not available_experts:
            # Fallback to general experts
            available_experts = [e for e in self.expert_profiles if e.availability]
        
        # Use genetic algorithm to optimize selection
        selected_experts = self.genetic_algorithm.optimize_expert_selection(
            available_experts, task_type, task_requirements, num_experts)
        
        # Log selection results
        logger.info(f"Selected experts for {task_type.value} task:")
        for i, expert in enumerate(selected_experts):
            fitness = self.genetic_algorithm.calculate_fitness(expert, task_type, task_requirements)
            logger.info(f"  {i+1}. {expert.name} (Fitness: {fitness:.4f})")
        
        return selected_experts
    
    def update_expert_performance(self, expert_id: str, performance_score: float, 
                                response_time: float, success: bool):
        """Update expert performance metrics."""
        for expert in self.expert_profiles:
            if expert.id == expert_id:
                expert.performance_history.append(performance_score)
                expert.performance_history = expert.performance_history[-10:]  # Keep last 10
                
                # Update response time (exponential moving average)
                alpha = 0.3
                expert.response_time_avg = (alpha * response_time + 
                                          (1 - alpha) * expert.response_time_avg)
                
                # Update success rate
                if len(expert.performance_history) > 0:
                    expert.success_rate = np.mean(expert.performance_history)
                
                break 

# Backward compatibility alias
EvaluationEngine = EnhancedEvaluationEngine 