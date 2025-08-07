from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
import structlog
import yaml
import json
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import openai
import anthropic
from .base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus, CircuitState

logger = structlog.get_logger()


class EvaluationMetrics:
    """Tracks evaluation-specific metrics and analytics."""
    
    def __init__(self):
        self.total_evaluations = 0
        self.category_scores = {}
        self.task_type_distribution = {}
        self.evaluation_times = []
        self.consistency_scores = []
        self.inter_evaluator_agreement = []
    
    def update_metrics(self, evaluation_result: Dict[str, Any], execution_time: float):
        """Update evaluation metrics."""
        self.total_evaluations += 1
        self.evaluation_times.append(execution_time)
        
        # Track category scores
        category_scores = evaluation_result.get("category_scores", {})
        for category, data in category_scores.items():
            if category not in self.category_scores:
                self.category_scores[category] = []
            self.category_scores[category].append(data["score"])
        
        # Track task type distribution
        task_type = evaluation_result.get("task_type", "unknown")
        self.task_type_distribution[task_type] = self.task_type_distribution.get(task_type, 0) + 1
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation analytics."""
        return {
            "total_evaluations": self.total_evaluations,
            "average_evaluation_time": np.mean(self.evaluation_times) if self.evaluation_times else 0.0,
            "category_performance": {
                category: {
                    "average_score": np.mean(scores),
                    "std_dev": np.std(scores),
                    "count": len(scores)
                }
                for category, scores in self.category_scores.items()
            },
            "task_type_distribution": self.task_type_distribution,
            "consistency_score": np.mean(self.consistency_scores) if self.consistency_scores else 0.0
        }


class EvaluatorAgent(BaseAgent):
    """Enhanced agent that scores Prover outputs using advanced evaluation rubric."""
    
    def __init__(self, config: AgentConfig, agent_id: str = None, rubric_config: Dict[str, Any] = None):
        super().__init__(config, agent_id)
        self.rubric_config = rubric_config or {}
        self.evaluation_weights = config.hyperparameters or {}
        self.specialization = self._get_specialization()
        
        # Enhanced evaluation features
        self.evaluation_metrics = EvaluationMetrics()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.evaluation_cache = {}
        self.consistency_threshold = 0.1
        
        # Multi-evaluator consensus (if multiple evaluators are used)
        self.consensus_enabled = config.hyperparameters.get("consensus_enabled", False)
        self.consensus_threshold = config.hyperparameters.get("consensus_threshold", 0.2)
        
        logger.info("Enhanced evaluator initialized", 
                   agent_id=self.agent_id,
                   specialization=self.specialization,
                   consensus_enabled=self.consensus_enabled)
        
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Execute the evaluator agent with enhanced evaluation capabilities."""
        try:
            context = context or {}
            prover_output = context.get("prover_output")
            original_task = context.get("original_task", task)
            evaluation_mode = context.get("evaluation_mode", "standard")
            
            if not prover_output:
                raise ValueError("No prover output provided for evaluation")
            
            # Extract the best variant from prover output
            best_variant = prover_output.get("best_variant", {})
            solution_content = best_variant.get("content", "")
            
            if not solution_content:
                raise ValueError("No solution content found in prover output")
            
            # Perform comprehensive evaluation with different modes
            if evaluation_mode == "parallel":
                evaluation_result = await self._evaluate_solution_parallel(
                    original_task, solution_content, prover_output
                )
            elif evaluation_mode == "consensus":
                evaluation_result = await self._evaluate_solution_consensus(
                    original_task, solution_content, prover_output, context
                )
            else:
                evaluation_result = await self._evaluate_solution_enhanced(
                    original_task, solution_content, prover_output
                )
            
            # Update evaluation metrics
            execution_time = context.get("execution_time", 0.0)
            self.evaluation_metrics.update_metrics(evaluation_result, execution_time)
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.name,
                result=evaluation_result,
                confidence_score=evaluation_result.get("overall_score", 0.5),
                execution_time=execution_time,
                status=AgentStatus.COMPLETED,
                metadata={
                    "specialization": self.specialization,
                    "evaluation_weights": self.evaluation_weights,
                    "rubric_used": list(self.rubric_config.get("categories", {}).keys()),
                    "evaluation_mode": evaluation_mode,
                    "consensus_enabled": self.consensus_enabled,
                    "analytics": self.evaluation_metrics.get_analytics()
                }
            )
            
        except Exception as e:
            logger.error("Evaluator execution failed", agent_id=self.agent_id, error=str(e))
            raise
    
    async def _evaluate_solution_enhanced(self, task: str, solution: str, prover_output: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced evaluation with advanced scoring and analysis."""
        categories = self.rubric_config.get("categories", {})
        task_type = self._determine_task_type(task)
        
        # Get task-specific weights and criteria
        task_config = self.rubric_config.get("task_types", {}).get(task_type, {})
        task_weights = task_config.get("weights", {})
        specific_criteria = task_config.get("specific_criteria", {})
        
        # Evaluate each category with enhanced analysis
        category_evaluations = []
        for category_name, category_config in categories.items():
            weight = task_weights.get(category_name, category_config.get("weight", 0.2))
            
            # Enhanced category evaluation
            evaluation = await self._evaluate_category_enhanced(
                category_name, task, solution, category_config, specific_criteria.get(category_name, {})
            )
            
            category_evaluations.append({
                "category": category_name,
                "score": evaluation["score"],
                "weight": weight,
                "weighted_score": evaluation["score"] * weight,
                "feedback": evaluation["feedback"],
                "confidence": evaluation["confidence"],
                "sub_scores": evaluation.get("sub_scores", {})
            })
        
        # Calculate overall score with confidence weighting
        total_weighted_score = sum(eval["weighted_score"] for eval in category_evaluations)
        total_confidence = sum(eval["confidence"] for eval in category_evaluations)
        overall_score = total_weighted_score / len(category_evaluations) if category_evaluations else 0.0
        
        # Generate comprehensive feedback
        overall_feedback = await self._generate_overall_feedback_enhanced(
            task, solution, category_evaluations, prover_output
        )
        
        # Determine performance level with confidence
        performance_level = self._determine_performance_level_enhanced(overall_score, total_confidence)
        
        # Calculate evaluation consistency
        consistency_score = self._calculate_consistency_score(category_evaluations)
        
        return {
            "overall_score": overall_score,
            "performance_level": performance_level,
            "category_evaluations": category_evaluations,
            "overall_feedback": overall_feedback,
            "task_type": task_type,
            "consistency_score": consistency_score,
            "confidence_score": total_confidence / len(category_evaluations) if category_evaluations else 0.0,
            "evaluation_metadata": {
                "evaluator": self.config.name,
                "timestamp": datetime.now().isoformat(),
                "prover_agent": prover_output.get("prover_type", "unknown"),
                "num_variants": len(prover_output.get("variants", [])),
                "evaluation_mode": "enhanced",
                "consistency_score": consistency_score
            }
        }
    
    async def _evaluate_solution_parallel(self, task: str, solution: str, prover_output: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel evaluation of all categories for improved performance."""
        categories = self.rubric_config.get("categories", {})
        task_type = self._determine_task_type(task)
        task_weights = self.rubric_config.get("task_types", {}).get(task_type, {}).get("weights", {})
        
        # Create evaluation tasks for parallel execution
        evaluation_tasks = []
        for category_name, category_config in categories.items():
            weight = task_weights.get(category_name, category_config.get("weight", 0.2))
            task = asyncio.create_task(
                self._evaluate_category_enhanced(category_name, task, solution, category_config, {})
            )
            evaluation_tasks.append((category_name, weight, task))
        
        # Execute all evaluations in parallel
        results = await asyncio.gather(*[task for _, _, task in evaluation_tasks], return_exceptions=True)
        
        # Process results
        category_evaluations = []
        for i, (category_name, weight, _) in enumerate(evaluation_tasks):
            if isinstance(results[i], Exception):
                logger.error(f"Category evaluation failed", category=category_name, error=str(results[i]))
                # Use fallback evaluation
                fallback_eval = await self._evaluate_category_fallback(category_name, task, solution)
                category_evaluations.append({
                    "category": category_name,
                    "score": fallback_eval["score"],
                    "weight": weight,
                    "weighted_score": fallback_eval["score"] * weight,
                    "feedback": fallback_eval["feedback"],
                    "confidence": 0.5,
                    "error": str(results[i])
                })
            else:
                evaluation = results[i]
                category_evaluations.append({
                    "category": category_name,
                    "score": evaluation["score"],
                    "weight": weight,
                    "weighted_score": evaluation["score"] * weight,
                    "feedback": evaluation["feedback"],
                    "confidence": evaluation["confidence"]
                })
        
        # Calculate overall score
        overall_score = sum(eval["weighted_score"] for eval in category_evaluations) / len(category_evaluations) if category_evaluations else 0.0
        
        return {
            "overall_score": overall_score,
            "performance_level": self._determine_performance_level_enhanced(overall_score, 0.8),
            "category_evaluations": category_evaluations,
            "task_type": task_type,
            "evaluation_mode": "parallel",
            "parallel_execution": True
        }
    
    async def _evaluate_solution_consensus(self, task: str, solution: str, prover_output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-evaluator consensus evaluation."""
        if not self.consensus_enabled:
            return await self._evaluate_solution_enhanced(task, solution, prover_output)
        
        # Get other evaluators from context
        other_evaluators = context.get("other_evaluators", [])
        if not other_evaluators:
            return await self._evaluate_solution_enhanced(task, solution, prover_output)
        
        # Perform evaluations with all evaluators
        evaluations = []
        
        # Self evaluation
        self_eval = await self._evaluate_solution_enhanced(task, solution, prover_output)
        evaluations.append(self_eval)
        
        # Other evaluators (simulated for now)
        for evaluator in other_evaluators[:2]:  # Limit to 2 additional evaluators
            other_eval = await self._simulate_other_evaluator(evaluator, task, solution, prover_output)
            evaluations.append(other_eval)
        
        # Calculate consensus
        consensus_result = self._calculate_consensus(evaluations)
        
        return consensus_result
    
    async def _evaluate_category_enhanced(self, category_name: str, task: str, solution: str, 
                                        category_config: Dict[str, Any], specific_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced category evaluation with sub-criteria and confidence scoring."""
        description = category_config.get("description", "")
        levels = category_config.get("levels", {})
        sub_criteria = category_config.get("sub_criteria", {})
        
        # Generate enhanced evaluation prompt
        prompt = self._generate_enhanced_evaluation_prompt(
            category_name, task, solution, description, levels, sub_criteria, specific_criteria
        )
        
        # Call LLM for evaluation
        llm_response = await self.call_llm(prompt, {"category": category_name})
        
        # Parse the enhanced response
        parsed_result = self._parse_enhanced_evaluation_response(llm_response.get("content", ""), levels, sub_criteria)
        
        return parsed_result
    
    def _generate_enhanced_evaluation_prompt(self, category_name: str, task: str, solution: str, 
                                           description: str, levels: Dict[str, str], 
                                           sub_criteria: Dict[str, Any], specific_criteria: Dict[str, Any]) -> str:
        """Generate enhanced evaluation prompt with detailed criteria."""
        prompt = f"""
You are evaluating a solution for the following task:

Task: {task}

Solution: {solution}

Evaluation Category: {category_name}
Description: {description}

Evaluation Levels:
{self._format_levels(levels)}

Sub-Criteria (with weights):
{self._format_sub_criteria(sub_criteria)}

Specific Criteria for this task type:
{self._format_specific_criteria(specific_criteria)}

Please provide your evaluation in the following JSON format:
{{
    "overall_score": 0.85,
    "level": "excellent",
    "reasoning": "detailed explanation",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "confidence": 0.9,
    "sub_scores": {{
        "sub_criteria1": 0.8,
        "sub_criteria2": 0.9
    }}
}}

Be objective, thorough, and provide specific examples from the solution.
"""
        return prompt
    
    def _format_sub_criteria(self, sub_criteria: Dict[str, Any]) -> str:
        """Format sub-criteria for the prompt."""
        if not sub_criteria:
            return "No specific sub-criteria provided."
        
        formatted = []
        for criterion, details in sub_criteria.items():
            weight = details.get("weight", 0.1)
            description = details.get("description", "")
            formatted.append(f"- {criterion} (weight: {weight}): {description}")
        
        return "\n".join(formatted)
    
    def _format_specific_criteria(self, specific_criteria: Dict[str, Any]) -> str:
        """Format task-specific criteria."""
        if not specific_criteria:
            return "No task-specific criteria provided."
        
        formatted = []
        for criterion, details in specific_criteria.items():
            description = details.get("description", "")
            formatted.append(f"- {criterion}: {description}")
        
        return "\n".join(formatted)
    
    def _parse_enhanced_evaluation_response(self, response: str, levels: Dict[str, str], 
                                          sub_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Parse enhanced evaluation response with JSON structure."""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                parsed = json.loads(response)
                score = float(parsed.get("overall_score", 0.5))
                confidence = float(parsed.get("confidence", 0.8))
                
                return {
                    "score": min(1.0, max(0.0, score)),
                    "feedback": {
                        "reasoning": parsed.get("reasoning", ""),
                        "strengths": parsed.get("strengths", []),
                        "weaknesses": parsed.get("weaknesses", []),
                        "suggestions": parsed.get("suggestions", []),
                        "level": parsed.get("level", "fair")
                    },
                    "confidence": confidence,
                    "sub_scores": parsed.get("sub_scores", {})
                }
            
            # Fallback to old parsing method
            return self._parse_evaluation_response_fallback(response, levels)
            
        except Exception as e:
            logger.warning("Failed to parse enhanced evaluation response", error=str(e))
            return self._parse_evaluation_response_fallback(response, levels)
    
    def _parse_evaluation_response_fallback(self, response: str, levels: Dict[str, str]) -> Dict[str, Any]:
        """Fallback parsing method for evaluation responses."""
        try:
            # Extract score
            score_match = None
            for line in response.split('\n'):
                if 'Score (0.0-1.0):' in line:
                    score_match = line.split(':')[1].strip()
                    break
            
            if score_match:
                score = float(score_match)
            else:
                score = self._estimate_score_from_response(response, levels)
            
            return {
                "score": min(1.0, max(0.0, score)),
                "feedback": {
                    "raw_response": response,
                    "parsed_score": score,
                    "evaluation_text": response
                },
                "confidence": 0.7
            }
            
        except Exception as e:
            logger.warning("Failed to parse evaluation response", error=str(e))
            return {
                "score": 0.5,
                "feedback": {"raw_response": response, "parse_error": str(e)},
                "confidence": 0.5
            }
    
    async def _evaluate_category_fallback(self, category_name: str, task: str, solution: str) -> Dict[str, Any]:
        """Fallback evaluation when primary evaluation fails."""
        return {
            "score": 0.5,
            "feedback": {
                "reasoning": "Fallback evaluation due to error",
                "strengths": ["Solution provided"],
                "weaknesses": ["Evaluation failed"],
                "suggestions": ["Improve evaluation system"]
            },
            "confidence": 0.3
        }
    
    async def _simulate_other_evaluator(self, evaluator_info: Dict[str, Any], task: str, solution: str, 
                                       prover_output: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate evaluation from another evaluator for consensus."""
        # This would integrate with actual other evaluators
        # For now, simulate with slight variations
        base_eval = await self._evaluate_solution_enhanced(task, solution, prover_output)
        
        # Add some variation to simulate different evaluator
        variation = np.random.normal(0, 0.1)  # ±10% variation
        base_eval["overall_score"] = max(0.0, min(1.0, base_eval["overall_score"] + variation))
        
        return base_eval
    
    def _calculate_consensus(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus from multiple evaluations."""
        if not evaluations:
            return {}
        
        scores = [eval.get("overall_score", 0.0) for eval in evaluations]
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Check for consensus
        consensus_achieved = std_score <= self.consensus_threshold
        
        # If no consensus, use weighted average or require human review
        if not consensus_achieved:
            logger.warning("No consensus achieved", std_score=std_score, threshold=self.consensus_threshold)
        
        return {
            "overall_score": avg_score,
            "consensus_achieved": consensus_achieved,
            "consensus_std": std_score,
            "individual_evaluations": evaluations,
            "evaluation_mode": "consensus"
        }
    
    def _calculate_consistency_score(self, category_evaluations: List[Dict[str, Any]]) -> float:
        """Calculate consistency score across category evaluations."""
        if len(category_evaluations) < 2:
            return 1.0
        
        scores = [eval["score"] for eval in category_evaluations]
        std_dev = np.std(scores)
        
        # Higher consistency = lower standard deviation
        consistency = max(0.0, 1.0 - (std_dev * 2))
        
        return consistency
    
    def _determine_performance_level_enhanced(self, score: float, confidence: float) -> str:
        """Enhanced performance level determination with confidence."""
        if score >= 0.8 and confidence >= 0.8:
            return "excellent"
        elif score >= 0.6 and confidence >= 0.6:
            return "good"
        elif score >= 0.4 and confidence >= 0.4:
            return "fair"
        else:
            return "poor"
    
    async def _generate_overall_feedback_enhanced(self, task: str, solution: str, 
                                                category_evaluations: List[Dict[str, Any]], 
                                                prover_output: Dict[str, Any]) -> str:
        """Generate enhanced overall feedback with detailed analysis."""
        # Identify strongest and weakest categories
        sorted_evaluations = sorted(category_evaluations, key=lambda x: x["score"], reverse=True)
        strongest = sorted_evaluations[0] if sorted_evaluations else None
        weakest = sorted_evaluations[-1] if sorted_evaluations else None
        
        # Calculate overall statistics
        avg_score = sum(eval["score"] for eval in category_evaluations) / len(category_evaluations) if category_evaluations else 0.0
        consistency = self._calculate_consistency_score(category_evaluations)
        
        prompt = f"""
Based on the comprehensive evaluation of a solution for the task "{task}", provide detailed overall feedback:

Overall Score: {avg_score:.2f}
Consistency Score: {consistency:.2f}

Strongest Category: {strongest["category"]} (score: {strongest["score"]:.2f})
Weakest Category: {weakest["category"]} (score: {weakest["score"]:.2f})

Category Breakdown:
{self._format_category_breakdown(category_evaluations)}

Please provide:
1. Overall assessment of the solution quality
2. Key strengths and areas for improvement
3. Specific recommendations for enhancement
4. Confidence in the solution's effectiveness
5. Suggestions for addressing the weakest areas
6. Assessment of solution consistency across categories

Keep your feedback constructive, actionable, and specific.
"""
        
        llm_response = await self.call_llm(prompt, {"feedback_type": "overall"})
        return llm_response.get("content", "No feedback generated.")
    
    def _format_category_breakdown(self, category_evaluations: List[Dict[str, Any]]) -> str:
        """Format category breakdown for feedback."""
        breakdown = []
        for eval in category_evaluations:
            breakdown.append(f"- {eval['category']}: {eval['score']:.2f} (weight: {eval['weight']:.2f})")
        return "\n".join(breakdown)
    
    def _determine_task_type(self, task: str) -> str:
        """Enhanced task type determination with more categories."""
        task_lower = task.lower()
        
        # Enhanced keyword matching
        task_keywords = {
            "problem_solving": ["solve", "problem", "calculate", "compute", "analyze", "find"],
            "creative_writing": ["write", "story", "creative", "narrative", "compose", "generate"],
            "technical_design": ["design", "architecture", "system", "technical", "implement", "build"],
            "strategic_planning": ["plan", "strategy", "roadmap", "approach", "framework", "methodology"],
            "data_analysis": ["analyze", "data", "statistics", "chart", "graph", "visualization"],
            "code_generation": ["code", "program", "function", "algorithm", "script", "implementation"]
        }
        
        for task_type, keywords in task_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                return task_type
        
        return "general"
    
    async def _call_llm_implementation(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced LLM calling implementation for evaluators."""
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
        """Enhanced OpenAI calling with better error handling and response processing."""
        client = await self._llm_manager.get_client("openai", {"timeout": self.config.timeout})
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}  # Request JSON for better parsing
            )
            
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "confidence": 0.85,  # Evaluators have high confidence
                "reasoning": "Evaluation performed using OpenAI GPT-4",
                "model": "gpt-4",
                "usage": response.usage.dict() if hasattr(response, 'usage') else {}
            }
            
        except Exception as e:
            logger.error("OpenAI API call failed", error=str(e))
            raise
    
    async def _call_anthropic_enhanced(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Anthropic calling with better error handling."""
        client = await self._llm_manager.get_client("anthropic", {"timeout": self.config.timeout})
        
        try:
            response = await client.messages.create(
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
                "confidence": 0.85,  # Evaluators have high confidence
                "reasoning": "Evaluation performed using Anthropic Claude 3",
                "model": "claude-3-sonnet-20240229",
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error("Anthropic API call failed", error=str(e))
            raise
    
    def _get_specialization(self) -> str:
        """Get the evaluator's specialization based on its name."""
        name_lower = self.config.name.lower()
        if "quality" in name_lower:
            return "Quality assessment"
        elif "creativity" in name_lower:
            return "Creativity evaluation"
        elif "feasibility" in name_lower:
            return "Feasibility analysis"
        elif "technical" in name_lower:
            return "Technical evaluation"
        else:
            return "General evaluation"
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        """Get the evaluation criteria used by this evaluator."""
        return {
            "specialization": self.specialization,
            "categories": list(self.rubric_config.get("categories", {}).keys()),
            "weights": self.evaluation_weights,
            "scoring_scale": self.rubric_config.get("scoring", {}),
            "consensus_enabled": self.consensus_enabled,
            "consensus_threshold": self.consensus_threshold,
            "analytics": self.evaluation_metrics.get_analytics()
        }
    
    async def generate_prompt(self, task: str, context: Dict[str, Any] = None) -> str:
        """Generate the prompt for the LLM based on the task and context."""
        # For evaluator, this is mainly used for category-specific evaluations
        return f"{self.config.prompt}\n\nTask: {task}"