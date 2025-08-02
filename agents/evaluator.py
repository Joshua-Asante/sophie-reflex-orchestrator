from typing import Dict, Any, List, Optional, Tuple
import asyncio
import structlog
import yaml
from datetime import datetime
import openai
import anthropic
from .base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus

logger = structlog.get_logger()


class EvaluatorAgent(BaseAgent):
    """Agent that scores Prover outputs using evaluation rubric."""
    
    def __init__(self, config: AgentConfig, agent_id: str = None, rubric_config: Dict[str, Any] = None):
        super().__init__(config, agent_id)
        self.rubric_config = rubric_config or {}
        self.evaluation_weights = config.hyperparameters or {}
        self.specialization = self._get_specialization()
        
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Execute the evaluator agent to score prover outputs."""
        try:
            context = context or {}
            prover_output = context.get("prover_output")
            original_task = context.get("original_task", task)
            
            if not prover_output:
                raise ValueError("No prover output provided for evaluation")
            
            # Extract the best variant from prover output
            best_variant = prover_output.get("best_variant", {})
            solution_content = best_variant.get("content", "")
            
            if not solution_content:
                raise ValueError("No solution content found in prover output")
            
            # Perform comprehensive evaluation
            evaluation_result = await self._evaluate_solution(
                original_task, solution_content, prover_output
            )
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.name,
                result=evaluation_result,
                confidence_score=evaluation_result.get("overall_score", 0.5),
                execution_time=0.0,  # Will be set by execute_with_retry
                status=AgentStatus.COMPLETED,
                metadata={
                    "specialization": self.specialization,
                    "evaluation_weights": self.evaluation_weights,
                    "rubric_used": list(self.rubric_config.get("categories", {}).keys())
                }
            )
            
        except Exception as e:
            logger.error("Evaluator execution failed", agent_id=self.agent_id, error=str(e))
            raise
    
    async def _evaluate_solution(self, task: str, solution: str, prover_output: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the solution."""
        categories = self.rubric_config.get("categories", {})
        task_type = self._determine_task_type(task)
        
        # Get task-specific weights if available
        task_weights = self.rubric_config.get("task_types", {}).get(task_type, {}).get("weights", {})
        
        # Evaluate each category
        category_scores = {}
        category_feedback = {}
        
        for category_name, category_config in categories.items():
            # Use task-specific weight if available, otherwise use default
            weight = task_weights.get(category_name, category_config.get("weight", 0.2))
            
            # Evaluate this category
            score, feedback = await self._evaluate_category(
                category_name, task, solution, category_config
            )
            
            category_scores[category_name] = {
                "score": score,
                "weight": weight,
                "weighted_score": score * weight
            }
            category_feedback[category_name] = feedback
        
        # Calculate overall score
        overall_score = sum(item["weighted_score"] for item in category_scores.values())
        
        # Generate overall feedback
        overall_feedback = await self._generate_overall_feedback(
            task, solution, category_scores, category_feedback
        )
        
        # Determine performance level
        performance_level = self._determine_performance_level(overall_score)
        
        return {
            "overall_score": overall_score,
            "performance_level": performance_level,
            "category_scores": category_scores,
            "category_feedback": category_feedback,
            "overall_feedback": overall_feedback,
            "task_type": task_type,
            "evaluation_metadata": {
                "evaluator": self.config.name,
                "timestamp": datetime.now().isoformat(),
                "prover_agent": prover_output.get("prover_type", "unknown"),
                "num_variants": len(prover_output.get("variants", []))
            }
        }
    
    async def _evaluate_category(self, category_name: str, task: str, solution: str, category_config: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate a specific category of the solution."""
        description = category_config.get("description", "")
        levels = category_config.get("levels", {})
        
        # Generate evaluation prompt
        prompt = f"""
You are evaluating a solution for the following task:

Task: {task}

Solution: {solution}

Evaluation Category: {category_name}
Description: {description}

Please evaluate the solution based on the following levels:
{self._format_levels(levels)}

Provide your evaluation in the following format:
1. Score (0.0-1.0): [numerical score]
2. Level: [excellent/good/fair/poor]
3. Reasoning: [detailed explanation of your evaluation]
4. Strengths: [list of strengths in this category]
5. Weaknesses: [list of weaknesses in this category]
6. Suggestions: [specific suggestions for improvement]

Be objective and thorough in your evaluation.
"""
        
        # Call LLM for evaluation
        llm_response = await self.call_llm(prompt)
        
        # Parse the response
        score, feedback = self._parse_evaluation_response(llm_response.get("content", ""), levels)
        
        return score, feedback
    
    def _format_levels(self, levels: Dict[str, str]) -> str:
        """Format evaluation levels for the prompt."""
        formatted = []
        for level, description in levels.items():
            formatted.append(f"- {level.title()}: {description}")
        return "\n".join(formatted)
    
    def _parse_evaluation_response(self, response: str, levels: Dict[str, str]) -> Tuple[float, str]:
        """Parse the LLM evaluation response."""
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
                # Fallback: estimate score from level
                score = self._estimate_score_from_response(response, levels)
            
            # Generate feedback
            feedback = {
                "raw_response": response,
                "parsed_score": score,
                "evaluation_text": response
            }
            
            return min(1.0, max(0.0, score)), feedback
            
        except Exception as e:
            logger.warning("Failed to parse evaluation response", error=str(e))
            return 0.5, {"raw_response": response, "parse_error": str(e)}
    
    def _estimate_score_from_response(self, response: str, levels: Dict[str, str]) -> float:
        """Estimate score from response when explicit score is not found."""
        response_lower = response.lower()
        
        # Look for level indicators
        if "excellent" in response_lower:
            return 0.9
        elif "good" in response_lower:
            return 0.7
        elif "fair" in response_lower:
            return 0.5
        elif "poor" in response_lower:
            return 0.3
        else:
            return 0.6  # Default middle score
    
    async def _generate_overall_feedback(self, task: str, solution: str, category_scores: Dict[str, Any], category_feedback: Dict[str, Any]) -> str:
        """Generate overall feedback for the solution."""
        # Identify strongest and weakest categories
        categories_by_score = sorted(
            category_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        strongest = categories_by_score[0] if categories_by_score else None
        weakest = categories_by_score[-1] if categories_by_score else None
        
        prompt = f"""
Based on the evaluation of a solution for the task "{task}", provide overall feedback:

Strongest Category: {strongest[0]} (score: {strongest[1]['score']:.2f})
Weakest Category: {weakest[0]} (score: {weakest[1]['score']:.2f})

Overall Score: {sum(item['weighted_score'] for item in category_scores.values()):.2f}

Please provide:
1. Overall assessment of the solution
2. Key strengths and areas for improvement
3. Specific recommendations for enhancement
4. Confidence in the solution's effectiveness

Keep your feedback constructive and actionable.
"""
        
        llm_response = await self.call_llm(prompt)
        return llm_response.get("content", "No feedback generated.")
    
    def _determine_task_type(self, task: str) -> str:
        """Determine the type of task based on the task description."""
        task_lower = task.lower()
        
        if any(keyword in task_lower for keyword in ["solve", "problem", "calculate", "compute"]):
            return "problem_solving"
        elif any(keyword in task_lower for keyword in ["write", "story", "creative", "narrative"]):
            return "creative_writing"
        elif any(keyword in task_lower for keyword in ["design", "architecture", "system", "technical"]):
            return "technical_design"
        elif any(keyword in task_lower for keyword in ["plan", "strategy", "roadmap", "approach"]):
            return "strategic_planning"
        else:
            return "general"
    
    def _determine_performance_level(self, score: float) -> str:
        """Determine performance level based on score."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    async def generate_prompt(self, task: str, context: Dict[str, Any] = None) -> str:
        """Generate the prompt for the LLM based on the task and context."""
        # For evaluator, this is mainly used for category-specific evaluations
        return f"{self.config.prompt}\n\nTask: {task}"
    
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
                "confidence": 0.8,  # Evaluators typically have high confidence
                "reasoning": "Evaluation performed using OpenAI GPT-4",
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
                "confidence": 0.8,  # Evaluators typically have high confidence
                "reasoning": "Evaluation performed using Anthropic Claude 3",
                "model": "claude-3-sonnet-20240229"
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
        else:
            return "General evaluation"
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        """Get the evaluation criteria used by this evaluator."""
        return {
            "specialization": self.specialization,
            "categories": list(self.rubric_config.get("categories", {}).keys()),
            "weights": self.evaluation_weights,
            "scoring_scale": self.rubric_config.get("scoring", {})
        }