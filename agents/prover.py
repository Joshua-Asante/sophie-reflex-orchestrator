from typing import Dict, Any, List, Optional
import asyncio
import structlog
from datetime import datetime
import openai
import anthropic
from .base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus

logger = structlog.get_logger()


class ProverAgent(BaseAgent):
    """Agent that executes plans and generates multiple variants of responses."""
    
    def __init__(self, config: AgentConfig, agent_id: str = None):
        super().__init__(config, agent_id)
        self.max_variants = config.hyperparameters.get('max_variants', 3)
        self.creativity = config.hyperparameters.get('creativity', 0.7)
        self.detail_level = config.hyperparameters.get('detail_level', 0.7)
        
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Execute the prover agent to generate solution variants."""
        try:
            # Generate multiple variants
            variants = []
            for i in range(self.max_variants):
                variant_prompt = await self.generate_prompt(task, context, variant_id=i)
                variant_result = await self.call_llm(variant_prompt)
                variants.append({
                    "variant_id": i,
                    "content": variant_result.get("content", ""),
                    "confidence": variant_result.get("confidence", 0.5),
                    "reasoning": variant_result.get("reasoning", "")
                })
            
            # Calculate overall confidence
            overall_confidence = sum(v["confidence"] for v in variants) / len(variants)
            
            # Select best variant
            best_variant = max(variants, key=lambda x: x["confidence"])
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.name,
                result={
                    "variants": variants,
                    "best_variant": best_variant,
                    "task": task,
                    "prover_type": self.config.name
                },
                confidence_score=overall_confidence,
                execution_time=0.0,  # Will be set by execute_with_retry
                status=AgentStatus.COMPLETED,
                metadata={
                    "num_variants": len(variants),
                    "creativity": self.creativity,
                    "detail_level": self.detail_level
                }
            )
            
        except Exception as e:
            logger.error("Prover execution failed", agent_id=self.agent_id, error=str(e))
            raise
    
    async def generate_prompt(self, task: str, context: Dict[str, Any] = None, variant_id: int = 0) -> str:
        """Generate the prompt for the LLM based on the task and context."""
        context = context or {}
        
        # Add variant-specific instructions
        variant_instructions = self._get_variant_instructions(variant_id)
        
        # Build the complete prompt
        prompt = f"""{self.config.prompt}

Task: {task}

{variant_instructions}

Additional Context:
{self._format_context(context)}

Please provide a detailed solution that addresses all aspects of the task.
Consider different approaches and provide reasoning for your choices.

Response should be well-structured and include:
1. Main solution
2. Alternative approaches (if applicable)
3. Reasoning and justification
4. Potential challenges and mitigation strategies
"""
        
        return prompt
    
    def _get_variant_instructions(self, variant_id: int) -> str:
        """Get variant-specific instructions to encourage diversity."""
        variant_prompts = [
            "Focus on a creative and innovative approach.",
            "Emphasize practicality and feasibility.",
            "Balance creativity with analytical rigor.",
            "Prioritize efficiency and optimization.",
            "Consider long-term sustainability and scalability."
        ]
        
        return variant_prompts[variant_id % len(variant_prompts)]
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for the prompt."""
        if not context:
            return "No additional context provided."
        
        formatted_context = []
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                formatted_context.append(f"{key}: {str(value)}")
            else:
                formatted_context.append(f"{key}: {value}")
        
        return "\n".join(formatted_context)
    
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
            
            # Calculate confidence based on response quality indicators
            confidence = self._calculate_confidence_openai(response)
            
            return {
                "content": content,
                "confidence": confidence,
                "reasoning": "Generated using OpenAI GPT-4",
                "model": "gpt-4",
                "usage": response.usage.dict() if hasattr(response, 'usage') else {}
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
            
            # Calculate confidence based on response quality
            confidence = self._calculate_confidence_anthropic(response)
            
            return {
                "content": content,
                "confidence": confidence,
                "reasoning": "Generated using Anthropic Claude 3",
                "model": "claude-3-sonnet-20240229",
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error("Anthropic API call failed", error=str(e))
            raise
    
    def _calculate_confidence_openai(self, response) -> float:
        """Calculate confidence score for OpenAI response."""
        # Base confidence on model and response characteristics
        base_confidence = 0.7
        
        # Adjust based on response length and completeness
        if hasattr(response, 'usage'):
            usage = response.usage
            if usage.completion_tokens > 100:
                base_confidence += 0.1
            if usage.completion_tokens > 500:
                base_confidence += 0.1
        
        # Adjust based on temperature setting
        if self.config.temperature < 0.5:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _calculate_confidence_anthropic(self, response) -> float:
        """Calculate confidence score for Anthropic response."""
        # Base confidence on model and response characteristics
        base_confidence = 0.75
        
        # Adjust based on response length
        if hasattr(response, 'usage'):
            if response.usage.output_tokens > 100:
                base_confidence += 0.1
            if response.usage.output_tokens > 500:
                base_confidence += 0.1
        
        # Adjust based on temperature setting
        if self.config.temperature < 0.5:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def get_specialization_info(self) -> Dict[str, Any]:
        """Get information about the prover's specialization."""
        return {
            "type": "prover",
            "max_variants": self.max_variants,
            "creativity": self.creativity,
            "detail_level": self.detail_level,
            "specialization": self._get_specialization()
        }
    
    def _get_specialization(self) -> str:
        """Get the prover's specialization based on its name."""
        name_lower = self.config.name.lower()
        if "creative" in name_lower:
            return "Creative problem solving"
        elif "analytical" in name_lower:
            return "Analytical reasoning"
        elif "practical" in name_lower:
            return "Practical implementation"
        else:
            return "General problem solving"