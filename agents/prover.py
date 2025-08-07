from typing import Dict, Any, List, Optional, Union
import asyncio
import structlog
import json
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import openai
import anthropic
import google.generativeai as genai
from .base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus, CircuitState

logger = structlog.get_logger()


class VariantGenerator:
    """Manages variant generation with diversity and quality optimization."""
    
    def __init__(self, max_variants: int = 3, diversity_threshold: float = 0.3):
        self.max_variants = max_variants
        self.diversity_threshold = diversity_threshold
        self.variant_strategies = [
            "creative_innovative",
            "practical_feasible", 
            "analytical_rigorous",
            "efficient_optimized",
            "sustainable_scalable"
        ]
    
    def get_variant_strategy(self, variant_id: int, task_type: str) -> Dict[str, Any]:
        """Get variant-specific strategy with task adaptation."""
        strategy = self.variant_strategies[variant_id % len(self.variant_strategies)]
        
        # Adapt strategy based on task type
        adaptations = {
            "problem_solving": {
                "creative_innovative": {"temperature": 0.8, "focus": "out-of-the-box thinking"},
                "practical_feasible": {"temperature": 0.6, "focus": "implementable solutions"},
                "analytical_rigorous": {"temperature": 0.4, "focus": "logical analysis"},
                "efficient_optimized": {"temperature": 0.5, "focus": "performance optimization"},
                "sustainable_scalable": {"temperature": 0.7, "focus": "long-term viability"}
            },
            "creative_writing": {
                "creative_innovative": {"temperature": 0.9, "focus": "artistic expression"},
                "practical_feasible": {"temperature": 0.7, "focus": "readable content"},
                "analytical_rigorous": {"temperature": 0.5, "focus": "structured narrative"},
                "efficient_optimized": {"temperature": 0.6, "focus": "concise writing"},
                "sustainable_scalable": {"temperature": 0.8, "focus": "engaging storytelling"}
            }
        }
        
        task_adaptations = adaptations.get(task_type, {})
        strategy_config = task_adaptations.get(strategy, {"temperature": 0.7, "focus": "balanced approach"})
        
        return {
            "strategy": strategy,
            "temperature_adjustment": strategy_config["temperature"],
            "focus": strategy_config["focus"],
            "variant_id": variant_id
        }


class ProverAgent(BaseAgent):
    """Enhanced agent that executes plans and generates multiple variants with advanced features."""
    
    def __init__(self, config: AgentConfig, agent_id: str = None):
        super().__init__(config, agent_id)
        self.max_variants = config.hyperparameters.get('max_variants', 3)
        self.creativity = config.hyperparameters.get('creativity', 0.7)
        self.detail_level = config.hyperparameters.get('detail_level', 0.7)
        
        # Enhanced prover features
        self.variant_generator = VariantGenerator(self.max_variants)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.collaboration_enabled = config.hyperparameters.get('collaboration_enabled', False)
        self.memory_integration_enabled = config.hyperparameters.get('memory_integration_enabled', True)
        self.quality_threshold = config.hyperparameters.get('quality_threshold', 0.6)
        
        # Performance tracking
        self.variant_quality_history = []
        self.collaboration_usage = 0
        
        logger.info("Enhanced prover initialized", 
                   agent_id=self.agent_id,
                   max_variants=self.max_variants,
                   collaboration_enabled=self.collaboration_enabled)
        
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Execute the prover agent with enhanced variant generation and collaboration."""
        try:
            context = context or {}
            task_type = self._determine_task_type(task)
            
            # Get memory context if enabled
            memory_context = {}
            if self.memory_integration_enabled:
                memory_context = await self._get_memory_context(task, context)
            
            # Generate variants with enhanced strategies
            variants = []
            variant_tasks = []
            
            for i in range(self.max_variants):
                variant_task = asyncio.create_task(
                    self._generate_variant_enhanced(task, context, i, task_type, memory_context)
                )
                variant_tasks.append(variant_task)
            
            # Execute variants in parallel
            variant_results = await asyncio.gather(*variant_tasks, return_exceptions=True)
            
            # Process results and filter by quality
            for i, result in enumerate(variant_results):
                if isinstance(result, Exception):
                    logger.warning(f"Variant {i} generation failed", error=str(result))
                    continue
                
                if result.get("quality_score", 0.0) >= self.quality_threshold:
                    variants.append(result)
                else:
                    logger.info(f"Variant {i} filtered out due to low quality", 
                              quality_score=result.get("quality_score", 0.0))
            
            if not variants:
                logger.warning("No high-quality variants generated, using fallback")
                fallback_variant = await self._generate_fallback_variant(task, context)
                variants.append(fallback_variant)
            
            # Select best variant with enhanced criteria
            best_variant = self._select_best_variant(variants, task_type)
            
            # Apply collaboration if enabled
            if self.collaboration_enabled and len(variants) > 1:
                best_variant = await self._apply_collaboration_enhancement(best_variant, variants, task, context)
                self.collaboration_usage += 1
            
            # Calculate overall confidence with quality weighting
            overall_confidence = self._calculate_weighted_confidence(variants, best_variant)
            
            # Update performance metrics
            self._update_prover_metrics(variants, best_variant, task_type)
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.name,
                result={
                    "variants": variants,
                    "best_variant": best_variant,
                    "task": task,
                    "task_type": task_type,
                    "prover_type": self.config.name,
                    "memory_context_used": bool(memory_context),
                    "collaboration_applied": self.collaboration_enabled and len(variants) > 1
                },
                confidence_score=overall_confidence,
                execution_time=0.0,  # Will be set by execute_with_retry
                status=AgentStatus.COMPLETED,
                metadata={
                    "num_variants": len(variants),
                    "creativity": self.creativity,
                    "detail_level": self.detail_level,
                    "quality_threshold": self.quality_threshold,
                    "task_type": task_type,
                    "collaboration_usage": self.collaboration_usage
                }
            )
            
        except Exception as e:
            logger.error("Prover execution failed", agent_id=self.agent_id, error=str(e))
            # Return a failed result instead of raising
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.name,
                result={
                    "variants": [],
                    "best_variant": {
                        "content": f"Error: {str(e)}",
                        "confidence": 0.0,
                        "strategy": "error_fallback"
                    },
                    "task": task,
                    "task_type": "error",
                    "prover_type": self.config.name,
                    "memory_context_used": False,
                    "collaboration_applied": False,
                    "error": str(e)
                },
                confidence_score=0.0,
                execution_time=0.0,
                status=AgentStatus.FAILED,
                metadata={
                    "num_variants": 0,
                    "creativity": self.creativity,
                    "detail_level": self.detail_level,
                    "quality_threshold": self.quality_threshold,
                    "task_type": "error",
                    "collaboration_usage": self.collaboration_usage,
                    "error": str(e)
                }
            )
    
    async def _generate_variant_enhanced(self, task: str, context: Dict[str, Any], 
                                       variant_id: int, task_type: str, memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced variant with strategy and quality assessment."""
        try:
            # Get variant strategy
            strategy = self.variant_generator.get_variant_strategy(variant_id, task_type)
            
            # Generate enhanced prompt
            variant_prompt = await self.generate_prompt_enhanced(
                task, context, variant_id, strategy, memory_context
            )
            
            # Call LLM with strategy-specific parameters
            variant_result = await self._call_llm_implementation(variant_prompt, {
                "variant_id": variant_id,
                "strategy": strategy,
                "task_type": task_type
            })
            
            # Assess variant quality
            quality_score = await self._assess_variant_quality(
                variant_result.get("content", ""), task, task_type, strategy
            )
            
            # Calculate variant-specific confidence
            variant_confidence = self._calculate_variant_confidence(
                variant_result, quality_score, strategy
            )
            
            return {
                "variant_id": variant_id,
                "content": variant_result.get("content", ""),
                "confidence": variant_confidence,
                "reasoning": variant_result.get("reasoning", ""),
                "strategy": strategy,
                "quality_score": quality_score,
                "task_type": task_type,
                "generation_metadata": {
                    "model": variant_result.get("model", "unknown"),
                    "usage": variant_result.get("usage", {}),
                    "temperature_used": strategy["temperature_adjustment"]
                }
            }
            
        except Exception as e:
            logger.error(f"Variant {variant_id} generation failed", error=str(e))
            return {
                "variant_id": variant_id,
                "content": f"Error generating variant: {str(e)}",
                "confidence": 0.0,
                "quality_score": 0.0,
                "error": str(e)
            }
    
    async def generate_prompt_enhanced(self, task: str, context: Dict[str, Any], 
                                     variant_id: int, strategy: Dict[str, Any], 
                                     memory_context: Dict[str, Any]) -> str:
        """Generate enhanced prompt with strategy and memory integration."""
        context = context or {}
        
        # Build strategy-specific instructions
        strategy_instructions = self._get_strategy_instructions(strategy)
        
        # Build memory context section
        memory_section = self._format_memory_context(memory_context)
        
        # Build the complete prompt
        prompt = f"""{self.config.prompt}

Task: {task}

Strategy: {strategy['strategy']}
Focus: {strategy['focus']}

{strategy_instructions}

Additional Context:
{self._format_context(context)}

{memory_section}

Please provide a detailed solution that addresses all aspects of the task.
Consider the specific strategy and focus area while maintaining high quality.

Response should be well-structured and include:
1. Main solution with clear reasoning
2. Alternative approaches (if applicable)
3. Implementation considerations
4. Potential challenges and mitigation strategies
5. Quality indicators and confidence assessment
"""
        
        return prompt
    
    def _get_strategy_instructions(self, strategy: Dict[str, Any]) -> str:
        """Get strategy-specific instructions."""
        strategy_instructions = {
            "creative_innovative": """
Focus on creative and innovative approaches. Think outside the box and consider unconventional solutions.
Emphasize originality and novel perspectives while maintaining relevance to the task.
""",
            "practical_feasible": """
Emphasize practicality and feasibility. Focus on implementable solutions that can be executed effectively.
Consider real-world constraints and practical considerations.
""",
            "analytical_rigorous": """
Balance creativity with analytical rigor. Provide well-reasoned solutions with clear logical structure.
Include detailed analysis and systematic approach to problem-solving.
""",
            "efficient_optimized": """
Prioritize efficiency and optimization. Focus on solutions that maximize effectiveness while minimizing resource usage.
Consider performance, scalability, and optimization opportunities.
""",
            "sustainable_scalable": """
Consider long-term sustainability and scalability. Focus on solutions that can grow and adapt over time.
Emphasize maintainability, extensibility, and future-proofing.
"""
        }
        
        return strategy_instructions.get(strategy["strategy"], "Provide a balanced and comprehensive solution.")
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt integration."""
        if not context:
            return ""
        
        context_section = "\nAdditional Context:\n"
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                context_section += f"{key}: {value}\n"
            elif isinstance(value, dict):
                context_section += f"{key}: {json.dumps(value, indent=2)}\n"
            elif isinstance(value, list):
                context_section += f"{key}: {', '.join(str(item) for item in value)}\n"
        
        return context_section
    
    def _format_memory_context(self, memory_context: Dict[str, Any]) -> str:
        """Format memory context for prompt integration."""
        if not memory_context:
            return ""
        
        relevant_memories = memory_context.get("relevant_memories", [])
        if not relevant_memories:
            return ""
        
        memory_section = "\nRelevant Previous Solutions:\n"
        for i, memory in enumerate(relevant_memories[:3]):  # Limit to top 3
            memory_section += f"{i+1}. {memory.get('content', '')[:200]}...\n"
        
        return memory_section
    
    async def _get_memory_context(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant memory context for the task."""
        try:
            # This would integrate with the memory system
            # For now, return empty context
            return {
                "relevant_memories": [],
                "similar_tasks": [],
                "successful_patterns": []
            }
        except Exception as e:
            logger.warning("Memory context retrieval failed", error=str(e))
            return {}
    
    async def _assess_variant_quality(self, content: str, task: str, task_type: str, 
                                    strategy: Dict[str, Any]) -> float:
        """Assess the quality of a generated variant."""
        try:
            # Simple quality assessment based on content characteristics
            quality_factors = {
                "length": min(1.0, len(content) / 1000),  # Prefer longer responses
                "structure": self._assess_structure_quality(content),
                "relevance": self._assess_relevance_quality(content, task),
                "completeness": self._assess_completeness_quality(content, task_type)
            }
            
            # Weighted quality score
            weights = {"length": 0.2, "structure": 0.3, "relevance": 0.3, "completeness": 0.2}
            quality_score = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning("Quality assessment failed", error=str(e))
            return 0.5
    
    def _assess_structure_quality(self, content: str) -> float:
        """Assess the structural quality of the content."""
        # Simple heuristics for structure quality
        lines = content.split('\n')
        has_numbered_sections = any(line.strip().startswith(('1.', '2.', '3.')) for line in lines)
        has_bullet_points = any(line.strip().startswith(('-', '•', '*')) for line in lines)
        has_clear_sections = len([line for line in lines if line.strip().isupper()]) > 0
        
        structure_score = 0.0
        if has_numbered_sections:
            structure_score += 0.4
        if has_bullet_points:
            structure_score += 0.3
        if has_clear_sections:
            structure_score += 0.3
        
        return structure_score
    
    def _assess_relevance_quality(self, content: str, task: str) -> float:
        """Assess the relevance of content to the task."""
        # Simple keyword matching
        task_keywords = set(task.lower().split())
        content_keywords = set(content.lower().split())
        
        if not task_keywords:
            return 0.5
        
        keyword_overlap = len(task_keywords.intersection(content_keywords))
        relevance_score = min(1.0, keyword_overlap / len(task_keywords))
        
        return relevance_score
    
    def _assess_completeness_quality(self, content: str, task_type: str) -> float:
        """Assess the completeness of the content for the task type."""
        # Task-specific completeness criteria
        completeness_indicators = {
            "problem_solving": ["solution", "approach", "analysis", "result"],
            "creative_writing": ["narrative", "story", "character", "plot"],
            "technical_design": ["design", "architecture", "implementation", "specification"],
            "strategic_planning": ["strategy", "plan", "timeline", "milestone"]
        }
        
        indicators = completeness_indicators.get(task_type, ["solution", "approach"])
        content_lower = content.lower()
        
        indicator_count = sum(1 for indicator in indicators if indicator in content_lower)
        completeness_score = min(1.0, indicator_count / len(indicators))
        
        return completeness_score
    
    def _calculate_variant_confidence(self, variant_result: Dict[str, Any], 
                                   quality_score: float, strategy: Dict[str, Any]) -> float:
        """Calculate confidence for a specific variant."""
        base_confidence = variant_result.get("confidence", 0.5)
        
        # Adjust based on quality score
        quality_adjustment = quality_score * 0.3
        
        # Adjust based on strategy effectiveness
        strategy_adjustment = 0.1 if strategy["strategy"] in ["analytical_rigorous", "practical_feasible"] else 0.0
        
        # Adjust based on content length
        content_length = len(variant_result.get("content", ""))
        length_adjustment = min(0.1, content_length / 10000)  # Cap at 0.1 for very long content
        
        final_confidence = base_confidence + quality_adjustment + strategy_adjustment + length_adjustment
        
        return min(1.0, max(0.0, final_confidence))
    
    def _select_best_variant(self, variants: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """Select the best variant using multi-criteria selection."""
        if not variants:
            return {}
        
        # Calculate composite scores for each variant
        variant_scores = []
        for variant in variants:
            confidence = variant.get("confidence", 0.0)
            quality = variant.get("quality_score", 0.0)
            
            # Weight factors based on task type
            if task_type == "problem_solving":
                weights = {"confidence": 0.4, "quality": 0.6}
            elif task_type == "creative_writing":
                weights = {"confidence": 0.3, "quality": 0.7}
            else:
                weights = {"confidence": 0.5, "quality": 0.5}
            
            composite_score = confidence * weights["confidence"] + quality * weights["quality"]
            variant_scores.append((composite_score, variant))
        
        # Select variant with highest composite score
        best_variant = max(variant_scores, key=lambda x: x[0])[1]
        
        return best_variant
    
    async def _apply_collaboration_enhancement(self, best_variant: Dict[str, Any], 
                                            all_variants: List[Dict[str, Any]], 
                                            task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply collaboration enhancement to the best variant."""
        try:
            # Get insights from other variants
            variant_insights = []
            for variant in all_variants:
                if variant["variant_id"] != best_variant["variant_id"]:
                    insight = self._extract_variant_insight(variant, task)
                    if insight:
                        variant_insights.append(insight)
            
            if not variant_insights:
                return best_variant
            
            # Generate collaboration prompt
            collaboration_prompt = f"""
Enhance the following solution by incorporating insights from other approaches:

Original Solution:
{best_variant['content']}

Insights from Other Approaches:
{self._format_variant_insights(variant_insights)}

Task: {task}

Please enhance the original solution by incorporating the best insights from other approaches.
Maintain the core structure while adding valuable elements from other variants.
"""
            
            # Get enhanced solution
            enhanced_result = await self.call_llm(collaboration_prompt, {"enhancement_type": "collaboration"})
            
            # Create enhanced variant
            enhanced_variant = best_variant.copy()
            enhanced_variant["content"] = enhanced_result.get("content", best_variant["content"])
            enhanced_variant["collaboration_applied"] = True
            enhanced_variant["insights_incorporated"] = len(variant_insights)
            
            return enhanced_variant
            
        except Exception as e:
            logger.warning("Collaboration enhancement failed", error=str(e))
            return best_variant
    
    def _extract_variant_insight(self, variant: Dict[str, Any], task: str) -> Optional[str]:
        """Extract key insight from a variant."""
        content = variant.get("content", "")
        if not content:
            return None
        
        # Simple insight extraction - in production, this would be more sophisticated
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ["key", "important", "critical", "essential", "notable"]):
                return line
        
        # Fallback: return first substantial line
        for line in lines:
            if len(line.strip()) > 50:
                return line[:200] + "..."
        
        return None
    
    def _format_variant_insights(self, insights: List[str]) -> str:
        """Format variant insights for collaboration prompt."""
        if not insights:
            return "No additional insights available."
        
        formatted = []
        for i, insight in enumerate(insights, 1):
            formatted.append(f"{i}. {insight}")
        
        return "\n".join(formatted)
    
    def _calculate_weighted_confidence(self, variants: List[Dict[str, Any]], 
                                    best_variant: Dict[str, Any]) -> float:
        """Calculate weighted confidence across all variants."""
        if not variants:
            return 0.0
        
        # Weight by quality and confidence
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for variant in variants:
            confidence = variant.get("confidence", 0.0)
            quality = variant.get("quality_score", 0.0)
            weight = confidence * quality
            
            total_weighted_confidence += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return sum(v.get("confidence", 0.0) for v in variants) / len(variants)
        
        return total_weighted_confidence / total_weight
    
    def _update_prover_metrics(self, variants: List[Dict[str, Any]], 
                             best_variant: Dict[str, Any], task_type: str):
        """Update prover performance metrics."""
        # Track variant quality
        quality_scores = [v.get("quality_score", 0.0) for v in variants]
        if quality_scores:
            self.variant_quality_history.append({
                "timestamp": datetime.now(),
                "average_quality": np.mean(quality_scores),
                "best_quality": max(quality_scores),
                "task_type": task_type,
                "num_variants": len(variants)
            })
        
        # Keep history manageable
        if len(self.variant_quality_history) > 100:
            self.variant_quality_history = self.variant_quality_history[-50:]
    
    async def _generate_fallback_variant(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fallback variant when all others fail."""
        try:
            fallback_prompt = f"""
Generate a basic solution for the following task:

Task: {task}

Provide a straightforward, practical solution that addresses the core requirements.
Focus on clarity and completeness rather than innovation.
"""
            
            fallback_result = await self.call_llm(fallback_prompt, {"fallback": True})
            
            return {
                "variant_id": -1,  # Indicates fallback
                "content": fallback_result.get("content", "Fallback solution generated."),
                "confidence": 0.5,
                "quality_score": 0.5,
                "strategy": {"strategy": "fallback", "focus": "basic_completion"},
                "fallback": True
            }
            
        except Exception as e:
            logger.error("Fallback variant generation failed", error=str(e))
            return {
                "variant_id": -1,
                "content": "Error: Unable to generate solution.",
                "confidence": 0.0,
                "quality_score": 0.0,
                "error": str(e)
            }
    
    def _determine_task_type(self, task: str) -> str:
        """Determine task type for strategy adaptation."""
        task_lower = task.lower()
        
        task_keywords = {
            "problem_solving": ["solve", "problem", "calculate", "compute", "analyze"],
            "creative_writing": ["write", "story", "creative", "narrative", "compose"],
            "technical_design": ["design", "architecture", "system", "technical", "implement"],
            "strategic_planning": ["plan", "strategy", "roadmap", "approach", "framework"]
        }
        
        for task_type, keywords in task_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                return task_type
        
        return "general"
    
    async def _call_llm_implementation(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced LLM calling implementation for provers."""
        try:
            if self.config.model == "openai":
                return await self._call_openai_enhanced(prompt, context)
            elif self.config.model == "google":
                return await self._call_google_enhanced(prompt, context)
            elif self.config.model == "xai":
                return await self._call_xai_enhanced(prompt, context)
            elif self.config.model == "mistral":
                return await self._call_mistral_enhanced(prompt, context)
            elif self.config.model == "deepseek":
                return await self._call_deepseek_enhanced(prompt, context)
            elif self.config.model == "kimi":
                return await self._call_kimi_enhanced(prompt, context)
            elif self.config.model == "glm":
                return await self._call_glm_enhanced(prompt, context)
            else:
                raise ValueError(f"Unsupported model: {self.config.model}")
                
        except Exception as e:
            logger.error("LLM call failed", agent_id=self.agent_id, model=self.config.model, error=str(e))
            raise
    
    async def _call_openai_enhanced(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced OpenAI calling with strategy-specific parameters."""
        client = await self._llm_manager.get_client("openai", {"timeout": self.config.timeout})
        
        # Adjust parameters based on context
        temperature = self.config.temperature
        if context.get("strategy"):
            strategy = context["strategy"]
            temperature = strategy.get("temperature_adjustment", temperature)
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence_openai_enhanced(response, context)
            
            return {
                "content": content,
                "confidence": confidence,
                "reasoning": "Generated using OpenAI GPT-4 with enhanced parameters",
                "model": "gpt-4",
                "usage": response.usage.dict() if hasattr(response, 'usage') else {},
                "temperature_used": temperature
            }
            
        except Exception as e:
            logger.error("OpenAI API call failed", error=str(e))
            raise
    
    async def _call_google_enhanced(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Google Gemini calling with strategy-specific parameters."""
        client = await self._llm_manager.get_client("google", {"timeout": self.config.timeout})
        
        # Adjust parameters based on context
        temperature = self.config.temperature
        if context.get("strategy"):
            strategy = context["strategy"]
            temperature = strategy.get("temperature_adjustment", temperature)
        
        try:
            response = await client.generate_content_async(
                f"{self.config.prompt}\n\n{prompt}",
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=self.config.max_tokens
                )
            )
            
            content = response.text
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence_google_enhanced(response, context)
            
            return {
                "content": content,
                "confidence": confidence,
                "reasoning": "Generated using Google Gemini Pro with enhanced parameters",
                "model": "gemini-pro",
                "usage": {
                    "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
                },
                "temperature_used": temperature
            }
            
        except Exception as e:
            logger.error("Google API call failed", error=str(e))
            raise

    async def _call_xai_enhanced(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced XAI Grok calling with strategy-specific parameters."""
        client = await self._llm_manager.get_client("xai", {"timeout": self.config.timeout})
        
        # Adjust parameters based on context
        temperature = self.config.temperature
        if context.get("strategy"):
            strategy = context["strategy"]
            temperature = strategy.get("temperature_adjustment", temperature)
        
        try:
            response = await client.chat.completions.create(
                model="grok-2-1212",
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence_xai_enhanced(response, context)
            
            return {
                "content": content,
                "confidence": confidence,
                "reasoning": "Generated using XAI Grok with enhanced parameters",
                "model": "grok-2-1212",
                "usage": response.usage.dict() if hasattr(response, 'usage') else {},
                "temperature_used": temperature
            }
            
        except Exception as e:
            logger.error("XAI API call failed", error=str(e))
            raise

    async def _call_mistral_enhanced(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Mistral AI calling with strategy-specific parameters."""
        client = await self._llm_manager.get_client("mistral", {"timeout": self.config.timeout})
        
        # Adjust parameters based on context
        temperature = self.config.temperature
        if context.get("strategy"):
            strategy = context["strategy"]
            temperature = strategy.get("temperature_adjustment", temperature)
        
        try:
            response = await client.chat.completions.create(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence_mistral_enhanced(response, context)
            
            return {
                "content": content,
                "confidence": confidence,
                "reasoning": "Generated using Mistral AI with enhanced parameters",
                "model": "mistral-large-latest",
                "usage": response.usage.dict() if hasattr(response, 'usage') else {},
                "temperature_used": temperature
            }
            
        except Exception as e:
            logger.error("Mistral API call failed", error=str(e))
            raise

    async def _call_deepseek_enhanced(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced DeepSeek calling with strategy-specific parameters."""
        client = await self._llm_manager.get_client("deepseek", {"timeout": self.config.timeout})
        
        # Adjust parameters based on context
        temperature = self.config.temperature
        if context.get("strategy"):
            strategy = context["strategy"]
            temperature = strategy.get("temperature_adjustment", temperature)
        
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence_deepseek_enhanced(response, context)
            
            return {
                "content": content,
                "confidence": confidence,
                "reasoning": "Generated using DeepSeek with enhanced parameters",
                "model": "deepseek-chat",
                "usage": response.usage.dict() if hasattr(response, 'usage') else {},
                "temperature_used": temperature
            }
            
        except Exception as e:
            logger.error("DeepSeek API call failed", error=str(e))
            raise

    async def _call_kimi_enhanced(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Kimi calling with strategy-specific parameters."""
        client = await self._llm_manager.get_client("kimi", {"timeout": self.config.timeout})
        
        # Adjust parameters based on context
        temperature = self.config.temperature
        if context.get("strategy"):
            strategy = context["strategy"]
            temperature = strategy.get("temperature_adjustment", temperature)
        
        try:
            response = await client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence_kimi_enhanced(response, context)
            
            return {
                "content": content,
                "confidence": confidence,
                "reasoning": "Generated using Kimi with enhanced parameters",
                "model": "moonshot-v1-8k",
                "usage": response.usage.dict() if hasattr(response, 'usage') else {},
                "temperature_used": temperature
            }
            
        except Exception as e:
            logger.error("Kimi API call failed", error=str(e))
            raise

    async def _call_glm_enhanced(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced GLM calling with strategy-specific parameters."""
        client = await self._llm_manager.get_client("glm", {"timeout": self.config.timeout})
        
        # Adjust parameters based on context
        temperature = self.config.temperature
        if context.get("strategy"):
            strategy = context["strategy"]
            temperature = strategy.get("temperature_adjustment", temperature)
        
        try:
            response = await client.chat.completions.create(
                model="glm-4.5",
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence_glm_enhanced(response, context)
            
            return {
                "content": content,
                "confidence": confidence,
                "reasoning": "Generated using GLM 4.5 with enhanced parameters",
                "model": "glm-4.5",
                "usage": response.usage.dict() if hasattr(response, 'usage') else {},
                "temperature_used": temperature
            }
            
        except Exception as e:
            logger.error("GLM API call failed", error=str(e))
            raise
    
    def _calculate_confidence_openai_enhanced(self, response, context: Dict[str, Any]) -> float:
        """Enhanced confidence calculation for OpenAI responses."""
        base_confidence = 0.7
        
        # Adjust based on response length and completeness
        if hasattr(response, 'usage'):
            usage = response.usage
            if usage.completion_tokens > 200:
                base_confidence += 0.1
            if usage.completion_tokens > 800:
                base_confidence += 0.1
        
        # Adjust based on strategy
        if context.get("strategy"):
            strategy = context["strategy"]["strategy"]
            if strategy in ["analytical_rigorous", "practical_feasible"]:
                base_confidence += 0.1
        
        # Adjust based on temperature
        temperature_used = context.get("temperature_used", self.config.temperature)
        if temperature_used < 0.5:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _calculate_confidence_google_enhanced(self, response, context: Dict[str, Any]) -> float:
        """Enhanced confidence calculation for Google Gemini responses."""
        base_confidence = 0.75
        
        # Adjust based on response length
        if hasattr(response, 'usage_metadata'):
            if response.usage_metadata.candidates_token_count > 200:
                base_confidence += 0.1
            if response.usage_metadata.candidates_token_count > 800:
                base_confidence += 0.1
        
        # Adjust based on strategy
        if context.get("strategy"):
            strategy = context["strategy"]["strategy"]
            if strategy in ["analytical_rigorous", "practical_feasible"]:
                base_confidence += 0.1
        
        # Adjust based on temperature
        temperature_used = context.get("temperature_used", self.config.temperature)
        if temperature_used < 0.5:
            base_confidence += 0.1
        
        # Task-specific confidence adjustments
        task_type = context.get("task_type", "general")
        if task_type == "research":
            base_confidence += 0.08  # Gemini excels at research
        elif task_type == "analysis":
            base_confidence += 0.06  # Gemini good at analysis
        elif task_type == "synthesis":
            base_confidence += 0.05  # Gemini decent at synthesis
        
        # Response quality indicators
        content = getattr(response, 'text', '')
        if content:
            # Check for citations (indicates research quality)
            if any(marker in content for marker in ['[', ']', '(', ')', 'http']):
                base_confidence += 0.05
            # Check for structured analysis
            if any(marker in content for marker in ['##', '###', '1.', '2.', '3.']):
                base_confidence += 0.03
            # Check for research terms
            research_terms = ['study', 'research', 'analysis', 'findings', 'conclusion']
            if any(term in content.lower() for term in research_terms):
                base_confidence += 0.02
        
        return min(1.0, base_confidence)

    def _calculate_confidence_xai_enhanced(self, response, context: Dict[str, Any]) -> float:
        """Enhanced confidence calculation for XAI Grok responses."""
        base_confidence = 0.8  # Higher base confidence for real-time reasoning
        
        # Adjust based on response length
        if hasattr(response, 'usage'):
            if response.usage.completion_tokens > 200:
                base_confidence += 0.1
            if response.usage.completion_tokens > 800:
                base_confidence += 0.1
        
        # Adjust based on strategy - XAI excels at real-time reasoning
        if context.get("strategy"):
            strategy = context["strategy"]["strategy"]
            if strategy in ["creative_innovative", "practical_feasible"]:
                base_confidence += 0.15  # Higher boost for real-time capabilities
        
        # Adjust based on temperature
        temperature_used = context.get("temperature_used", self.config.temperature)
        if temperature_used < 0.5:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

    def _calculate_confidence_mistral_enhanced(self, response, context: Dict[str, Any]) -> float:
        """Enhanced confidence calculation for Mistral AI responses."""
        base_confidence = 0.75
        
        # Adjust based on response length
        if hasattr(response, 'usage'):
            if response.usage.completion_tokens > 200:
                base_confidence += 0.1
            if response.usage.completion_tokens > 800:
                base_confidence += 0.1
        
        # Adjust based on strategy - Mistral excels at multilingual tasks
        if context.get("strategy"):
            strategy = context["strategy"]["strategy"]
            if strategy in ["analytical_rigorous", "structured_reasoning"]:
                base_confidence += 0.1
        
        # Adjust based on temperature
        temperature_used = context.get("temperature_used", self.config.temperature)
        if temperature_used < 0.5:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

    def _calculate_confidence_deepseek_enhanced(self, response, context: Dict[str, Any]) -> float:
        """Enhanced confidence calculation for DeepSeek responses."""
        base_confidence = 0.8  # Higher base confidence for technical tasks
        
        # Adjust based on response length
        if hasattr(response, 'usage'):
            if response.usage.completion_tokens > 200:
                base_confidence += 0.1
            if response.usage.completion_tokens > 800:
                base_confidence += 0.1
        
        # Adjust based on strategy - DeepSeek excels at technical tasks
        if context.get("strategy"):
            strategy = context["strategy"]["strategy"]
            if strategy in ["analytical_rigorous", "efficient_optimized"]:
                base_confidence += 0.15  # Higher boost for technical capabilities
        
        # Adjust based on temperature
        temperature_used = context.get("temperature_used", self.config.temperature)
        if temperature_used < 0.5:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

    def _calculate_confidence_kimi_enhanced(self, response, context: Dict[str, Any]) -> float:
        """Enhanced confidence calculation for Kimi responses."""
        base_confidence = 0.8  # Higher base confidence for long-context understanding
        
        # Adjust based on response length
        if hasattr(response, 'usage'):
            if response.usage.completion_tokens > 200:
                base_confidence += 0.1
            if response.usage.completion_tokens > 800:
                base_confidence += 0.1
        
        # Adjust based on strategy - Kimi excels at comprehensive analysis
        if context.get("strategy"):
            strategy = context["strategy"]["strategy"]
            if strategy in ["creative_innovative", "sustainable_scalable"]:
                base_confidence += 0.15  # Higher boost for comprehensive capabilities
        
        # Adjust based on temperature
        temperature_used = context.get("temperature_used", self.config.temperature)
        if temperature_used < 0.5:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

    def _calculate_confidence_glm_enhanced(self, response, context: Dict[str, Any]) -> float:
        """Enhanced confidence calculation for GLM responses."""
        base_confidence = 0.85  # Higher base confidence for advanced reasoning
        
        # Adjust based on response length
        if hasattr(response, 'usage'):
            if response.usage.completion_tokens > 200:
                base_confidence += 0.1
            if response.usage.completion_tokens > 800:
                base_confidence += 0.1
        
        # Adjust based on strategy - GLM excels at advanced reasoning and code generation
        if context.get("strategy"):
            strategy = context["strategy"]["strategy"]
            if strategy in ["analytical_rigorous", "efficient_optimized", "technical_design"]:
                base_confidence += 0.15  # Higher boost for technical capabilities
        
        # Adjust based on temperature
        temperature_used = context.get("temperature_used", self.config.temperature)
        if temperature_used < 0.5:
            base_confidence += 0.1
        
        # Task-specific confidence adjustments
        task_type = context.get("task_type", "general")
        if task_type == "code_generation":
            base_confidence += 0.05  # GLM excels at code generation
        elif task_type == "system_design":
            base_confidence += 0.08  # GLM good at architecture
        elif task_type == "debugging":
            base_confidence += 0.03  # GLM decent at debugging
        
        # Response quality indicators
        content = getattr(response, 'choices', [{}])[0].get('message', {}).get('content', '')
        if content:
            # Check for code blocks (indicates technical response)
            if '```' in content:
                base_confidence += 0.05
            # Check for structured output
            if any(marker in content for marker in ['##', '###', '- ', '* ']):
                base_confidence += 0.03
            # Check for technical terms
            tech_terms = ['function', 'class', 'interface', 'component', 'api', 'database']
            if any(term in content.lower() for term in tech_terms):
                base_confidence += 0.02
        
        return min(1.0, base_confidence)

    def _calculate_confidence_anthropic_enhanced(self, response, context: Dict[str, Any]) -> float:
        """Enhanced confidence calculation for Anthropic responses."""
        base_confidence = 0.75
        
        # Adjust based on response length
        if hasattr(response, 'usage'):
            if response.usage.output_tokens > 200:
                base_confidence += 0.1
            if response.usage.output_tokens > 800:
                base_confidence += 0.1
        
        # Adjust based on strategy
        if context.get("strategy"):
            strategy = context["strategy"]["strategy"]
            if strategy in ["analytical_rigorous", "practical_feasible"]:
                base_confidence += 0.1
        
        # Adjust based on temperature
        temperature_used = context.get("temperature_used", self.config.temperature)
        if temperature_used < 0.5:
            base_confidence += 0.1
        
        # Task-specific confidence adjustments
        task_type = context.get("task_type", "general")
        if task_type == "security":
            base_confidence += 0.10  # Claude excels at security
        elif task_type == "general_assistance":
            base_confidence += 0.08  # Claude good at general tasks
        elif task_type == "writing":
            base_confidence += 0.06  # Claude decent at writing
        
        # Response quality indicators
        content = getattr(response, 'content', [{}])[0].get('text', '')
        if content:
            # Check for security considerations
            security_terms = ['security', 'authentication', 'authorization', 'encryption', 'vulnerability']
            if any(term in content.lower() for term in security_terms):
                base_confidence += 0.05
            # Check for structured advice
            if any(marker in content for marker in ['##', '###', '1.', '2.', '3.']):
                base_confidence += 0.03
            # Check for practical considerations
            practical_terms = ['consider', 'ensure', 'verify', 'validate', 'implement']
            if any(term in content.lower() for term in practical_terms):
                base_confidence += 0.02
        
        return min(1.0, base_confidence)
    
    def get_specialization_info(self) -> Dict[str, Any]:
        """Get enhanced information about the prover's specialization."""
        return {
            "type": "prover",
            "max_variants": self.max_variants,
            "creativity": self.creativity,
            "detail_level": self.detail_level,
            "specialization": self._get_specialization(),
            "collaboration_enabled": self.collaboration_enabled,
            "memory_integration_enabled": self.memory_integration_enabled,
            "quality_threshold": self.quality_threshold,
            "collaboration_usage": self.collaboration_usage,
            "average_variant_quality": np.mean([h["average_quality"] for h in self.variant_quality_history]) if self.variant_quality_history else 0.0
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
        elif "strategic" in name_lower:
            return "Strategic planning"
        elif "technical" in name_lower:
            return "Technical design"
        else:
            return "General problem solving"
    
    async def generate_prompt(self, task: str, context: Dict[str, Any] = None) -> str:
        """Generate the prompt for the LLM based on the task and context."""
        # This is now handled by generate_prompt_enhanced
        return await self.generate_prompt_enhanced(task, context, 0, {}, {})