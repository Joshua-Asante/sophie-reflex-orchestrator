from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
import structlog
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re
from concurrent.futures import ThreadPoolExecutor
import functools

logger = structlog.get_logger()

# Modularization imports (scaffold)
try:
    from governance.policy_rules import parse_trust_condition  # type: ignore
    from governance.policy_cache import PolicyCache  # type: ignore
except Exception:
    parse_trust_condition = None  # type: ignore
    PolicyCache = None  # type: ignore

class PolicyDecision(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_HUMAN_REVIEW = "require_human_review"
    MODIFY = "modify"
    WARN = "warn"


@dataclass
class PolicyResult:
    """Result of a policy evaluation."""
    decision: PolicyDecision
    reason: str
    confidence: float
    modifications: Dict[str, Any] = None
    conditions_met: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.modifications is None:
            self.modifications = {}
        if self.conditions_met is None:
            self.conditions_met = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PolicyContext:
    """Context for policy evaluation."""
    agent_id: str
    agent_type: str
    action: str
    content: str
    trust_score: float
    confidence_score: float
    iteration_count: int
    timestamp: datetime
    additional_context: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_context is None:
            self.additional_context = {}


class PolicyEngine:
    """Applies policies from policies.yaml to gate/refine agent behavior."""

    def __init__(self, policies_config: Dict[str, Any]):
        self.policies_config = policies_config
        self.hitl_policies = policies_config.get("hitl", {})
        self.agent_lifecycle_policies = policies_config.get("agent_lifecycle", {})
        self.trust_policies = policies_config.get("trust", {})
        self.resource_limits = policies_config.get("resource_limits", {})
        self.security_policies = policies_config.get("security", {})
        self.performance_policies = policies_config.get("performance", {})

        # Policy caches with improved TTL management
        self.cache_ttl = 300  # 5 minutes
        self.cache_stats = {"hits": 0, "misses": 0}
        self.policy_cache = PolicyCache(self.cache_ttl) if PolicyCache else {}

        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Compile regex patterns for better performance
        self._compile_regex_patterns()

        logger.info("Policy engine initialized",
                   policies_loaded=len(self.policies_config),
                   cache_ttl=self.cache_ttl)

    def _compile_regex_patterns(self):
        """Compile regex patterns for condition evaluation."""
        self.patterns = {
            "trust_score": re.compile(r"trust_score\s*([<>=]+)\s*([\d.]+)"),
            "confidence_score": re.compile(r"confidence_score\s*([<>=]+)\s*([\d.]+)"),
            "iteration_count": re.compile(r"iteration_count\s*([<>=]+)\s*(\d+)")
        }

    async def evaluate_action(self, context: PolicyContext) -> PolicyResult:
        """Evaluate an action against all applicable policies."""
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(context)

            # Check cache first
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.cache_stats["hits"] += 1
                return cached_result

            self.cache_stats["misses"] += 1

            # Evaluate policies concurrently for better performance
            policy_tasks = [
                self._evaluate_hitl_policies(context),
                self._evaluate_agent_lifecycle_policies(context),
                self._evaluate_trust_policies(context),
                self._evaluate_resource_policies(context),
                self._evaluate_security_policies(context),
                self._evaluate_performance_policies(context)
            ]

            # Wait for all policy evaluations to complete
            results = await asyncio.gather(*policy_tasks, return_exceptions=True)

            # Filter out exceptions and log them
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Policy evaluation failed",
                               policy_type=["hitl", "lifecycle", "trust", "resource", "security", "performance"][i],
                               error=str(result))
                else:
                    valid_results.append(result)

            # Combine results
            final_result = self._combine_policy_results(valid_results, context)

            # Cache result
            self._cache_result(cache_key, final_result)

            logger.info(
                "Policy evaluation completed",
                agent_id=context.agent_id,
                action=context.action,
                decision=final_result.decision.value,
                reason=final_result.reason,
                confidence=final_result.confidence
            )

            return final_result

        except Exception as e:
            logger.error("Policy evaluation failed",
                        agent_id=context.agent_id,
                        action=context.action,
                        error=str(e))
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                reason=f"Policy evaluation error: {str(e)}",
                confidence=0.0
            )

    async def _evaluate_hitl_policies(self, context: PolicyContext) -> PolicyResult:
        """Evaluate Human-in-the-Loop policies."""
        if not self.hitl_policies.get("enabled", True):
            return PolicyResult(
                decision=PolicyDecision.ALLOW,
                reason="HITL policies disabled",
                confidence=1.0
            )

        try:
            conditions_met = []
            modifications = {}

            # Check if human review is required
            require_review_conditions = self.hitl_policies.get("require_human_review", [])

            for condition in require_review_conditions:
                if await self._evaluate_condition(condition, context):
                    conditions_met.append(condition)

            # Check auto-approval conditions
            auto_approve_conditions = self.hitl_policies.get("auto_approve", [])
            can_auto_approve = True

            for condition in auto_approve_conditions:
                if not await self._evaluate_condition(condition, context):
                    can_auto_approve = False
                    break

            # Make decision
            if conditions_met:
                return PolicyResult(
                    decision=PolicyDecision.REQUIRE_HUMAN_REVIEW,
                    reason=f"Human review required: {', '.join(conditions_met)}",
                    confidence=0.9,
                    conditions_met=conditions_met,
                    modifications={"requires_human_review": True}
                )
            elif can_auto_approve:
                return PolicyResult(
                    decision=PolicyDecision.ALLOW,
                    reason="Auto-approval conditions met",
                    confidence=0.8,
                    conditions_met=auto_approve_conditions
                )
            else:
                # Check approval threshold
                approval_threshold = self.hitl_policies.get("approval_threshold", 0.7)
                if context.trust_score >= approval_threshold:
                    return PolicyResult(
                        decision=PolicyDecision.ALLOW,
                        reason=f"Trust score {context.trust_score} meets approval threshold {approval_threshold}",
                        confidence=0.7
                    )
                else:
                    return PolicyResult(
                        decision=PolicyDecision.REQUIRE_HUMAN_REVIEW,
                        reason=f"Trust score {context.trust_score} below approval threshold {approval_threshold}",
                        confidence=0.8
                    )

        except Exception as e:
            logger.error("HITL policy evaluation failed", error=str(e))
            return PolicyResult(
                decision=PolicyDecision.REQUIRE_HUMAN_REVIEW,
                reason=f"HITL policy evaluation error: {str(e)}",
                confidence=0.5
            )

    async def _evaluate_agent_lifecycle_policies(self, context: PolicyContext) -> PolicyResult:
        """Evaluate agent lifecycle policies."""
        conditions_met = []
        modifications = {}

        # Check pruning conditions
        prune_conditions = self.agent_lifecycle_policies.get("prune_agents", [])
        should_prune = False

        for condition in prune_conditions:
            if await self._evaluate_condition(condition, context):
                conditions_met.append(condition)
                should_prune = True

        if should_prune:
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                reason=f"Agent should be pruned: {', '.join(conditions_met)}",
                confidence=0.8,
                conditions_met=conditions_met,
                modifications={"should_prune": True}
            )

        # Check forking conditions
        fork_conditions = self.agent_lifecycle_policies.get("fork_agents", [])
        should_fork = False

        for condition in fork_conditions:
            if await self._evaluate_condition(condition, context):
                conditions_met.append(condition)
                should_fork = True

        if should_fork:
            return PolicyResult(
                decision=PolicyDecision.MODIFY,
                reason=f"Agent should be forked: {', '.join(conditions_met)}",
                confidence=0.7,
                conditions_met=conditions_met,
                modifications={"should_fork": True}
            )

        # Check mutation conditions
        mutation_conditions = self.agent_lifecycle_policies.get("mutate_agents", [])
        should_mutate = False

        for condition in mutation_conditions:
            if await self._evaluate_condition(condition, context):
                conditions_met.append(condition)
                should_mutate = True

        if should_mutate:
            return PolicyResult(
                decision=PolicyDecision.MODIFY,
                reason=f"Agent should be mutated: {', '.join(conditions_met)}",
                confidence=0.6,
                conditions_met=conditions_met,
                modifications={"should_mutate": True}
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="No agent lifecycle actions required",
            confidence=0.9
        )

    async def _evaluate_trust_policies(self, context: PolicyContext) -> PolicyResult:
        """Evaluate trust management policies."""
        conditions_met = []
        modifications = {}

        # Check trust score bounds
        min_score = self.trust_policies.get("min_score", 0.0)
        max_score = self.trust_policies.get("max_score", 1.0)

        if context.trust_score < min_score:
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                reason=f"Trust score {context.trust_score} below minimum {min_score}",
                confidence=0.9,
                modifications={"trust_violation": True}
            )

        if context.trust_score > max_score:
            modifications["trust_score"] = max_score

        # Apply trust decay if enabled
        if self.trust_policies.get("decay", {}).get("enabled", False):
            decay_rate = self.trust_policies.get("decay", {}).get("rate", 0.01)
            max_decay = self.trust_policies.get("decay", {}).get("max_decay", 0.3)

            # Calculate decay (simplified - in practice would track time)
            decay_amount = min(max_decay, decay_rate)
            modifications["trust_decay"] = decay_amount

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="Trust policies satisfied",
            confidence=0.8,
            modifications=modifications
        )

    async def _evaluate_resource_policies(self, context: PolicyContext) -> PolicyResult:
        """Evaluate resource limit policies."""
        conditions_met = []
        modifications = {}

        # Check concurrent agents limit
        max_concurrent = self.resource_limits.get("max_concurrent_agents", 10)
        current_concurrent = context.additional_context.get("current_concurrent_agents", 0)

        if current_concurrent >= max_concurrent:
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                reason=f"Concurrent agent limit reached: {current_concurrent}/{max_concurrent}",
                confidence=1.0,
                modifications={"resource_limit_exceeded": True}
            )

        # Check execution time
        max_execution_time = self.resource_limits.get("max_execution_time", 300)
        current_execution_time = context.additional_context.get("execution_time", 0)

        if current_execution_time >= max_execution_time:
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                reason=f"Execution time limit exceeded: {current_execution_time}/{max_execution_time}s",
                confidence=1.0,
                modifications={"timeout_exceeded": True}
            )

        # Check iteration count
        max_iterations = self.resource_limits.get("max_total_iterations", 100)
        if context.iteration_count >= max_iterations:
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                reason=f"Iteration limit reached: {context.iteration_count}/{max_iterations}",
                confidence=1.0,
                modifications={"iteration_limit_exceeded": True}
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="Resource limits satisfied",
            confidence=0.9
        )

    async def _evaluate_security_policies(self, context: PolicyContext) -> PolicyResult:
        """Evaluate security policies."""
        conditions_met = []
        modifications = {}

        # Content filtering
        if self.security_policies.get("content_filtering", {}).get("enabled", True):
            blocked_categories = self.security_policies.get("content_filtering", {}).get("block_categories", [])

            for category in blocked_categories:
                if await self._check_content_category(context.content, category):
                    conditions_met.append(f"blocked_content:{category}")

            if conditions_met:
                return PolicyResult(
                    decision=PolicyDecision.BLOCK,
                    reason=f"Content blocked: {', '.join(conditions_met)}",
                    confidence=0.9,
                    conditions_met=conditions_met
                )

        # Access control
        if self.security_policies.get("access_control", {}).get("require_authentication", False):
            is_authenticated = context.additional_context.get("is_authenticated", False)

            if not is_authenticated:
                return PolicyResult(
                    decision=PolicyDecision.BLOCK,
                    reason="Authentication required",
                    confidence=1.0,
                    modifications={"authentication_required": True}
                )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="Security policies satisfied",
            confidence=0.8
        )

    async def _evaluate_performance_policies(self, context: PolicyContext) -> PolicyResult:
        """Evaluate performance policies."""
        conditions_met = []
        modifications = {}

        # Check performance alerts
        alerts = self.performance_policies.get("alerts", {})

        # Low trust score alert
        low_trust_threshold = alerts.get("low_trust_score", 0.3)
        if context.trust_score < low_trust_threshold:
            conditions_met.append("low_trust_score")
            modifications["performance_alert"] = "low_trust_score"

        # High failure rate alert
        failure_rate = context.additional_context.get("failure_rate", 0.0)
        high_failure_threshold = alerts.get("high_failure_rate", 0.5)
        if failure_rate > high_failure_threshold:
            conditions_met.append("high_failure_rate")
            modifications["performance_alert"] = "high_failure_rate"

        # Resource exhaustion alert
        resource_usage = context.additional_context.get("resource_usage", 0.0)
        resource_threshold = alerts.get("resource_exhaustion", 0.8)
        if resource_usage > resource_threshold:
            conditions_met.append("resource_exhaustion")
            modifications["performance_alert"] = "resource_exhaustion"

        if conditions_met:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason=f"Performance alerts: {', '.join(conditions_met)}",
                confidence=0.8,
                conditions_met=conditions_met,
                modifications=modifications
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="Performance policies satisfied",
            confidence=0.9
        )

    async def _evaluate_condition(self, condition: str, context: PolicyContext) -> bool:
        """Evaluate a single condition."""
        try:
            # Parse condition
            if condition.startswith("trust_score"):
                # Prefer centralized parser if available
                if parse_trust_condition:
                    parsed = parse_trust_condition(condition)  # type: ignore[misc]
                    if parsed:
                        operator, threshold = parsed
                        if operator == "<":
                            return context.trust_score < threshold
                        if operator == "<=":
                            return context.trust_score <= threshold
                        if operator == ">":
                            return context.trust_score > threshold
                        if operator == ">=":
                            return context.trust_score >= threshold
                        if operator == "==":
                            return context.trust_score == threshold
                # Fallback to local regex
                match = self.patterns["trust_score"].match(condition)
                if match:
                    operator, value = match.groups()
                    threshold = float(value)
                    if operator == "<":
                        return context.trust_score < threshold
                    elif operator == "<=":
                        return context.trust_score <= threshold
                    elif operator == ">":
                        return context.trust_score > threshold
                    elif operator == ">=":
                        return context.trust_score >= threshold
                    elif operator == "==":
                        return context.trust_score == threshold

            elif condition.startswith("confidence_score"):
                # Confidence score conditions
                match = self.patterns["confidence_score"].match(condition)
                if match:
                    operator, value = match.groups()
                    threshold = float(value)

                    if operator == "<":
                        return context.confidence_score < threshold
                    elif operator == "<=":
                        return context.confidence_score <= threshold
                    elif operator == ">":
                        return context.confidence_score > threshold
                    elif operator == ">=":
                        return context.confidence_score >= threshold

            elif condition.startswith("iteration_count"):
                # Iteration count conditions
                match = self.patterns["iteration_count"].match(condition)
                if match:
                    operator, value = match.groups()
                    threshold = int(value)

                    if operator == "<":
                        return context.iteration_count < threshold
                    elif operator == "<=":
                        return context.iteration_count <= threshold
                    elif operator == ">":
                        return context.iteration_count > threshold
                    elif operator == ">=":
                        return context.iteration_count >= threshold

            elif condition == "contains_sensitive_content":
                # Check for sensitive content (simplified)
                sensitive_keywords = ["password", "secret", "key", "token", "auth"]
                content_lower = context.content.lower()
                return any(keyword in content_lower for keyword in sensitive_keywords)

            elif condition == "high_risk_task":
                # Check for high-risk task indicators
                risk_keywords = ["delete", "remove", "destroy", "overwrite", "format"]
                content_lower = context.content.lower()
                return any(keyword in content_lower for keyword in risk_keywords)

            elif condition == "no_interventions":
                # Check if there have been no interventions
                return context.additional_context.get("intervention_count", 0) == 0

            elif condition == "high_performance":
                # Check if agent is high performing
                return context.trust_score > 0.8 and context.confidence_score > 0.7

            elif condition == "unique_solution":
                # Check if solution is unique (simplified)
                return context.additional_context.get("solution_uniqueness", 0.0) > 0.7

            elif condition == "stagnant_performance":
                # Check if performance is stagnant
                return context.additional_context.get("performance_trend", "stable") == "stagnant"

            elif condition == "diversity_needed":
                # Check if diversity is needed
                return context.additional_context.get("diversity_score", 0.0) < 0.3

            else:
                # Unknown condition
                logger.warning("Unknown condition", condition=condition)
                return False

        except Exception as e:
            logger.error("Condition evaluation failed", condition=condition, error=str(e))
            return False

    async def _check_content_category(self, content: str, category: str) -> bool:
        """Check if content belongs to a blocked category."""
        # Simplified content checking - in practice would use more sophisticated methods
        category_keywords = {
            "hate_speech": ["hate", "discriminate", "racist", "sexist"],
            "violence": ["violence", "kill", "harm", "attack"],
            "adult_content": ["adult", "explicit", "nsfw"],
            "illegal_activities": ["illegal", "crime", "hack", "steal"]
        }

        keywords = category_keywords.get(category, [])
        content_lower = content.lower()

        return any(keyword in content_lower for keyword in keywords)

    def _combine_policy_results(self, results: List[PolicyResult], context: PolicyContext) -> PolicyResult:
        """Combine multiple policy results into a final decision."""
        # Priority order: BLOCK > REQUIRE_HUMAN_REVIEW > WARN > MODIFY > ALLOW

        # Check for any BLOCK decisions
        block_results = [r for r in results if r.decision == PolicyDecision.BLOCK]
        if block_results:
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                reason=f"Blocked by policies: {', '.join(r.reason for r in block_results)}",
                confidence=max(r.confidence for r in block_results),
                modifications={k: v for r in block_results for k, v in r.modifications.items()},
                conditions_met=[c for r in block_results for c in r.conditions_met]
            )

        # Check for any REQUIRE_HUMAN_REVIEW decisions
        review_results = [r for r in results if r.decision == PolicyDecision.REQUIRE_HUMAN_REVIEW]
        if review_results:
            return PolicyResult(
                decision=PolicyDecision.REQUIRE_HUMAN_REVIEW,
                reason=f"Human review required: {', '.join(r.reason for r in review_results)}",
                confidence=max(r.confidence for r in review_results),
                modifications={k: v for r in review_results for k, v in r.modifications.items()},
                conditions_met=[c for r in review_results for c in r.conditions_met]
            )

        # Check for any WARN decisions
        warn_results = [r for r in results if r.decision == PolicyDecision.WARN]
        if warn_results:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason=f"Warning: {', '.join(r.reason for r in warn_results)}",
                confidence=max(r.confidence for r in warn_results),
                modifications={k: v for r in warn_results for k, v in r.modifications.items()},
                conditions_met=[c for r in warn_results for c in r.conditions_met]
            )

        # Check for any MODIFY decisions
        modify_results = [r for r in results if r.decision == PolicyDecision.MODIFY]
        if modify_results:
            return PolicyResult(
                decision=PolicyDecision.MODIFY,
                reason=f"Modification required: {', '.join(r.reason for r in modify_results)}",
                confidence=max(r.confidence for r in modify_results),
                modifications={k: v for r in modify_results for k, v in r.modifications.items()},
                conditions_met=[c for r in modify_results for c in r.conditions_met]
            )

        # If all decisions are ALLOW, return ALLOW
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="All policies satisfied",
            confidence=min(r.confidence for r in results) if results else 1.0
        )

    def _generate_cache_key(self, context: PolicyContext) -> str:
        """Generate a cache key for the policy context."""
        key_data = {
            "agent_id": context.agent_id,
            "action": context.action,
            "trust_score": context.trust_score,
            "confidence_score": context.confidence_score,
            "iteration_count": context.iteration_count,
            "timestamp": context.timestamp.isoformat()
        }
        return hash(str(key_data))

    def _get_cached_result(self, cache_key: str) -> Optional[PolicyResult]:
        """Get a cached policy result."""
        if isinstance(self.policy_cache, dict):
            if cache_key in self.policy_cache:
                cached_data = self.policy_cache[cache_key]
                if datetime.now() - cached_data["timestamp"] < timedelta(seconds=self.cache_ttl):
                    return cached_data["result"]
                else:
                    del self.policy_cache[cache_key]
            return None
        # New PolicyCache path
        value = self.policy_cache.get(cache_key)  # type: ignore[attr-defined]
        return value

    def _cache_result(self, cache_key: str, result: PolicyResult):
        """Cache a policy result."""
        if isinstance(self.policy_cache, dict):
            self.policy_cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now()
            }
        else:
            self.policy_cache.set(cache_key, result)  # type: ignore[attr-defined]

    def get_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about policy engine usage."""
        return {
            "cache_size": len(self.policy_cache),
            "cache_ttl": self.cache_ttl,
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "policies_configured": {
                "hitl": bool(self.hitl_policies),
                "agent_lifecycle": bool(self.agent_lifecycle_policies),
                "trust": bool(self.trust_policies),
                "resource_limits": bool(self.resource_limits),
                "security": bool(self.security_policies),
                "performance": bool(self.performance_policies)
            }
        }

    def clear_cache(self):
        """Clear the policy cache."""
        self.policy_cache.clear()
        logger.info("Policy cache cleared")

    async def validate_policies(self) -> Dict[str, Any]:
        """Validate the policy configuration."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Validate HITL policies
        if self.hitl_policies.get("enabled", True):
            approval_threshold = self.hitl_policies.get("approval_threshold", 0.7)
            rejection_threshold = self.hitl_policies.get("rejection_threshold", 0.4)

            if approval_threshold <= rejection_threshold:
                validation_results["errors"].append(
                    f"Approval threshold ({approval_threshold}) must be greater than rejection threshold ({rejection_threshold})"
                )
                validation_results["valid"] = False

        # Validate trust policies
        min_score = self.trust_policies.get("min_score", 0.0)
        max_score = self.trust_policies.get("max_score", 1.0)

        if min_score >= max_score:
            validation_results["errors"].append(
                f"Min trust score ({min_score}) must be less than max trust score ({max_score})"
            )
            validation_results["valid"] = False

        # Validate resource limits
        if self.resource_limits.get("max_concurrent_agents", 10) <= 0:
            validation_results["errors"].append("Max concurrent agents must be positive")
            validation_results["valid"] = False

        return validation_results

    def evaluate_hitl_requirement(self, trust_score: float, confidence_score: float,
                                 content: str = "", agent_id: str = None) -> bool:
        """Evaluate if human-in-the-loop review is required."""
        try:
            if not self.hitl_policies.get("enabled", False):
                return False

            approval_threshold = self.hitl_policies.get("approval_threshold", 0.7)

            # Check if trust score is below threshold
            if trust_score < approval_threshold:
                return True

            # Check if confidence score is below threshold
            if confidence_score < approval_threshold:
                return True

            # Check for sensitive content patterns
            sensitive_patterns = self.hitl_policies.get("sensitive_patterns", [])
            for pattern in sensitive_patterns:
                if pattern.lower() in content.lower():
                    return True

            return False

        except Exception as e:
            logger.error("HITL requirement evaluation failed", error=str(e))
            return True  # Default to requiring HITL on error

    def validate_trust_score(self, trust_score: float) -> bool:
        """Validate if a trust score is within acceptable range."""
        try:
            min_score = self.trust_policies.get("min_trust_score", 0.0)
            max_score = self.trust_policies.get("max_trust_score", 1.0)
            return min_score <= trust_score <= max_score
        except Exception as e:
            logger.error("Trust score validation failed", error=str(e))
            return False

    def check_execution_policy(self, agent_id: str, retry_count: int) -> bool:
        """Check if execution is allowed based on policies."""
        try:
            max_retries = self.resource_limits.get("max_retries", 3)
            return retry_count < max_retries
        except Exception as e:
            logger.error("Execution policy check failed", error=str(e))
            return False
