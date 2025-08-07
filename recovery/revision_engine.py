import yaml
import json
import time
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FixPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FixType(Enum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    USABILITY = "usability"
    RELIABILITY = "reliability"

@dataclass
class TrustDelta:
    """Enhanced trust delta calculation with detailed metrics."""
    base_trust: float
    proposed_trust: float
    delta: float
    confidence: float
    risk_factor: float
    impact_score: float
    user_feedback_weight: float
    execution_history_weight: float
    security_impact: float
    performance_impact: float

@dataclass
class PlanRefinerFix:
    """Enhanced fix proposal with detailed analysis."""
    id: str
    description: str
    fix_type: FixType
    priority: FixPriority
    trust_delta: TrustDelta
    affected_components: List[str]
    estimated_effort: int  # hours
    risk_assessment: Dict[str, float]
    user_feedback_score: float
    execution_history_score: float
    proposed_changes: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    timestamp: float
    author: str

class EnhancedPlanRefiner:
    """Enhanced PlanRefiner with sophisticated trust delta calculation and prioritization."""
    
    def __init__(self, trust_threshold: float = 0.7):
        self.trust_threshold = trust_threshold
        self.fix_history = []
        self.user_feedback_cache = {}
        self.execution_history_cache = {}
        self.trust_metrics = {}
        
        # Enhanced weighting factors
        self.weighting_factors = {
            "user_feedback": 0.4,
            "execution_history": 0.3,
            "security_impact": 0.15,
            "performance_impact": 0.1,
            "risk_factor": 0.05
        }
        
        # Priority thresholds
        self.priority_thresholds = {
            FixPriority.CRITICAL: 0.9,
            FixPriority.HIGH: 0.7,
            FixPriority.MEDIUM: 0.5,
            FixPriority.LOW: 0.3
        }
    
    def calculate_trust_delta(self, 
                            current_trust: float,
                            proposed_changes: Dict[str, Any],
                            user_feedback: Dict[str, Any],
                            execution_history: Dict[str, Any],
                            security_context: Dict[str, Any]) -> TrustDelta:
        """
        Calculate sophisticated trust delta with multiple factors.
        
        Args:
            current_trust: Current trust score
            proposed_changes: Proposed changes to implement
            user_feedback: Recent user feedback data
            execution_history: Historical execution data
            security_context: Security-related context
            
        Returns:
            TrustDelta object with detailed analysis
        """
        # Calculate base proposed trust
        proposed_trust = self._calculate_proposed_trust(proposed_changes, execution_history)
        
        # Calculate delta
        delta = proposed_trust - current_trust
        
        # Calculate confidence in the prediction
        confidence = self._calculate_prediction_confidence(proposed_changes, execution_history)
        
        # Calculate risk factor
        risk_factor = self._calculate_risk_factor(proposed_changes, security_context)
        
        # Calculate impact scores
        security_impact = self._calculate_security_impact(proposed_changes, security_context)
        performance_impact = self._calculate_performance_impact(proposed_changes, execution_history)
        
        # Calculate user feedback weight
        user_feedback_weight = self._calculate_user_feedback_weight(user_feedback)
        
        # Calculate execution history weight
        execution_history_weight = self._calculate_execution_history_weight(execution_history)
        
        # Calculate overall impact score
        impact_score = self._calculate_impact_score(
            delta, confidence, risk_factor, security_impact, performance_impact
        )
        
        return TrustDelta(
            base_trust=current_trust,
            proposed_trust=proposed_trust,
            delta=delta,
            confidence=confidence,
            risk_factor=risk_factor,
            impact_score=impact_score,
            user_feedback_weight=user_feedback_weight,
            execution_history_weight=execution_history_weight,
            security_impact=security_impact,
            performance_impact=performance_impact
        )
    
    def _calculate_proposed_trust(self, proposed_changes: Dict[str, Any], 
                                execution_history: Dict[str, Any]) -> float:
        """Calculate proposed trust based on changes and history."""
        base_trust = 0.5
        
        # Factor in change complexity
        complexity_score = self._assess_change_complexity(proposed_changes)
        base_trust += complexity_score * 0.2
        
        # Factor in historical success of similar changes
        historical_success = self._assess_historical_success(proposed_changes, execution_history)
        base_trust += historical_success * 0.3
        
        # Factor in change safety
        safety_score = self._assess_change_safety(proposed_changes)
        base_trust += safety_score * 0.2
        
        # Factor in testing coverage
        testing_score = self._assess_testing_coverage(proposed_changes)
        base_trust += testing_score * 0.1
        
        return max(0.0, min(1.0, base_trust))
    
    def _assess_change_complexity(self, proposed_changes: Dict[str, Any]) -> float:
        """Assess the complexity of proposed changes."""
        complexity_score = 0.0
        
        # Count affected components
        affected_components = len(proposed_changes.get("affected_components", []))
        complexity_score += min(affected_components / 10.0, 0.3)
        
        # Assess change type
        change_types = proposed_changes.get("change_types", [])
        for change_type in change_types:
            if change_type in ["security", "authentication", "encryption"]:
                complexity_score += 0.2
            elif change_type in ["performance", "optimization"]:
                complexity_score += 0.1
            elif change_type in ["ui", "cosmetic"]:
                complexity_score += 0.05
        
        # Assess code changes
        code_changes = proposed_changes.get("code_changes", {})
        total_lines = sum(code_changes.values())
        complexity_score += min(total_lines / 1000.0, 0.2)
        
        return min(1.0, complexity_score)
    
    def _assess_historical_success(self, proposed_changes: Dict[str, Any], 
                                 execution_history: Dict[str, Any]) -> float:
        """Assess historical success of similar changes."""
        if not execution_history:
            return 0.5
        
        # Find similar changes in history
        similar_changes = []
        for change in execution_history.get("changes", []):
            similarity = self._calculate_change_similarity(proposed_changes, change)
            if similarity > 0.7:  # 70% similarity threshold
                similar_changes.append((similarity, change.get("success_rate", 0.5)))
        
        if not similar_changes:
            return 0.5
        
        # Weighted average of success rates
        total_weight = sum(similarity for similarity, _ in similar_changes)
        weighted_success = sum(similarity * success_rate for similarity, success_rate in similar_changes)
        
        return weighted_success / total_weight if total_weight > 0 else 0.5
    
    def _calculate_change_similarity(self, change1: Dict[str, Any], change2: Dict[str, Any]) -> float:
        """Calculate similarity between two changes."""
        similarity = 0.0
        
        # Compare change types
        types1 = set(change1.get("change_types", []))
        types2 = set(change2.get("change_types", []))
        if types1 and types2:
            type_similarity = len(types1.intersection(types2)) / len(types1.union(types2))
            similarity += type_similarity * 0.4
        
        # Compare affected components
        components1 = set(change1.get("affected_components", []))
        components2 = set(change2.get("affected_components", []))
        if components1 and components2:
            component_similarity = len(components1.intersection(components2)) / len(components1.union(components2))
            similarity += component_similarity * 0.3
        
        # Compare complexity
        complexity1 = change1.get("complexity_score", 0.5)
        complexity2 = change2.get("complexity_score", 0.5)
        complexity_similarity = 1.0 - abs(complexity1 - complexity2)
        similarity += complexity_similarity * 0.3
        
        return similarity
    
    def _assess_change_safety(self, proposed_changes: Dict[str, Any]) -> float:
        """Assess the safety of proposed changes."""
        safety_score = 0.8  # Base safety score
        
        # Check for security-sensitive changes
        if any(change_type in ["authentication", "authorization", "encryption"] 
               for change_type in proposed_changes.get("change_types", [])):
            safety_score -= 0.2
        
        # Check for database changes
        if "database" in proposed_changes.get("affected_components", []):
            safety_score -= 0.1
        
        # Check for API changes
        if "api" in proposed_changes.get("affected_components", []):
            safety_score -= 0.1
        
        # Check for configuration changes
        if "config" in proposed_changes.get("change_types", []):
            safety_score += 0.1
        
        return max(0.0, min(1.0, safety_score))
    
    def _assess_testing_coverage(self, proposed_changes: Dict[str, Any]) -> float:
        """Assess testing coverage for proposed changes."""
        testing_score = 0.5  # Base testing score
        
        # Check if tests are included
        if proposed_changes.get("includes_tests", False):
            testing_score += 0.3
        
        # Check test coverage
        test_coverage = proposed_changes.get("test_coverage", 0.0)
        testing_score += test_coverage * 0.2
        
        return min(1.0, testing_score)
    
    def _calculate_prediction_confidence(self, proposed_changes: Dict[str, Any], 
                                       execution_history: Dict[str, Any]) -> float:
        """Calculate confidence in the trust prediction."""
        confidence = 0.5  # Base confidence
        
        # Factor in historical data availability
        if execution_history and len(execution_history.get("changes", [])) > 10:
            confidence += 0.2
        
        # Factor in change complexity (simpler changes are more predictable)
        complexity = self._assess_change_complexity(proposed_changes)
        confidence += (1.0 - complexity) * 0.2
        
        # Factor in testing coverage
        testing_coverage = self._assess_testing_coverage(proposed_changes)
        confidence += testing_coverage * 0.1
        
        return min(1.0, confidence)
    
    def _calculate_risk_factor(self, proposed_changes: Dict[str, Any], 
                             security_context: Dict[str, Any]) -> float:
        """Calculate risk factor for proposed changes."""
        risk_factor = 0.1  # Base risk
        
        # Security-sensitive changes
        if any(change_type in ["authentication", "authorization", "encryption"] 
               for change_type in proposed_changes.get("change_types", [])):
            risk_factor += 0.3
        
        # Database changes
        if "database" in proposed_changes.get("affected_components", []):
            risk_factor += 0.2
        
        # API changes
        if "api" in proposed_changes.get("affected_components", []):
            risk_factor += 0.15
        
        # Production environment
        if security_context.get("environment") == "production":
            risk_factor += 0.2
        
        # High-traffic system
        if security_context.get("traffic_level") == "high":
            risk_factor += 0.1
        
        return min(1.0, risk_factor)
    
    def _calculate_security_impact(self, proposed_changes: Dict[str, Any], 
                                 security_context: Dict[str, Any]) -> float:
        """Calculate security impact of proposed changes."""
        security_impact = 0.0
        
        # Security-related changes
        if any(change_type in ["authentication", "authorization", "encryption", "security"] 
               for change_type in proposed_changes.get("change_types", [])):
            security_impact += 0.5
        
        # Data handling changes
        if "data" in proposed_changes.get("affected_components", []):
            security_impact += 0.3
        
        # User management changes
        if "user" in proposed_changes.get("affected_components", []):
            security_impact += 0.2
        
        return min(1.0, security_impact)
    
    def _calculate_performance_impact(self, proposed_changes: Dict[str, Any], 
                                    execution_history: Dict[str, Any]) -> float:
        """Calculate performance impact of proposed changes."""
        performance_impact = 0.0
        
        # Performance-related changes
        if any(change_type in ["optimization", "performance", "caching"] 
               for change_type in proposed_changes.get("change_types", [])):
            performance_impact += 0.4
        
        # Database changes
        if "database" in proposed_changes.get("affected_components", []):
            performance_impact += 0.3
        
        # API changes
        if "api" in proposed_changes.get("affected_components", []):
            performance_impact += 0.2
        
        return min(1.0, performance_impact)
    
    def _calculate_user_feedback_weight(self, user_feedback: Dict[str, Any]) -> float:
        """Calculate weight based on user feedback."""
        if not user_feedback:
            return 0.5
        
        # Recent feedback weight
        recent_feedback = user_feedback.get("recent_feedback", [])
        if recent_feedback:
            avg_rating = sum(f.get("rating", 3) for f in recent_feedback) / len(recent_feedback)
            return max(0.1, min(1.0, avg_rating / 5.0))
        
        return 0.5
    
    def _calculate_execution_history_weight(self, execution_history: Dict[str, Any]) -> float:
        """Calculate weight based on execution history."""
        if not execution_history:
            return 0.5
        
        # Success rate weight
        success_rate = execution_history.get("success_rate", 0.5)
        return max(0.1, min(1.0, success_rate))
    
    def _calculate_impact_score(self, delta: float, confidence: float, risk_factor: float,
                              security_impact: float, performance_impact: float) -> float:
        """Calculate overall impact score."""
        # Base impact from trust delta
        impact = abs(delta) * 0.4
        
        # Factor in confidence
        impact += confidence * 0.2
        
        # Factor in risk (negative impact)
        impact -= risk_factor * 0.2
        
        # Factor in security impact
        impact += security_impact * 0.1
        
        # Factor in performance impact
        impact += performance_impact * 0.1
        
        return max(0.0, min(1.0, impact))
    
    def prioritize_fixes(self, fixes: List[PlanRefinerFix]) -> List[PlanRefinerFix]:
        """Prioritize fixes based on trust delta and other factors."""
        for fix in fixes:
            # Calculate priority score
            priority_score = self._calculate_priority_score(fix)
            fix.priority = self._determine_priority(priority_score)
        
        # Sort by priority and score
        fixes.sort(key=lambda f: (self._priority_to_numeric(f.priority), f.trust_delta.impact_score), reverse=True)
        
        return fixes
    
    def _calculate_priority_score(self, fix: PlanRefinerFix) -> float:
        """Calculate priority score for a fix."""
        # Base score from trust delta
        base_score = fix.trust_delta.impact_score
        
        # Factor in fix type
        type_multipliers = {
            FixType.SECURITY: 1.5,
            FixType.PERFORMANCE: 1.2,
            FixType.ACCURACY: 1.1,
            FixType.RELIABILITY: 1.3,
            FixType.USABILITY: 0.9
        }
        type_multiplier = type_multipliers.get(fix.fix_type, 1.0)
        
        # Factor in user feedback
        feedback_multiplier = 1.0 + (fix.user_feedback_score - 0.5) * 0.4
        
        # Factor in execution history
        history_multiplier = 1.0 + (fix.execution_history_score - 0.5) * 0.3
        
        # Factor in risk assessment
        risk_multiplier = 1.0 - fix.trust_delta.risk_factor * 0.2
        
        priority_score = base_score * type_multiplier * feedback_multiplier * history_multiplier * risk_multiplier
        
        return max(0.0, min(1.0, priority_score))
    
    def _determine_priority(self, priority_score: float) -> FixPriority:
        """Determine priority level based on score."""
        if priority_score >= self.priority_thresholds[FixPriority.CRITICAL]:
            return FixPriority.CRITICAL
        elif priority_score >= self.priority_thresholds[FixPriority.HIGH]:
            return FixPriority.HIGH
        elif priority_score >= self.priority_thresholds[FixPriority.MEDIUM]:
            return FixPriority.MEDIUM
        else:
            return FixPriority.LOW
    
    def _priority_to_numeric(self, priority: FixPriority) -> int:
        """Convert priority to numeric for sorting."""
        priority_map = {
            FixPriority.CRITICAL: 4,
            FixPriority.HIGH: 3,
            FixPriority.MEDIUM: 2,
            FixPriority.LOW: 1
        }
        return priority_map.get(priority, 0)
    
    def suggest_fix(self, issue_description: str, current_trust: float,
                   user_feedback: Dict[str, Any], execution_history: Dict[str, Any],
                   security_context: Dict[str, Any]) -> PlanRefinerFix:
        """Suggest a fix with enhanced analysis."""
        # Generate fix ID
        fix_id = self._generate_fix_id(issue_description)
        
        # Analyze issue and propose changes
        proposed_changes = self._analyze_issue_and_propose_changes(issue_description)
        
        # Calculate trust delta
        trust_delta = self.calculate_trust_delta(
            current_trust, proposed_changes, user_feedback, execution_history, security_context
        )
        
        # Determine fix type
        fix_type = self._determine_fix_type(issue_description, proposed_changes)
        
        # Calculate user feedback score
        user_feedback_score = self._calculate_user_feedback_weight(user_feedback)
        
        # Calculate execution history score
        execution_history_score = self._calculate_execution_history_weight(execution_history)
        
        # Assess risk
        risk_assessment = self._assess_fix_risk(proposed_changes, security_context)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(issue_description, proposed_changes)
        
        # Create fix proposal
        fix = PlanRefinerFix(
            id=fix_id,
            description=issue_description,
            fix_type=fix_type,
            priority=FixPriority.MEDIUM,  # Will be calculated by prioritize_fixes
            trust_delta=trust_delta,
            affected_components=proposed_changes.get("affected_components", []),
            estimated_effort=self._estimate_effort(proposed_changes),
            risk_assessment=risk_assessment,
            user_feedback_score=user_feedback_score,
            execution_history_score=execution_history_score,
            proposed_changes=proposed_changes,
            alternatives=alternatives,
            timestamp=time.time(),
            author="PlanRefiner"
        )
        
        # Prioritize the fix
        prioritized_fixes = self.prioritize_fixes([fix])
        return prioritized_fixes[0]
    
    def _generate_fix_id(self, issue_description: str) -> str:
        """Generate unique fix ID."""
        timestamp = str(int(time.time()))
        hash_input = f"{issue_description}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    def _analyze_issue_and_propose_changes(self, issue_description: str) -> Dict[str, Any]:
        """Analyze issue and propose specific changes."""
        # This would typically involve AI analysis
        # For now, return a structured proposal
        return {
            "change_types": ["optimization", "bug_fix"],
            "affected_components": ["api", "orchestrator"],
            "code_changes": {"api": 50, "orchestrator": 30},
            "includes_tests": True,
            "test_coverage": 0.8,
            "complexity_score": 0.6
        }
    
    def _determine_fix_type(self, issue_description: str, proposed_changes: Dict[str, Any]) -> FixType:
        """Determine the type of fix based on issue and changes."""
        description_lower = issue_description.lower()
        change_types = proposed_changes.get("change_types", [])
        
        if any(word in description_lower for word in ["security", "vulnerability", "auth", "encrypt"]):
            return FixType.SECURITY
        elif any(word in description_lower for word in ["performance", "slow", "timeout", "optimize"]):
            return FixType.PERFORMANCE
        elif any(word in description_lower for word in ["accuracy", "correct", "wrong", "error"]):
            return FixType.ACCURACY
        elif any(word in description_lower for word in ["reliability", "crash", "fail", "stable"]):
            return FixType.RELIABILITY
        else:
            return FixType.USABILITY
    
    def _assess_fix_risk(self, proposed_changes: Dict[str, Any], 
                         security_context: Dict[str, Any]) -> Dict[str, float]:
        """Assess various risk factors for the fix."""
        return {
            "security_risk": self._calculate_security_impact(proposed_changes, security_context),
            "performance_risk": self._calculate_performance_impact(proposed_changes, {}),
            "deployment_risk": 0.3,
            "rollback_risk": 0.2
        }
    
    def _estimate_effort(self, proposed_changes: Dict[str, Any]) -> int:
        """Estimate effort required for the fix in hours."""
        base_effort = 2
        
        # Add effort based on complexity
        complexity = self._assess_change_complexity(proposed_changes)
        base_effort += int(complexity * 8)
        
        # Add effort based on affected components
        affected_components = len(proposed_changes.get("affected_components", []))
        base_effort += affected_components * 1
        
        # Add effort based on code changes
        code_changes = proposed_changes.get("code_changes", {})
        total_lines = sum(code_changes.values())
        base_effort += int(total_lines / 100)
        
        return max(1, min(40, base_effort))  # Between 1 and 40 hours
    
    def _generate_alternatives(self, issue_description: str, 
                             proposed_changes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative solutions."""
        alternatives = []
        
        # Alternative 1: Minimal change approach
        alternatives.append({
            "description": "Minimal change approach",
            "changes": {**proposed_changes, "complexity_score": proposed_changes.get("complexity_score", 0.6) * 0.5},
            "pros": ["Lower risk", "Faster implementation"],
            "cons": ["May not fully address the issue"]
        })
        
        # Alternative 2: Comprehensive approach
        alternatives.append({
            "description": "Comprehensive solution",
            "changes": {**proposed_changes, "complexity_score": proposed_changes.get("complexity_score", 0.6) * 1.5},
            "pros": ["Thorough solution", "Future-proof"],
            "cons": ["Higher risk", "More effort required"]
        })
        
        return alternatives

def generate_revised_step(original_step: Dict[str, Any], 
                         feedback: Dict[str, Any],
                         context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a revised step based on feedback and context.
    
    Args:
        original_step: The original step that needs revision
        feedback: Feedback about the original step
        context: Additional context for the revision
        
    Returns:
        Revised step with improvements
    """
    revised_step = original_step.copy()
    
    # Apply feedback-based improvements
    if feedback.get("accuracy_issues"):
        revised_step["validation_checks"] = revised_step.get("validation_checks", []) + [
            "accuracy_verification",
            "fact_checking"
        ]
    
    if feedback.get("performance_issues"):
        revised_step["optimizations"] = revised_step.get("optimizations", []) + [
            "caching_strategy",
            "parallel_processing"
        ]
    
    if feedback.get("security_concerns"):
        revised_step["security_measures"] = revised_step.get("security_measures", []) + [
            "input_validation",
            "output_sanitization",
            "access_control"
        ]
    
    # Add context-specific improvements
    if context.get("high_priority"):
        revised_step["priority"] = "high"
        revised_step["timeout"] = context.get("timeout", 30)
    
    if context.get("user_experience_focus"):
        revised_step["user_feedback_integration"] = True
        revised_step["progress_tracking"] = True
    
    # Add metadata about the revision
    revised_step["revision_metadata"] = {
        "original_step_id": original_step.get("id"),
        "revision_timestamp": time.time(),
        "feedback_applied": list(feedback.keys()),
        "context_factors": list(context.keys())
    }
    
    return revised_step