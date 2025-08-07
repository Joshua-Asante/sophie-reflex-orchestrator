"""
Performance Monitor - Advanced monitoring and analytics for agent performance.
Provides real-time metrics, predictive analytics, and optimization recommendations.
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import structlog
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
import statistics
import threading
import time

from .base_agent import BaseAgent, AgentResult, AgentStatus

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    SUCCESS_RATE = "success_rate"
    CONFIDENCE_SCORE = "confidence_score"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    QUALITY_SCORE = "quality_score"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    agent_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert notification."""
    alert_id: str
    agent_id: str
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    threshold_value: float
    actual_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class AgentPerformanceProfile:
    """Comprehensive performance profile for an agent."""
    agent_id: str
    agent_type: str
    creation_time: datetime
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    average_confidence: float = 0.0
    resource_efficiency: float = 0.0
    quality_trend: List[float] = field(default_factory=list)
    performance_history: List[PerformanceMetric] = field(default_factory=list)
    anomaly_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """
    Advanced performance monitoring system for agent ecosystem.
    Provides real-time monitoring, analytics, and predictive insights.
    """
    
    def __init__(self, 
                 monitoring_interval: int = 60,
                 history_retention_days: int = 30):
        self.monitoring_interval = monitoring_interval
        self.history_retention_days = history_retention_days
        
        # Data storage
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.alerts: List[PerformanceAlert] = []
        self.performance_thresholds: Dict[MetricType, Dict[str, float]] = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_callbacks: List[callable] = []
        
        # Analytics
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.predictor = PerformancePredictor()
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
        # Statistics
        self.monitor_stats = {
            "metrics_collected": 0,
            "alerts_generated": 0,
            "anomalies_detected": 0,
            "predictions_made": 0
        }
        
        logger.info("Performance monitor initialized",
                   monitoring_interval=monitoring_interval,
                   retention_days=history_retention_days)
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Performance monitoring stopped")
    
    def record_agent_execution(self, 
                             agent: BaseAgent,
                             result: AgentResult,
                             execution_context: Dict[str, Any] = None):
        """Record agent execution for performance tracking."""
        try:
            # Get or create agent profile
            profile = self._get_or_create_profile(agent)
            
            # Update basic counters
            profile.total_executions += 1
            if result.status == AgentStatus.COMPLETED:
                profile.successful_executions += 1
            else:
                profile.failed_executions += 1
            
            # Record metrics
            metrics = [
                PerformanceMetric(
                    agent_id=agent.agent_id,
                    metric_type=MetricType.EXECUTION_TIME,
                    value=result.execution_time,
                    timestamp=datetime.now(),
                    context=execution_context or {}
                ),
                PerformanceMetric(
                    agent_id=agent.agent_id,
                    metric_type=MetricType.CONFIDENCE_SCORE,
                    value=result.confidence_score,
                    timestamp=datetime.now(),
                    context=execution_context or {}
                ),
                PerformanceMetric(
                    agent_id=agent.agent_id,
                    metric_type=MetricType.SUCCESS_RATE,
                    value=1.0 if result.status == AgentStatus.COMPLETED else 0.0,
                    timestamp=datetime.now(),
                    context=execution_context or {}
                )
            ]
            
            # Add metrics to buffer and profile
            for metric in metrics:
                self.metrics_buffer.append(metric)
                profile.performance_history.append(metric)
                self.monitor_stats["metrics_collected"] += 1
            
            # Update derived metrics
            self._update_derived_metrics(profile)
            
            # Check for alerts
            self._check_performance_alerts(agent.agent_id, metrics)
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(
                agent.agent_id, metrics, profile.performance_history
            )
            
            if anomalies:
                profile.anomaly_count += len(anomalies)
                self.monitor_stats["anomalies_detected"] += len(anomalies)
                
                for anomaly in anomalies:
                    self._generate_anomaly_alert(agent.agent_id, anomaly)
            
            profile.last_updated = datetime.now()
            
            logger.debug("Agent execution recorded",
                        agent_id=agent.agent_id,
                        execution_time=result.execution_time,
                        success=result.status == AgentStatus.COMPLETED)
            
        except Exception as e:
            logger.error("Failed to record agent execution",
                        agent_id=agent.agent_id,
                        error=str(e))
    
    def get_agent_performance_summary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive performance summary for an agent."""
        if agent_id not in self.agent_profiles:
            return None
        
        profile = self.agent_profiles[agent_id]
        
        # Calculate recent performance (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_metrics = [
            m for m in profile.performance_history 
            if m.timestamp >= recent_cutoff
        ]
        
        # Calculate trends
        trends = self.trend_analyzer.analyze_trends(agent_id, profile.performance_history)
        
        # Get predictions
        predictions = self.predictor.predict_performance(agent_id, profile.performance_history)
        
        summary = {
            "agent_id": agent_id,
            "agent_type": profile.agent_type,
            "uptime_days": (datetime.now() - profile.creation_time).days,
            "total_executions": profile.total_executions,
            "success_rate": profile.successful_executions / max(1, profile.total_executions),
            "average_execution_time": profile.average_execution_time,
            "average_confidence": profile.average_confidence,
            "resource_efficiency": profile.resource_efficiency,
            "anomaly_count": profile.anomaly_count,
            "recent_performance": {
                "executions_24h": len(recent_metrics),
                "avg_execution_time_24h": np.mean([m.value for m in recent_metrics 
                                                  if m.metric_type == MetricType.EXECUTION_TIME]) 
                                         if recent_metrics else 0.0,
                "success_rate_24h": np.mean([m.value for m in recent_metrics 
                                           if m.metric_type == MetricType.SUCCESS_RATE]) 
                                   if recent_metrics else 0.0
            },
            "trends": trends,
            "predictions": predictions,
            "health_score": self._calculate_health_score(profile),
            "recommendations": self._generate_recommendations(profile, trends)
        }
        
        return summary
    
    def get_system_performance_overview(self) -> Dict[str, Any]:
        """Get system-wide performance overview."""
        if not self.agent_profiles:
            return {"message": "No agent data available"}
        
        # Aggregate metrics
        total_agents = len(self.agent_profiles)
        total_executions = sum(p.total_executions for p in self.agent_profiles.values())
        total_successful = sum(p.successful_executions for p in self.agent_profiles.values())
        
        # Calculate system-wide metrics
        system_success_rate = total_successful / max(1, total_executions)
        avg_execution_time = np.mean([p.average_execution_time for p in self.agent_profiles.values()])
        avg_confidence = np.mean([p.average_confidence for p in self.agent_profiles.values()])
        
        # Get top performers and underperformers
        sorted_agents = sorted(
            self.agent_profiles.values(),
            key=lambda p: p.successful_executions / max(1, p.total_executions),
            reverse=True
        )
        
        top_performers = sorted_agents[:3]
        underperformers = sorted_agents[-3:]
        
        # Recent activity (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.metrics_buffer if m.timestamp >= recent_cutoff]
        
        overview = {
            "system_health": {
                "total_agents": total_agents,
                "active_agents": len([p for p in self.agent_profiles.values() 
                                    if p.last_updated >= datetime.now() - timedelta(hours=1)]),
                "system_success_rate": system_success_rate,
                "average_execution_time": avg_execution_time,
                "average_confidence": avg_confidence,
                "total_executions": total_executions
            },
            "recent_activity": {
                "metrics_last_hour": len(recent_metrics),
                "active_alerts": len([a for a in self.alerts if not a.acknowledged]),
                "anomalies_detected": self.monitor_stats["anomalies_detected"]
            },
            "top_performers": [
                {
                    "agent_id": p.agent_id,
                    "success_rate": p.successful_executions / max(1, p.total_executions),
                    "executions": p.total_executions
                } for p in top_performers
            ],
            "underperformers": [
                {
                    "agent_id": p.agent_id,
                    "success_rate": p.successful_executions / max(1, p.total_executions),
                    "executions": p.total_executions
                } for p in underperformers
            ],
            "alerts_summary": self._get_alerts_summary(),
            "monitor_stats": self.monitor_stats
        }
        
        return overview
    
    def set_performance_threshold(self, 
                                metric_type: MetricType,
                                warning_threshold: float,
                                critical_threshold: float):
        """Set performance thresholds for alerting."""
        self.performance_thresholds[metric_type] = {
            "warning": warning_threshold,
            "critical": critical_threshold
        }
        
        logger.info("Performance threshold set",
                   metric_type=metric_type.value,
                   warning=warning_threshold,
                   critical=critical_threshold)
    
    def add_alert_callback(self, callback: callable):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a performance alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info("Alert acknowledged", alert_id=alert_id)
                return True
        
        return False
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        while self.monitoring_active:
            try:
                # Cleanup old data
                self._cleanup_old_data()
                
                # Update real-time metrics
                self._update_realtime_metrics()
                
                # Check for threshold violations
                self._check_system_health()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                time.sleep(self.monitoring_interval)
    
    def _get_or_create_profile(self, agent: BaseAgent) -> AgentPerformanceProfile:
        """Get existing profile or create new one for agent."""
        if agent.agent_id not in self.agent_profiles:
            self.agent_profiles[agent.agent_id] = AgentPerformanceProfile(
                agent_id=agent.agent_id,
                agent_type=agent.__class__.__name__,
                creation_time=datetime.now()
            )
        
        return self.agent_profiles[agent.agent_id]
    
    def _update_derived_metrics(self, profile: AgentPerformanceProfile):
        """Update derived metrics for agent profile."""
        if profile.total_executions == 0:
            return
        
        # Update averages
        execution_times = [
            m.value for m in profile.performance_history 
            if m.metric_type == MetricType.EXECUTION_TIME
        ]
        if execution_times:
            profile.average_execution_time = np.mean(execution_times)
        
        confidence_scores = [
            m.value for m in profile.performance_history 
            if m.metric_type == MetricType.CONFIDENCE_SCORE
        ]
        if confidence_scores:
            profile.average_confidence = np.mean(confidence_scores)
        
        # Calculate resource efficiency (inverse of execution time)
        if profile.average_execution_time > 0:
            profile.resource_efficiency = 1.0 / profile.average_execution_time
        
        # Update quality trend
        recent_confidence = confidence_scores[-10:] if confidence_scores else []
        if recent_confidence:
            profile.quality_trend = recent_confidence
    
    def _check_performance_alerts(self, agent_id: str, metrics: List[PerformanceMetric]):
        """Check metrics against thresholds and generate alerts."""
        for metric in metrics:
            if metric.metric_type not in self.performance_thresholds:
                continue
            
            thresholds = self.performance_thresholds[metric.metric_type]
            
            # Check critical threshold
            if metric.value <= thresholds.get("critical", 0):
                alert = PerformanceAlert(
                    alert_id=f"{agent_id}_{metric.metric_type.value}_{int(time.time())}",
                    agent_id=agent_id,
                    severity=AlertSeverity.CRITICAL,
                    metric_type=metric.metric_type,
                    message=f"Critical performance threshold violated for {metric.metric_type.value}",
                    threshold_value=thresholds["critical"],
                    actual_value=metric.value
                )
                self._add_alert(alert)
            
            # Check warning threshold
            elif metric.value <= thresholds.get("warning", 0):
                alert = PerformanceAlert(
                    alert_id=f"{agent_id}_{metric.metric_type.value}_{int(time.time())}",
                    agent_id=agent_id,
                    severity=AlertSeverity.WARNING,
                    metric_type=metric.metric_type,
                    message=f"Warning performance threshold violated for {metric.metric_type.value}",
                    threshold_value=thresholds["warning"],
                    actual_value=metric.value
                )
                self._add_alert(alert)
    
    def _add_alert(self, alert: PerformanceAlert):
        """Add alert and notify callbacks."""
        self.alerts.append(alert)
        self.monitor_stats["alerts_generated"] += 1
        
        # Limit alert history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))
        
        logger.warning("Performance alert generated",
                      alert_id=alert.alert_id,
                      severity=alert.severity.value,
                      message=alert.message)
    
    def _generate_anomaly_alert(self, agent_id: str, anomaly: Dict[str, Any]):
        """Generate alert for detected anomaly."""
        alert = PerformanceAlert(
            alert_id=f"{agent_id}_anomaly_{int(time.time())}",
            agent_id=agent_id,
            severity=AlertSeverity.WARNING,
            metric_type=MetricType.EXECUTION_TIME,  # Default
            message=f"Performance anomaly detected: {anomaly.get('description', 'Unknown')}",
            threshold_value=anomaly.get("expected_value", 0),
            actual_value=anomaly.get("actual_value", 0)
        )
        self._add_alert(alert)
    
    def _calculate_health_score(self, profile: AgentPerformanceProfile) -> float:
        """Calculate overall health score for agent (0-100)."""
        if profile.total_executions == 0:
            return 50.0  # Neutral score for new agents
        
        # Components of health score
        success_rate = profile.successful_executions / profile.total_executions
        confidence_score = profile.average_confidence
        efficiency_score = min(1.0, profile.resource_efficiency / 10.0)  # Normalize
        anomaly_penalty = max(0, 1.0 - (profile.anomaly_count / 100.0))
        
        # Weighted health score
        health_score = (
            success_rate * 0.4 +
            confidence_score * 0.3 +
            efficiency_score * 0.2 +
            anomaly_penalty * 0.1
        ) * 100
        
        return max(0, min(100, health_score))
    
    def _generate_recommendations(self, 
                                profile: AgentPerformanceProfile,
                                trends: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        success_rate = profile.successful_executions / max(1, profile.total_executions)
        
        if success_rate < 0.7:
            recommendations.append("Consider adjusting agent parameters to improve success rate")
        
        if profile.average_execution_time > 30.0:
            recommendations.append("Optimize agent configuration for better response time")
        
        if profile.average_confidence < 0.6:
            recommendations.append("Review agent prompts and training to improve confidence")
        
        if profile.anomaly_count > 10:
            recommendations.append("Investigate frequent anomalies and consider agent retraining")
        
        # Trend-based recommendations
        if trends.get("execution_time_trend") == "increasing":
            recommendations.append("Performance degradation detected - consider agent refresh")
        
        if trends.get("confidence_trend") == "decreasing":
            recommendations.append("Confidence declining - review recent task assignments")
        
        return recommendations
    
    def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts."""
        unacknowledged = [a for a in self.alerts if not a.acknowledged]
        
        by_severity = defaultdict(int)
        for alert in unacknowledged:
            by_severity[alert.severity.value] += 1
        
        return {
            "total_unacknowledged": len(unacknowledged),
            "by_severity": dict(by_severity),
            "recent_alerts": [
                {
                    "alert_id": a.alert_id,
                    "agent_id": a.agent_id,
                    "severity": a.severity.value,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in sorted(unacknowledged, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }
    
    def _initialize_default_thresholds(self):
        """Initialize default performance thresholds."""
        self.performance_thresholds = {
            MetricType.SUCCESS_RATE: {"warning": 0.7, "critical": 0.5},
            MetricType.EXECUTION_TIME: {"warning": 30.0, "critical": 60.0},
            MetricType.CONFIDENCE_SCORE: {"warning": 0.6, "critical": 0.4},
            MetricType.ERROR_RATE: {"warning": 0.3, "critical": 0.5}
        }
    
    def _cleanup_old_data(self):
        """Clean up old performance data."""
        cutoff_date = datetime.now() - timedelta(days=self.history_retention_days)
        
        # Clean up metrics buffer
        self.metrics_buffer = deque(
            [m for m in self.metrics_buffer if m.timestamp >= cutoff_date],
            maxlen=self.metrics_buffer.maxlen
        )
        
        # Clean up agent profiles
        for profile in self.agent_profiles.values():
            profile.performance_history = [
                m for m in profile.performance_history if m.timestamp >= cutoff_date
            ]
        
        # Clean up old alerts
        alert_cutoff = datetime.now() - timedelta(days=7)
        self.alerts = [a for a in self.alerts if a.timestamp >= alert_cutoff]
    
    def _update_realtime_metrics(self):
        """Update real-time system metrics."""
        # This would update system-wide metrics
        pass
    
    def _check_system_health(self):
        """Check overall system health and generate alerts if needed."""
        # This would perform system-wide health checks
        pass


class TrendAnalyzer:
    """Analyzes performance trends over time."""
    
    def analyze_trends(self, agent_id: str, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance trends for an agent."""
        if len(metrics) < 10:
            return {"insufficient_data": True}
        
        trends = {}
        
        # Group metrics by type
        by_type = defaultdict(list)
        for metric in metrics[-50:]:  # Last 50 data points
            by_type[metric.metric_type].append(metric.value)
        
        # Analyze each metric type
        for metric_type, values in by_type.items():
            if len(values) >= 5:
                trend = self._calculate_trend(values)
                trends[f"{metric_type.value}_trend"] = trend
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 3:
            return "stable"
        
        # Simple linear regression slope
        x = list(range(len(values)))
        slope = np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"


class AnomalyDetector:
    """Detects performance anomalies using statistical methods."""
    
    def detect_anomalies(self, 
                        agent_id: str,
                        recent_metrics: List[PerformanceMetric],
                        historical_metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Detect anomalies in recent metrics compared to historical data."""
        anomalies = []
        
        if len(historical_metrics) < 20:
            return anomalies  # Need sufficient history
        
        # Group by metric type
        historical_by_type = defaultdict(list)
        for metric in historical_metrics[:-len(recent_metrics)]:  # Exclude recent
            historical_by_type[metric.metric_type].append(metric.value)
        
        # Check each recent metric
        for metric in recent_metrics:
            historical_values = historical_by_type.get(metric.metric_type, [])
            
            if len(historical_values) >= 10:
                anomaly = self._detect_statistical_anomaly(metric, historical_values)
                if anomaly:
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_statistical_anomaly(self, 
                                   metric: PerformanceMetric,
                                   historical_values: List[float]) -> Optional[Dict[str, Any]]:
        """Detect anomaly using statistical methods."""
        if len(historical_values) < 10:
            return None
        
        mean_val = statistics.mean(historical_values)
        std_val = statistics.stdev(historical_values)
        
        # Z-score anomaly detection
        z_score = abs(metric.value - mean_val) / max(std_val, 0.001)
        
        if z_score > 3.0:  # 3 standard deviations
            return {
                "type": "statistical_outlier",
                "metric_type": metric.metric_type.value,
                "z_score": z_score,
                "expected_value": mean_val,
                "actual_value": metric.value,
                "description": f"Value {metric.value:.2f} is {z_score:.1f} standard deviations from mean {mean_val:.2f}"
            }
        
        return None


class PerformancePredictor:
    """Predicts future performance based on historical data."""
    
    def predict_performance(self, 
                          agent_id: str,
                          metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Predict future performance metrics."""
        if len(metrics) < 20:
            return {"insufficient_data": True}
        
        predictions = {}
        
        # Group by metric type
        by_type = defaultdict(list)
        for metric in metrics[-100:]:  # Last 100 data points
            by_type[metric.metric_type].append(metric.value)
        
        # Make predictions for each metric type
        for metric_type, values in by_type.items():
            if len(values) >= 10:
                prediction = self._simple_trend_prediction(values)
                predictions[f"{metric_type.value}_prediction"] = prediction
        
        return predictions
    
    def _simple_trend_prediction(self, values: List[float]) -> Dict[str, Any]:
        """Simple trend-based prediction."""
        if len(values) < 5:
            return {"error": "insufficient_data"}
        
        # Calculate trend
        recent_avg = np.mean(values[-5:])
        older_avg = np.mean(values[-10:-5]) if len(values) >= 10 else np.mean(values[:-5])
        
        trend = recent_avg - older_avg
        
        # Predict next value
        predicted_value = recent_avg + trend
        
        # Calculate confidence based on variance
        variance = np.var(values[-10:])
        confidence = max(0.1, min(0.9, 1.0 / (1.0 + variance)))
        
        return {
            "predicted_value": predicted_value,
            "confidence": confidence,
            "trend": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable"
        }


# Global monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor
