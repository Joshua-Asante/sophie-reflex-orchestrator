"""
Performance Monitor

Tracks performance metrics and identifies bottlenecks for system optimization.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class PerformanceTracker:
    """Tracks performance metrics for components."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.active_timers: Dict[str, float] = {}
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            tags=tags or {}
        )
        
        self.metrics[name].append(metric)
        logger.debug(f"Recorded metric {name}: {value}")
    
    def start_timer(self, name: str):
        """Start a timer for measuring duration."""
        self.active_timers[name] = time.time()
    
    def end_timer(self, name: str, tags: Dict[str, str] = None):
        """End a timer and record the duration."""
        if name not in self.active_timers:
            logger.warning(f"Timer {name} was not started")
            return
        
        duration = time.time() - self.active_timers[name]
        self.record_metric(f"{name}_duration", duration, tags)
        del self.active_timers[name]
    
    def get_metric_stats(self, name: str, window: int = None) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = [m.value for m in self.metrics[name]]
        
        if window:
            values = values[-window:]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "latest": values[-1]
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            name: self.get_metric_stats(name)
            for name in self.metrics.keys()
        }


class BottleneckDetector:
    """Detects performance bottlenecks in the system."""
    
    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
        self.thresholds = {
            "duration": 5.0,  # seconds
            "error_rate": 0.1,  # 10%
            "response_time": 2.0,  # seconds
        }
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        all_metrics = self.tracker.get_all_metrics()
        
        for metric_name, stats in all_metrics.items():
            if not stats:
                continue
            
            # Check for high duration
            if "duration" in metric_name and stats["mean"] > self.thresholds["duration"]:
                bottlenecks.append({
                    "type": "high_duration",
                    "metric": metric_name,
                    "value": stats["mean"],
                    "threshold": self.thresholds["duration"],
                    "severity": "high" if stats["mean"] > self.thresholds["duration"] * 2 else "medium"
                })
            
            # Check for high response time
            if "response_time" in metric_name and stats["mean"] > self.thresholds["response_time"]:
                bottlenecks.append({
                    "type": "high_response_time",
                    "metric": metric_name,
                    "value": stats["mean"],
                    "threshold": self.thresholds["response_time"],
                    "severity": "high" if stats["mean"] > self.thresholds["response_time"] * 2 else "medium"
                })
            
            # Check for high variance (indicating instability)
            if stats["std"] > stats["mean"] * 0.5:  # High coefficient of variation
                bottlenecks.append({
                    "type": "high_variance",
                    "metric": metric_name,
                    "value": stats["std"],
                    "threshold": stats["mean"] * 0.5,
                    "severity": "medium"
                })
        
        return bottlenecks
    
    def get_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Get recommendations for fixing bottlenecks."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "high_duration":
                recommendations.append(
                    f"Optimize {bottleneck['metric']} - current duration {bottleneck['value']:.2f}s "
                    f"exceeds threshold of {bottleneck['threshold']}s"
                )
            elif bottleneck["type"] == "high_response_time":
                recommendations.append(
                    f"Improve response time for {bottleneck['metric']} - current time {bottleneck['value']:.2f}s "
                    f"exceeds threshold of {bottleneck['threshold']}s"
                )
            elif bottleneck["type"] == "high_variance":
                recommendations.append(
                    f"Stabilize {bottleneck['metric']} - high variance indicates inconsistent performance"
                )
        
        return recommendations


class PerformanceMonitor:
    """High-level performance monitoring system."""
    
    def __init__(self):
        self.tracker = PerformanceTracker()
        self.detector = BottleneckDetector(self.tracker)
        self.component_timers: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def start_component_timer(self, component: str, operation: str):
        """Start timing a component operation."""
        timer_name = f"{component}_{operation}"
        self.tracker.start_timer(timer_name)
        self.component_timers[component][operation] = timer_name
    
    def end_component_timer(self, component: str, operation: str, tags: Dict[str, str] = None):
        """End timing a component operation."""
        if component in self.component_timers and operation in self.component_timers[component]:
            timer_name = self.component_timers[component][operation]
            self.tracker.end_timer(timer_name, tags)
            del self.component_timers[component][operation]
    
    def record_component_metric(self, component: str, metric: str, value: float, tags: Dict[str, str] = None):
        """Record a metric for a component."""
        metric_name = f"{component}_{metric}"
        self.tracker.record_metric(metric_name, value, tags)
    
    def get_component_performance(self, component: str) -> Dict[str, Any]:
        """Get performance statistics for a component."""
        component_metrics = {}
        
        for metric_name, stats in self.tracker.get_all_metrics().items():
            if metric_name.startswith(f"{component}_"):
                metric_type = metric_name[len(f"{component}_"):]
                component_metrics[metric_type] = stats
        
        return {
            "component": component,
            "metrics": component_metrics,
            "active_timers": list(self.component_timers.get(component, {}).keys())
        }
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance."""
        bottlenecks = self.detector.detect_bottlenecks()
        recommendations = self.detector.get_recommendations(bottlenecks)
        
        return {
            "overall_metrics": self.tracker.get_all_metrics(),
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "active_timers": {
                component: list(timers.keys())
                for component, timers in self.component_timers.items()
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a performance summary."""
        system_perf = self.get_system_performance()
        
        # Calculate overall health score
        total_metrics = len(system_perf["overall_metrics"])
        total_bottlenecks = len(system_perf["bottlenecks"])
        
        health_score = max(0, 100 - (total_bottlenecks * 20))  # Deduct 20 points per bottleneck
        
        return {
            "health_score": health_score,
            "total_metrics": total_metrics,
            "total_bottlenecks": total_bottlenecks,
            "critical_bottlenecks": len([b for b in system_perf["bottlenecks"] if b["severity"] == "high"]),
            "recommendations": system_perf["recommendations"][:5]  # Top 5 recommendations
        }


class AsyncPerformanceDecorator:
    """Decorator for measuring async function performance."""
    
    def __init__(self, monitor: PerformanceMonitor, component: str, operation: str):
        self.monitor = monitor
        self.component = component
        self.operation = operation
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            self.monitor.start_component_timer(self.component, self.operation)
            
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success
                self.monitor.record_component_metric(
                    self.component, 
                    f"{self.operation}_success", 
                    1.0,
                    {"duration": str(duration)}
                )
                
                return result
                
            except Exception as e:
                # Record failure
                self.monitor.record_component_metric(
                    self.component, 
                    f"{self.operation}_failure", 
                    1.0,
                    {"error": str(e)}
                )
                raise
            finally:
                self.monitor.end_component_timer(self.component, self.operation)
        
        return wrapper


# Global instance
performance_monitor = PerformanceMonitor()


def monitor_performance(component: str, operation: str):
    """Decorator for monitoring async function performance."""
    return AsyncPerformanceDecorator(performance_monitor, component, operation) 