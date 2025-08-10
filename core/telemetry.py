"""
core/telemetry.py
Opinionated structured logging + Prometheus metrics.
Public API unchanged:
    get_logger(name)      -> structlog.BoundLogger
    get_meter(name)       -> opentelemetry.metrics.Meter
"""

from __future__ import annotations

import os
import structlog
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from sophie_shared.telemetry import record_action
from prometheus_client import start_http_server

import json
import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import asyncio
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest

logger = logging.getLogger(__name__)

# Public API functions
def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)

def get_meter(name: str) -> metrics.Meter:
    """Get a metrics meter instance."""
    return metrics.get_meter(name)

# Prometheus metrics
REQUEST_COUNT = Counter('sophie_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('sophie_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
MODEL_INFERENCE_DURATION = Histogram('sophie_model_inference_duration_seconds', 'Model inference duration', ['model_name'])
CIRCUIT_BREAKER_STATE = Gauge('sophie_circuit_breaker_state', 'Circuit breaker state', ['breaker_name'])
EXPERT_SELECTION_DURATION = Histogram('sophie_expert_selection_duration_seconds', 'Expert selection duration')
DATABASE_CONNECTIONS = Gauge('sophie_database_connections', 'Database connections')
REDIS_HITS = Counter('sophie_redis_hits_total', 'Redis cache hits')
REDIS_MISSES = Counter('sophie_redis_misses_total', 'Redis cache misses')
MEMORY_USAGE = Gauge('sophie_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('sophie_cpu_usage_percent', 'CPU usage percentage')
ERROR_RATE = Counter('sophie_errors_total', 'Total errors', ['error_type', 'component'])

@dataclass
class TelemetryEvent:
    """Enhanced telemetry event with comprehensive metrics."""
    timestamp: float
    event_type: str
    component: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    duration: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    throughput_rps: float
    error_rate: float
    success_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    database_connections: int
    redis_hit_rate: float
    circuit_breaker_trips: int
    model_inference_time: float
    expert_selection_time: float

class EnhancedTelemetryManager:
    """Enhanced telemetry manager with comprehensive monitoring."""
    
    def __init__(self, log_file: str = "telemetry.jsonl", max_events: int = 10000):
        self.log_file = log_file
        self.max_events = max_events
        self.events = deque(maxlen=max_events)
        self.metrics_cache = {}
        self.performance_history = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.duration_history = defaultdict(lambda: deque(maxlen=100))
        
        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitoring_thread.start()
        
        # Initialize system metrics
        self._update_system_metrics()
    
    def log_event(self, event: TelemetryEvent):
        """Log a telemetry event with enhanced metrics."""
        try:
            # Add to in-memory queue
            self.events.append(event)
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(event)
            
            # Write to file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(asdict(event)) + '\n')
            
            # Update performance history
            if event.duration is not None:
                self.duration_history[event.event_type].append(event.duration)
            
            # Update success/error counts
            if event.success is not None:
                if event.success:
                    self.success_counts[event.event_type] += 1
                else:
                    self.error_counts[event.event_type] += 1
            
            # Emit lightweight JSONL record via shared package (scaffold)
            _ = record_action(event.event_type, {
                "component": event.component,
                "success": event.success,
                "metadata": event.metadata,
            })
            logger.debug(f"Telemetry event logged: {event.event_type}")
            
        except Exception as e:
            logger.error(f"Failed to log telemetry event: {e}")
    
    def _update_prometheus_metrics(self, event: TelemetryEvent):
        """Update Prometheus metrics based on event."""
        try:
            # Update request metrics
            if event.event_type == "api_request":
                method = event.metadata.get("method", "unknown")
                endpoint = event.metadata.get("endpoint", "unknown")
                status = event.metadata.get("status", "unknown")
                
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
                
                if event.duration:
                    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(event.duration)
            
            # Update model inference metrics
            elif event.event_type == "model_inference":
                model_name = event.metadata.get("model_name", "unknown")
                if event.duration:
                    MODEL_INFERENCE_DURATION.labels(model_name=model_name).observe(event.duration)
            
            # Update expert selection metrics
            elif event.event_type == "expert_selection":
                if event.duration:
                    EXPERT_SELECTION_DURATION.observe(event.duration)
            
            # Update error metrics
            if not event.success and event.error_message:
                error_type = event.metadata.get("error_type", "unknown")
                component = event.component
                ERROR_RATE.labels(error_type=error_type, component=component).inc()
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
    
    def _background_monitoring(self):
        """Background monitoring thread."""
        while True:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Update circuit breaker states
                self._update_circuit_breaker_metrics()
                
                # Update database metrics
                self._update_database_metrics()
                
                # Update Redis metrics
                self._update_redis_metrics()
                
                # Sleep for monitoring interval
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)
            
            # Store in cache
            self.metrics_cache["memory_usage_mb"] = memory.used / (1024 * 1024)
            self.metrics_cache["cpu_usage_percent"] = cpu_percent
            self.metrics_cache["memory_percent"] = memory.percent
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def _update_circuit_breaker_metrics(self):
        """Update circuit breaker metrics."""
        try:
            # This would typically check actual circuit breaker states
            # For now, simulate based on error rates
            for event_type, error_count in self.error_counts.items():
                success_count = self.success_counts.get(event_type, 0)
                total_count = error_count + success_count
                
                if total_count > 0:
                    error_rate = error_count / total_count
                    # Simulate circuit breaker state (0=closed, 1=open, 0.5=half-open)
                    breaker_state = 1.0 if error_rate > 0.5 else (0.5 if error_rate > 0.2 else 0.0)
                    CIRCUIT_BREAKER_STATE.labels(breaker_name=event_type).set(breaker_state)
            
        except Exception as e:
            logger.error(f"Failed to update circuit breaker metrics: {e}")
    
    def _update_database_metrics(self):
        """Update database connection metrics."""
        try:
            # This would typically check actual database connections
            # For now, simulate based on request patterns
            total_requests = sum(self.success_counts.values()) + sum(self.error_counts.values())
            estimated_connections = min(10, max(1, total_requests // 100))
            DATABASE_CONNECTIONS.set(estimated_connections)
            
        except Exception as e:
            logger.error(f"Failed to update database metrics: {e}")
    
    def _update_redis_metrics(self):
        """Update Redis cache metrics."""
        try:
            # This would typically check actual Redis metrics
            # For now, simulate based on cache patterns
            cache_events = [e for e in self.events if e.event_type == "cache_access"]
            if cache_events:
                hits = sum(1 for e in cache_events if e.metadata.get("cache_hit", False))
                total = len(cache_events)
                hit_rate = hits / total if total > 0 else 0.0
                
                # Update Prometheus metrics
                REDIS_HITS.inc(hits)
                REDIS_MISSES.inc(total - hits)
                
                # Store in cache
                self.metrics_cache["redis_hit_rate"] = hit_rate
            
        except Exception as e:
            logger.error(f"Failed to update Redis metrics: {e}")
    
    def get_performance_metrics(self, time_window: str = "1h") -> PerformanceMetrics:
        """Get comprehensive performance metrics."""
        try:
            # Calculate time window
            now = time.time()
            if time_window == "1h":
                cutoff_time = now - 3600
            elif time_window == "24h":
                cutoff_time = now - 86400
            else:
                cutoff_time = now - 3600  # Default to 1 hour
            
            # Filter events by time window
            recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
            
            # Calculate response time percentiles
            durations = [e.duration for e in recent_events if e.duration is not None]
            if durations:
                sorted_durations = sorted(durations)
                p50_idx = int(len(sorted_durations) * 0.5)
                p95_idx = int(len(sorted_durations) * 0.95)
                p99_idx = int(len(sorted_durations) * 0.99)
                
                response_time_p50 = sorted_durations[p50_idx] if p50_idx < len(sorted_durations) else 0
                response_time_p95 = sorted_durations[p95_idx] if p95_idx < len(sorted_durations) else 0
                response_time_p99 = sorted_durations[p99_idx] if p99_idx < len(sorted_durations) else 0
            else:
                response_time_p50 = response_time_p95 = response_time_p99 = 0
            
            # Calculate throughput
            total_requests = len(recent_events)
            time_span = now - cutoff_time
            throughput_rps = total_requests / time_span if time_span > 0 else 0
            
            # Calculate error rate
            errors = sum(1 for e in recent_events if e.success is False)
            error_rate = errors / total_requests if total_requests > 0 else 0
            success_rate = 1 - error_rate
            
            # Get system metrics from cache
            memory_usage_mb = self.metrics_cache.get("memory_usage_mb", 0)
            cpu_usage_percent = self.metrics_cache.get("cpu_usage_percent", 0)
            database_connections = self.metrics_cache.get("database_connections", 0)
            redis_hit_rate = self.metrics_cache.get("redis_hit_rate", 0)
            
            # Calculate circuit breaker trips
            circuit_breaker_trips = sum(1 for e in recent_events 
                                      if e.event_type == "circuit_breaker_trip")
            
            # Calculate model inference time
            model_events = [e for e in recent_events if e.event_type == "model_inference"]
            model_inference_time = sum(e.duration for e in model_events) / len(model_events) if model_events else 0
            
            # Calculate expert selection time
            expert_events = [e for e in recent_events if e.event_type == "expert_selection"]
            expert_selection_time = sum(e.duration for e in expert_events) / len(expert_events) if expert_events else 0
            
            return PerformanceMetrics(
                response_time_p50=response_time_p50,
                response_time_p95=response_time_p95,
                response_time_p99=response_time_p99,
                throughput_rps=throughput_rps,
                error_rate=error_rate,
                success_rate=success_rate,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                database_connections=database_connections,
                redis_hit_rate=redis_hit_rate,
                circuit_breaker_trips=circuit_breaker_trips,
                model_inference_time=model_inference_time,
                expert_selection_time=expert_selection_time
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return PerformanceMetrics(
                response_time_p50=0, response_time_p95=0, response_time_p99=0,
                throughput_rps=0, error_rate=0, success_rate=0,
                memory_usage_mb=0, cpu_usage_percent=0, database_connections=0,
                redis_hit_rate=0, circuit_breaker_trips=0,
                model_inference_time=0, expert_selection_time=0
            )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            # Get performance metrics
            performance = self.get_performance_metrics()
            
            # Get recent events summary
            recent_events = list(self.events)[-100:]  # Last 100 events
            event_summary = defaultdict(int)
            for event in recent_events:
                event_summary[event.event_type] += 1
            
            # Get error summary
            error_summary = defaultdict(int)
            for event in recent_events:
                if not event.success and event.error_message:
                    error_type = event.metadata.get("error_type", "unknown")
                    error_summary[error_type] += 1
            
            # Get user activity
            user_activity = defaultdict(int)
            for event in recent_events:
                if event.user_id:
                    user_activity[event.user_id] += 1
            
            # Get component performance
            component_performance = defaultdict(lambda: {"success": 0, "errors": 0, "avg_duration": 0})
            for event in recent_events:
                component = event.component
                if event.success is not None:
                    if event.success:
                        component_performance[component]["success"] += 1
                    else:
                        component_performance[component]["errors"] += 1
                
                if event.duration:
                    current_avg = component_performance[component]["avg_duration"]
                    count = component_performance[component]["success"] + component_performance[component]["errors"]
                    component_performance[component]["avg_duration"] = (current_avg * (count - 1) + event.duration) / count
            
            return {
                "performance": asdict(performance),
                "event_summary": dict(event_summary),
                "error_summary": dict(error_summary),
                "user_activity": dict(user_activity),
                "component_performance": {k: dict(v) for k, v in component_performance.items()},
                "system_metrics": self.metrics_cache,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        try:
            return generate_latest()
        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {e}")
            return ""
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        try:
            if format == "json":
                return json.dumps(self.get_dashboard_data(), indent=2)
            elif format == "prometheus":
                return self.get_prometheus_metrics()
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return ""

# Global telemetry manager instance
telemetry_manager = EnhancedTelemetryManager()

def log_telemetry_event(event_type: str, component: str, **kwargs):
    """Convenience function to log telemetry events."""
    event = TelemetryEvent(
        timestamp=time.time(),
        event_type=event_type,
        component=component,
        **kwargs
    )
    telemetry_manager.log_event(event)

def log_api_request(method: str, endpoint: str, duration: float, status: int, user_id: str = None):
    """Log API request telemetry."""
    log_telemetry_event(
        event_type="api_request",
        component="api",
        duration=duration,
        success=status < 400,
        metadata={
            "method": method,
            "endpoint": endpoint,
            "status": status
        },
        user_id=user_id
    )

def log_model_inference(model_name: str, duration: float, success: bool, tokens_used: int = None):
    """Log model inference telemetry."""
    log_telemetry_event(
        event_type="model_inference",
        component="llm",
        duration=duration,
        success=success,
        metadata={
            "model_name": model_name,
            "tokens_used": tokens_used
        }
    )

def log_expert_selection(duration: float, experts_selected: List[str], task_type: str):
    """Log expert selection telemetry."""
    log_telemetry_event(
        event_type="expert_selection",
        component="orchestrator",
        duration=duration,
        success=True,
        metadata={
            "experts_selected": experts_selected,
            "task_type": task_type
        }
    )

def log_cache_access(cache_hit: bool, cache_key: str, duration: float):
    """Log cache access telemetry."""
    log_telemetry_event(
        event_type="cache_access",
        component="cache",
        duration=duration,
        success=True,
        metadata={
            "cache_hit": cache_hit,
            "cache_key": cache_key
        }
    )

def log_circuit_breaker_trip(breaker_name: str, failure_count: int):
    """Log circuit breaker trip telemetry."""
    log_telemetry_event(
        event_type="circuit_breaker_trip",
        component="resilience",
        success=False,
        metadata={
            "breaker_name": breaker_name,
            "failure_count": failure_count
        }
    )