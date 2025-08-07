"""
Orchestration Coordinator - Advanced workflow management and agent coordination.
Handles complex multi-agent workflows, dependency management, and execution optimization.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Set
import asyncio
import structlog
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import uuid

from .base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus
from .agent_registry import AgentRegistry, AgentCapability, get_agent_registry

logger = structlog.get_logger()


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(Enum):
    """Task priority levels for scheduling."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class WorkflowTask:
    """Individual task within a workflow."""
    task_id: str
    agent_type: str
    task_description: str
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: int = 300  # seconds
    retry_count: int = 3
    required_capabilities: Set[AgentCapability] = field(default_factory=set)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[AgentResult] = None
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None
    assigned_agent: Optional[BaseAgent] = None


@dataclass
class WorkflowDefinition:
    """Complete workflow definition with tasks and execution parameters."""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    global_timeout: int = 1800  # 30 minutes
    max_parallel_tasks: int = 5
    failure_strategy: str = "fail_fast"  # "fail_fast", "continue_on_error", "retry_failed"
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionContext:
    """Context for workflow execution with shared state."""
    workflow_id: str
    shared_data: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)


class OrchestrationCoordinator:
    """
    Advanced coordinator for managing complex multi-agent workflows.
    Provides dependency resolution, parallel execution, and intelligent scheduling.
    """
    
    def __init__(self, agent_registry: Optional[AgentRegistry] = None):
        self.agent_registry = agent_registry or get_agent_registry()
        self.active_workflows: Dict[str, WorkflowDefinition] = {}
        self.execution_contexts: Dict[str, ExecutionContext] = {}
        self.task_scheduler = TaskScheduler()
        self.dependency_resolver = DependencyResolver()
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Performance tracking
        self.orchestration_stats = {
            "workflows_executed": 0,
            "tasks_completed": 0,
            "average_workflow_time": 0.0,
            "success_rate": 0.0,
            "resource_efficiency": 0.0
        }
        
        logger.info("Orchestration coordinator initialized")
    
    async def execute_workflow(self, 
                             workflow_def: WorkflowDefinition,
                             initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a complete workflow with advanced orchestration."""
        workflow_id = workflow_def.workflow_id
        start_time = datetime.now()
        
        try:
            # Initialize execution context
            context = ExecutionContext(
                workflow_id=workflow_id,
                shared_data=initial_context or {}
            )
            self.execution_contexts[workflow_id] = context
            self.active_workflows[workflow_id] = workflow_def
            
            logger.info("Starting workflow execution", 
                       workflow_id=workflow_id,
                       task_count=len(workflow_def.tasks))
            
            # Build dependency graph
            dependency_graph = self.dependency_resolver.build_dependency_graph(workflow_def.tasks)
            
            # Validate workflow
            validation_result = await self._validate_workflow(workflow_def, dependency_graph)
            if not validation_result["valid"]:
                raise ValueError(f"Workflow validation failed: {validation_result['errors']}")
            
            # Execute workflow with intelligent scheduling
            execution_result = await self._execute_workflow_with_scheduling(
                workflow_def, dependency_graph, context
            )
            
            # Calculate final metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            success_rate = self._calculate_workflow_success_rate(workflow_def.tasks)
            
            # Update orchestration stats
            self._update_orchestration_stats(execution_time, success_rate)
            
            result = {
                "workflow_id": workflow_id,
                "status": WorkflowStatus.COMPLETED if success_rate > 0.5 else WorkflowStatus.FAILED,
                "execution_time": execution_time,
                "success_rate": success_rate,
                "task_results": {task.task_id: task.result for task in workflow_def.tasks},
                "shared_data": context.shared_data,
                "performance_metrics": context.performance_metrics
            }
            
            logger.info("Workflow execution completed", 
                       workflow_id=workflow_id,
                       status=result["status"],
                       execution_time=execution_time)
            
            return result
            
        except Exception as e:
            logger.error("Workflow execution failed", 
                        workflow_id=workflow_id, 
                        error=str(e))
            return {
                "workflow_id": workflow_id,
                "status": WorkflowStatus.FAILED,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
        
        finally:
            # Cleanup
            self.active_workflows.pop(workflow_id, None)
            self.execution_contexts.pop(workflow_id, None)
    
    async def _execute_workflow_with_scheduling(self,
                                              workflow_def: WorkflowDefinition,
                                              dependency_graph: nx.DiGraph,
                                              context: ExecutionContext) -> Dict[str, Any]:
        """Execute workflow with intelligent task scheduling."""
        completed_tasks = set()
        running_tasks = {}
        failed_tasks = set()
        
        # Get execution order from dependency graph
        execution_batches = self.dependency_resolver.get_execution_batches(dependency_graph)
        
        for batch_idx, batch_tasks in enumerate(execution_batches):
            logger.info("Executing task batch", 
                       batch_idx=batch_idx,
                       task_count=len(batch_tasks))
            
            # Execute tasks in current batch with parallelism control
            batch_results = await self._execute_task_batch(
                batch_tasks, workflow_def, context, workflow_def.max_parallel_tasks
            )
            
            # Process batch results
            for task_id, result in batch_results.items():
                task = next(t for t in workflow_def.tasks if t.task_id == task_id)
                task.result = result
                
                if result.status == AgentStatus.COMPLETED:
                    completed_tasks.add(task_id)
                    # Update shared context with task results
                    context.shared_data[f"task_{task_id}_result"] = result.result
                else:
                    failed_tasks.add(task_id)
                    
                    # Handle failure based on strategy
                    if workflow_def.failure_strategy == "fail_fast":
                        raise Exception(f"Task {task_id} failed: {result.error_message}")
                    elif workflow_def.failure_strategy == "retry_failed":
                        # Implement retry logic
                        retry_result = await self._retry_failed_task(task, context)
                        if retry_result.status == AgentStatus.COMPLETED:
                            completed_tasks.add(task_id)
                            failed_tasks.discard(task_id)
            
            # Check if we should continue
            if workflow_def.failure_strategy == "fail_fast" and failed_tasks:
                break
        
        return {
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "total_tasks": len(workflow_def.tasks)
        }
    
    async def _execute_task_batch(self,
                                batch_tasks: List[str],
                                workflow_def: WorkflowDefinition,
                                context: ExecutionContext,
                                max_parallel: int) -> Dict[str, AgentResult]:
        """Execute a batch of tasks with controlled parallelism."""
        # Get task objects
        tasks_to_execute = [
            next(t for t in workflow_def.tasks if t.task_id == task_id)
            for task_id in batch_tasks
        ]
        
        # Sort by priority
        tasks_to_execute.sort(key=lambda t: self._get_priority_value(t.priority), reverse=True)
        
        # Execute with semaphore for parallelism control
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_single_task(task: WorkflowTask) -> Tuple[str, AgentResult]:
            async with semaphore:
                return task.task_id, await self._execute_single_task(task, context)
        
        # Execute all tasks in batch
        task_coroutines = [execute_single_task(task) for task in tasks_to_execute]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        # Process results
        batch_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error("Task execution exception", error=str(result))
                continue
            
            task_id, agent_result = result
            batch_results[task_id] = agent_result
        
        return batch_results
    
    async def _execute_single_task(self, 
                                 task: WorkflowTask, 
                                 context: ExecutionContext) -> AgentResult:
        """Execute a single task with agent assignment and monitoring."""
        try:
            task.execution_start = datetime.now()
            task.status = WorkflowStatus.RUNNING
            
            # Find optimal agent for task
            agent = await self._assign_optimal_agent(task)
            if not agent:
                raise Exception(f"No suitable agent found for task {task.task_id}")
            
            task.assigned_agent = agent
            
            # Prepare task context
            task_context = {
                **context.shared_data,
                "task_id": task.task_id,
                "workflow_id": context.workflow_id,
                "dependencies_data": self._get_dependencies_data(task, context)
            }
            
            # Execute task with timeout
            result = await asyncio.wait_for(
                agent.execute(task.task_description, task_context),
                timeout=task.timeout
            )
            
            task.execution_end = datetime.now()
            task.status = WorkflowStatus.COMPLETED if result.status == AgentStatus.COMPLETED else WorkflowStatus.FAILED
            
            # Update context performance metrics
            execution_time = (task.execution_end - task.execution_start).total_seconds()
            context.performance_metrics[task.task_id] = {
                "execution_time": execution_time,
                "agent_used": agent.agent_id,
                "success": result.status == AgentStatus.COMPLETED
            }
            
            logger.info("Task completed", 
                       task_id=task.task_id,
                       execution_time=execution_time,
                       status=result.status)
            
            return result
            
        except asyncio.TimeoutError:
            task.status = WorkflowStatus.FAILED
            error_result = AgentResult(
                agent_id=task.assigned_agent.agent_id if task.assigned_agent else "unknown",
                agent_name=task.agent_type,
                result=None,
                confidence_score=0.0,
                execution_time=task.timeout,
                status=AgentStatus.FAILED,
                error_message=f"Task timeout after {task.timeout} seconds"
            )
            return error_result
            
        except Exception as e:
            task.status = WorkflowStatus.FAILED
            error_result = AgentResult(
                agent_id=task.assigned_agent.agent_id if task.assigned_agent else "unknown",
                agent_name=task.agent_type,
                result=None,
                confidence_score=0.0,
                execution_time=0.0,
                status=AgentStatus.FAILED,
                error_message=str(e)
            )
            return error_result
    
    async def _assign_optimal_agent(self, task: WorkflowTask) -> Optional[BaseAgent]:
        """Assign the most suitable agent for a task."""
        # Discover agents by capability
        suitable_agents = self.agent_registry.discover_agents_by_capability(
            task.required_capabilities,
            performance_tier=None  # Let registry decide
        )
        
        if not suitable_agents:
            # Fallback to agent type if no capability match
            return self.agent_registry.create_agent_instance(task.agent_type)
        
        # Get optimal combination (single agent in this case)
        optimal_agents = self.agent_registry.get_optimal_agent_combination(
            {
                "capabilities": list(task.required_capabilities),
                "resource_constraints": task.resource_requirements,
                "complexity": "medium"
            },
            max_agents=1
        )
        
        if optimal_agents:
            agent_name = optimal_agents[0]["agent_name"]
            return self.agent_registry.create_agent_instance(agent_name)
        
        return None
    
    def _get_dependencies_data(self, 
                             task: WorkflowTask, 
                             context: ExecutionContext) -> Dict[str, Any]:
        """Get data from completed dependency tasks."""
        dependencies_data = {}
        
        for dep_task_id in task.dependencies:
            dep_data_key = f"task_{dep_task_id}_result"
            if dep_data_key in context.shared_data:
                dependencies_data[dep_task_id] = context.shared_data[dep_data_key]
        
        return dependencies_data
    
    async def _retry_failed_task(self, 
                               task: WorkflowTask, 
                               context: ExecutionContext) -> AgentResult:
        """Retry a failed task with exponential backoff."""
        for attempt in range(task.retry_count):
            try:
                # Wait with exponential backoff
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                
                logger.info("Retrying failed task", 
                           task_id=task.task_id,
                           attempt=attempt + 1)
                
                # Re-execute task
                result = await self._execute_single_task(task, context)
                
                if result.status == AgentStatus.COMPLETED:
                    logger.info("Task retry successful", task_id=task.task_id)
                    return result
                    
            except Exception as e:
                logger.warning("Task retry failed", 
                              task_id=task.task_id,
                              attempt=attempt + 1,
                              error=str(e))
        
        # All retries failed
        return AgentResult(
            agent_id="retry_system",
            agent_name=task.agent_type,
            result=None,
            confidence_score=0.0,
            execution_time=0.0,
            status=AgentStatus.FAILED,
            error_message=f"Task failed after {task.retry_count} retries"
        )
    
    async def _validate_workflow(self, 
                               workflow_def: WorkflowDefinition,
                               dependency_graph: nx.DiGraph) -> Dict[str, Any]:
        """Validate workflow definition and dependencies."""
        errors = []
        
        # Check for circular dependencies
        if not nx.is_directed_acyclic_graph(dependency_graph):
            errors.append("Circular dependencies detected")
        
        # Validate task dependencies exist
        task_ids = {task.task_id for task in workflow_def.tasks}
        for task in workflow_def.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    errors.append(f"Task {task.task_id} depends on non-existent task {dep}")
        
        # Check agent availability
        for task in workflow_def.tasks:
            suitable_agents = self.agent_registry.discover_agents_by_capability(
                task.required_capabilities
            )
            if not suitable_agents and task.agent_type not in self.agent_registry.registered_agents:
                errors.append(f"No suitable agent found for task {task.task_id}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _calculate_workflow_success_rate(self, tasks: List[WorkflowTask]) -> float:
        """Calculate success rate for completed workflow."""
        if not tasks:
            return 0.0
        
        successful_tasks = sum(
            1 for task in tasks 
            if task.result and task.result.status == AgentStatus.COMPLETED
        )
        
        return successful_tasks / len(tasks)
    
    def _update_orchestration_stats(self, execution_time: float, success_rate: float):
        """Update orchestration performance statistics."""
        self.orchestration_stats["workflows_executed"] += 1
        
        # Update average execution time
        current_avg = self.orchestration_stats["average_workflow_time"]
        workflow_count = self.orchestration_stats["workflows_executed"]
        self.orchestration_stats["average_workflow_time"] = (
            (current_avg * (workflow_count - 1) + execution_time) / workflow_count
        )
        
        # Update success rate
        current_success = self.orchestration_stats["success_rate"]
        self.orchestration_stats["success_rate"] = (
            (current_success * (workflow_count - 1) + success_rate) / workflow_count
        )
    
    def _get_priority_value(self, priority: TaskPriority) -> int:
        """Convert priority enum to numeric value for sorting."""
        priority_values = {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        return priority_values.get(priority, 1)
    
    def create_workflow_from_template(self, 
                                    template_name: str,
                                    parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Create workflow from predefined template."""
        # This would load from a template registry
        # For now, return a simple example
        workflow_id = str(uuid.uuid4())
        
        if template_name == "analyze_and_refine":
            return WorkflowDefinition(
                workflow_id=workflow_id,
                name="Analyze and Refine",
                description="Analyze input and refine results",
                tasks=[
                    WorkflowTask(
                        task_id="analyze",
                        agent_type="prover",
                        task_description=parameters.get("analysis_task", "Analyze the input"),
                        input_data=parameters.get("input_data", {}),
                        required_capabilities={AgentCapability.PROBLEM_SOLVING}
                    ),
                    WorkflowTask(
                        task_id="evaluate",
                        agent_type="evaluator", 
                        task_description="Evaluate the analysis results",
                        input_data={},
                        dependencies=["analyze"],
                        required_capabilities={AgentCapability.EVALUATION}
                    ),
                    WorkflowTask(
                        task_id="refine",
                        agent_type="refiner",
                        task_description="Refine based on evaluation",
                        input_data={},
                        dependencies=["analyze", "evaluate"],
                        required_capabilities={AgentCapability.REFINEMENT}
                    )
                ]
            )
        
        raise ValueError(f"Unknown template: {template_name}")
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics."""
        return {
            **self.orchestration_stats,
            "active_workflows": len(self.active_workflows),
            "registry_stats": self.agent_registry.get_registry_stats()
        }


class TaskScheduler:
    """Advanced task scheduler with priority and resource management."""
    
    def __init__(self):
        self.scheduling_strategies = {
            "priority_first": self._priority_first_scheduling,
            "resource_aware": self._resource_aware_scheduling,
            "deadline_driven": self._deadline_driven_scheduling
        }
    
    def _priority_first_scheduling(self, tasks: List[WorkflowTask]) -> List[WorkflowTask]:
        """Schedule tasks by priority."""
        return sorted(tasks, key=lambda t: self._get_priority_value(t.priority), reverse=True)
    
    def _resource_aware_scheduling(self, tasks: List[WorkflowTask]) -> List[WorkflowTask]:
        """Schedule tasks considering resource requirements."""
        # Sort by resource requirements (lighter tasks first)
        return sorted(tasks, key=lambda t: sum(t.resource_requirements.values()))
    
    def _deadline_driven_scheduling(self, tasks: List[WorkflowTask]) -> List[WorkflowTask]:
        """Schedule tasks by deadline/timeout."""
        return sorted(tasks, key=lambda t: t.timeout)
    
    def _get_priority_value(self, priority: TaskPriority) -> int:
        """Convert priority to numeric value."""
        return {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3, 
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }.get(priority, 1)


class DependencyResolver:
    """Resolves task dependencies and creates execution order."""
    
    def build_dependency_graph(self, tasks: List[WorkflowTask]) -> nx.DiGraph:
        """Build directed graph of task dependencies."""
        graph = nx.DiGraph()
        
        # Add all tasks as nodes
        for task in tasks:
            graph.add_node(task.task_id, task=task)
        
        # Add dependency edges
        for task in tasks:
            for dependency in task.dependencies:
                graph.add_edge(dependency, task.task_id)
        
        return graph
    
    def get_execution_batches(self, dependency_graph: nx.DiGraph) -> List[List[str]]:
        """Get batches of tasks that can be executed in parallel."""
        batches = []
        remaining_nodes = set(dependency_graph.nodes())
        
        while remaining_nodes:
            # Find nodes with no dependencies in remaining set
            current_batch = []
            for node in remaining_nodes:
                dependencies = set(dependency_graph.predecessors(node))
                if dependencies.issubset(set(dependency_graph.nodes()) - remaining_nodes):
                    current_batch.append(node)
            
            if not current_batch:
                # This shouldn't happen with a valid DAG
                raise Exception("Circular dependency detected")
            
            batches.append(current_batch)
            remaining_nodes -= set(current_batch)
        
        return batches
