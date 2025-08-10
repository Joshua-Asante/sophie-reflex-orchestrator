# Repository Organization

## Overview

Note: In the monorepo, the Orchestrator lives under `apps/orchestrator/`. Paths in this document are relative to that directory.

The Sophie Reflex Orchestrator repository has been organized into logical directories to improve maintainability and discoverability.

## Directory Structure

### Core System (`core/`)

Core system components and utilities:

- `reasoning.py` - Core cognitive engine with ReAct loop
- `reasoning_modes.py` - Reasoning strategy definitions
- `adapter.py` - Unified tool execution interface
- `telemetry.py` - Logging and Prometheus metrics
- `tool_registry.py` - High-throughput tool registry with caching
- `graph_utils.py` - Graph algorithms including topological sorting

### Recovery & Safety (`recovery/`)

Failure recovery and safety mechanisms:

- `recovery_manager.py` - Manages failure recovery with LLM-based revisions
- `revision_engine.py` - Provides LLM-based step revision for failed tasks
- `reflective_pause.py` - Triggers system-wide reflective pause for safety

### Governance (`governance/`)

Governance and trust management:

- `feedback_handler.py` - Processes human feedback for trust management
- `bootstrap_engine.py` - Provides process improvement logging and ratification
- `trust_audit_log.py` - Comprehensive trust audit logging
- `trust_manager.py` - Trust management for governance operations
- `policy_engine.py` - Policy evaluation and enforcement
- `audit_log.py` - Audit logging system

### Security (`security/`)

Security and credential management:

- `security_manager.py` - Comprehensive security and credential management
- `store_credentials.py` - Credential setup and management interface

### Configuration (`configs/`)

Configuration files and schemas:

- `llm_registry.py` - LLM registry configuration
- `agents.yaml` - Agent configurations
- `policies.yaml` - Policy configurations
- `rubric.yaml` - Evaluation rubric
- `system.yaml` - System configuration
- `schemas/` - JSON schemas for validation

### Tools (`tools/`)

Tool ecosystem:

- `adapters/` - Tool adapter implementations
- `definitions/` - YAML tool definitions

### Planning (`planning/`)

Planning and execution:

- `plan_executor.py` - Parallel DAG-aware plan executor
- `plan_loader.py` - Fast, cached plan loader with schema parsing

### Service (`service/`)

Service layer components:

- `council_orchestrator.py` - Core coordinator for Council Mode execution
- `reflex_router.py` - Central cognitive router for the system
- `plan_generator.py` - Generates YAML execution plans
- `query_complexity_assessor.py` - Analyzes query complexity
- `consensus_tracker.py` - Evaluates agreement, dissent, and quorum

### Memory (`memory/`)

Memory management:

- `episodic_memory.py` - Stores discrete, timestamped episodes
- `longitudinal_memory.py` - Tracks long-range knowledge persistence
- `norm_memory.py` - Stores human-curated normative memory
- `semantic_memory.py` - Extracts and maintains high-level intent categories
- `working_memory.py` - Short-term, thread-safe memory buffer
- `memory_utils.py` - Common utilities for memory operations
- `memory_manager.py` - Unified memory manager

### Agents (`agents/`)

Agent implementations:

- `base_agent.py` - Base agent class
- `evaluator.py` - Evaluation agent
- `prover.py` - Proof generation agent
- `refiner.py` - Refinement agent

### Orchestrator (`orchestrator/`)

Main orchestration system:

- `core.py` - Main orchestrator
- `components/` - Orchestrator components

### UI (`ui/`)

User interface components:

- `cli_components.py` - CLI components
- `interactive_ui.py` - Interactive UI
- `webhook_server.py` - Webhook server
- `templates/` - UI templates
- `static/` - Static assets

### Data (`data/`)

Data storage:

- `trust_tracker.db` - Trust tracking database

### Documentation (`docs/`)

Documentation files:

- `webhook_server_fixes.md` - Webhook server fixes
- `ui_optimization_analysis.md` - UI optimization analysis
- `repository_organization.md` - This file

### Setup (`setup/`)

Setup and initialization:

- `setup_api_keys.py` - API key setup

### Tests (`tests/`)

Test files:

- `test_*.py` - Various test files
- `test_modules/` - Test modules

### Other Directories

- `logs/` - Log files
- `plans/` - Execution plans
- `models/` - Data models
- `utils/` - Utility functions
- `llm/` - LLM-related components
- `model_gateway/` - Model gateway components
- `council_mode/` - Council mode components
- `constitution/` - Constitutional components
- `chroma_db/` - Vector database
- `explainability/` - Explainability components

## Root Files

The following files remain in the root directory as they are the main entry points or configuration files:

- `main.py` - Main application entry point
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `docker-compose.yaml` - Docker Compose configuration
- `Dockerfile` - Docker configuration
- `LICENSE` - Project license

## Import Paths

After reorganization, some import paths may need to be updated. Common patterns:

```python
# Before
from reasoning import Reasoner
from reasoning_modes import ReasoningMode

# After
from core.reasoning import Reasoner
from core.reasoning_modes import ReasoningMode
```

## Benefits

1. **Improved Discoverability**: Related files are grouped together
2. **Better Maintainability**: Clear separation of concerns
3. **Easier Navigation**: Logical directory structure
4. **Reduced Clutter**: Root directory is much cleaner
5. **Clear Dependencies**: Dependencies between components are clearer

## Migration Notes

- All core system files moved to `core/`
- Recovery and safety files moved to `recovery/`
- Governance files moved to `governance/`
- Configuration files moved to `configs/`
- Tool files organized in `tools/`
- Documentation moved to `docs/`
- Setup files moved to `setup/`
- Data files moved to `data/`

Import statements in existing code may need to be updated to reflect the new structure.
