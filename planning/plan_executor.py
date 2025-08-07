"""
planning/plan_executor.py
Parallel DAG-aware plan executor.
Public API: `async execute_plan(plan_def, initial_context=None) -> dict`
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Tuple

from core.adapter import execute
from core.telemetry import get_logger
from core.graph_utils import topological_sort  # reused DAG ordering

_logger = get_logger("plan_executor")

# ---------- placeholder resolver ----------
_PLACEHOLDER_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}")

def _resolve_params(params: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """In-place replacement of {{ steps.x.y }} placeholders."""
    resolved = {}
    for k, v in params.items():
        if isinstance(v, str) and _PLACEHOLDER_RE.search(v):
            keys = _PLACEHOLDER_RE.search(v).group(1).split(".")
            try:
                ref = state
                for part in keys:
                    ref = ref[part]
                resolved[k] = ref
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Unresolved placeholder: {'.'.join(keys)}") from exc
        else:
            resolved[k] = v
    return resolved


# ---------- public executor ----------
async def execute_plan(
    plan_def: Dict[str, Any], initial_context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Execute a validated plan in parallel where dependencies allow."""
    state = {"context": initial_context or {}, "steps": {}}
    plan_name = plan_def.get("plan_name", "untitled")

    # Build DAG if we ever add depends_on; today the schema is linear
    tasks = {step["name"]: step for step in plan_def["steps"]}
    order = topological_sort(
        {k: {"depends_on": []} for k in tasks}
    )  # returns names in correct order

    _logger.info("Starting plan", plan=plan_name, steps=len(order))

    for batch in _batches_in_topological_order(order, tasks):
        coros = [_run_single_step(step, state) for step in batch]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for maybe_exc in results:
            if isinstance(maybe_exc, Exception):
                _logger.error("Step failed", exc=maybe_exc)
                raise maybe_exc

    _logger.info("Plan finished", plan=plan_name)
    return state


# ---------- internal helpers ----------
async def _run_single_step(step: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Run one step and mutate state."""
    name = step["name"]
    tool = step["tool"]
    _logger.debug("Running step", step=name, tool=tool)

    resolved = _resolve_params(step.get("params", {}), state)
    output = await execute(tool, resolved)
    state["steps"][name] = {"output": output}


def _batches_in_topological_order(
    ordered_names: List[str], task_map: Dict[str, Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    """
    Today the schema is linear (no depends_on); we return one batch per step
    to keep determinism. When DAG is introduced we can group ready nodes.
    """
    return [[task_map[name]] for name in ordered_names]