"""
Plan Schema Module
Provides schema definitions for SOPHIE plans
"""

import json
from typing import Dict, Any

def get_plan_schema() -> Dict[str, Any]:
    """Get the plan schema definition."""
    return {
        "type": "object",
        "properties": {
            "agents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "goal": {"type": "string"},
                        "provider": {"type": "string", "enum": ["gemini", "openai"]},
                        "tools": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["name", "goal"]
                }
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "goal": {"type": "string"},
                        "agent": {"type": "string"},
                        "context": {"type": "object"}
                    },
                    "required": ["name", "goal", "agent"]
                }
            }
        },
        "required": ["agents", "tasks"]
    }

def validate_plan(plan_data: Dict[str, Any]) -> bool:
    """Validate a plan against the schema."""
    schema = get_plan_schema()
    
    # Basic validation
    if "agents" not in plan_data or "tasks" not in plan_data:
        return False
    
    # Validate agents
    for agent in plan_data["agents"]:
        if "name" not in agent or "goal" not in agent:
            return False
    
    # Validate tasks
    for task in plan_data["tasks"]:
        if "name" not in task or "goal" not in task or "agent" not in task:
            return False
    
    return True
