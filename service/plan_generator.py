# modules/planning/plan_generator.py

import os
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Mock model_router for now
def model_router(provider):
    class MockRouter:
        def query(self, system_prompt, user_prompt):
            return f"Mock response from {provider}: {user_prompt[:50]}..."
    return MockRouter()

load_dotenv()

def generate_plan(goal: str, output_dir: Path, provider: Optional[str] = None) -> Path:
    """
    Generates a YAML execution plan from a natural language goal using the specified LLM provider.
    Saves the plan to a file in output_dir and returns the Path to the saved file.
    """
    try:
        print(f"[SOPHIE] Generating plan with provider: {provider or 'auto'}")
        response = model_router(provider or "auto").query(
            system_prompt="""
You are a task planning assistant for an agent named SOPHIE.
Your job is to generate a valid YAML execution plan based on a user's goal.
Use proper indentation and ensure the YAML is correct.
            """.strip(),
            user_prompt=goal,
        )

        # Ensure we only parse YAML content if wrapped in code block
        if "```yaml" in response:
            yaml_block = response.split("```yaml")[1].split("```")[0].strip()
        else:
            yaml_block = response.strip()

        parsed_yaml = yaml.safe_load(yaml_block)
        plan_path = output_dir / "generated_plan.yaml"
        plan_path.write_text(yaml.dump(parsed_yaml, sort_keys=False), encoding="utf-8")

        print(f"[SOPHIE] Plan written to {plan_path}")
        return plan_path

    except Exception as e:
        raise RuntimeError(f"Plan generation failed using provider {provider}: {e}")
