import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from sophie_shared.openrouter.capabilities import preferred_model_for

class LLMRegistry:
    """Loads and provides access to the LLM registry config."""
    _instance = None
    _models: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        """Loads models from the YAML registry file."""
        self._models.clear()
        try:
            registry_path = Path("configs/llm_registry.yaml")
            with registry_path.open() as f:
                models_list = yaml.safe_load(f)
            for model in models_list:
                self._models[model["name"]] = model
        except FileNotFoundError:
            # In a real app, this might be a fatal error.
            print("ERROR: llm_registry.yaml not found.")
        except yaml.YAMLError as e:
            print(f"ERROR: Could not parse llm_registry.yaml: {e}")

    def get_model(self, name: str) -> Dict[str, Any]:
        """Gets a model by name or resolves capability:NAME to preferred model."""
        if name.startswith("capability:"):
            cap = name.split(":", 1)[1]
            resolved: Optional[str] = preferred_model_for(cap)
            if not resolved:
                raise KeyError(f"No preferred model configured for capability: {cap}")
            name = resolved
        return self._models[name]

    def list_models(self) -> List[Dict[str, Any]]:
        """Lists all available models."""
        return list(self._models.values())
