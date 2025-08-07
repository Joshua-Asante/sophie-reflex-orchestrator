import yaml
from pathlib import Path
from typing import Dict, List, Any

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
        """Gets a model by name."""
        return self._models[name]

    def list_models(self) -> List[Dict[str, Any]]:
        """Lists all available models."""
        return list(self._models.values())