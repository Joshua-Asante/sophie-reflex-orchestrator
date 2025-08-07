from datetime import datetime
from pathlib import Path
import uuid
import yaml


class BootstrapEngine:
    def __init__(self, log_path='constitution/process_improvements.yaml'):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure the file exists and is a valid YAML list
        if not self.log_path.exists():
            with self.log_path.open('w', encoding='utf-8') as f:
                yaml.dump([], f)

    def log_improvement(self, entry: dict):
        """
        Appends a validated, versioned entry to the process improvement log
        under a read-modify-write safety model.
        """
        entry_id = f"PI-{str(uuid.uuid4())[:8].upper()}"
        entry['entry_id'] = entry_id
        entry['date'] = datetime.utcnow().isoformat() + "Z"

        try:
            with self.log_path.open('r', encoding='utf-8') as f:
                log_data = yaml.safe_load(f) or []

            log_data.append(entry)

            with self.log_path.open('w', encoding='utf-8') as f:
                yaml.dump(log_data, f, sort_keys=False, indent=2)

            print(f"[✓] Logged improvement: {entry_id}")
        except Exception as e:
            print(f"[!] Failed to log improvement: {e}")

    def stage_improvement_for_ratification(self, entry: dict, stage_path='logs/staged_improvements.yaml'):
        """
        Stages a process improvement entry for manual ratification rather than 
        immediately writing it to the immutable constitution. For future separation of powers.
        """
        stage_file = Path(stage_path)
        stage_file.parent.mkdir(parents=True, exist_ok=True)
        entry['staged'] = True

        try:
            existing = []
            if stage_file.exists():
                with stage_file.open('r', encoding='utf-8') as f:
                    existing = yaml.safe_load(f) or []

            existing.append(entry)

            with stage_file.open('w', encoding='utf-8') as f:
                yaml.dump(existing, f, sort_keys=False, indent=2)

            print(f"[⚖] Improvement staged for ratification.")
        except Exception as e:
            print(f"[!] Failed to stage improvement: {e}")
