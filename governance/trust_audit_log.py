# core/trust_audit_log.py

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict


class TrustAuditLog:
    """
    Logs every trust update to a persistent JSON Lines file.
    Also supports logging optional protected attributes and decision outcomes.
    """

    def __init__(self, log_dir: str = "logs"):
        self.is_enabled = os.getenv("ENABLE_AUDIT_LOG", "True").lower() == "true"
        self.log_file_path: Optional[Path] = None

        if self.is_enabled:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.log_file_path = log_path / "trust_audit.jsonl"
            print(f"✅ TrustAuditLog is ENABLED. Logging to: {self.log_file_path}")
        else:
            print("⚠️ TrustAuditLog is DISABLED.")

    def log_update(
        self,
        source: str,
        old_score: float,
        new_score: float,
        reason: str,
        session_id: str,
        protected_attributes: Optional[Dict[str, str]] = None,
        outcome: Optional[str] = None
    ):
        """
        Logs a single trust update event. Skips if logging is disabled.
        """
        if not self.is_enabled or self.log_file_path is None:
            return

        log_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "source": source,
            "reason": reason,
            "trust_before": round(old_score, 4),
            "trust_after": round(new_score, 4)
        }

        if protected_attributes:
            log_entry["protected_attributes"] = protected_attributes
        if outcome:
            log_entry["outcome"] = outcome

        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"❌ Failed to write audit log: {e}")
