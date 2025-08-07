"""
Trust Audit Component

Integrated trust audit logging that combines with the existing audit system.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from governance.audit_log import AuditLog


class TrustAuditLog:
    """
    Integrated trust audit logging that works with the existing audit system.
    Logs every trust update to both the main audit log and a specialized trust log.
    """

    def __init__(self, log_dir: str = "logs"):
        self.is_enabled = os.getenv("ENABLE_TRUST_AUDIT", "True").lower() == "true"
        self.log_file_path: Optional[Path] = None
        self.main_audit_log: Optional[AuditLog] = None

        if self.is_enabled:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.log_file_path = log_path / "trust_audit.jsonl"
            print(f"✅ TrustAuditLog is ENABLED. Logging to: {self.log_file_path}")
        else:
            print("⚠️ TrustAuditLog is DISABLED.")

    def set_main_audit_log(self, audit_log: AuditLog):
        """Set the main audit log for integration."""
        self.main_audit_log = audit_log

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
        Logs a single trust update event to both specialized and main audit logs.
        """
        if not self.is_enabled:
            return

        # Create log entry
        log_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "source": source,
            "reason": reason,
            "trust_before": round(old_score, 4),
            "trust_after": round(new_score, 4),
            "trust_change": round(new_score - old_score, 4)
        }

        if protected_attributes:
            log_entry["protected_attributes"] = protected_attributes
        if outcome:
            log_entry["outcome"] = outcome

        # Log to specialized trust audit file
        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"❌ Failed to write trust audit log: {e}")

        # Log to main audit system
        if self.main_audit_log:
            try:
                self.main_audit_log.log_event(
                    event_type="trust_update",
                    event_data=log_entry,
                    session_id=session_id
                )
            except Exception as e:
                print(f"❌ Failed to write to main audit log: {e}")

    def get_trust_statistics(self) -> Dict[str, Any]:
        """Get statistics from the trust audit log."""
        if not self.log_file_path or not self.log_file_path.exists():
            return {"total_updates": 0, "sources": {}}

        try:
            updates = []
            sources = {}
            
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        updates.append(entry)
                        
                        source = entry.get("source", "unknown")
                        if source not in sources:
                            sources[source] = {"updates": 0, "total_change": 0.0}
                        
                        sources[source]["updates"] += 1
                        sources[source]["total_change"] += entry.get("trust_change", 0.0)

            return {
                "total_updates": len(updates),
                "sources": sources,
                "recent_updates": updates[-10:] if updates else []
            }
        except Exception as e:
            print(f"❌ Failed to read trust audit statistics: {e}")
            return {"total_updates": 0, "sources": {}, "error": str(e)} 