"""
Path configurations for Sophie Reflex Orchestrator
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
MEMORY_DIR = BASE_DIR / "memory"
TESTS_DIR = BASE_DIR / "tests"

# Memory-specific paths
EPISODIC_LOG_PATH = LOGS_DIR / "episodic_memory.jsonl"
LONGITUDINAL_LOG_PATH = LOGS_DIR / "longitudinal_memory.jsonl"
NORM_MEMORY_PATH = LOGS_DIR / "norm_memory.json"

# Database paths
TRUST_TRACKER_DB = MEMORY_DIR / "trust_tracker.db"
AUDIT_LOG_DB = MEMORY_DIR / "audit_log.db"
MEMORY_DB = MEMORY_DIR / "memory.db"

# Vector store paths
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)
CHROMA_DB_PATH.mkdir(exist_ok=True) 