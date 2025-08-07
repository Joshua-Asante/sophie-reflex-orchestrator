import os
import sys
import re
from getpass import getpass

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from security.security_manager import retrieve_credential, store_credential, generate_user_key

OPENAI_SERVICE = "openai"
BIGQUERY_SERVICE = "bigquery"
DEFAULT_USER = "default_user"


def sanitize_path(path: str) -> str:
    """Remove surrounding quotes and expand user paths"""
    # Remove surrounding quotes if present
    path = re.sub(r'^[\'"](.*)[\'"]$', r'\1', path.strip())
    # Expand ~ to home directory
    path = os.path.expanduser(path)
    # Normalize path separators
    path = os.path.normpath(path)
    return path


def ensure_credentials():
    generate_user_key()

    # BigQuery credentials
    bq_path = retrieve_credential(BIGQUERY_SERVICE, DEFAULT_USER)
    if bq_path:
        bq_path = sanitize_path(bq_path)

    if not bq_path or not os.path.exists(bq_path):
        print("BigQuery credentials not found or invalid.")
        while True:
            bq_path = input("Enter full path to your Google BigQuery JSON credentials file: ")
            bq_path = sanitize_path(bq_path)

            if os.path.isfile(bq_path) and bq_path.lower().endswith(".json"):
                store_credential(BIGQUERY_SERVICE, DEFAULT_USER, bq_path)
                break
            print(f"[✖] Invalid path: '{bq_path}'. Must be an existing JSON file.")

    # OpenAI API key
    openai_key = retrieve_credential(OPENAI_SERVICE, DEFAULT_USER)
    if not openai_key:
        openai_key = getpass("Enter your OpenAI API key (input hidden): ").strip()
        if not openai_key:
            print("[✖] OpenAI API key cannot be empty.")
            exit(1)
        store_credential(OPENAI_SERVICE, DEFAULT_USER, openai_key)

    print("[✔] Credentials ready.")
    print(f"BigQuery path: {bq_path}")
    print(f"OpenAI key: {'*' * min(len(openai_key), 8)}...")  # Show first 8 chars as stars


def main():
    print("=== Agent Smith Credential Setup ===")
    ensure_credentials()


if __name__ == "__main__":
    main()