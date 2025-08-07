import os
import keyring
import json
import base64
from cryptography.fernet import Fernet

# Paths
CONFIG_DIR = os.path.expanduser("~/.agent_smith")
KEY_FILE = os.path.join(CONFIG_DIR, "encryption.key")
CLOUD_SYNC_FILE = os.path.join(CONFIG_DIR, "cloud_sync.json")


def get_bigquery_credentials(username="default_user"):
    try:
        # Import only when needed
        from google.oauth2 import service_account
    except ImportError:
        raise RuntimeError("Google authentication requires 'google-auth' package. Run: pip install google-auth")

    path = retrieve_credential("bigquery", username)
    if not path:
        raise FileNotFoundError(f"[✖] No credential path found in keyring for user '{username}'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"[✖] Credential file not found at: {path}. Did you move or delete it?")

    return service_account.Credentials.from_service_account_file(path)
# Ensure config directory exists
os.makedirs(CONFIG_DIR, exist_ok=True)

# -----------------------------
# User Provisioning Flow
# -----------------------------
def generate_user_key():
    """Generate and store encryption key locally if it doesn't exist."""
    try:
        if not os.path.exists(KEY_FILE):
            key = Fernet.generate_key()
            with open(KEY_FILE, 'wb') as f:
                f.write(key)
            print("[✔] New encryption key generated and stored locally.")
        else:
            print("[ℹ] Encryption key already exists.")
    except Exception as e:
        print(f"[✖] Failed to generate user key: {e}")

def load_user_key():
    """Load the encryption key from local storage."""
    try:
        with open(KEY_FILE, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError("Encryption key not found. Run provisioning flow.")
    except Exception as e:
        print(f"[✖] Failed to load encryption key: {e}")
        raise

# -----------------------------
# Secure Credential Manager
# -----------------------------
def store_credential(service_name: str, username: str, password: str):
    """Store credentials securely using keyring."""
    if not all([service_name, username, password]):
        print("[✖] Cannot store empty credential values.")
        return
    try:
        keyring.set_password(service_name, username, password)
        print(f"[✔] Credential for {service_name} stored securely.")
    except Exception as e:
        print(f"[✖] Failed to store credential: {e}")

def retrieve_credential(service_name: str, username: str):
    """Retrieve stored credentials."""
    try:
        return keyring.get_password(service_name, username)
    except Exception as e:
        print(f"[✖] Failed to retrieve credential: {e}")
        return None

# -----------------------------
# Cloud Data Encryption Layer
# -----------------------------
def encrypt_data(data: dict) -> str:
    """Encrypt a dictionary using Fernet."""
    try:
        key = load_user_key()
        f = Fernet(key)
        encoded = json.dumps(data).encode()
        encrypted = f.encrypt(encoded)
        return encrypted.decode()
    except Exception as e:
        print(f"[✖] Data encryption failed: {e}")
        return ""

def decrypt_data(encrypted_data: str) -> dict:
    """Decrypt a Fernet-encrypted string into a dictionary."""
    try:
        key = load_user_key()
        f = Fernet(key)
        decrypted = f.decrypt(encrypted_data.encode())
        return json.loads(decrypted.decode())
    except Exception as e:
        print(f"[✖] Data decryption failed: {e}")
        return {}

# -----------------------------
# Simulated Cloud Sync Layer
# -----------------------------
def sync_to_cloud(data: dict):
    """Encrypt and save data to simulated cloud (local file)."""
    try:
        encrypted = encrypt_data(data)
        if not encrypted:
            print("[✖] Failed to encrypt data. Aborting cloud sync.")
            return
        with open(CLOUD_SYNC_FILE, 'w') as f:
            json.dump({"encrypted_data": encrypted}, f)
        print("[✔] Data synced to cloud (simulated as local file).")
    except Exception as e:
        print(f"[✖] Cloud sync failed: {e}")

def load_from_cloud() -> dict:
    """Load and decrypt cloud-synced data from local file."""
    try:
        with open(CLOUD_SYNC_FILE, 'r') as f:
            encrypted = json.load(f)["encrypted_data"]
        return decrypt_data(encrypted)
    except Exception as e:
        print(f"[✖] Failed to load from cloud: {e}")
        return {}
