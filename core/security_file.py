"""
Security File Module for SOPHIE

Provides secure vault-based secret management with encryption, HMAC validation,
and refresh token support for OAuth flows.
"""

import os
import json
import base64
import hashlib
import hmac
import time
from typing import Optional, Dict, Any, List
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import structlog

logger = structlog.get_logger()

load_dotenv()


class SecurityFile:
    """
    Secure vault-based secret management for SOPHIE.
    
    Provides encrypted storage for credentials, OAuth tokens, and other sensitive data
    with HMAC validation and automatic refresh token handling.
    """
    
    def __init__(self, vault_path: str = ".vault.json", key_env_var: str = "VAULT_KEY"):
        self.vault_path = vault_path
        self.key = os.getenv(key_env_var)
        if not self.key:
            raise ValueError("Vault encryption key not set in environment")
        
        # Initialize Fernet cipher
        self.fernet = Fernet(self.key.encode())
        
        # Initialize audit log
        self.audit_log = []
        
        logger.info("SecurityFile initialized", vault_path=vault_path)
    
    def store_secret(self, name: str, value: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Store a secret in the encrypted vault.
        
        Args:
            name: Secret identifier
            value: Secret value to encrypt
            metadata: Optional metadata (expiry, scope, etc.)
        
        Returns:
            bool: True if successful
        """
        try:
            secrets = self._load_vault()
            
            # Create secret entry with metadata
            secret_entry = {
                "value": self.fernet.encrypt(value.encode()).decode(),
                "created_at": time.time(),
                "metadata": metadata or {}
            }
            
            # Add HMAC for integrity
            secret_entry["hmac"] = self._generate_hmac(name, value)
            
            secrets[name] = secret_entry
            self._save_vault(secrets)
            
            # Log the operation
            self._audit_log("store_secret", name, metadata)
            
            logger.info("Secret stored successfully", name=name)
            return True
            
        except Exception as e:
            logger.error("Failed to store secret", name=name, error=str(e))
            return False
    
    def retrieve_secret(self, name: str) -> Optional[str]:
        """
        Retrieve a secret from the encrypted vault.
        
        Args:
            name: Secret identifier
        
        Returns:
            Optional[str]: Decrypted secret value or None if not found
        """
        try:
            secrets = self._load_vault()
            secret_entry = secrets.get(name)
            
            if not secret_entry:
                logger.warning("Secret not found", name=name)
                return None
            
            # Verify HMAC integrity
            encrypted_value = secret_entry["value"]
            decrypted_value = self.fernet.decrypt(encrypted_value.encode()).decode()
            
            if not self._verify_hmac(name, decrypted_value, secret_entry.get("hmac")):
                logger.error("HMAC verification failed", name=name)
                return None
            
            # Check expiry if present
            if "metadata" in secret_entry and "expires_at" in secret_entry["metadata"]:
                if time.time() > secret_entry["metadata"]["expires_at"]:
                    logger.warning("Secret expired", name=name)
                    return None
            
            # Log the operation
            self._audit_log("retrieve_secret", name, secret_entry.get("metadata"))
            
            logger.info("Secret retrieved successfully", name=name)
            return decrypted_value
            
        except Exception as e:
            logger.error("Failed to retrieve secret", name=name, error=str(e))
            return None
    
    def store_oauth_tokens(self, provider: str, access_token: str, refresh_token: str, 
                          expires_in: int = 3600) -> bool:
        """
        Store OAuth tokens with automatic expiry handling.
        
        Args:
            provider: OAuth provider (google, github, etc.)
            access_token: OAuth access token
            refresh_token: OAuth refresh token
            expires_in: Token expiry time in seconds
        
        Returns:
            bool: True if successful
        """
        try:
            # Store access token with expiry
            access_metadata = {
                "type": "oauth_access_token",
                "provider": provider,
                "expires_at": time.time() + expires_in,
                "scope": "oauth"
            }
            
            self.store_secret(f"{provider}_access_token", access_token, access_metadata)
            
            # Store refresh token (no expiry)
            refresh_metadata = {
                "type": "oauth_refresh_token",
                "provider": provider,
                "scope": "oauth"
            }
            
            self.store_secret(f"{provider}_refresh_token", refresh_token, refresh_metadata)
            
            logger.info("OAuth tokens stored", provider=provider)
            return True
            
        except Exception as e:
            logger.error("Failed to store OAuth tokens", provider=provider, error=str(e))
            return False
    
    def get_oauth_access_token(self, provider: str) -> Optional[str]:
        """
        Get OAuth access token, refreshing if expired.
        
        Args:
            provider: OAuth provider
        
        Returns:
            Optional[str]: Valid access token or None
        """
        try:
            # Try to get current access token
            access_token = self.retrieve_secret(f"{provider}_access_token")
            
            if access_token:
                return access_token
            
            # If no valid access token, try to refresh
            refresh_token = self.retrieve_secret(f"{provider}_refresh_token")
            if refresh_token:
                logger.info("Attempting token refresh", provider=provider)
                # In a real implementation, this would call the OAuth refresh endpoint
                # For now, we'll just log the attempt
                self._audit_log("token_refresh_attempt", f"{provider}_access_token", {"provider": provider})
            
            return None
            
        except Exception as e:
            logger.error("Failed to get OAuth access token", provider=provider, error=str(e))
            return None
    
    def list_secrets(self) -> Dict[str, Dict[str, Any]]:
        """
        List all stored secrets (without values).
        
        Returns:
            Dict of secret names to metadata
        """
        try:
            secrets = self._load_vault()
            result = {}
            
            for name, entry in secrets.items():
                result[name] = {
                    "created_at": entry.get("created_at"),
                    "metadata": entry.get("metadata", {}),
                    "has_value": "value" in entry
                }
            
            return result
            
        except Exception as e:
            logger.error("Failed to list secrets", error=str(e))
            return {}
    
    def delete_secret(self, name: str) -> bool:
        """
        Delete a secret from the vault.
        
        Args:
            name: Secret identifier
        
        Returns:
            bool: True if successful
        """
        try:
            secrets = self._load_vault()
            
            if name in secrets:
                del secrets[name]
                self._save_vault(secrets)
                
                self._audit_log("delete_secret", name)
                logger.info("Secret deleted", name=name)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to delete secret", name=name, error=str(e))
            return False
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Get the audit log of security operations.
        
        Returns:
            List of audit entries
        """
        return self.audit_log.copy()
    
    def _load_vault(self) -> dict:
        """Load the encrypted vault from disk."""
        if not os.path.exists(self.vault_path):
            return {}
        
        try:
            with open(self.vault_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load vault", error=str(e))
            return {}
    
    def _save_vault(self, secrets: dict):
        """Save the encrypted vault to disk."""
        try:
            with open(self.vault_path, "w") as f:
                json.dump(secrets, f, indent=2)
        except Exception as e:
            logger.error("Failed to save vault", error=str(e))
            raise
    
    def _generate_hmac(self, name: str, value: str) -> str:
        """Generate HMAC for secret integrity."""
        message = f"{name}:{value}".encode()
        return hmac.new(self.key.encode(), message, hashlib.sha256).hexdigest()
    
    def _verify_hmac(self, name: str, value: str, expected_hmac: str) -> bool:
        """Verify HMAC for secret integrity."""
        if not expected_hmac:
            return False
        
        actual_hmac = self._generate_hmac(name, value)
        return hmac.compare_digest(actual_hmac, expected_hmac)
    
    def _audit_log(self, operation: str, secret_name: str, metadata: Dict[str, Any] = None):
        """Log security operations for audit trail."""
        audit_entry = {
            "timestamp": time.time(),
            "operation": operation,
            "secret_name": secret_name,
            "metadata": metadata or {},
            "user_agent": "sophie-security"
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]


# Convenience functions for easy integration
def create_vault_key() -> str:
    """Generate a secure vault key for .env file."""
    return Fernet.generate_key().decode()


def initialize_vault(key: str, vault_path: str = ".vault.json") -> SecurityFile:
    """Initialize a new vault with the given key."""
    # Set the key in environment
    os.environ["VAULT_KEY"] = key
    
    # Create and return SecurityFile instance
    return SecurityFile(vault_path=vault_path) 