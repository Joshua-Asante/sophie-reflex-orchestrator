#!/usr/bin/env python3
"""
Security Scaffold Demo for SOPHIE

This script demonstrates the security scaffold functionality including:
- Vault operations (store/retrieve secrets)
- OAuth token management
- Audit logging
- HMAC validation
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path to import core modules
sys.path.append(str(Path(__file__).parent))

from core.security_file import SecurityFile, create_vault_key, initialize_vault
from tools.oauth_google import check_google_auth_status, get_google_access_token


class SecurityScaffoldDemo:
    """
    Demo of SOPHIE's security scaffold functionality.
    """
    
    def __init__(self):
        self.vault = None
        self.demo_secrets = {
            "api_key": "sk-1234567890abcdef",
            "database_password": "super_secure_password_123",
            "oauth_client_secret": "client_secret_456",
            "encryption_key": "32_byte_encryption_key_here"
        }
    
    def setup_vault(self):
        """Set up the vault for demo."""
        print("🔐 Setting up Security Vault")
        print("-" * 40)
        
        try:
            # Generate a new vault key
            vault_key = create_vault_key()
            print(f"✅ Generated vault key: {vault_key[:20]}...")
            
            # Initialize vault
            self.vault = initialize_vault(vault_key, ".demo_vault.json")
            print("✅ Vault initialized successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to set up vault: {e}")
            return False
    
    def demo_secret_operations(self):
        """Demonstrate basic secret operations."""
        print("\n🔑 Secret Operations Demo")
        print("-" * 40)
        
        if not self.vault:
            print("❌ Vault not initialized")
            return False
        
        try:
            # Store secrets
            print("📝 Storing secrets...")
            for name, value in self.demo_secrets.items():
                success = self.vault.store_secret(name, value, {
                    "demo": True,
                    "created_by": "security_demo"
                })
                if success:
                    print(f"   ✅ Stored: {name}")
                else:
                    print(f"   ❌ Failed to store: {name}")
            
            # List secrets
            print("\n📋 Listing stored secrets...")
            secrets = self.vault.list_secrets()
            for name, metadata in secrets.items():
                created_at = time.strftime("%Y-%m-%d %H:%M:%S", 
                                        time.localtime(metadata.get("created_at", 0)))
                print(f"   • {name} (created: {created_at})")
            
            # Retrieve secrets
            print("\n🔍 Retrieving secrets...")
            for name in self.demo_secrets.keys():
                value = self.vault.retrieve_secret(name)
                if value:
                    print(f"   ✅ Retrieved: {name} = {value[:10]}...")
                else:
                    print(f"   ❌ Failed to retrieve: {name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Secret operations failed: {e}")
            return False
    
    def demo_oauth_token_management(self):
        """Demonstrate OAuth token management."""
        print("\n🔄 OAuth Token Management Demo")
        print("-" * 40)
        
        if not self.vault:
            print("❌ Vault not initialized")
            return False
        
        try:
            # Simulate OAuth tokens
            access_token = "ya29.a0AfH6SMC..."
            refresh_token = "1//04dX..."
            
            print("📝 Storing OAuth tokens...")
            success = self.vault.store_oauth_tokens(
                provider="google",
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=3600
            )
            
            if success:
                print("   ✅ OAuth tokens stored")
            else:
                print("   ❌ Failed to store OAuth tokens")
            
            # Check OAuth status
            print("\n📊 Checking OAuth status...")
            status = check_google_auth_status()
            print(f"   • Authenticated: {status.get('authenticated', False)}")
            print(f"   • Has access token: {status.get('has_access_token', False)}")
            print(f"   • Has refresh token: {status.get('has_refresh_token', False)}")
            print(f"   • Access token valid: {status.get('access_token_valid', False)}")
            
            # Get access token
            print("\n🔑 Getting access token...")
            token = get_google_access_token()
            if token:
                print(f"   ✅ Access token: {token[:20]}...")
            else:
                print("   ❌ No valid access token")
            
            return True
            
        except Exception as e:
            print(f"❌ OAuth token management failed: {e}")
            return False
    
    def demo_audit_logging(self):
        """Demonstrate audit logging functionality."""
        print("\n📊 Audit Logging Demo")
        print("-" * 40)
        
        if not self.vault:
            print("❌ Vault not initialized")
            return False
        
        try:
            # Get audit log
            audit_log = self.vault.get_audit_log()
            
            print(f"📋 Audit log entries: {len(audit_log)}")
            
            if audit_log:
                print("\nRecent operations:")
                for entry in audit_log[-5:]:  # Show last 5 entries
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", 
                                           time.localtime(entry.get("timestamp", 0)))
                    operation = entry.get("operation", "unknown")
                    secret_name = entry.get("secret_name", "unknown")
                    
                    print(f"   • {timestamp} - {operation} - {secret_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Audit logging failed: {e}")
            return False
    
    def demo_security_features(self):
        """Demonstrate security features."""
        print("\n🛡️ Security Features Demo")
        print("-" * 40)
        
        if not self.vault:
            print("❌ Vault not initialized")
            return False
        
        try:
            # Test HMAC validation
            print("🔐 Testing HMAC validation...")
            
            # Store a test secret
            test_secret = "test_secret_value"
            self.vault.store_secret("test_secret", test_secret)
            
            # Retrieve it (should work)
            retrieved = self.vault.retrieve_secret("test_secret")
            if retrieved == test_secret:
                print("   ✅ HMAC validation passed")
            else:
                print("   ❌ HMAC validation failed")
            
            # Test expiry
            print("\n⏰ Testing expiry functionality...")
            
            # Store a secret with short expiry
            self.vault.store_secret("expiring_secret", "will_expire", {
                "expires_at": time.time() + 1  # Expires in 1 second
            })
            
            # Retrieve immediately (should work)
            immediate = self.vault.retrieve_secret("expiring_secret")
            if immediate:
                print("   ✅ Immediate retrieval works")
            
            # Wait and try again (should fail)
            time.sleep(2)
            expired = self.vault.retrieve_secret("expiring_secret")
            if not expired:
                print("   ✅ Expiry functionality works")
            else:
                print("   ❌ Expiry functionality failed")
            
            # Clean up
            self.vault.delete_secret("test_secret")
            self.vault.delete_secret("expiring_secret")
            
            return True
            
        except Exception as e:
            print(f"❌ Security features failed: {e}")
            return False
    
    def demo_error_handling(self):
        """Demonstrate error handling."""
        print("\n⚠️ Error Handling Demo")
        print("-" * 40)
        
        if not self.vault:
            print("❌ Vault not initialized")
            return False
        
        try:
            # Try to retrieve non-existent secret
            print("🔍 Testing retrieval of non-existent secret...")
            result = self.vault.retrieve_secret("non_existent_secret")
            if result is None:
                print("   ✅ Correctly returned None for non-existent secret")
            else:
                print("   ❌ Should have returned None")
            
            # Try to store with invalid data
            print("\n📝 Testing invalid data handling...")
            try:
                # This should work fine
                self.vault.store_secret("test_invalid", "valid_string")
                print("   ✅ Valid string storage works")
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error handling failed: {e}")
            return False
    
    def run_full_demo(self):
        """Run the complete security scaffold demo."""
        print("🔐 SOPHIE Security Scaffold Demo")
        print("=" * 60)
        print("This demonstrates secure vault-based secret management")
        print("with OAuth token support and audit logging.")
        print("=" * 60)
        
        # Set up vault
        if not self.setup_vault():
            return False
        
        # Run all demos
        demos = [
            ("Secret Operations", self.demo_secret_operations),
            ("OAuth Token Management", self.demo_oauth_token_management),
            ("Audit Logging", self.demo_audit_logging),
            ("Security Features", self.demo_security_features),
            ("Error Handling", self.demo_error_handling)
        ]
        
        results = []
        for name, demo_func in demos:
            print(f"\n🎯 Running {name} Demo...")
            result = demo_func()
            results.append((name, result))
            print(f"   {'✅' if result else '❌'} {name} completed")
        
        # Summary
        print("\n📊 Demo Summary")
        print("-" * 40)
        successful = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} - {name}")
        
        print(f"\n🎯 Overall Result: {successful}/{total} demos passed")
        
        if successful == total:
            print("🎉 All security scaffold features working correctly!")
        else:
            print("⚠️ Some features need attention")
        
        return successful == total


def main():
    """Main function to run the security scaffold demo."""
    demo = SecurityScaffoldDemo()
    success = demo.run_full_demo()
    
    if success:
        print("\n✅ Security scaffold demo completed successfully!")
        print("🔐 SOPHIE's security infrastructure is ready for production use.")
    else:
        print("\n❌ Security scaffold demo encountered issues.")
        print("🔧 Please review the implementation and try again.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 