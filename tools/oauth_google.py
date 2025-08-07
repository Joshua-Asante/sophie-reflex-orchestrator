"""
Google OAuth Scaffold for SOPHIE

Handles Google OAuth authentication flow, token exchange, and secure storage
using the SecurityFile module for vault-backed credential management.
"""

import os
import requests
import webbrowser
import urllib.parse as urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time
from typing import Optional, Dict, Any
import structlog

# Add the parent directory to the path to import core modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.security_file import SecurityFile

logger = structlog.get_logger()

# OAuth Configuration
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "your-client-id.apps.googleusercontent.com")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "your-client-secret")
REDIRECT_URI = "http://localhost:8080/oauth2callback"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPES = ["openid", "email", "profile"]


class OAuthHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""
    
    def __init__(self, *args, **kwargs):
        self.auth_code = None
        self.state = None
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle OAuth callback."""
        try:
            # Parse query parameters
            query = urlparse.urlparse(self.path).query
            params = urlparse.parse_qs(query)
            
            # Extract authorization code and state
            self.auth_code = params.get("code", [None])[0]
            self.state = params.get("state", [None])[0]
            
            if self.auth_code:
                # Send success response
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                
                success_html = """
                <html>
                <head><title>Authentication Successful</title></head>
                <body>
                <h1>✅ Authentication Successful</h1>
                <p>SOPHIE has successfully authenticated with Google.</p>
                <p>You may close this window and return to SOPHIE.</p>
                <script>setTimeout(function() { window.close(); }, 3000);</script>
                </body>
                </html>
                """
                
                self.wfile.write(success_html.encode())
                
                logger.info("OAuth callback received", has_code=bool(self.auth_code))
                
            else:
                # Send error response
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                
                error_html = """
                <html>
                <head><title>Authentication Failed</title></head>
                <body>
                <h1>❌ Authentication Failed</h1>
                <p>No authorization code received from Google.</p>
                <p>Please try again.</p>
                </body>
                </html>
                """
                
                self.wfile.write(error_html.encode())
                
                logger.error("OAuth callback failed - no authorization code")
                
        except Exception as e:
            logger.error("Error handling OAuth callback", error=str(e))
            self.send_response(500)
            self.end_headers()


def start_google_oauth() -> Optional[str]:
    """
    Start Google OAuth flow.
    
    Returns:
        Optional[str]: Authorization code or None if failed
    """
    try:
        # Generate state parameter for security
        state = f"sophie_auth_{int(time.time())}"
        
        # Build authorization URL
        params = {
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "state": state,
            "access_type": "offline",
            "prompt": "consent"
        }
        
        auth_url = AUTH_URL + "?" + urlparse.urlencode(params)
        
        logger.info("Starting Google OAuth flow", auth_url=auth_url)
        
        # Open browser for user authentication
        print("🔐 Opening Google Sign-In in your browser...")
        webbrowser.open(auth_url)
        
        # Start local server to handle callback
        server = HTTPServer(("localhost", 8080), OAuthHandler)
        server.timeout = 300  # 5 minute timeout
        
        print("⏳ Waiting for authentication...")
        server.handle_request()
        
        # Get the authorization code from the handler
        handler = server.RequestHandlerClass
        auth_code = getattr(handler, 'auth_code', None)
        
        if auth_code:
            logger.info("OAuth flow completed successfully")
            return auth_code
        else:
            logger.error("OAuth flow failed - no authorization code")
            return None
            
    except Exception as e:
        logger.error("Failed to start Google OAuth flow", error=str(e))
        return None


def exchange_code_for_tokens(auth_code: str) -> Optional[Dict[str, Any]]:
    """
    Exchange authorization code for access and refresh tokens.
    
    Args:
        auth_code: Authorization code from OAuth flow
    
    Returns:
        Optional[Dict[str, Any]]: Token response or None if failed
    """
    try:
        # Prepare token exchange request
        data = {
            "code": auth_code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        
        logger.info("Exchanging authorization code for tokens")
        
        # Make token exchange request
        response = requests.post(TOKEN_URL, data=data)
        response.raise_for_status()
        
        tokens = response.json()
        
        logger.info("Token exchange successful", 
                   has_access_token="access_token" in tokens,
                   has_refresh_token="refresh_token" in tokens)
        
        return tokens
        
    except Exception as e:
        logger.error("Failed to exchange code for tokens", error=str(e))
        return None


def store_google_tokens(tokens: Dict[str, Any]) -> bool:
    """
    Store Google OAuth tokens securely in vault.
    
    Args:
        tokens: Token response from Google
    
    Returns:
        bool: True if successful
    """
    try:
        vault = SecurityFile()
        
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        expires_in = tokens.get("expires_in", 3600)
        
        if not access_token or not refresh_token:
            logger.error("Missing required tokens", 
                        has_access_token=bool(access_token),
                        has_refresh_token=bool(refresh_token))
            return False
        
        # Store tokens securely
        success = vault.store_oauth_tokens(
            provider="google",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in
        )
        
        if success:
            print("✅ Google tokens securely stored in vault")
            logger.info("Google tokens stored successfully")
        else:
            print("❌ Failed to store Google tokens")
            logger.error("Failed to store Google tokens")
        
        return success
        
    except Exception as e:
        logger.error("Failed to store Google tokens", error=str(e))
        return False


def get_google_access_token() -> Optional[str]:
    """
    Get Google access token from vault, refreshing if needed.
    
    Returns:
        Optional[str]: Valid access token or None
    """
    try:
        vault = SecurityFile()
        return vault.get_oauth_access_token("google")
        
    except Exception as e:
        logger.error("Failed to get Google access token", error=str(e))
        return None


def check_google_auth_status() -> Dict[str, Any]:
    """
    Check Google authentication status.
    
    Returns:
        Dict with authentication status and metadata
    """
    try:
        vault = SecurityFile()
        
        # Check for stored tokens
        secrets = vault.list_secrets()
        
        has_access_token = "google_access_token" in secrets
        has_refresh_token = "google_refresh_token" in secrets
        
        # Try to get current access token
        access_token = vault.get_oauth_access_token("google")
        
        return {
            "authenticated": bool(access_token),
            "has_access_token": has_access_token,
            "has_refresh_token": has_refresh_token,
            "access_token_valid": bool(access_token),
            "secrets_count": len(secrets)
        }
        
    except Exception as e:
        logger.error("Failed to check Google auth status", error=str(e))
        return {
            "authenticated": False,
            "error": str(e)
        }


def main():
    """Main function for Google OAuth flow."""
    print("🔐 SOPHIE Google OAuth Authentication")
    print("=" * 50)
    
    # Check current auth status
    print("📊 Checking current authentication status...")
    status = check_google_auth_status()
    
    if status.get("authenticated"):
        print("✅ Already authenticated with Google")
        print(f"   • Access token: {'Valid' if status.get('access_token_valid') else 'Invalid'}")
        print(f"   • Refresh token: {'Available' if status.get('has_refresh_token') else 'Missing'}")
        return
    
    print("❌ Not authenticated with Google")
    print("🔄 Starting OAuth flow...")
    
    # Start OAuth flow
    auth_code = start_google_oauth()
    
    if not auth_code:
        print("❌ OAuth flow failed - no authorization code received")
        return
    
    print("✅ Authorization code received")
    print("🔄 Exchanging for tokens...")
    
    # Exchange code for tokens
    tokens = exchange_code_for_tokens(auth_code)
    
    if not tokens:
        print("❌ Token exchange failed")
        return
    
    print("✅ Tokens received from Google")
    print("🔐 Storing tokens securely...")
    
    # Store tokens securely
    if store_google_tokens(tokens):
        print("✅ Google authentication completed successfully!")
        print("🔐 Tokens stored securely in vault")
        
        # Verify storage
        access_token = get_google_access_token()
        if access_token:
            print("✅ Access token retrieved successfully")
        else:
            print("❌ Failed to retrieve access token")
    else:
        print("❌ Failed to store Google tokens")


if __name__ == "__main__":
    main() 