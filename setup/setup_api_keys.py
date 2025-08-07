#!/usr/bin/env python3
"""
API Key Setup and Configuration Helper for Sophie Reflex Orchestrator

This script helps users configure their LLM API keys and test connectivity.
"""

import os
import sys
import asyncio
import socket
import ssl
import urllib.request
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

def check_api_keys():
    """Check which API keys are currently set."""
    print("🔍 CHECKING CURRENT API KEY STATUS")
    print("=" * 50)
    
    api_keys = {
        "OPENAI_API_KEY": "OpenAI GPT",
        "GOOGLE_API_KEY": "Google Gemini", 
        "XAI_API_KEY": "XAI Grok",
        "MISTRAL_API_KEY": "Mistral AI",
        "DEEPSEEK_API_KEY": "DeepSeek",
        "KIMI_API_KEY": "Kimi"
    }
    
    found_keys = []
    missing_keys = []
    
    for key_name, service_name in api_keys.items():
        key_value = os.getenv(key_name)
        if key_value and key_value.strip():
            # Show first few characters of key for verification
            masked_key = key_value[:8] + "..." + key_value[-4:] if len(key_value) > 12 else "***"
            print(f"✅ {service_name}: {key_name} = {masked_key}")
            found_keys.append(service_name)
        else:
            print(f"❌ {service_name}: {key_name} = NOT SET")
            missing_keys.append(service_name)
    
    print(f"\n📊 Summary: {len(found_keys)}/{len(api_keys)} API keys configured")
    
    if missing_keys:
        print(f"\n⚠️  Missing API keys for: {', '.join(missing_keys)}")
        print("   You can set these using environment variables or a .env file")
    
    return found_keys, missing_keys

def test_network_connectivity():
    """Test basic network connectivity to API endpoints."""
    print("\n🌐 TESTING NETWORK CONNECTIVITY")
    print("=" * 50)
    
    api_endpoints = {
        "OpenAI": "api.openai.com",
        "Google": "generativelanguage.googleapis.com", 
        "XAI": "api.x.ai",
        "Mistral": "api.mistral.ai",
        "DeepSeek": "api.deepseek.com",
        "Kimi": "api.moonshot.cn"
    }
    
    results = {}
    
    for service, host in api_endpoints.items():
        try:
            # Test DNS resolution
            ip = socket.gethostbyname(host)
            print(f"✅ {service}: DNS resolved to {ip}")
            
            # Test TCP connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, 443))
            sock.close()
            
            if result == 0:
                print(f"✅ {service}: TCP connection successful")
                results[service] = True
            else:
                print(f"❌ {service}: TCP connection failed (error code: {result})")
                results[service] = False
                
        except socket.gaierror as e:
            print(f"❌ {service}: DNS resolution failed - {e}")
            results[service] = False
        except Exception as e:
            print(f"❌ {service}: Network test failed - {e}")
            results[service] = False
    
    return results

def test_http_connectivity():
    """Test basic HTTP connectivity to API endpoints."""
    print("\n🌐 TESTING HTTP CONNECTIVITY")
    print("=" * 50)
    
    import urllib.request
    import urllib.error
    
    test_urls = {
        "OpenAI": "https://api.openai.com/v1/models",
        "XAI": "https://api.x.ai/v1/models", 
        "Mistral": "https://api.mistral.ai/v1/models",
        "DeepSeek": "https://api.deepseek.com/v1/models",
        "Kimi": "https://api.moonshot.cn/v1/models"
    }
    
    results = {}
    
    for service, url in test_urls.items():
        try:
            print(f"  🔧 Testing HTTP GET to {service}...")
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Sophie-Reflex-Orchestrator/1.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.getcode()
                if status == 401:  # Expected - requires auth
                    print(f"✅ {service}: HTTP connection successful (401 - auth required)")
                    results[service] = True
                else:
                    print(f"✅ {service}: HTTP connection successful (status: {status})")
                    results[service] = True
                    
        except urllib.error.HTTPError as e:
            if e.code == 401:  # Expected - requires auth
                print(f"✅ {service}: HTTP connection successful (401 - auth required)")
                results[service] = True
            else:
                print(f"❌ {service}: HTTP error - {e.code}: {e.reason}")
                results[service] = False
        except urllib.error.URLError as e:
            print(f"❌ {service}: URL error - {e.reason}")
            results[service] = False
        except Exception as e:
            print(f"❌ {service}: HTTP test failed - {e}")
            results[service] = False
    
    return results

def check_environment_issues():
    """Check for common environment issues that might affect connectivity."""
    print("\n🔍 CHECKING ENVIRONMENT ISSUES")
    print("=" * 50)
    
    issues = []
    
    # Check for proxy settings
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"]
    proxy_settings = {var: os.getenv(var) for var in proxy_vars if os.getenv(var)}
    if proxy_settings:
        print(f"⚠️  Proxy settings detected:")
        for var, value in proxy_settings.items():
            print(f"   {var}: {value}")
        issues.append("Proxy settings may interfere with API connections")
    
    # Check for corporate firewall indicators
    corporate_indicators = [
        "corporate", "company", "enterprise", "internal", "intranet"
    ]
    
    # Check hostname for corporate indicators
    import socket
    hostname = socket.gethostname().lower()
    for indicator in corporate_indicators:
        if indicator in hostname:
            print(f"⚠️  Corporate environment detected (hostname: {hostname})")
            issues.append("Corporate firewall may block external API access")
            break
    
    # Check for VPN indicators
    vpn_indicators = ["vpn", "tunnel", "secure"]
    for indicator in vpn_indicators:
        if indicator in hostname:
            print(f"⚠️  VPN environment detected (hostname: {hostname})")
            issues.append("VPN may interfere with API connections")
            break
    
    # Check Windows Defender or antivirus
    try:
        import subprocess
        result = subprocess.run(['netsh', 'advfirewall', 'show', 'allprofiles'], 
                              capture_output=True, text=True, timeout=5)
        if "ON" in result.stdout:
            print("⚠️  Windows Firewall is active")
            issues.append("Windows Firewall may block API connections")
    except:
        pass
    
    if not issues:
        print("✅ No obvious environment issues detected")
    else:
        print(f"\n📋 Potential issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    return issues

def test_openai_library_connectivity():
    """Test basic connectivity using the same libraries as OpenAI client."""
    print("\n🔧 TESTING OPENAI LIBRARY CONNECTIVITY")
    print("=" * 50)
    
    try:
        import httpx
        import openai
        
        print("  🔧 Testing httpx connectivity...")
        async def test_httpx():
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    response = await client.get("https://api.openai.com/v1/models")
                    print(f"  ✅ httpx GET successful: {response.status_code}")
                    return True
                except Exception as e:
                    print(f"  ❌ httpx GET failed: {type(e).__name__}: {e}")
                    return False
        
        import asyncio
        httpx_result = asyncio.run(test_httpx())
        
        print("  🔧 Testing OpenAI client with minimal config...")
        async def test_openai_client():
            try:
                # Test with minimal configuration
                client = openai.AsyncOpenAI(
                    api_key="test-key",  # Will fail auth but should connect
                    timeout=10.0
                )
                print("  ✅ OpenAI client created successfully")
                
                # Try to make a request (will fail auth but should connect)
                try:
                    await client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                except openai.AuthenticationError:
                    print("  ✅ OpenAI client connected (auth failed as expected)")
                    return True
                except Exception as e:
                    print(f"  ❌ OpenAI client failed: {type(e).__name__}: {e}")
                    return False
                    
            except Exception as e:
                print(f"  ❌ OpenAI client creation failed: {type(e).__name__}: {e}")
                return False
        
        openai_result = asyncio.run(test_openai_client())
        return openai_result
            
    except ImportError as e:
        print(f"  ❌ Required libraries not available: {e}")
        return False

def create_env_template():
    """Create a template .env file for API keys."""
    template_content = """# Sophie Reflex Orchestrator - API Keys Configuration
# Copy this file to .env and fill in your actual API keys

# OpenAI API Key (for GPT models)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Google API Key (for Gemini models)  
# Get from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# XAI API Key (for Grok models)
# Get from: https://console.x.ai/
XAI_API_KEY=your_xai_api_key_here

# Mistral AI API Key
# Get from: https://console.mistral.ai/
MISTRAL_API_KEY=your_mistral_api_key_here

# DeepSeek API Key
# Get from: https://platform.deepseek.com/
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Kimi API Key
# Get from: https://kimi.moonshot.cn/
KIMI_API_KEY=your_kimi_api_key_here

# Optional: Database configuration
# CHROMA_HOST=localhost
# CHROMA_PORT=8000
"""
    
    try:
        with open('.env.template', 'w') as f:
            f.write(template_content)
        print("✅ Created .env.template file")
        print("   Copy this to .env and fill in your actual API keys")
    except Exception as e:
        print(f"❌ Failed to create .env.template: {e}")

def test_connectivity():
    """Test connectivity to available APIs."""
    print("\n🔍 TESTING API CONNECTIVITY")
    print("=" * 50)
    
    # Import here to avoid issues if dependencies aren't installed
    try:
        from agents.prover import ProverAgent
        from agents.base_agent import AgentConfig
    except ImportError as e:
        print(f"❌ Cannot test connectivity: {e}")
        print("   Make sure all dependencies are installed: pip install -r requirements.txt")
        return
    
    async def test_single_api(provider, config):
        """Test a single API provider with detailed debugging."""
        try:
            print(f"  🔧 Creating test config for {provider}...")
            # Create test config
            test_config = AgentConfig(
                name=f"test_{provider}",
                prompt="Test prompt",
                model=provider,
                temperature=0.7,
                max_tokens=20,
                timeout=10
            )
            
            print(f"  🔧 Creating test agent for {provider}...")
            # Create test agent
            test_agent = ProverAgent(test_config, f"test_{provider}")
            
            print(f"  🔧 Getting client for {provider}...")
            # Get client
            client = await test_agent._llm_manager.get_client(provider, {"timeout": 10})
            
            print(f"  🔧 Making API call to {provider}...")
            # Test API call
            if provider in ["openai", "xai", "mistral", "deepseek", "kimi"]:
                # Add proxy debugging
                import os
                proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
                proxy_info = {var: os.getenv(var) for var in proxy_vars if os.getenv(var)}
                if proxy_info:
                    print(f"  🔧 Proxy detected: {proxy_info}")
                
                # Add SSL debugging
                import ssl
                print(f"  🔧 SSL version: {ssl.OPENSSL_VERSION}")
                print(f"  🔧 Default SSL context: {ssl.get_default_verify_paths()}")
                
                # Try with explicit timeout and different settings
                try:
                    response = await client.chat.completions.create(
                        model=config["model"],
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5,
                        timeout=30.0
                    )
                    return True, "API call successful"
                except Exception as inner_e:
                    print(f"  🔧 Detailed error: {type(inner_e).__name__}: {str(inner_e)}")
                    # Try to get more details about the connection
                    if hasattr(inner_e, '__cause__') and inner_e.__cause__:
                        print(f"  🔧 Root cause: {type(inner_e.__cause__).__name__}: {str(inner_e.__cause__)}")
                    raise
                
            elif provider == "google":
                import google.generativeai as genai
                response = await client.generate_content_async(
                    "Hello",
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=5
                    )
                )
                return True, "API call successful"
                
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Enhanced error categorization with debugging info
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                return False, f"Authentication failed (invalid API key) - {error_type}: {error_msg}"
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                return False, f"Access forbidden - {error_type}: {error_msg}"
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                return True, f"Rate limited (API key valid but quota exceeded) - {error_type}: {error_msg}"
            elif "timeout" in error_msg.lower():
                return False, f"Connection timeout - {error_type}: {error_msg}"
            elif "connection" in error_msg.lower() or "connect" in error_msg.lower():
                return False, f"Network connection failed - {error_type}: {error_msg}"
            elif "dns" in error_msg.lower():
                return False, f"DNS resolution failed - {error_type}: {error_msg}"
            elif "ssl" in error_msg.lower() or "certificate" in error_msg.lower():
                return False, f"SSL/TLS error - {error_type}: {error_msg}"
            elif "proxy" in error_msg.lower():
                return False, f"Proxy connection error - {error_type}: {error_msg}"
            else:
                return False, f"API call failed - {error_type}: {error_msg}"
    
    # Test each available API
    llm_providers = {
        "openai": {"name": "OpenAI GPT", "model": "gpt-3.5-turbo"},
        "google": {"name": "Google Gemini", "model": "gemini-1.5-pro"},
                 "xai": {"name": "XAI Grok", "model": "grok-2-1212"},
        "mistral": {"name": "Mistral AI", "model": "mistral-large-latest"},
        "deepseek": {"name": "DeepSeek", "model": "deepseek-chat"},
        "kimi": {"name": "Kimi", "model": "moonshot-v1-8k"}
    }
    
    async def run_connectivity_tests():
        results = {}
        for provider, config in llm_providers.items():
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            if api_key and api_key.strip():
                print(f"\n🔍 Testing {config['name']}...")
                success, message = await test_single_api(provider, config)
                if success:
                    print(f"✅ {config['name']}: {message}")
                else:
                    print(f"❌ {config['name']}: {message}")
                results[provider] = success
            else:
                print(f"⏭️  {config['name']}: Skipped (no API key)")
                results[provider] = None
        
        return results
    
    # Run the tests
    try:
        results = asyncio.run(run_connectivity_tests())
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 CONNECTIVITY TEST SUMMARY")
        print("=" * 50)
        
        successful = [k for k, v in results.items() if v is True]
        failed = [k for k, v in results.items() if v is False]
        skipped = [k for k, v in results.items() if v is None]
        
        if successful:
            print(f"✅ Successful: {len(successful)}")
            for api in successful:
                print(f"   • {llm_providers[api]['name']}")
        
        if failed:
            print(f"❌ Failed: {len(failed)}")
            for api in failed:
                print(f"   • {llm_providers[api]['name']}")
        
        if skipped:
            print(f"⏭️  Skipped: {len(skipped)}")
            for api in skipped:
                print(f"   • {llm_providers[api]['name']}")
        
        if successful:
            print(f"\n🎉 Ready to run tests with {len(successful)} connected APIs!")
            print("   Run: python test_orchestrator_real_enhanced.py")
        else:
            print(f"\n⚠️  No APIs are connected")
            print("   Run: python test_orchestrator_mock.py")
        
    except Exception as e:
        print(f"❌ Connectivity test failed: {e}")

def main():
    """Main setup function."""
    print("🔧 SOPHIE REFLEX ORCHESTRATOR - API SETUP")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Check current API key status
    found_keys, missing_keys = check_api_keys()
    
    # Test network connectivity first
    network_results = test_network_connectivity()
    
    # Test HTTP connectivity
    http_results = test_http_connectivity()
    
    # Check for environment issues
    environment_issues = check_environment_issues()
    
    # Test OpenAI library connectivity
    test_openai_library_connectivity()
    
    # Create .env template if needed
    if missing_keys:
        print(f"\n📝 Creating .env template...")
        create_env_template()
    
    # Test connectivity for available APIs
    if found_keys:
        test_connectivity()
    else:
        print(f"\n⚠️  No API keys found")
        print("   Set up your API keys first, then run this script again")
    
    print(f"\n" + "=" * 60)
    print("📚 NEXT STEPS")
    print("=" * 60)
    print("1. Configure your API keys (see .env.template)")
    print("2. Run this script again to test connectivity")
    print("3. Run tests:")
    print("   • Mock tests: python test_orchestrator_mock.py")
    print("   • Real tests: python test_orchestrator_real_enhanced.py")
    print("   • All tests: python run_tests.py --module orchestrator")

if __name__ == "__main__":
    main() 