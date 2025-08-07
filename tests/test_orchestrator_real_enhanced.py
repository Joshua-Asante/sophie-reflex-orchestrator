#!/usr/bin/env python3
"""
Enhanced Real Agent Test Runner for Sophie Reflex Orchestrator

This script runs the orchestrator tests with real LLM API calls,
providing detailed connectivity testing and error reporting.
"""

import asyncio
import logging
import os
import sys
import tempfile
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from test_modules.orchestrator_tests import OrchestratorTestSuite
    from utils.resource_manager import ResourceManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('real_agent_test.log')
    ]
)
logger = logging.getLogger(__name__)

async def test_api_connectivity_detailed():
    """Test connectivity to all LLM APIs with detailed reporting."""
    print("🔍 DETAILED LLM API CONNECTIVITY TEST")
    print("=" * 60)
    
    # Test each LLM provider with detailed error reporting
    llm_providers = {
        "openai": {
            "name": "OpenAI GPT",
            "model": "gpt-3.5-turbo",
            "base_url": "https://api.openai.com/v1"
        },
        "google": {
            "name": "Google Gemini",
            "model": "gemini-1.5-pro",
            "base_url": "https://generativelanguage.googleapis.com"
        },
                 "xai": {
             "name": "XAI Grok",
             "model": "grok-2-1212",
             "base_url": "https://api.x.ai/v1"
         },
        "mistral": {
            "name": "Mistral AI",
            "model": "mistral-large-latest",
            "base_url": "https://api.mistral.ai/v1"
        },
        "deepseek": {
            "name": "DeepSeek",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1"
        },
        "kimi": {
            "name": "Kimi",
            "model": "moonshot-v1-8k",
            "base_url": "https://api.moonshot.cn/v1"
        }
    }
    
    connectivity_results = {}
    api_key_status = {}
    
    for provider, config in llm_providers.items():
        print(f"\n🔍 Testing {config['name']} ({provider.upper()})...")
        
        # Check API key availability
        api_key_name = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_name)
        
        if not api_key:
            print(f"❌ {config['name']}: No API key found ({api_key_name})")
            connectivity_results[provider] = False
            api_key_status[provider] = "MISSING"
            continue
        elif api_key == "":
            print(f"❌ {config['name']}: Empty API key ({api_key_name})")
            connectivity_results[provider] = False
            api_key_status[provider] = "EMPTY"
            continue
        else:
            print(f"✅ {config['name']}: API key found")
            api_key_status[provider] = "PRESENT"
        
        # Test connectivity
        try:
            from agents.prover import ProverAgent
            from agents.base_agent import AgentConfig
            
            # Create test config
            test_config = AgentConfig(
                name=f"test_{provider}",
                prompt="Test prompt",
                model=provider,
                temperature=0.7,
                max_tokens=50,  # Small token limit for testing
                timeout=15  # Shorter timeout for connectivity test
            )
            
            # Create test agent
            test_agent = ProverAgent(test_config, f"test_{provider}")
            
            # Test API call - use exact same logic as setup_api_keys.py
            if provider in ["openai", "xai", "mistral", "deepseek", "kimi"]:
                # Get OpenAI-style client
                client = await test_agent._llm_manager.get_client(provider, {"timeout": 15})
                try:
                    # Use the exact same model names and logic as setup_api_keys.py
                    model_map = {
                        "openai": "gpt-3.5-turbo",
                        "xai": "grok-2-1212",
                        "mistral": "mistral-large-latest",
                        "deepseek": "deepseek-chat",
                        "kimi": "moonshot-v1-8k"
                    }
                    
                    response = await client.chat.completions.create(
                        model=model_map[provider],
                        messages=[{"role": "user", "content": "Hello, this is a connectivity test."}],
                        max_tokens=10
                    )
                    print(f"✅ {config['name']}: API call successful")
                    connectivity_results[provider] = True
                    
                except Exception as e:
                    error_msg = str(e)
                    # Use exact same error handling as setup_api_keys.py
                    if "401" in error_msg or "unauthorized" in error_msg.lower():
                        print(f"❌ {config['name']}: Authentication failed (invalid API key)")
                    elif "429" in error_msg or "rate limit" in error_msg.lower():
                        print(f"⚠️  {config['name']}: Rate limited (API key valid but quota exceeded)")
                        connectivity_results[provider] = True  # Consider rate limits as "connected"
                    elif "timeout" in error_msg.lower():
                        print(f"⏰ {config['name']}: Connection timeout")
                    elif "connection" in error_msg.lower():
                        print(f"🌐 {config['name']}: Network connection failed")
                    else:
                        print(f"❌ {config['name']}: API call failed - {error_msg}")
                    connectivity_results[provider] = False
                    
            elif provider == "google":
                # Get Google GenerativeModel client
                client = await test_agent._llm_manager.get_client(provider, {"timeout": 15})
                try:
                    import google.generativeai as genai
                    # For Google API, client is a GenerativeModel, not OpenAI-style client
                    response = await client.generate_content_async(
                        "Hello, this is a connectivity test.",
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=10
                        )
                    )
                    print(f"✅ {config['name']}: API call successful")
                    connectivity_results[provider] = True
                    
                except Exception as e:
                    error_msg = str(e)
                    if "401" in error_msg or "unauthorized" in error_msg.lower():
                        print(f"❌ {config['name']}: Authentication failed (invalid API key)")
                    elif "429" in error_msg or "rate limit" in error_msg.lower():
                        print(f"⚠️  {config['name']}: Rate limited (API key valid but quota exceeded)")
                        connectivity_results[provider] = True  # Consider rate limits as "connected"
                    elif "timeout" in error_msg.lower():
                        print(f"⏰ {config['name']}: Connection timeout")
                    elif "connection" in error_msg.lower():
                        print(f"🌐 {config['name']}: Network connection failed")
                    else:
                        print(f"❌ {config['name']}: API call failed - {error_msg}")
                    connectivity_results[provider] = False
                    
        except Exception as e:
            print(f"❌ {config['name']}: Setup failed - {str(e)}")
            connectivity_results[provider] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 CONNECTIVITY TEST SUMMARY")
    print("=" * 60)
    
    connected_apis = [k for k, v in connectivity_results.items() if v]
    failed_apis = [k for k, v in connectivity_results.items() if not v]
    
    print(f"✅ Connected APIs: {len(connected_apis)}")
    if connected_apis:
        for api in connected_apis:
            print(f"   • {llm_providers[api]['name']} ({api.upper()})")
    
    print(f"❌ Failed APIs: {len(failed_apis)}")
    if failed_apis:
        for api in failed_apis:
            status = api_key_status.get(api, "UNKNOWN")
            print(f"   • {llm_providers[api]['name']} ({api.upper()}) - {status}")
    
    print(f"\n📈 Success Rate: {(len(connected_apis) / len(llm_providers)) * 100:.1f}%")
    
    return connectivity_results

async def main():
    """Main test runner with enhanced real agent testing."""
    print("🧪 ENHANCED REAL AGENT TEST RUNNER")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Test API connectivity first
    connectivity_results = await test_api_connectivity_detailed()
    
    # Create temporary directory for tests
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n📁 Using temporary directory: {temp_dir}")
        
        # Initialize resource manager
        resource_manager = ResourceManager()
        
        try:
            # Create test suite with real agents
            test_suite = OrchestratorTestSuite(temp_dir, use_mock_agents=False)
            
            # Run all tests
            print("\n🚀 Running Orchestrator Tests with Real Agents")
            print("-" * 50)
            
            success = await test_suite.run_all_tests()
            
            # Print final summary
            print("\n" + "=" * 60)
            print("🎯 FINAL TEST SUMMARY")
            print("=" * 60)
            
            if success:
                print("✅ All tests PASSED")
            else:
                print("❌ Some tests FAILED")
            
            # Provide recommendations based on connectivity results
            connected_apis = [k for k, v in connectivity_results.items() if v]
            if not connected_apis:
                print("\n⚠️  RECOMMENDATIONS:")
                print("   • No LLM APIs are connected")
                print("   • Consider using mock agents for testing: python test_orchestrator_mock.py")
                print("   • Check your API keys and network connectivity")
                print("   • Verify API service availability")
            elif len(connected_apis) < len(connectivity_results):
                print(f"\n⚠️  PARTIAL CONNECTIVITY:")
                print(f"   • {len(connected_apis)}/{len(connectivity_results)} APIs connected")
                print("   • Tests will use available APIs with fallbacks")
            
            return success
            
        except Exception as e:
            logger.error(f"Test runner failed: {e}")
            print(f"💥 Test runner failed: {e}")
            return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1) 