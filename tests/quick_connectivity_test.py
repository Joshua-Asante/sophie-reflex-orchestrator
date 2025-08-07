#!/usr/bin/env python3
"""
Quick connectivity test to verify which APIs are working.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.prover import ProverAgent
from agents.base_agent import AgentConfig

async def test_connectivity():
    """Test connectivity using the same logic as setup_api_keys.py"""
    print("🔍 QUICK CONNECTIVITY TEST")
    print("=" * 40)
    
    # Test the APIs that should be working
    test_apis = ["google", "mistral", "kimi"]
    
    for provider in test_apis:
        print(f"\n🔍 Testing {provider.upper()}...")
        
        try:
            # Create test config
            test_config = AgentConfig(
                name=f"test_{provider}",
                prompt="Test prompt",
                model=provider,
                temperature=0.7,
                max_tokens=50,
                timeout=15
            )
            
            # Create test agent
            test_agent = ProverAgent(test_config, f"test_{provider}")
            
            # Get client
            client = await test_agent._llm_manager.get_client(provider, {"timeout": 15})
            
            # Test API call
            if provider == "google":
                try:
                    import google.generativeai as genai
                    response = await client.generate_content_async(
                        "Hello, this is a connectivity test.",
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=10
                        )
                    )
                    print(f"✅ {provider.upper()}: API call successful")
                    
                except Exception as e:
                    print(f"❌ {provider.upper()}: API call failed - {str(e)}")
                    
            elif provider == "mistral":
                try:
                    response = await client.chat.completions.create(
                        model="mistral-large-latest",
                        messages=[{"role": "user", "content": "Hello, this is a connectivity test."}],
                        max_tokens=10
                    )
                    print(f"✅ {provider.upper()}: API call successful")
                    
                except Exception as e:
                    print(f"❌ {provider.upper()}: API call failed - {str(e)}")
                    
            elif provider == "kimi":
                try:
                    response = await client.chat.completions.create(
                        model="moonshot-v1-8k",
                        messages=[{"role": "user", "content": "Hello, this is a connectivity test."}],
                        max_tokens=10
                    )
                    print(f"✅ {provider.upper()}: API call successful")
                    
                except Exception as e:
                    print(f"❌ {provider.upper()}: API call failed - {str(e)}")
                    
        except Exception as e:
            print(f"❌ {provider.upper()}: Setup failed - {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_connectivity()) 