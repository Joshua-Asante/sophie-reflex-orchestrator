#!/usr/bin/env python3
"""
Debug test for Mistral API connectivity issues.
"""

import asyncio
import os
import sys
import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_mistral_direct():
    """Test Mistral API directly without the agent framework"""
    print("🔍 MISTRAL API DEBUG TEST")
    print("=" * 40)
    
    # Get API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ No Mistral API key found")
        return
    
    print(f"✅ Mistral API key found: {api_key[:10]}...")
    
    # Test 1: Direct HTTP request
    print("\n🔍 Test 1: Direct HTTP request to Mistral API")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            print(f"✅ HTTP Status: {response.status_code}")
            print(f"✅ Response: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ HTTP request failed: {str(e)}")
    
    # Test 2: OpenAI client with Mistral base URL
    print("\n🔍 Test 2: OpenAI client with Mistral configuration")
    try:
        import openai
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1"
        )
        
        response = await client.chat.completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"✅ OpenAI client with Mistral: Success")
        print(f"✅ Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ OpenAI client with Mistral failed: {str(e)}")
    
    # Test 3: Test through agent framework
    print("\n🔍 Test 3: Through agent framework")
    try:
        from agents.prover import ProverAgent
        from agents.base_agent import AgentConfig
        
        test_config = AgentConfig(
            name="test_mistral",
            prompt="Test prompt",
            model="mistral",
            temperature=0.7,
            max_tokens=50,
            timeout=30
        )
        
        test_agent = ProverAgent(test_config, "test_mistral")
        client = await test_agent._llm_manager.get_client("mistral", {"timeout": 30})
        
        response = await client.chat.completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"✅ Agent framework: Success")
        
    except Exception as e:
        print(f"❌ Agent framework failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_mistral_direct()) 