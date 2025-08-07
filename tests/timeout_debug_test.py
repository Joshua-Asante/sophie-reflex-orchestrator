#!/usr/bin/env python3
"""
Debug test for timeout configuration differences.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_timeout_configs():
    """Test different timeout configurations"""
    print("🔍 TIMEOUT CONFIGURATION DEBUG")
    print("=" * 40)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ No Mistral API key found")
        return
    
    # Test 1: Working configuration (simple timeout)
    print("\n🔍 Test 1: Working configuration (simple timeout)")
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
        print(f"✅ Simple timeout: Success")
        
    except Exception as e:
        print(f"❌ Simple timeout failed: {str(e)}")
    
    # Test 2: Agent framework configuration (aiohttp timeout)
    print("\n🔍 Test 2: Agent framework configuration (aiohttp timeout)")
    try:
        import openai
        import aiohttp
        
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        response = await client.chat.completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"✅ aiohttp timeout: Success")
        
    except Exception as e:
        print(f"❌ aiohttp timeout failed: {str(e)}")
    
    # Test 3: Different aiohttp timeout configurations
    print("\n🔍 Test 3: Different aiohttp timeout configurations")
    timeout_configs = [
        ("total=30", aiohttp.ClientTimeout(total=30)),
        ("connect=10, total=30", aiohttp.ClientTimeout(connect=10, total=30)),
        ("connect=30, total=60", aiohttp.ClientTimeout(connect=30, total=60)),
        ("connect=5, total=15", aiohttp.ClientTimeout(connect=5, total=15))
    ]
    
    for name, timeout_config in timeout_configs:
        try:
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1",
                timeout=timeout_config
            )
            
            response = await client.chat.completions.create(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print(f"✅ {name}: Success")
            
        except Exception as e:
            print(f"❌ {name}: Failed - {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_timeout_configs()) 