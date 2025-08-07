#!/usr/bin/env python3
"""
Simple test to verify new API keys are loaded correctly.
"""

import os
import asyncio
from dotenv import load_dotenv

async def test_single_api(provider: str, api_key: str, base_url: str = None, model: str = None):
    """Test a single API with the new key"""
    print(f"\n🔍 Testing {provider} with new API key...")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        import openai
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            
        client = openai.AsyncOpenAI(**client_kwargs)
        
        response = await client.chat.completions.create(
            model=model or "gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print(f"✅ {provider}: SUCCESS!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ {provider}: FAILED - {str(e)}")
        return False

async def main():
    """Test the new API keys"""
    print("🔍 TESTING NEW API KEYS")
    print("=" * 40)
    
    # Force reload environment variables
    load_dotenv(override=True)
    
    # Test each API with new keys
    tests = [
        ("OpenAI", os.getenv("OPENAI_API_KEY"), None, "gpt-3.5-turbo"),
        ("XAI", os.getenv("XAI_API_KEY"), "https://api.x.ai/v1", "grok-beta"),
        ("DeepSeek", os.getenv("DEEPSEEK_API_KEY"), "https://api.deepseek.com/v1", "deepseek-chat"),
        ("Kimi", os.getenv("KIMI_API_KEY"), "https://api.moonshot.cn/v1", "moonshot-v1-8k")
    ]
    
    results = []
    for provider, api_key, base_url, model in tests:
        if api_key:
            success = await test_single_api(provider, api_key, base_url, model)
            results.append((provider, success))
        else:
            print(f"❌ {provider}: No API key found")
            results.append((provider, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 RESULTS SUMMARY")
    print("=" * 40)
    
    working = [p for p, s in results if s]
    failed = [p for p, s in results if not s]
    
    print(f"✅ Working: {len(working)}")
    for provider in working:
        print(f"   • {provider}")
    
    print(f"❌ Failed: {len(failed)}")
    for provider in failed:
        print(f"   • {provider}")

if __name__ == "__main__":
    asyncio.run(main()) 