#!/usr/bin/env python3
"""
Detailed API debug test to identify specific issues with each API.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_api_detailed(provider: str, api_key: str, base_url: str = None, model: str = None):
    """Test a specific API with detailed error reporting"""
    print(f"\n🔍 Testing {provider.upper()} with detailed diagnostics...")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        import openai
        
        # Create client
        client_kwargs = {
            "api_key": api_key
        }
        
        if base_url:
            client_kwargs["base_url"] = base_url
            
        client = openai.AsyncOpenAI(**client_kwargs)
        print(f"✅ Client created successfully")
        
        # Test API call
        response = await client.chat.completions.create(
            model=model or "gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        
        print(f"✅ API call successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        print(f"❌ API call failed")
        print(f"Error Type: {error_type}")
        print(f"Error Message: {error_msg}")
        
        # Provide specific guidance based on error type
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            print(f"💡 This appears to be an authentication issue. Please check:")
            print(f"   - API key is correct and active")
            print(f"   - Account has sufficient credits/quota")
            print(f"   - API key has proper permissions")
        elif "404" in error_msg or "not found" in error_msg.lower():
            print(f"💡 This appears to be a model access issue. Please check:")
            print(f"   - Model name is correct")
            print(f"   - Your account has access to this model")
            print(f"   - Model is available in your region")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            print(f"💡 This appears to be a rate limiting issue.")
        elif "connection" in error_msg.lower():
            print(f"💡 This appears to be a network connectivity issue.")
        else:
            print(f"💡 Unknown error type. Please check the API documentation.")
        
        return False

async def main():
    """Test all APIs with detailed diagnostics"""
    print("🔍 DETAILED API DEBUG TEST")
    print("=" * 50)
    
    # Test configurations
    apis_to_test = [
        {
            "provider": "OpenAI",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-3.5-turbo"
        },
        {
            "provider": "XAI",
            "api_key": os.getenv("XAI_API_KEY"),
            "base_url": "https://api.x.ai/v1",
            "model": "grok-beta"
        },
        {
            "provider": "DeepSeek",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        },
        {
            "provider": "Kimi",
            "api_key": os.getenv("KIMI_API_KEY"),
            "base_url": "https://api.moonshot.cn/v1",
            "model": "moonshot-v1-8k"
        }
    ]
    
    results = {}
    
    for api_config in apis_to_test:
        if not api_config["api_key"]:
            print(f"❌ {api_config['provider']}: No API key found")
            results[api_config["provider"]] = False
            continue
            
        success = await test_api_detailed(
            provider=api_config["provider"],
            api_key=api_config["api_key"],
            base_url=api_config.get("base_url"),
            model=api_config.get("model")
        )
        
        results[api_config["provider"]] = success
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEBUG TEST SUMMARY")
    print("=" * 50)
    
    working_apis = [k for k, v in results.items() if v]
    failed_apis = [k for k, v in results.items() if not v]
    
    print(f"✅ Working APIs: {len(working_apis)}")
    for api in working_apis:
        print(f"   • {api}")
    
    print(f"❌ Failed APIs: {len(failed_apis)}")
    for api in failed_apis:
        print(f"   • {api}")
    
    if failed_apis:
        print(f"\n💡 For failed APIs, please check:")
        print(f"   - API keys are correct and active")
        print(f"   - Account has sufficient credits/quota")
        print(f"   - Model access permissions")
        print(f"   - Account status and billing")

if __name__ == "__main__":
    asyncio.run(main()) 