#!/usr/bin/env python3
"""
Direct Kimi API test to debug authentication issues.
"""

import asyncio
import os
import sys
import openai
from dotenv import load_dotenv

# Force reload environment variables
load_dotenv(override=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_kimi_direct():
    """Test Kimi API directly with detailed error reporting."""
    print("🔍 DIRECT KIMI API TEST")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv("KIMI_API_KEY")
    if not api_key:
        print("❌ KIMI_API_KEY not found in environment")
        return
    
    print(f"✅ KIMI_API_KEY found (length: {len(api_key)})")
    print(f"   Key starts with: {api_key[:10]}...")
    print(f"   Key ends with: ...{api_key[-4:]}")
    
    try:
        # Create client directly
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        
        print("✅ Kimi client created successfully")
        
        # Test API call
        print("🔍 Testing API call...")
        response = await client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        
        print("✅ Kimi API call successful!")
        print(f"   Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Kimi API call failed: {type(e).__name__}: {str(e)}")
        
        # Detailed error analysis
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            print("🔍 This is an authentication error. Possible causes:")
            print("   • API key is invalid or expired")
            print("   • API key doesn't have proper permissions")
            print("   • API key format is incorrect")
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            print("🔍 This is a permission error. Possible causes:")
            print("   • API key doesn't have access to this model")
            print("   • Account doesn't have sufficient credits")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            print("🔍 This is a rate limit error.")
        else:
            print("🔍 Unknown error type. Check the error message above.")

if __name__ == "__main__":
    asyncio.run(test_kimi_direct()) 