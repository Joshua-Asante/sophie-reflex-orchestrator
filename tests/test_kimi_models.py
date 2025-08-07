#!/usr/bin/env python3
"""
Test Kimi API models and permissions.
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

async def test_kimi_models():
    """Test Kimi API models and permissions."""
    print("🔍 KIMI API MODELS TEST")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv("KIMI_API_KEY")
    if not api_key:
        print("❌ KIMI_API_KEY not found in environment")
        return
    
    print(f"✅ KIMI_API_KEY found (length: {len(api_key)})")
    
    try:
        # Create client
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        
        print("✅ Kimi client created successfully")
        
        # Test 1: List available models
        print("\n🔍 Testing 1: List available models...")
        try:
            models = await client.models.list()
            print("✅ Models list successful!")
            print("Available models:")
            for model in models.data:
                print(f"   • {model.id}")
        except Exception as e:
            print(f"❌ Models list failed: {type(e).__name__}: {str(e)}")
        
        # Test 2: Try different model names
        print("\n🔍 Testing 2: Try different model names...")
        test_models = [
            "moonshot-v1-8k",
            "moonshot-v1-32k", 
            "moonshot-v1-128k",
            "moonshot-v1-8k-chat",
            "moonshot-v1-32k-chat",
            "moonshot-v1-128k-chat"
        ]
        
        for model_name in test_models:
            try:
                print(f"   Testing model: {model_name}")
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                print(f"   ✅ {model_name}: SUCCESS!")
                break
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    print(f"   ❌ {model_name}: Authentication failed")
                elif "404" in error_msg or "not found" in error_msg.lower():
                    print(f"   ❌ {model_name}: Model not found")
                else:
                    print(f"   ❌ {model_name}: {error_msg}")
        
        # Test 3: Check account status
        print("\n🔍 Testing 3: Check account status...")
        try:
            # Try to get billing info or account status
            response = await client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            print("✅ Account appears to be active")
        except Exception as e:
            print(f"❌ Account test failed: {str(e)}")
            
    except Exception as e:
        print(f"❌ Setup failed: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_kimi_models()) 