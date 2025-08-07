#!/usr/bin/env python3
"""
Test to check available XAI models.
"""

import os
import asyncio
from dotenv import load_dotenv

async def check_xai_models():
    """Check what models are available for XAI"""
    print("🔍 CHECKING XAI MODELS")
    print("=" * 40)
    
    load_dotenv(override=True)
    api_key = os.getenv("XAI_API_KEY")
    
    if not api_key:
        print("❌ No XAI API key found")
        return
    
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        
        # List available models
        print("\n🔍 Listing available models...")
        models = await client.models.list()
        
        print(f"✅ Found {len(models.data)} models:")
        for model in models.data:
            print(f"   • {model.id}")
        
        # Test with first available model
        if models.data:
            first_model = models.data[0].id
            print(f"\n🔍 Testing with model: {first_model}")
            
            response = await client.chat.completions.create(
                model=first_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            print(f"✅ Success with {first_model}!")
            print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(check_xai_models()) 