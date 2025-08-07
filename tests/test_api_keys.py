#!/usr/bin/env python3
"""
Simple test to check if API keys are being loaded from environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_api_keys():
    """Test if API keys are accessible."""
    print("🔑 Testing API Key Access")
    print("=" * 40)
    
    # Check each API key
    keys_to_check = [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY", 
        "XAI_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPSEEK_API_KEY",
        "KIMI_API_KEY"
    ]
    
    all_keys_present = True
    
    for key_name in keys_to_check:
        key_value = os.getenv(key_name)
        if key_value:
            # Show first 10 characters and last 4 characters for security
            masked_key = f"{key_value[:10]}...{key_value[-4:]}" if len(key_value) > 14 else "***"
            print(f"✅ {key_name}: {masked_key}")
        else:
            print(f"❌ {key_name}: NOT FOUND")
            all_keys_present = False
    
    print("=" * 40)
    if all_keys_present:
        print("🎉 All API keys are accessible!")
        return True
    else:
        print("⚠️ Some API keys are missing!")
        return False

if __name__ == "__main__":
    test_api_keys() 