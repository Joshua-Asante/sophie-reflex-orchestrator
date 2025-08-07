#!/usr/bin/env python3
"""
Core Dependencies Test Script
Tests all critical dependencies for SOPHIE orchestrator
"""

import sys
import importlib
from typing import List, Dict, Any

def test_import(module_name: str, package_name: str = None) -> Dict[str, Any]:
    """Test importing a module and return results."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        return {
            "success": True,
            "module": module_name,
            "version": version,
            "error": None
        }
    except ImportError as e:
        return {
            "success": False,
            "module": module_name,
            "version": None,
            "error": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "module": module_name,
            "version": None,
            "error": f"Unexpected error: {str(e)}"
        }

def main():
    """Test all core dependencies."""
    print("🧪 Testing SOPHIE Core Dependencies")
    print("=" * 50)
    
    # Core dependencies to test
    dependencies = [
        # Core Framework
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        
        # Async and HTTP
        ("aiohttp", "aiohttp"),
        ("httpx", "httpx"),
        ("asyncio", "asyncio"),
        
        # Database
        ("sqlalchemy", "SQLAlchemy"),
        ("aiosqlite", "aiosqlite"),
        ("alembic", "Alembic"),
        
        # AI and ML
        ("openai", "OpenAI"),
        ("google.generativeai", "Google Generative AI"),
        ("sentence_transformers", "Sentence Transformers"),
        
        # Caching and Storage
        ("redis", "Redis"),
        # ("aioredis", "aioredis"),  # Skip due to compatibility issues
        ("chromadb", "ChromaDB"),
        
        # Monitoring
        ("prometheus_client", "Prometheus Client"),
        ("structlog", "Structlog"),
        
        # Security
        ("cryptography", "Cryptography"),
        ("bcrypt", "bcrypt"),
        ("keyring", "keyring"),
        
        # Data Processing
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "scikit-learn"),  # Fixed import name
        
        # Configuration
        ("yaml", "PyYAML"),
        ("dotenv", "python-dotenv"),
        
        # Testing
        ("pytest", "pytest"),
        ("pytest_asyncio", "pytest-asyncio"),
    ]
    
    results = []
    success_count = 0
    total_count = len(dependencies)
    
    for module_name, display_name in dependencies:
        print(f"Testing {display_name}...", end=" ")
        result = test_import(module_name)
        results.append(result)
        
        if result["success"]:
            print(f"✅ {result['version']}")
            success_count += 1
        else:
            print(f"❌ {result['error']}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {success_count}/{total_count} dependencies successful")
    
    if success_count == total_count:
        print("🎉 All core dependencies are working!")
        return True
    else:
        print("⚠️  Some dependencies failed to import:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['module']}: {result['error']}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
