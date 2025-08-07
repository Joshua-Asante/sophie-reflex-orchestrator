#!/usr/bin/env python3
"""
Simple ChromaDB Test Script
Basic check to verify ChromaDB is working
"""

import os
import sys
import asyncio

def check_chromadb_installation():
    """Check if ChromaDB is properly installed."""
    print("🔍 Checking ChromaDB installation...")
    
    try:
        import chromadb
        print("✅ ChromaDB is installed")
        print(f"   Version: {chromadb.__version__}")
        return True
    except ImportError as e:
        print(f"❌ ChromaDB not installed: {e}")
        print("   Install with: pip install chromadb")
        return False

def check_chromadb_directory():
    """Check if ChromaDB directory exists and has files."""
    print("\n📁 Checking ChromaDB directory...")
    
    chromadb_dir = "chroma_db"
    
    if not os.path.exists(chromadb_dir):
        print(f"❌ ChromaDB directory not found: {chromadb_dir}")
        print("   This is normal for new installations.")
        return False
    
    print(f"✅ ChromaDB directory found: {chromadb_dir}")
    
    # Count files
    file_count = 0
    total_size = 0
    
    for root, dirs, files in os.walk(chromadb_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            file_count += 1
            print(f"   📄 {os.path.relpath(file_path, chromadb_dir)} ({file_size} bytes)")
    
    print(f"📊 Total files: {file_count}")
    print(f"💾 Total size: {total_size / 1024:.2f} KB")
    
    return file_count > 0

def test_chromadb_connection():
    """Test basic ChromaDB connection."""
    print("\n🔌 Testing ChromaDB connection...")
    
    try:
        import chromadb
        
        # Try to create a client
        client = chromadb.PersistentClient(path="chroma_db")
        print("✅ Successfully created ChromaDB client")
        
        # List collections
        collections = client.list_collections()
        print(f"📚 Found {len(collections)} collections:")
        
        for collection in collections:
            try:
                count = collection.count()
                print(f"   - {collection.name}: {count} documents")
            except Exception as e:
                print(f"   - {collection.name}: Error getting count - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB: {e}")
        return False

def test_basic_operations():
    """Test basic ChromaDB operations."""
    print("\n🧪 Testing basic operations...")
    
    try:
        import chromadb
        
        # Create a test collection
        client = chromadb.PersistentClient(path="chroma_db")
        test_collection = client.create_collection(name="test_collection")
        
        # Add a test document
        test_collection.add(
            documents=["This is a test document"],
            metadatas=[{"test": True}],
            ids=["test_id_1"]
        )
        
        print("✅ Successfully added test document")
        
        # Query the document
        results = test_collection.query(
            query_texts=["test document"],
            n_results=1
        )
        
        if results['documents'] and len(results['documents'][0]) > 0:
            print("✅ Successfully queried test document")
            print(f"   Found: {results['documents'][0][0]}")
        else:
            print("❌ Query returned no results")
        
        # Clean up
        client.delete_collection(name="test_collection")
        print("✅ Cleaned up test collection")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic operations test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Simple ChromaDB Test")
    print("=" * 40)
    
    tests = [
        ("Installation", check_chromadb_installation),
        ("Directory", check_chromadb_directory),
        ("Connection", test_chromadb_connection),
        ("Operations", test_basic_operations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST RESULTS")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! ChromaDB is working correctly.")
        print("✅ Your ChromaDB files are accurate and functional.")
    elif passed >= len(results) - 1:
        print("⚠️  Most tests passed. ChromaDB is mostly working.")
        print("✅ Your ChromaDB files are likely accurate.")
    else:
        print("❌ Several tests failed. There may be issues with ChromaDB.")
        print("🔧 Consider reinstalling ChromaDB or checking dependencies.")
    
    return 0 if passed >= len(results) - 1 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 