#!/usr/bin/env python3
"""
ChromaDB Files Diagnostic Script
Quick check to verify ChromaDB files are accessible and working
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from memory.vector_store import VectorStore, MemoryEntry
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies: pip install chromadb")
    sys.exit(1)


class ChromaDBDiagnostic:
    """Diagnostic tool for ChromaDB files."""
    
    def __init__(self):
        self.chromadb_dir = "chroma_db"
        self.results = {}
    
    def check_file_structure(self) -> Dict[str, Any]:
        """Check if ChromaDB directory and files exist."""
        print("🔍 Checking ChromaDB file structure...")
        
        results = {
            "directory_exists": False,
            "files_found": [],
            "total_size": 0,
            "file_count": 0
        }
        
        # Check if directory exists
        if os.path.exists(self.chromadb_dir):
            results["directory_exists"] = True
            print(f"✅ ChromaDB directory found: {self.chromadb_dir}")
            
            # List all files
            try:
                for root, dirs, files in os.walk(self.chromadb_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        results["files_found"].append({
                            "path": file_path,
                            "size": file_size,
                            "relative_path": os.path.relpath(file_path, self.chromadb_dir)
                        })
                        results["total_size"] += file_size
                        results["file_count"] += 1
                
                print(f"📁 Found {results['file_count']} files")
                print(f"💾 Total size: {results['total_size'] / 1024:.2f} KB")
                
                # Show file types
                file_extensions = {}
                for file_info in results["files_found"]:
                    ext = os.path.splitext(file_info["relative_path"])[1]
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1
                
                print("📋 File types found:")
                for ext, count in file_extensions.items():
                    print(f"   {ext or 'no extension'}: {count} files")
                
            except Exception as e:
                print(f"❌ Error reading directory: {e}")
                results["error"] = str(e)
        else:
            print(f"❌ ChromaDB directory not found: {self.chromadb_dir}")
        
        return results
    
    def check_chromadb_connectivity(self) -> Dict[str, Any]:
        """Check if ChromaDB can connect to existing files."""
        print("\n🔌 Testing ChromaDB connectivity...")
        
        results = {
            "can_connect": False,
            "collections": [],
            "error": None
        }
        
        try:
            # Try to connect to existing ChromaDB
            client = chromadb.PersistentClient(path=self.chromadb_dir)
            
            # List collections
            collections = client.list_collections()
            results["collections"] = [col.name for col in collections]
            results["can_connect"] = True
            
            print(f"✅ Successfully connected to ChromaDB")
            print(f"📚 Found {len(collections)} collections:")
            
            for collection in collections:
                try:
                    count = collection.count()
                    print(f"   - {collection.name}: {count} documents")
                except Exception as e:
                    print(f"   - {collection.name}: Error getting count - {e}")
            
        except Exception as e:
            print(f"❌ Failed to connect to ChromaDB: {e}")
            results["error"] = str(e)
        
        return results
    
    async def test_vector_store_integration(self) -> Dict[str, Any]:
        """Test VectorStore integration with existing ChromaDB."""
        print("\n🧪 Testing VectorStore integration...")
        
        results = {
            "can_initialize": False,
            "can_add_entry": False,
            "can_retrieve_entries": False,
            "entry_count": 0,
            "error": None
        }
        
        try:
            # Initialize VectorStore with existing directory
            config = {
                'backend': 'chroma',
                'collection_name': 'test_diagnostic',
                'persist_directory': self.chromadb_dir
            }
            
            vector_store = VectorStore(config)
            results["can_initialize"] = True
            print("✅ VectorStore initialized successfully")
            
            # Test adding an entry
            test_entry = MemoryEntry(
                content="Diagnostic test entry",
                metadata={"diagnostic": True, "timestamp": "2024-01-01"},
                embedding=[0.1] * 100  # 100-dimensional vector
            )
            
            add_result = await vector_store.add_entry(test_entry)
            if add_result:
                results["can_add_entry"] = True
                print("✅ Successfully added test entry")
            else:
                print("❌ Failed to add test entry")
            
            # Test retrieving entries
            entries = await vector_store.get_entries(limit=10)
            results["entry_count"] = len(entries)
            if len(entries) > 0:
                results["can_retrieve_entries"] = True
                print(f"✅ Successfully retrieved {len(entries)} entries")
            else:
                print("⚠️  No entries found (this might be normal for new installations)")
            
        except Exception as e:
            print(f"❌ VectorStore integration test failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def check_permissions(self) -> Dict[str, Any]:
        """Check file permissions and accessibility."""
        print("\n🔐 Checking file permissions...")
        
        results = {
            "readable": False,
            "writable": False,
            "executable": False,
            "permissions": None
        }
        
        if os.path.exists(self.chromadb_dir):
            try:
                # Check directory permissions
                stat_info = os.stat(self.chromadb_dir)
                results["permissions"] = oct(stat_info.st_mode)[-3:]
                
                # Test readability
                test_file = os.path.join(self.chromadb_dir, "test_permissions")
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    results["writable"] = True
                    os.remove(test_file)
                    print("✅ Directory is writable")
                except Exception as e:
                    print(f"❌ Directory is not writable: {e}")
                
                results["readable"] = True
                print("✅ Directory is readable")
                
            except Exception as e:
                print(f"❌ Permission check failed: {e}")
                results["error"] = str(e)
        
        return results
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        print("\n" + "=" * 60)
        print("🏥 CHROMADB HEALTH DIAGNOSTIC REPORT")
        print("=" * 60)
        
        # Run all checks
        file_structure = self.check_file_structure()
        connectivity = self.check_chromadb_connectivity()
        permissions = self.check_permissions()
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        integration = loop.run_until_complete(self.test_vector_store_integration())
        loop.close()
        
        # Compile results
        health_report = {
            "file_structure": file_structure,
            "connectivity": connectivity,
            "integration": integration,
            "permissions": permissions,
            "overall_status": "UNKNOWN"
        }
        
        # Determine overall status
        all_good = (
            file_structure.get("directory_exists", False) and
            connectivity.get("can_connect", False) and
            integration.get("can_initialize", False) and
            permissions.get("readable", False)
        )
        
        if all_good:
            health_report["overall_status"] = "HEALTHY"
            print("✅ ChromaDB is HEALTHY and working correctly!")
        elif file_structure.get("directory_exists", False):
            health_report["overall_status"] = "PARTIAL"
            print("⚠️  ChromaDB has some issues but files exist")
        else:
            health_report["overall_status"] = "UNHEALTHY"
            print("❌ ChromaDB is UNHEALTHY - files may be missing or corrupted")
        
        # Print detailed status
        print(f"\n📊 Detailed Status:")
        print(f"   File Structure: {'✅' if file_structure.get('directory_exists') else '❌'}")
        print(f"   Connectivity: {'✅' if connectivity.get('can_connect') else '❌'}")
        print(f"   Integration: {'✅' if integration.get('can_initialize') else '❌'}")
        print(f"   Permissions: {'✅' if permissions.get('readable') else '❌'}")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        if health_report["overall_status"] == "HEALTHY":
            print("✅ Your ChromaDB files are working correctly!")
            print("✅ You can safely use the vector store functionality.")
            print("✅ No action needed.")
        elif health_report["overall_status"] == "PARTIAL":
            print("⚠️  Some issues detected:")
            if not connectivity.get("can_connect"):
                print("   - ChromaDB connection issues - check file permissions")
            if not integration.get("can_initialize"):
                print("   - VectorStore integration issues - check dependencies")
        else:
            print("❌ Significant issues detected:")
            if not file_structure.get("directory_exists"):
                print("   - ChromaDB directory missing - run the application to create it")
            print("   - Consider running: python test_chromadb_integration.py")
        
        return health_report


def main():
    """Main diagnostic function."""
    print("🔍 ChromaDB Files Diagnostic Tool")
    print("=" * 40)
    
    diagnostic = ChromaDBDiagnostic()
    health_report = diagnostic.generate_health_report()
    
    # Save report to file
    report_file = "chromadb_diagnostic_report.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(health_report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")
    
    return 0 if health_report["overall_status"] == "HEALTHY" else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 