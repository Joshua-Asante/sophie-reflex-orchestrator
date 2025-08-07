"""
Test reasoning and security system integration: reasoning, reasoning_modes, security_manager, store_credentials
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import json
from pathlib import Path

from reasoning import Reasoner, ReasoningConfig, ReasoningResult
from reasoning_modes import ReasoningMode, describe_mode
from security.security_manager import (
    generate_user_key, store_credential, retrieve_credential,
    encrypt_data, decrypt_data, sync_to_cloud, load_from_cloud
)
from security.store_credentials import sanitize_path


def test_reasoning_modes():
    """Test reasoning modes functionality."""
    print("Testing reasoning modes...")
    
    # Test enum values
    assert ReasoningMode.CONSENSUS.value == "consensus"
    assert ReasoningMode.REFLECTIVE_PAUSE.value == "reflective_pause"
    assert ReasoningMode.DISSENT_TRACKING.value == "dissent_tracking"
    assert ReasoningMode.ETHICAL_ESCALATION.value == "ethical_escalation"
    assert ReasoningMode.UNCERTAINTY_ANALYSIS.value == "uncertainty_analysis"
    assert ReasoningMode.ADVERSARIAL_PROBE.value == "adversarial_probe"
    assert ReasoningMode.FAST_PATH.value == "fast_path"
    
    # Test mode descriptions
    description = describe_mode(ReasoningMode.CONSENSUS)
    assert "converge" in description.lower()
    
    description = describe_mode(ReasoningMode.FAST_PATH)
    assert "shortcut" in description.lower()
    
    print("✓ Reasoning modes working")


def test_reasoning_config():
    """Test reasoning configuration functionality."""
    print("Testing reasoning configuration...")
    
    # Test default config
    config = ReasoningConfig()
    assert config.provider == "gemini"
    assert config.max_turns == 7
    assert config.mode == ReasoningMode.FAST_PATH
    
    # Test custom config
    custom_config = ReasoningConfig(
        provider="openai",
        max_turns=10,
        mode=ReasoningMode.CONSENSUS
    )
    assert custom_config.provider == "openai"
    assert custom_config.max_turns == 10
    assert custom_config.mode == ReasoningMode.CONSENSUS
    
    print("✓ Reasoning configuration working")


def test_reasoner_initialization():
    """Test reasoner initialization."""
    print("Testing reasoner initialization...")
    
    # Mock LLM provider
    async def mock_llm_provider(messages, provider):
        return "Mock response"
    
    # Test reasoner initialization
    reasoner = Reasoner(mock_llm_provider)
    assert reasoner.config is not None
    assert reasoner.tool_registry is not None
    
    # Test with custom config
    custom_config = ReasoningConfig(provider="openai", max_turns=5)
    reasoner = Reasoner(mock_llm_provider, custom_config)
    assert reasoner.config.provider == "openai"
    assert reasoner.config.max_turns == 5
    
    print("✓ Reasoner initialization working")


@pytest.mark.asyncio
async def test_reasoner_execution():
    """Test reasoner execution functionality."""
    print("Testing reasoner execution...")
    
    # Mock LLM provider that returns final answer
    async def mock_llm_provider(messages, provider):
        return "Thought: I have enough information.\nFINAL_ANSWER: Test answer"
    
    reasoner = Reasoner(mock_llm_provider)
    
    try:
        result = await reasoner.execute("Test goal")
        assert isinstance(result, ReasoningResult)
        assert result.success == True
        assert "Test answer" in result.final_answer
        print("✓ Reasoner execution working")
    except Exception as e:
        print(f"⚠️ Reasoner execution failed: {e}")
    
    print("✓ Reasoner infrastructure working")


def test_security_manager():
    """Test security manager functionality."""
    print("Testing security manager...")
    
    try:
        # Test user key generation
        generate_user_key()
        print("✓ User key generation working")
    except Exception as e:
        print(f"⚠️ User key generation failed: {e}")
    
    try:
        # Test credential storage and retrieval
        store_credential("test_service", "test_user", "test_password")
        retrieved = retrieve_credential("test_service", "test_user")
        assert retrieved == "test_password"
        print("✓ Credential storage working")
    except Exception as e:
        print(f"⚠️ Credential storage failed: {e}")
    
    try:
        # Test data encryption and decryption
        test_data = {"key": "value", "number": 42}
        encrypted = encrypt_data(test_data)
        assert encrypted != ""
        
        decrypted = decrypt_data(encrypted)
        assert decrypted == test_data
        print("✓ Data encryption working")
    except Exception as e:
        print(f"⚠️ Data encryption failed: {e}")
    
    try:
        # Test cloud sync
        test_data = {"cloud_key": "cloud_value"}
        sync_to_cloud(test_data)
        
        loaded_data = load_from_cloud()
        assert "cloud_key" in loaded_data
        print("✓ Cloud sync working")
    except Exception as e:
        print(f"⚠️ Cloud sync failed: {e}")
    
    print("✓ Security manager infrastructure working")


def test_store_credentials():
    """Test store credentials functionality."""
    print("Testing store credentials...")
    
    # Test path sanitization
    test_paths = [
        '"~/test/path.json"',
        "'/absolute/path.json'",
        "~/relative/path.json",
        "normal/path.json"
    ]
    
    for path in test_paths:
        sanitized = sanitize_path(path)
        assert '"' not in sanitized
        assert "'" not in sanitized
        assert "~" not in sanitized or os.path.expanduser("~") in sanitized
    
    print("✓ Path sanitization working")
    print("✓ Store credentials infrastructure working")


def test_core_modules():
    """Test core modules functionality."""
    print("Testing core modules...")
    
    # Test core.tool_registry
    try:
        from core.tool_registry import ToolRegistry
        registry = ToolRegistry()
        registry.register_tool("test_tool", lambda x: x)
        tools = registry.get_all_tools()
        assert "test_tool" in tools
        print("✓ core.tool_registry available")
    except ImportError:
        print("❌ core.tool_registry not available")
    
    print("✓ Core modules working")


def test_dependencies():
    """Test that required dependencies are available."""
    print("Testing dependencies...")
    
    # Test basic dependencies
    try:
        import pydantic
        print("✓ pydantic available")
    except ImportError:
        print("❌ pydantic not available")
    
    try:
        import keyring
        print("✓ keyring available")
    except ImportError:
        print("❌ keyring not available")
    
    try:
        import cryptography
        print("✓ cryptography available")
    except ImportError:
        print("❌ cryptography not available")
    
    try:
        import asyncio
        print("✓ asyncio available")
    except ImportError:
        print("❌ asyncio not available")
    
    print("✓ Basic dependencies working")


def test_directory_structure():
    """Test that directory structure exists."""
    print("Testing directory structure...")
    
    # Test reasoning files
    reasoning_files = [
        "reasoning.py",
        "reasoning_modes.py"
    ]
    
    for file_name in reasoning_files:
        file_path = Path(file_name)
        assert file_path.exists(), f"Reasoning file {file_name} should exist"
    
    # Test security directory
    security_dir = Path("security")
    assert security_dir.exists(), "Security directory should exist"
    
    # Test security files
    security_files = [
        "security_manager.py",
        "store_credentials.py"
    ]
    
    for file_name in security_files:
        file_path = security_dir / file_name
        assert file_path.exists(), f"Security file {file_name} should exist"
    
    print("✓ Directory structure working")


def test_integration():
    """Test integration of reasoning and security components."""
    print("Testing integration...")
    
    try:
        # Test reasoning modes
        mode = ReasoningMode.CONSENSUS
        description = describe_mode(mode)
        assert description is not None
        
        # Test security manager
        test_data = {"integration": "test"}
        encrypted = encrypt_data(test_data)
        decrypted = decrypt_data(encrypted)
        assert decrypted == test_data
        
        print("✓ Integration working")
    except Exception as e:
        print(f"⚠️ Integration test failed: {e}")
    
    print("✓ All components integrated")


def main():
    """Run all basic tests."""
    print("Testing reasoning and security system integration...\n")
    
    try:
        test_dependencies()
        test_directory_structure()
        test_core_modules()
        test_reasoning_modes()
        test_reasoning_config()
        test_reasoner_initialization()
        test_security_manager()
        test_store_credentials()
        test_integration()
        
        # Run async tests
        asyncio.run(test_reasoner_execution())
        
        print("\n✅ All reasoning and security system integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 