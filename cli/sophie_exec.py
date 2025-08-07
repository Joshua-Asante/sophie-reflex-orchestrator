#!/usr/bin/env python3
"""
SOPHIE Exec CLI

This module provides the CLI interface for SOPHIE's unified execution engine.
It implements the sophie exec command that routes through SOPHIE's full execution loop.
"""

import asyncio
import argparse
import json
import sys
from typing import Dict, Any
from pathlib import Path

# Add the parent directory to the path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.unified_executor import sophie_exec, unified_executor
from core.reflexive_moe_orchestrator import ReflexiveMoEOrchestrator
from core.constitutional_executor import ConstitutionalExecutor


def print_execution_header():
    """Print the SOPHIE exec header."""
    print("🧠 SOPHIE Unified Execution Engine v1.0")
    print("=" * 60)
    print("Principled executor for any tool, any API, any environment")
    print("=" * 60)


def print_execution_flow():
    """Print the execution flow diagram."""
    print("\n🔄 Execution Flow:")
    print("User Intent → Intent Parser → Planner → Diff → Approval → Execution → Audit")
    print()


def print_supported_types():
    """Print supported execution types."""
    print("📋 Supported Execution Types:")
    print("  • CLI: Command line interface operations")
    print("  • API: REST API calls and web services")
    print("  • FILESYSTEM: File and directory operations")
    print("  • DATABASE: SQL and database operations")
    print("  • CLOUD: Cloud platform operations (GCP, AWS, etc.)")
    print("  • SHELL: Shell commands and scripts")
    print("  • PYTHON: Python code execution")
    print()


def print_trust_info(trust_score: float, risk_level: str):
    """Print trust and risk information."""
    print(f"🛡️ Trust Score: {trust_score:.1%}")
    print(f"⚠️ Risk Level: {risk_level}")
    print()


def print_diff_summary(diff_summary: str):
    """Print the diff summary."""
    print("📋 Changes Summary:")
    print("-" * 40)
    print(diff_summary)
    print("-" * 40)
    print()


def print_execution_result(result: Dict[str, Any]):
    """Print the execution result."""
    print("✅ Execution Result:")
    print(f"  • Status: {result.get('status', 'unknown')}")
    print(f"  • Plan ID: {result.get('plan_id', 'unknown')}")
    print(f"  • Type: {result.get('execution_type', 'unknown')}")
    print(f"  • Trust Score: {result.get('trust_score', 0):.1%}")
    print(f"  • Risk Level: {result.get('risk_level', 'unknown')}")
    
    if result.get('execution_result'):
        exec_result = result['execution_result']
        print(f"  • Commands Executed: {exec_result.get('commands_executed', 0)}")
        print(f"  • Successful Commands: {exec_result.get('successful_commands', 0)}")
    
    print()


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SOPHIE Unified Execution Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sophie exec "list all GCP compute instances"
  sophie exec "read config.yaml and update database"
  sophie exec "deploy application to staging" --auto-approve
  sophie exec "analyze logs and generate report" --context '{"env": "production"}'
        """
    )
    
    parser.add_argument(
        "intent",
        help="User intent or command to execute"
    )
    
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip approval prompts for trusted plans"
    )
    
    parser.add_argument(
        "--context",
        type=str,
        help="Additional context as JSON string"
    )
    
    parser.add_argument(
        "--show-flow",
        action="store_true",
        help="Show detailed execution flow"
    )
    
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List supported execution types"
    )
    
    parser.add_argument(
        "--trust-info",
        action="store_true",
        help="Show current trust information"
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_types:
        print_execution_header()
        print_supported_types()
        return
    
    if args.trust_info:
        print_execution_header()
        trust_score = unified_executor.trust_tracker.get_current_trust_score()
        print(f"🛡️ Current Trust Score: {trust_score:.1%}")
        print(f"📊 Role Trust Scores:")
        for role, score in unified_executor.moe_orchestrator.role_trust_scores.items():
            print(f"  • {role.title()}: {score:.1%}")
        return
    
    # Parse context if provided
    context = None
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError:
            print("❌ Error: Invalid JSON in --context argument")
            return 1
    
    # Show execution header
    print_execution_header()
    
    if args.show_flow:
        print_execution_flow()
    
    # Execute the command
    print(f"🎯 User Intent: {args.intent}")
    print()
    
    try:
        result = await sophie_exec(
            user_intent=args.intent,
            context=context,
            auto_approve=args.auto_approve
        )
        
        # Print trust and risk information
        print_trust_info(
            result.get('trust_score', 0),
            result.get('risk_level', 'unknown')
        )
        
        # Print diff summary if available
        if 'diff_summary' in result:
            print_diff_summary(result['diff_summary'])
        
        # Print execution result
        print_execution_result(result)
        
        # Return appropriate exit code
        if result.get('status') == 'completed':
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 