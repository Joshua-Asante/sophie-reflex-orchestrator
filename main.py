#main.py
"""
Sophie Reflex Orchestrator - Main Entry Point

A minimal but powerful swarm-based orchestration system to demonstrate reflexive AI coordination.
Supports GA-style agent loop (Prover → Evaluator → Refiner), memory persistence, trust scoring,
and basic HITL override functionality.

Usage:
    python main.py --task "Your task here" [--config path/to/config.yaml]
    python main.py --interactive  # Interactive mode
    python main.py --server       # Start HITL server only
    python main.py --help         # Show help
"""

import asyncio
import argparse
import sys
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from orchestrator import SophieReflexOrchestrator
from ui.webhook_server import WebhookServer
from ui.cli_components import TaskExecutor, ServerManager, InteractiveController, ResultDisplay
from configs.config_manager import ConfigManager
from utils.resource_manager import ResourceManager, AsyncConfigLoader
from utils.mock_orchestrator import MockOrchestratorFactory
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    context_class=dict,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class SophieCLI:
    """Command Line Interface for Sophie Reflex Orchestrator."""
    
    def __init__(self, use_mock: bool = False):
        self.orchestrator: Optional[SophieReflexOrchestrator] = None
        self.resource_manager = ResourceManager()
        self.config_manager = ConfigManager()
        self.session_id: Optional[str] = None
        self.use_mock = use_mock
    
    async def initialize_orchestrator(self, config_path: str = "configs/system.yaml"):
        """Initialize the orchestrator with proper configuration validation."""
        try:
            # Load and validate configurations
            await self.config_manager.load_and_validate_all()
            
            # Initialize orchestrator (real or mock)
            if self.use_mock:
                self.orchestrator = MockOrchestratorFactory.create_mock_orchestrator(config_path)
                logger.info("Using mock orchestrator for testing")
            else:
                self.orchestrator = SophieReflexOrchestrator(config_path)
                self.resource_manager.register_resource(self.orchestrator)
            
            logger.info("Orchestrator initialized successfully", config_path=config_path)
            
        except Exception as e:
            logger.error("Failed to initialize orchestrator", error=str(e))
            raise
    
    async def run_task(self, task: str, config_path: str = "configs/system.yaml") -> bool:
        """Run a single task with the orchestrator."""
        try:
            logger.info("Starting task execution", task=task, config=config_path)
            
            # Initialize orchestrator
            await self.initialize_orchestrator(config_path)
            
            # Create task executor
            task_executor = TaskExecutor(self.orchestrator)
            
            # Run the task
            success = await task_executor.run_task(task)
            
            if success:
                self.session_id = task_executor.session_id
                logger.info("Task completed successfully", session_id=self.session_id)
            
            return success
            
        except Exception as e:
            logger.error("Task execution failed", error=str(e))
            return False
    
    async def run_interactive(self, config_path: str = "configs/system.yaml"):
        """Run interactive mode with enhanced UI."""
        try:
            # Initialize orchestrator
            await self.initialize_orchestrator(config_path)
            
            # Create components
            server_manager = ServerManager()
            interactive_controller = InteractiveController(self.orchestrator, server_manager)
            
            # Register with resource manager
            self.resource_manager.register_resource(server_manager)
            
            # Run interactive mode
            await interactive_controller.run_interactive()
            
        except Exception as e:
            logger.error("Interactive mode failed", error=str(e))
            print(f"Error: {e}")
        finally:
            # Cleanup resources
            await self.resource_manager.cleanup_resources()
    
    async def run_server_only(self, config_path: str = "configs/system.yaml"):
        """Run only the HITL server."""
        try:
            server_manager = ServerManager()
            self.resource_manager.register_resource(server_manager)
            
            await server_manager.start_server()
            
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
        except Exception as e:
            logger.error("Server failed", error=str(e))
            print(f"❌ Server error: {e}")
        finally:
            await self.resource_manager.cleanup_resources()
    
    async def save_results_async(self, results: Dict[str, Any], output_path: str):
        """Save results asynchronously."""
        await self.resource_manager.save_results_async(results, output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sophie Reflex Orchestrator - Swarm-based AI coordination system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --task "Design a sustainable city"
  python main.py --interactive
  python main.py --server
  python main.py --task "Write a poem" --config custom_config.yaml
  python main.py --test  # Run with mock orchestrator for testing
        """
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Task to execute with the orchestrator"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/system.yaml",
        help="Path to configuration file (default: configs/system.yaml)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--server", "-s",
        action="store_true",
        help="Run only the HITL server"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with mock orchestrator for testing"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Sophie Reflex Orchestrator v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create CLI instance with mock option
    cli = SophieCLI(use_mock=args.test)
    
    try:
        if args.server:
            # Run server only
            asyncio.run(cli.run_server_only(args.config))
        
        elif args.interactive:
            # Run interactive mode
            asyncio.run(cli.run_interactive(args.config))
        
        elif args.task:
            # Run single task
            success = asyncio.run(cli.run_task(args.task, args.config))
            
            # Save results if output file specified
            if success and args.output and cli.orchestrator:
                results = cli.orchestrator.get_results()
                stats = asyncio.run(cli.orchestrator.get_statistics())
                
                output_data = {
                    "task": args.task,
                    "session_id": cli.session_id,
                    "results": results,
                    "statistics": stats,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Use async file I/O
                asyncio.run(cli.save_results_async(output_data, args.output))
                print(f"\n💾 Results saved to: {args.output}")
            
            # Exit with appropriate code
            sys.exit(0 if success else 1)
        
        else:
            # No arguments provided, show help
            parser.print_help()
            print("\n💡 Tip: Use --interactive for interactive mode or --task to run a specific task")
            print("💡 Use --test to run with mock orchestrator for testing")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Application failed", error=str(e))
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()