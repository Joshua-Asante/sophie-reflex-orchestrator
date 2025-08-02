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
from typing import Optional

from orchestrator import SophieReflexOrchestrator
from ui.webhook_server import WebhookServer
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
    
    def __init__(self):
        self.orchestrator: Optional[SophieReflexOrchestrator] = None
        self.hitl_server: Optional[WebhookServer] = None
        self.session_id: Optional[str] = None
    
    async def run_task(self, task: str, config_path: str = "configs/system.yaml") -> bool:
        """Run a single task with the orchestrator."""
        try:
            logger.info("Starting Sophie Reflex Orchestrator", task=task, config=config_path)
            
            # Initialize orchestrator
            self.orchestrator = SophieReflexOrchestrator(config_path)
            
            # Start the task
            self.session_id = await self.orchestrator.start_task(task)
            
            logger.info("Task started", session_id=self.session_id)
            
            # Run the complete task
            results = await self.orchestrator.run_complete_task(task)
            
            # Display results
            self.display_results(results)
            
            # Get final statistics
            stats = await self.orchestrator.get_statistics()
            self.display_statistics(stats)
            
            logger.info("Task completed successfully", session_id=self.session_id)
            return True
            
        except Exception as e:
            logger.error("Task execution failed", error=str(e))
            return False
    
    async def run_interactive(self, config_path: str = "configs/system.yaml"):
        """Run interactive mode."""
        try:
            print("🤖 Sophie Reflex Orchestrator - Interactive Mode")
            print("=" * 50)
            print("Type 'help' for available commands, 'quit' to exit")
            print()
            
            # Initialize orchestrator
            self.orchestrator = SophieReflexOrchestrator(config_path)
            
            # Start HITL server in background
            self.hitl_server = WebhookServer()
            hitl_task = asyncio.create_task(self.hitl_server.run_async())
            
            while True:
                try:
                    # Get user input
                    user_input = input("sophie> ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Process commands
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    elif user_input.lower() == 'help':
                        self.show_help()
                    elif user_input.lower() == 'status':
                        self.show_status()
                    elif user_input.lower() == 'stats':
                        await self.show_stats()
                    elif user_input.lower() == 'agents':
                        self.show_agents()
                    elif user_input.lower().startswith('task '):
                        task = user_input[5:].strip()
                        if task:
                            await self.run_interactive_task(task)
                        else:
                            print("Please provide a task after 'task'")
                    elif user_input.lower().startswith('run '):
                        task = user_input[4:].strip()
                        if task:
                            await self.run_interactive_task(task)
                        else:
                            print("Please provide a task after 'run'")
                    elif user_input.lower() == 'pause':
                        self.pause_orchestrator()
                    elif user_input.lower() == 'resume':
                        self.resume_orchestrator()
                    elif user_input.lower() == 'stop':
                        self.stop_orchestrator()
                    elif user_input.lower() == 'clear':
                        self.clear_screen()
                    else:
                        print(f"Unknown command: {user_input}")
                        print("Type 'help' for available commands")
                
                except KeyboardInterrupt:
                    print("\nUse 'quit' to exit")
                except EOFError:
                    break
            
            # Cleanup
            if self.orchestrator:
                self.orchestrator.stop()
            
            if hitl_task:
                hitl_task.cancel()
                try:
                    await hitl_task
                except asyncio.CancelledError:
                    pass
            
            print("\n👋 Goodbye!")
            
        except Exception as e:
            logger.error("Interactive mode failed", error=str(e))
            print(f"Error: {e}")
    
    async def run_interactive_task(self, task: str):
        """Run a task in interactive mode."""
        try:
            print(f"🚀 Running task: {task}")
            
            # Start task
            if not self.session_id:
                self.session_id = await self.orchestrator.start_task(task)
            
            # Run generations one by one
            while self.orchestrator.should_continue():
                result = await self.orchestrator.run_generation()
                
                print(f"📊 Generation {result.generation}:")
                print(f"   Best Score: {result.best_score:.3f}")
                print(f"   Average Score: {result.average_score:.3f}")
                print(f"   Execution Time: {result.execution_time:.2f}s")
                print(f"   Interventions: {len(result.interventions)}")
                
                if result.interventions:
                    print("   ⚠️  Human interventions required")
                
                # Ask user if they want to continue
                if self.orchestrator.should_continue():
                    continue_prompt = input("Continue to next generation? [Y/n]: ").strip().lower()
                    if continue_prompt in ['n', 'no']:
                        break
                else:
                    print("   ✅ Task completed or limits reached")
                    break
            
            # Finalize task
            await self.orchestrator.finalize_task()
            print("✅ Task completed!")
            
            # Show final results
            results = self.orchestrator.get_results()
            if results:
                best_result = max(results, key=lambda x: x['best_score'])
                print(f"🏆 Best solution score: {best_result['best_score']:.3f}")
                print(f"   Found in generation {best_result['generation']}")
            
        except Exception as e:
            logger.error("Interactive task failed", error=str(e))
            print(f"❌ Task failed: {e}")
    
    async def run_server_only(self, config_path: str = "configs/system.yaml"):
        """Run only the HITL server."""
        try:
            print("🌐 Starting Sophie HITL Server...")
            
            # Initialize and start server
            self.hitl_server = WebhookServer(
                host="0.0.0.0",
                port=8001
            )
            
            print(f"📡 Server running on http://0.0.0.0:8001")
            print("Press Ctrl+C to stop")
            
            await self.hitl_server.run_async()
            
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
        except Exception as e:
            logger.error("Server failed", error=str(e))
            print(f"❌ Server error: {e}")
    
    def display_results(self, results):
        """Display task results."""
        if not results:
            print("No results to display")
            return
        
        print("\n" + "=" * 60)
        print("📊 TASK RESULTS")
        print("=" * 60)
        
        best_result = max(results, key=lambda x: x.best_score)
        
        print(f"🏆 Best Solution:")
        print(f"   Generation: {best_result.generation}")
        print(f"   Score: {best_result.best_score:.3f}")
        print(f"   Execution Time: {best_result.execution_time:.2f}s")
        
        if best_result.best_solution:
            print(f"   Solution: {best_result.best_solution.get('overall_feedback', 'No feedback available')[:200]}...")
        
        print(f"\n📈 Performance Summary:")
        print(f"   Total Generations: {len(results)}")
        print(f"   Average Score: {sum(r.average_score for r in results) / len(results):.3f}")
        print(f"   Total Execution Time: {sum(r.execution_time for r in results):.2f}s")
        print(f"   Total Interventions: {sum(len(r.interventions) for r in results)}")
        
        # Show generation progression
        print(f"\n📊 Generation Progression:")
        for result in results[-10:]:  # Show last 10 generations
            status_icon = "🟢" if result.interventions else "⚪"
            print(f"   Gen {result.generation:2d}: {status_icon} {result.best_score:.3f} (avg: {result.average_score:.3f})")
    
    def display_statistics(self, stats):
        """Display comprehensive statistics."""
        print("\n" + "=" * 60)
        print("📈 COMPREHENSIVE STATISTICS")
        print("=" * 60)
        
        # Orchestrator stats
        orch_stats = stats.get("orchestrator", {})
        print(f"🤖 Orchestrator:")
        print(f"   Status: {orch_stats.get('status', 'Unknown')}")
        print(f"   Total Generations: {orch_stats.get('total_generations', 0)}")
        print(f"   Total Agents: {orch_stats.get('total_agents', 0)}")
        print(f"   Execution Time: {orch_stats.get('total_execution_time', 0):.2f}s")
        print(f"   HITL Enabled: {orch_stats.get('hitl_enabled', False)}")
        
        # Trust stats
        trust_stats = stats.get("trust", {})
        print(f"\n🎯 Trust System:")
        print(f"   Total Agents: {trust_stats.get('total_agents', 0)}")
        print(f"   Total Events: {trust_stats.get('total_events', 0)}")
        print(f"   Average Trust Score: {trust_stats.get('average_trust_score', 0):.3f}")
        
        # Audit stats
        audit_stats = stats.get("audit", {})
        print(f"\n📝 Audit Log:")
        print(f"   Total Events: {audit_stats.get('total_events', 0)}")
        print(f"   Plan Diffs: {audit_stats.get('total_plan_diffs', 0)}")
        print(f"   Performance Metrics: {audit_stats.get('total_metrics', 0)}")
        
        # Vector store stats
        vector_stats = stats.get("vector_store", {})
        print(f"\n🧠 Memory System:")
        print(f"   Backend: {vector_stats.get('backend', 'Unknown')}")
        print(f"   Total Entries: {vector_stats.get('total_entries', 0)}")
    
    def show_help(self):
        """Show help information."""
        print("\n📚 Available Commands:")
        print("  help              - Show this help message")
        print("  status            - Show orchestrator status")
        print("  stats             - Show detailed statistics")
        print("  agents            - Show current agents")
        print("  task <description> - Run a task")
        print("  run <description>  - Run a task (alias)")
        print("  pause             - Pause the orchestrator")
        print("  resume            - Resume the orchestrator")
        print("  stop              - Stop the orchestrator")
        print("  clear             - Clear the screen")
        print("  quit/exit/q       - Exit the program")
    
    def show_status(self):
        """Show orchestrator status."""
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        status = self.orchestrator.get_status()
        print(f"\n📊 Orchestrator Status:")
        print(f"   Status: {status['status']}")
        print(f"   Current Task: {status['current_task'] or 'None'}")
        print(f"   Current Generation: {status['current_generation']}")
        print(f"   Total Agents: {status['total_agents']}")
        print(f"   Best Score: {status['best_score']:.3f}")
        print(f"   Average Score: {status['average_score']:.3f}")
        
        if status['start_time']:
            start_time = datetime.fromisoformat(status['start_time'])
            elapsed = datetime.now() - start_time
            print(f"   Running Time: {elapsed.total_seconds():.0f}s")
    
    async def show_stats(self):
        """Show detailed statistics."""
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        stats = await self.orchestrator.get_statistics()
        self.display_statistics(stats)
    
    def show_agents(self):
        """Show current agents."""
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        agents = self.orchestrator.agents
        if not agents:
            print("No agents available")
            return
        
        print(f"\n🤖 Current Agents ({len(agents)}):")
        for i, agent in enumerate(agents, 1):
            info = agent.get_info()
            print(f"   {i}. {info['name']} ({info['agent_id']})")
            print(f"      Status: {info['status']}")
            print(f"      Trust Score: {info['trust_score']:.3f}")
            print(f"      Success Rate: {info['success_rate']:.3f}")
            print(f"      Model: {info['model']}")
    
    def pause_orchestrator(self):
        """Pause the orchestrator."""
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        self.orchestrator.pause()
        print("⏸️  Orchestrator paused")
    
    def resume_orchestrator(self):
        """Resume the orchestrator."""
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        self.orchestrator.resume()
        print("▶️  Orchestrator resumed")
    
    def stop_orchestrator(self):
        """Stop the orchestrator."""
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        self.orchestrator.stop()
        print("🛑 Orchestrator stopped")
    
    def clear_screen(self):
        """Clear the screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')


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
        "--version",
        action="version",
        version="Sophie Reflex Orchestrator v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create CLI instance
    cli = SophieCLI()
    
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
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                
                print(f"\n💾 Results saved to: {args.output}")
            
            # Exit with appropriate code
            sys.exit(0 if success else 1)
        
        else:
            # No arguments provided, show help
            parser.print_help()
            print("\n💡 Tip: Use --interactive for interactive mode or --task to run a specific task")
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