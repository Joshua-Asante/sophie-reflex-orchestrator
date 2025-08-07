"""
Enhanced Interactive UI for Sophie Reflex Orchestrator

Provides auto-complete functionality and progress indicators for better user experience.
"""

import asyncio
import readline
import sys
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class Command:
    """Represents a CLI command."""
    name: str
    description: str
    usage: str
    aliases: List[str] = None
    completer: Optional[Callable] = None


class CommandCompleter:
    """Provides auto-completion for commands."""
    
    def __init__(self, commands: List[Command]):
        self.commands = commands
        self.command_names = [cmd.name for cmd in commands]
        self.all_names = []
        
        # Build complete list of names including aliases
        for cmd in commands:
            self.all_names.append(cmd.name)
            if cmd.aliases:
                self.all_names.extend(cmd.aliases)
    
    def complete(self, text: str, state: int) -> Optional[str]:
        """Complete command based on input text."""
        matches = [name for name in self.all_names if name.startswith(text)]
        
        if state < len(matches):
            return matches[state]
        return None


class ProgressIndicator:
    """Shows progress indicators for long-running operations."""
    
    def __init__(self):
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current_spinner = 0
        self.is_running = False
    
    async def show_spinner(self, message: str, duration: float = None):
        """Show a spinning progress indicator."""
        self.is_running = True
        start_time = asyncio.get_event_loop().time()
        
        while self.is_running:
            if duration and (asyncio.get_event_loop().time() - start_time) > duration:
                break
            
            spinner = self.spinner_chars[self.current_spinner % len(self.spinner_chars)]
            print(f"\r{spinner} {message}", end="", flush=True)
            
            self.current_spinner += 1
            await asyncio.sleep(0.1)
        
        print()  # New line after spinner
    
    def stop_spinner(self):
        """Stop the spinner."""
        self.is_running = False
    
    def show_progress_bar(self, current: int, total: int, message: str = ""):
        """Show a progress bar."""
        if total == 0:
            return
        
        percentage = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        print(f"\r{message} |{bar}| {percentage:.1f}% ({current}/{total})", end="", flush=True)
    
    def clear_progress(self):
        """Clear the current progress display."""
        print("\r" + " " * 80 + "\r", end="", flush=True)


class EnhancedInteractiveUI:
    """Enhanced interactive UI with auto-complete and progress indicators."""
    
    def __init__(self):
        self.progress = ProgressIndicator()
        self.commands = self._initialize_commands()
        self.completer = CommandCompleter(self.commands)
        self.command_handlers: Dict[str, Callable] = {}
        self.is_running = False
    
    def _initialize_commands(self) -> List[Command]:
        """Initialize available commands."""
        return [
            Command("help", "Show this help message", "help"),
            Command("status", "Show orchestrator status", "status"),
            Command("stats", "Show detailed statistics", "stats"),
            Command("agents", "Show current agents", "agents"),
            Command("task", "Run a task", "task <description>", aliases=["run"]),
            Command("pause", "Pause the orchestrator", "pause"),
            Command("resume", "Resume the orchestrator", "resume"),
            Command("stop", "Stop the orchestrator", "stop"),
            Command("clear", "Clear the screen", "clear"),
            Command("quit", "Exit the program", "quit", aliases=["exit", "q"]),
        ]
    
    def register_command_handler(self, command_name: str, handler: Callable):
        """Register a command handler."""
        self.command_handlers[command_name] = handler
    
    def setup_readline(self):
        """Setup readline for auto-completion."""
        try:
            readline.set_completer(self.completer.complete)
            readline.parse_and_bind("tab: complete")
            readline.set_completer_delims(" \t\n")
        except Exception as e:
            logger.warning("Failed to setup readline", error=str(e))
    
    async def run_interactive(self, prompt: str = "sophie> "):
        """Run the interactive UI."""
        self.is_running = True
        self.setup_readline()
        
        print("🤖 Sophie Reflex Orchestrator - Enhanced Interactive Mode")
        print("=" * 60)
        print("Type 'help' for available commands, 'quit' to exit")
        print("Use TAB for auto-completion")
        print()
        
        while self.is_running:
            try:
                # Get user input
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                # Handle command
                await self._handle_command(command, args)
                
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except EOFError:
                break
            except Exception as e:
                logger.error("Interactive UI error", error=str(e))
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")
    
    async def _handle_command(self, command: str, args: List[str]):
        """Handle a command."""
        # Find the command
        cmd_obj = None
        for cmd in self.commands:
            if command == cmd.name or (cmd.aliases and command in cmd.aliases):
                cmd_obj = cmd
                break
        
        if not cmd_obj:
            print(f"❌ Unknown command: {command}")
            print("Type 'help' for available commands")
            return
        
        # Check if handler is registered
        if cmd_obj.name in self.command_handlers:
            try:
                await self.command_handlers[cmd_obj.name](*args)
            except Exception as e:
                logger.error("Command handler error", command=command, error=str(e))
                print(f"❌ Command failed: {e}")
        else:
            print(f"⚠️  Command '{command}' not implemented")
    
    def show_help(self):
        """Show help information."""
        print("\n📚 Available Commands:")
        print("=" * 40)
        
        for cmd in self.commands:
            aliases_str = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
            print(f"  {cmd.name:<12} - {cmd.description}{aliases_str}")
            if cmd.usage != cmd.name:
                print(f"              Usage: {cmd.usage}")
        
        print("\n💡 Tips:")
        print("  - Use TAB for auto-completion")
        print("  - Commands are case-insensitive")
        print("  - Use 'help' to see this message again")
    
    async def show_progress_operation(self, message: str, operation: Callable, *args, **kwargs):
        """Show progress for an operation."""
        try:
            # Start spinner
            spinner_task = asyncio.create_task(
                self.progress.show_spinner(message)
            )
            
            # Run operation
            result = await operation(*args, **kwargs)
            
            # Stop spinner
            self.progress.stop_spinner()
            await spinner_task
            
            return result
            
        except Exception as e:
            self.progress.stop_spinner()
            await spinner_task
            raise e
    
    def show_progress_bar_operation(self, message: str, total: int, operation: Callable, *args, **kwargs):
        """Show progress bar for an operation."""
        async def progress_wrapper():
            for i in range(total):
                self.progress.show_progress_bar(i + 1, total, message)
                await operation(*args, **kwargs)
                await asyncio.sleep(0.1)  # Simulate work
        
        return progress_wrapper()


class InputValidator:
    """Validates user input."""
    
    @staticmethod
    def validate_task_description(task: str) -> bool:
        """Validate task description."""
        if not task or len(task.strip()) < 3:
            return False
        return True
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Sanitize user input."""
        # Remove potentially dangerous characters
        dangerous_chars = [';', '|', '&', '>', '<', '`', '$', '(', ')']
        sanitized = input_str
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        return sanitized.strip()
    
    @staticmethod
    def validate_config_path(path: str) -> bool:
        """Validate configuration file path."""
        import os
        return os.path.exists(path) and os.path.isfile(path)


class ColorOutput:
    """Provides colored output for better UX."""
    
    # ANSI color codes
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'reset': '\033[0m'
    }
    
    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Add color to text."""
        if color in cls.COLORS:
            return f"{cls.COLORS[color]}{text}{cls.COLORS['reset']}"
        return text
    
    @classmethod
    def success(cls, text: str) -> str:
        """Format success message."""
        return cls.colorize(text, 'green')
    
    @classmethod
    def error(cls, text: str) -> str:
        """Format error message."""
        return cls.colorize(text, 'red')
    
    @classmethod
    def warning(cls, text: str) -> str:
        """Format warning message."""
        return cls.colorize(text, 'yellow')
    
    @classmethod
    def info(cls, text: str) -> str:
        """Format info message."""
        return cls.colorize(text, 'blue')
    
    @classmethod
    def highlight(cls, text: str) -> str:
        """Format highlighted text."""
        return cls.colorize(text, 'bold') 