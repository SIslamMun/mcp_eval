"""
Command-line interface for MCP evaluation infrastructure.
"""

import json
import sys
import signal
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from rich.progress import track

from . import EvaluationEngine, MarkdownPromptLoader, SessionManager
from .evaluation_engine import EvaluationConfig


console = Console()
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    console.print("\n[yellow]⚠️  Shutdown requested. Stopping evaluation...[/yellow]")
    console.print("[dim]Press Ctrl+C again to force quit[/dim]")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@click.group()
@click.version_option()
def main():
    """MCP Evaluation Infrastructure - Evaluate Claude Code and OpenCode agents."""
    pass


@main.command()
@click.option("--config", "-c", help="Configuration file path")
@click.option("--prompts-dir", "-p", default="prompts", help="Prompts directory")
@click.option("--db-path", "-d", default="evaluation_sessions.db", help="Database path")
def setup(config: Optional[str], prompts_dir: str, db_path: str):
    """Initialize evaluation infrastructure and validate setup."""
    console.print("[bold blue]MCP Evaluation Infrastructure Setup[/bold blue]\n")
    
    # Create configuration if not provided
    if config:
        engine = EvaluationEngine(config_path=config)
    else:
        engine_config = EvaluationConfig(
            prompts_dir=prompts_dir,
            db_path=db_path
        )
        engine = EvaluationEngine(config=engine_config)
    
    # Create prompts directory
    prompts_path = Path(prompts_dir)
    if not prompts_path.exists():
        prompts_path.mkdir(parents=True)
        console.print(f"✅ Created prompts directory: {prompts_dir}")
        
        # Create sample prompt
        loader = MarkdownPromptLoader(prompts_dir)
        sample_file = loader.create_sample_prompt(1)
        console.print(f"✅ Created sample prompt: {sample_file}")
    
    # Validate setup
    validation = engine.validate_setup()
    
    if validation["setup_valid"]:
        console.print("✅ [bold green]Setup validation passed![/bold green]")
    else:
        console.print("❌ [bold red]Setup validation failed![/bold red]")
        for issue in validation["issues"]:
            console.print(f"  • [red]{issue}[/red]")
    
    if validation["warnings"]:
        console.print("\n⚠️  [bold yellow]Warnings:[/bold yellow]")
        for warning in validation["warnings"]:
            console.print(f"  • [yellow]{warning}[/yellow]")
    
    if validation["recommendations"]:
        console.print("\n💡 [bold cyan]Recommendations:[/bold cyan]")
        for rec in validation["recommendations"]:
            console.print(f"  • [cyan]{rec}[/cyan]")
    
    # Display statistics
    stats = engine.get_evaluation_statistics()
    console.print("\n📊 [bold]Current Statistics:[/bold]")
    console.print(f"  • Total prompts: {stats['total_prompts']}")
    console.print(f"  • Total sessions: {stats['total_sessions']}")
    console.print(f"  • Database size: {stats['database_size_mb']:.2f} MB")


@main.command()
@click.option("--agent", "-a", type=click.Choice(["claude", "opencode", "both"]), default="both", help="Agent to check models for")
@click.option("--preference", "-p", type=click.Choice(["fast", "accurate", "cheap"]), help="Model preference for suggestions")
def models(agent: str, preference: Optional[str]):
    """List available models for agents and get suggestions."""
    from .unified_agent import UnifiedAgent
    
    console.print("[bold blue]Available Models for MCP Evaluation[/bold blue]\n")
    
    if agent in ["claude", "both"]:
        console.print("[bold blue]Claude Models:[/bold blue]")
        claude_agent = UnifiedAgent("claude")
        claude_models = claude_agent.get_supported_models()
        claude_suggestions = claude_agent.suggest_models("claude", preference)
        
        claude_table = Table(show_header=True, header_style="bold blue")
        claude_table.add_column("Model", style="cyan")
        claude_table.add_column("Status", justify="center")
        claude_table.add_column("Recommended", justify="center")
        
        for model in claude_models:
            status = "✅ Available"
            recommended = "⭐" if model in claude_suggestions[:3] else ""
            claude_table.add_row(model, status, recommended)
        
        console.print(claude_table)
        
        if preference:
            console.print(f"[dim]💡 Suggested models for '{preference}' preference: {', '.join(claude_suggestions[:3])}[/dim]")
        console.print()
    
    if agent in ["opencode", "both"]:
        console.print("[bold green]OpenCode Models:[/bold green]")
        
        with console.status("[bold green]Detecting available OpenCode models..."):
            try:
                opencode_agent = UnifiedAgent("opencode")
                opencode_models = opencode_agent.get_supported_models()
                opencode_suggestions = opencode_agent.suggest_models("opencode", preference)
                
                opencode_table = Table(show_header=True, header_style="bold green")
                opencode_table.add_column("Model", style="cyan")
                opencode_table.add_column("Provider", style="yellow")
                opencode_table.add_column("Status", justify="center")
                opencode_table.add_column("Recommended", justify="center")
                
                for model in opencode_models:
                    provider = "GitHub Copilot" if model.startswith("github-copilot/") else "OpenCode"
                    status = "✅ Available"
                    recommended = "⭐" if model in opencode_suggestions[:3] else ""
                    
                    # Shorten model name for display
                    display_model = model.replace("github-copilot/", "")
                    opencode_table.add_row(display_model, provider, status, recommended)
                
                console.print(opencode_table)
                
                if preference:
                    suggested_display = [m.replace("github-copilot/", "") for m in opencode_suggestions[:3]]
                    console.print(f"[dim]💡 Suggested models for '{preference}' preference: {', '.join(suggested_display)}[/dim]")
                
            except Exception as e:
                console.print(f"[red]❌ Could not detect OpenCode models: {e}[/red]")
                console.print("[dim]Make sure OpenCode is installed and accessible[/dim]")
        
        console.print()
    
    # Usage examples
    console.print("[bold yellow]Usage Examples:[/bold yellow]")
    console.print("# Run with specific models:")
    if agent in ["claude", "both"]:
        console.print("  uv run python -m mcp_evaluation run 1 --agent claude --claude-model haiku")
    if agent in ["opencode", "both"]:
        console.print("  uv run python -m mcp_evaluation run 1 --agent opencode --opencode-model gpt-4o")
    
    console.print("\n# Multi-model comparison:")
    if agent in ["claude", "both"]:
        console.print("  uv run python -m mcp_evaluation run 1 --claude-models 'sonnet,haiku,opus'")
    if agent in ["opencode", "both"]:
        console.print("  uv run python -m mcp_evaluation run 1 --opencode-models 'claude-3.5-sonnet,gpt-4o'")


@main.command()
def cleanup():
    """Clean up any stuck evaluation processes."""
    import subprocess
    
    console.print("[bold yellow]🧹 Cleaning up stuck evaluation processes...[/bold yellow]\n")
    
    processes_to_kill = [
        "mcp_evaluation",
        "opencode run",
        "claude --print"
    ]
    
    killed_any = False
    for process_pattern in processes_to_kill:
        try:
            result = subprocess.run(
                ["pkill", "-f", process_pattern],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                console.print(f"✅ Killed processes matching: {process_pattern}")
                killed_any = True
        except Exception as e:
            console.print(f"[dim]Could not kill {process_pattern}: {e}[/dim]")
    
    if not killed_any:
        console.print("✅ No stuck processes found")
    else:
        console.print("\n[green]✅ Cleanup completed[/green]")
        console.print("[dim]You can now run new evaluations safely[/dim]")


@main.command()
@click.option("--agent", "-a", type=click.Choice(["claude", "opencode"]), default="opencode", help="Agent to test")
@click.option("--skip-permissions", is_flag=True, help="Skip Claude permissions (for sandboxes only)")
def test(agent: str, skip_permissions: bool):
    """Quick test of agent functionality."""
    from .unified_agent import UnifiedAgent
    import subprocess
    import time
    
    console.print(f"[bold blue]🧪 Testing {agent} agent[/bold blue]\n")
    
    try:
        # Test model detection first
        if agent == "opencode":
            with console.status("Detecting OpenCode models..."):
                test_agent = UnifiedAgent("opencode")
                models = test_agent.get_supported_models()
                if models:
                    console.print(f"✅ Detected {len(models)} models")
                    suggested_model = models[0]
                else:
                    console.print("❌ No models detected")
                    return
        else:
            suggested_model = "sonnet"
            console.print("✅ Claude models: sonnet, haiku, opus")
        
        # Test simple prompt execution
        console.print(f"🚀 Testing {agent} with model: {suggested_model}")
        
        start_time = time.time()
        
        # Run test without timeout
        cmd = [
            "uv", "run", "python", "-m", "mcp_evaluation", 
            "run", "1", "--agent", agent,
            f"--{agent}-model", suggested_model
        ]
        
        # Add skip permissions if requested
        if skip_permissions:
            cmd.append("--skip-permissions")
        
        with console.status("Running test evaluation..."):
            result = subprocess.run(cmd, capture_output=True, text=True)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            console.print(f"✅ Test passed in {elapsed:.1f}s")
        else:
            console.print(f"❌ Test failed (exit code: {result.returncode})")
            if result.stderr:
                console.print(f"[dim]Error: {result.stderr}[/dim]")
        
    except Exception as e:
        console.print(f"❌ Test failed with exception: {e}")


@main.command()
@click.argument("prompt_id", type=int)
@click.option("--agent", "-a", type=click.Choice(["claude", "opencode", "both"]), default="both", help="Agent to use")
@click.option("--claude-model", default="sonnet", help="Claude model")
@click.option("--opencode-model", default="github-copilot/claude-3.5-sonnet", help="OpenCode model")
@click.option("--claude-models", help="Multiple Claude models (comma-separated): sonnet,haiku,opus")
@click.option("--opencode-models", help="Multiple OpenCode models (comma-separated): github-copilot/claude-3.5-sonnet,mistral:latest")
@click.option("--config", "-c", help="Configuration file path")
@click.option("--continue-session", is_flag=True, help="Continue previous session")
@click.option("--skip-permissions", is_flag=True, help="Skip Claude permissions (for sandboxes only)")
def run(
    prompt_id: int,
    agent: str,
    claude_model: str,
    opencode_model: str,
    claude_models: Optional[str],
    opencode_models: Optional[str],
    config: Optional[str],
    continue_session: bool,
    skip_permissions: bool
):
    """Execute evaluation for a specific prompt."""
    console.print(f"[bold blue]Running evaluation for prompt {prompt_id}[/bold blue]\n")
    
    # Initialize engine
    if config:
        engine = EvaluationEngine(config_path=config)
    else:
        # Create engine with in-memory storage
        engine = EvaluationEngine()

    # Validate models and provide suggestions
    from .unified_agent import UnifiedAgent
    
    def validate_and_suggest_models(agent_type: str, model_list: List[str]) -> List[str]:
        """Validate models and suggest alternatives if needed."""
        try:
            agent = UnifiedAgent(agent_type)
            supported = agent.get_supported_models()
            valid_models = []
            invalid_models = []
            
            for model in model_list:
                if agent.validate_model(model):
                    valid_models.append(model)
                else:
                    invalid_models.append(model)
            
            if invalid_models:
                console.print(f"[red]❌ Invalid {agent_type} models (skipping): {invalid_models}[/red]")
                suggestions = agent.suggest_models(agent_type)
                console.print(f"[dim]💡 Suggested {agent_type} models: {suggestions[:3]}[/dim]")
                console.print(f"[dim]   Use 'uv run python -m mcp_evaluation models --agent {agent_type}' for full list[/dim]")
                
            if not valid_models:
                console.print(f"[red]❌ No valid {agent_type} models provided. Using default.[/red]")
                default_model = agent.capabilities.default_model
                if default_model and agent.validate_model(default_model):
                    return [default_model]
                else:
                    suggestions = agent.suggest_models(agent_type)
                    return suggestions[:1] if suggestions else []
            
            return valid_models
        except Exception as e:
            console.print(f"[dim]Could not validate {agent_type} models: {e}[/dim]")
            return model_list

    try:
        # Check for multi-model instances
        if claude_models or opencode_models:
            # Multi-model instance evaluation
            console.print("[bold yellow]Multi-model instance evaluation detected[/bold yellow]\n")
            
            claude_model_list = []
            opencode_model_list = []
            
            if claude_models:
                claude_model_list = [m.strip() for m in claude_models.split(",")]
                claude_model_list = validate_and_suggest_models("claude", claude_model_list)
                if claude_model_list:
                    console.print(f"✅ Claude models to process: {claude_model_list}")
                else:
                    console.print("[red]❌ No valid Claude models found[/red]")
            elif agent in ["claude", "both"]:
                claude_model_list = validate_and_suggest_models("claude", [claude_model])
                
            if opencode_models:
                opencode_model_list = [m.strip() for m in opencode_models.split(",")]
                opencode_model_list = validate_and_suggest_models("opencode", opencode_model_list)
                if opencode_model_list:
                    console.print(f"✅ OpenCode models to process: {opencode_model_list}")
                else:
                    console.print("[red]❌ No valid OpenCode models found[/red]")
            elif agent in ["opencode", "both"]:
                opencode_model_list = validate_and_suggest_models("opencode", [opencode_model])
            
            # Check if we have any valid models to run
            if agent in ["claude", "both"] and not claude_model_list:
                console.print("[red]❌ No valid Claude models to run[/red]")
                return
            if agent in ["opencode", "both"] and not opencode_model_list:
                console.print("[red]❌ No valid OpenCode models to run[/red]")
                return
                
            result = engine.execute_multi_model_evaluation(
                prompt_id=prompt_id,
                claude_models=claude_model_list if agent in ["claude", "both"] else [],
                opencode_models=opencode_model_list if agent in ["opencode", "both"] else [],
                continue_session=continue_session,
                skip_permissions=skip_permissions
            )
            
            # Display multi-model results
            display_multi_model_result(result)
            
        elif agent == "both":
            # Comparative evaluation - validate both models
            validated_claude = validate_and_suggest_models("claude", [claude_model])
            validated_opencode = validate_and_suggest_models("opencode", [opencode_model])
            
            claude_config = {
                "type": "claude",
                "model": validated_claude[0] if validated_claude else claude_model,
                "continue_session": continue_session,
                "dangerously_skip_permissions": skip_permissions
            }
            
            opencode_config = {
                "type": "opencode", 
                "model": validated_opencode[0] if validated_opencode else opencode_model,
                "continue_session": continue_session,
                "enable_logs": True
            }
            
            result = engine.execute_comparative_evaluation(
                prompt_id=prompt_id,
                claude_config=claude_config,
                opencode_config=opencode_config
            )
            
            # Display results
            display_comparative_result(result)
            
        else:
            # Single agent evaluation - validate models
            selected_model = claude_model if agent == "claude" else opencode_model
            validated_models = validate_and_suggest_models(agent, [selected_model])
            validated_model = validated_models[0] if validated_models else selected_model
            
            agent_config = {
                "type": agent,
                "model": validated_model,
                "continue_session": continue_session
            }
            
            if agent == "claude":
                agent_config["output_format"] = "json"
                agent_config["dangerously_skip_permissions"] = skip_permissions
            else:
                agent_config["enable_logs"] = True
            
            result = engine.execute_evaluation(
                prompt_id=prompt_id,
                agent_config=agent_config
            )
            
            # Display result
            display_single_result(result)
            
    except FileNotFoundError as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [red]Unexpected error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("prompt_ids", nargs=-1, type=int, required=True)
@click.option("--agent", "-a", type=click.Choice(["claude", "opencode", "both"]), default="both", help="Agent to use")
@click.option("--claude-model", default="sonnet", help="Claude model")
@click.option("--opencode-model", default="github-copilot/claude-3.5-sonnet", help="OpenCode model")
@click.option("--config", "-c", help="Configuration file path")
@click.option("--continue-sessions", is_flag=True, help="Continue sessions between prompts")
@click.option("--skip-permissions", is_flag=True, help="Skip Claude permissions (for sandboxes only)")
def batch(
    prompt_ids: List[int],
    agent: str,
    claude_model: str,
    opencode_model: str,
    config: Optional[str],
    continue_sessions: bool,
    skip_permissions: bool
):
    """Execute batch evaluation for multiple prompts."""
    console.print(f"[bold blue]Running batch evaluation for prompts: {', '.join(map(str, prompt_ids))}[/bold blue]\n")
    
    # Initialize engine
    if config:
        engine = EvaluationEngine(config_path=config)
    else:
        engine = EvaluationEngine()
    
    if agent == "both":
        # Batch comparative evaluation
        claude_config = {
            "type": "claude",
            "model": claude_model,
            "output_format": "json",
            "dangerously_skip_permissions": skip_permissions
        }
        
        opencode_config = {
            "type": "opencode",
            "model": opencode_model,
            "enable_logs": True
        }
        
        results = engine.execute_batch_comparative_evaluation(
            prompt_ids=prompt_ids,
            claude_config=claude_config,
            opencode_config=opencode_config
        )
        
        # Display results
        for i, result in enumerate(track(results, description="Processing results...")):
            console.print(f"\n[bold]Result {i + 1}/{len(results)}:[/bold]")
            display_comparative_result(result)
            
    else:
        # Single agent batch evaluation
        agent_config = {
            "type": agent,
            "model": claude_model if agent == "claude" else opencode_model
        }
        
        if agent == "claude":
            agent_config["output_format"] = "json"
            agent_config["dangerously_skip_permissions"] = skip_permissions
        else:
            agent_config["enable_logs"] = True
        
        results = engine.execute_batch_evaluation(
            prompt_ids=prompt_ids,
            agent_config=agent_config,
            continue_sessions=continue_sessions
        )
        
        # Display results
        for i, result in enumerate(track(results, description="Processing results...")):
            console.print(f"\n[bold]Result {i + 1}/{len(results)}:[/bold]")
            display_single_result(result)


@main.command()
@click.option("--agent", "-a", type=click.Choice(["claude", "opencode", "both"]), default="both", help="Agent to use")
@click.option("--claude-model", default="sonnet", help="Claude model")
@click.option("--opencode-model", default="github-copilot/claude-3.5-sonnet", help="OpenCode model")
@click.option("--config", "-c", help="Configuration file path")
@click.option("--continue-sessions", is_flag=True, help="Continue sessions between prompts")
@click.option("--skip-permissions", is_flag=True, help="Skip Claude permissions (for sandboxes only)")
@click.option("--complexity", type=click.Choice(["low", "medium", "high"]), help="Filter by complexity level")
@click.option("--mcp-target", help="Filter by MCP target server")
def run_all(
    agent: str,
    claude_model: str,
    opencode_model: str,
    config: Optional[str],
    backend: str,
    continue_sessions: bool,
    skip_permissions: bool,
    complexity: Optional[str],
    mcp_target: Optional[str]
):
    """Run all available prompts automatically."""
    console.print("[bold blue]Running all available prompts[/bold blue]\n")
    
    # Initialize engine
    if config:
        engine = EvaluationEngine(config_path=config)
    else:
        # Create engine with backend override
        from .evaluation_engine import EvaluationConfig
        config_obj = EvaluationConfig(backend=backend)
        engine = EvaluationEngine(config=config_obj)
    
    # Load all prompts
    try:
        loader = engine.prompt_loader
        all_prompts = loader.load_all_prompts()
        
        if not all_prompts:
            console.print("[red]No prompts found in prompts directory[/red]")
            sys.exit(1)
        
        # Apply filters
        filtered_prompts = list(all_prompts.values())
        
        if complexity:
            filtered_prompts = [p for p in filtered_prompts if p.metadata.complexity == complexity]
            console.print(f"[cyan]Filtering by complexity: {complexity}[/cyan]")
        
        if mcp_target:
            filtered_prompts = [p for p in filtered_prompts if mcp_target in p.metadata.target_mcp]
            console.print(f"[cyan]Filtering by MCP target: {mcp_target}[/cyan]")
        
        if not filtered_prompts:
            console.print("[yellow]No prompts match the specified filters[/yellow]")
            sys.exit(0)
        
        # Sort by prompt ID
        prompt_ids = sorted([p.metadata.id for p in filtered_prompts])
        
        console.print(f"[green]Found {len(prompt_ids)} prompts to run: {prompt_ids}[/green]\n")
        
        # Run batch evaluation
        if agent == "both":
            claude_config = {
                "type": "claude",
                "model": claude_model,
                "output_format": "json",
                "dangerously_skip_permissions": skip_permissions
            }
            
            opencode_config = {
                "type": "opencode",
                "model": opencode_model,
                "enable_logs": True
            }
            
            results = engine.execute_batch_comparative_evaluation(
                prompt_ids=prompt_ids,
                claude_config=claude_config,
                opencode_config=opencode_config
            )
            
            # Display results
            for i, result in enumerate(track(results, description="Processing results...")):
                console.print(f"\n[bold]Result {i + 1}/{len(results)}:[/bold]")
                display_comparative_result(result)
                
        else:
            agent_config = {
                "type": agent,
                "model": claude_model if agent == "claude" else opencode_model
            }
            
            if agent == "claude":
                agent_config["output_format"] = "json"
                agent_config["dangerously_skip_permissions"] = skip_permissions
            else:
                agent_config["enable_logs"] = True
            
            results = engine.execute_batch_evaluation(
                prompt_ids=prompt_ids,
                agent_config=agent_config,
                continue_sessions=continue_sessions
            )
            
            # Display results
            for i, result in enumerate(track(results, description="Processing results...")):
                console.print(f"\n[bold]Result {i + 1}/{len(results)}:[/bold]")
                display_single_result(result)
        
        # Show summary
        console.print(f"\n[bold green]✅ Completed evaluation of {len(prompt_ids)} prompts[/bold green]")
        console.print("Run 'mcp-eval stats' to view detailed statistics")
        
    except Exception as e:
        console.print(f"❌ [red]Error running all prompts: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--config", "-c", help="Configuration file path")
def stats(config: Optional[str]):
    """Display evaluation statistics from InfluxDB data."""
    console.print("[bold blue]Evaluation Statistics[/bold blue]\n")
    
    # Initialize prompt loader for prompt data
    if config:
        engine = EvaluationEngine(config_path=config)
    else:
        engine = EvaluationEngine()
        
    # Get prompt information from engine
    prompt_data_dict = engine.prompt_loader.load_all_prompts()
    prompt_data = list(prompt_data_dict.values())
    
    try:
        # Use PostProcessor to get real statistics from InfluxDB
        from .post_processor import PostProcessor
        
        processor = PostProcessor()
        
        # Extract sessions without processing them
        claude_sessions = processor.extract_claude_sessions()
        opencode_sessions = processor._extract_opencode_sessions()
        all_sessions = claude_sessions + opencode_sessions
        
        # Calculate statistics from real InfluxDB data
        total_sessions = len(all_sessions)
        claude_count = len(claude_sessions)
        opencode_count = len(opencode_sessions)
        
        # Get detailed session metrics for success rate calculation
        from pathlib import Path
        import json
        reports_dir = Path("reports")
        
        # Calculate success rates by reading existing evaluation_metrics.json files
        claude_success_count = 0
        opencode_success_count = 0
        claude_session_details = []
        opencode_session_details = []
        
        # Scan Claude reports
        claude_reports_dir = reports_dir / "claude"
        if claude_reports_dir.exists():
            for session_dir in claude_reports_dir.iterdir():
                if session_dir.is_dir():
                    metrics_file = session_dir / "evaluation_metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics_data = json.load(f)
                                session_metrics = metrics_data.get('session_metrics', {})
                                
                                session_details = {
                                    'session_id': session_metrics.get('session_id', session_dir.name),
                                    'success': session_metrics.get('success', False),
                                    'created_at': session_metrics.get('created_at', 'Unknown'),
                                    'completed_at': session_metrics.get('completed_at', 'Unknown'),
                                    'execution_time': session_metrics.get('execution_time', 0),
                                    'prompt': session_metrics.get('prompt', 'Unknown'),
                                    'model': session_metrics.get('model', 'Unknown')
                                }
                                claude_session_details.append(session_details)
                                
                                if session_metrics.get('success', False):
                                    claude_success_count += 1
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not read {metrics_file}: {e}[/yellow]")
        
        # Scan OpenCode reports
        opencode_reports_dir = reports_dir / "opencode"
        if opencode_reports_dir.exists():
            for session_dir in opencode_reports_dir.iterdir():
                if session_dir.is_dir():
                    metrics_file = session_dir / "evaluation_metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics_data = json.load(f)
                                session_metrics = metrics_data.get('session_metrics', {})
                                
                                session_details = {
                                    'session_id': session_metrics.get('session_id', session_dir.name),
                                    'success': session_metrics.get('success', False),
                                    'created_at': session_metrics.get('created_at', 'Unknown'),
                                    'completed_at': session_metrics.get('completed_at', 'Unknown'),
                                    'execution_time': session_metrics.get('execution_time', 0),
                                    'prompt': session_metrics.get('prompt', 'Unknown'),
                                    'model': session_metrics.get('model', 'Unknown')
                                }
                                opencode_session_details.append(session_details)
                                
                                if session_metrics.get('success', False):
                                    opencode_success_count += 1
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not read {metrics_file}: {e}[/yellow]")
        
        # Calculate session percentages (distribution between agents)
        total_evaluated_sessions = len(claude_session_details) + len(opencode_session_details)
        claude_session_percentage = (len(claude_session_details) / total_evaluated_sessions * 100) if total_evaluated_sessions > 0 else 0
        opencode_session_percentage = (len(opencode_session_details) / total_evaluated_sessions * 100) if total_evaluated_sessions > 0 else 0
        
        # Calculate success rates for reference
        claude_success_rate = (claude_success_count / len(claude_session_details) * 100) if claude_session_details else 0
        opencode_success_rate = (opencode_success_count / len(opencode_session_details) * 100) if opencode_session_details else 0
        overall_success_rate = ((claude_success_count + opencode_success_count) / total_evaluated_sessions * 100) if total_evaluated_sessions > 0 else 0
        
        # Calculate prompt distribution across all sessions
        prompt_distribution = {}
        all_session_details = claude_session_details + opencode_session_details
        for session in all_session_details:
            prompt_id = session['prompt']
            if prompt_id not in prompt_distribution:
                prompt_distribution[prompt_id] = {'total': 0, 'claude': 0, 'opencode': 0}
            prompt_distribution[prompt_id]['total'] += 1
            
        # Add agent-specific counts
        for session in claude_session_details:
            prompt_id = session['prompt']
            prompt_distribution[prompt_id]['claude'] += 1
            
        for session in opencode_session_details:
            prompt_id = session['prompt']
            prompt_distribution[prompt_id]['opencode'] += 1
        
        # Recent sessions (last 24 hours)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        recent_sessions = 0
        total_cost = 0.0
        costs = []
        
        # Agent distribution
        agent_distribution = {'claude': claude_count, 'opencode': opencode_count}
        
        # Create statistics table
        table = Table(title="Session Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Sessions", str(total_sessions))
        table.add_row("Claude Sessions", str(claude_count))
        table.add_row("OpenCode Sessions", str(opencode_count))
        table.add_row("Total Prompts", str(len(prompt_data)))
        table.add_row("Database Type", "InfluxDB")
        table.add_row("Claude Session %", f"{claude_session_percentage:.1f}% ({len(claude_session_details)} sessions)")
        table.add_row("OpenCode Session %", f"{opencode_session_percentage:.1f}% ({len(opencode_session_details)} sessions)")
        table.add_row("Claude Success Rate", f"{claude_success_rate:.1f}% ({claude_success_count}/{len(claude_session_details)})" if claude_session_details else "0% (0/0)")
        table.add_row("OpenCode Success Rate", f"{opencode_success_rate:.1f}% ({opencode_success_count}/{len(opencode_session_details)})" if opencode_session_details else "0% (0/0)")
        
        console.print(table)
        
        # Agent distribution
        console.print("\n[bold]Agent Distribution:[/bold]")
        for agent, count in agent_distribution.items():
            console.print(f"  • {agent}: {count} sessions")
        
        # Session Details by Agent
        if claude_session_details:
            console.print("\n[bold]Claude Session Details:[/bold]")
            claude_table = Table()
            claude_table.add_column("Session ID", style="cyan", max_width=30)
            claude_table.add_column("Success", style="green")
            claude_table.add_column("Prompt", style="yellow")
            claude_table.add_column("Model", style="blue")
            claude_table.add_column("Duration", style="magenta")
            claude_table.add_column("Created At", style="dim")
            
            for session in sorted(claude_session_details, key=lambda x: x['created_at'], reverse=True):
                success_icon = "✅" if session['success'] else "❌"
                duration_str = f"{session['execution_time']:.1f}s" if session['execution_time'] else "N/A"
                # Truncate session ID for display
                session_id_short = session['session_id'][:20] + "..." if len(session['session_id']) > 23 else session['session_id']
                created_at_short = session['created_at'][:19] if session['created_at'] != 'Unknown' else 'Unknown'
                
                claude_table.add_row(
                    session_id_short,
                    success_icon,
                    str(session['prompt']),
                    session['model'],
                    duration_str,
                    created_at_short
                )
            console.print(claude_table)
        
        if opencode_session_details:
            console.print("\n[bold]OpenCode Session Details:[/bold]")
            opencode_table = Table()
            opencode_table.add_column("Session ID", style="cyan", max_width=30)
            opencode_table.add_column("Success", style="green")
            opencode_table.add_column("Prompt", style="yellow")
            opencode_table.add_column("Model", style="blue")
            opencode_table.add_column("Duration", style="magenta")
            opencode_table.add_column("Created At", style="dim")
            
            for session in sorted(opencode_session_details, key=lambda x: x['created_at'], reverse=True):
                success_icon = "✅" if session['success'] else "❌"
                duration_str = f"{session['execution_time']:.1f}s" if session['execution_time'] else "N/A"
                # Truncate session ID for display
                session_id_short = session['session_id'][:20] + "..." if len(session['session_id']) > 23 else session['session_id']
                created_at_short = session['created_at'][:19] if session['created_at'] != 'Unknown' else 'Unknown'
                
                opencode_table.add_row(
                    session_id_short,
                    success_icon,
                    str(session['prompt']),
                    session['model'],
                    duration_str,
                    created_at_short
                )
            console.print(opencode_table)
        
        # Prompt Distribution
        if prompt_distribution:
            console.print("\n[bold]Prompt Distribution Across Sessions:[/bold]")
            prompt_table = Table()
            prompt_table.add_column("Prompt ID", style="cyan")
            prompt_table.add_column("Total Sessions", style="magenta")
            prompt_table.add_column("Percentage", style="green")
            prompt_table.add_column("Claude", style="blue")
            prompt_table.add_column("OpenCode", style="yellow")
            
            for prompt_id, counts in sorted(prompt_distribution.items()):
                percentage = (counts['total'] / total_evaluated_sessions * 100) if total_evaluated_sessions > 0 else 0
                prompt_table.add_row(
                    str(prompt_id),
                    str(counts['total']),
                    f"{percentage:.1f}%",
                    str(counts['claude']),
                    str(counts['opencode'])
                )
            console.print(prompt_table)
        
        # Prompt Statistics
        if prompt_data:
            console.print("\n[bold]Prompt Statistics:[/bold]")
            
            # Complexity distribution
            complexity_stats = {}
            category_stats = {}
            timeout_stats = {'total': 0, 'avg': 0, 'min': float('inf'), 'max': 0}
            tag_stats = {}
            
            for prompt in prompt_data:
                # Access metadata attributes correctly
                metadata = getattr(prompt, 'metadata', None)
                if metadata:
                    # Complexity distribution
                    complexity = getattr(metadata, 'complexity', 'unknown')
                    complexity_stats[complexity] = complexity_stats.get(complexity, 0) + 1
                    
                    # Category distribution
                    category = getattr(metadata, 'category', 'unknown')
                    category_stats[category] = category_stats.get(category, 0) + 1
                    
                    # Timeout statistics
                    timeout = getattr(metadata, 'timeout', 0)
                    if timeout and timeout > 0:
                        timeout_stats['total'] += timeout
                        timeout_stats['min'] = min(timeout_stats['min'], timeout)
                        timeout_stats['max'] = max(timeout_stats['max'], timeout)
                    
                    # Tag statistics
                    tags = getattr(metadata, 'tags', [])
                    if tags:
                        for tag in tags:
                            tag_stats[tag] = tag_stats.get(tag, 0) + 1
            
            # Calculate average timeout
            if len(prompt_data) > 0:
                timeout_stats['avg'] = timeout_stats['total'] / len(prompt_data)
                if timeout_stats['min'] == float('inf'):
                    timeout_stats['min'] = 0
            
            # Display complexity distribution
            if complexity_stats:
                console.print(f"\n[bold cyan]Complexity Distribution:[/bold cyan]")
                for complexity, count in sorted(complexity_stats.items()):
                    percentage = (count / len(prompt_data) * 100)
                    console.print(f"  • {complexity}: {count} prompts ({percentage:.1f}%)")
            
            # Display category distribution
            if category_stats:
                console.print(f"\n[bold cyan]Category Distribution:[/bold cyan]")
                for category, count in sorted(category_stats.items()):
                    percentage = (count / len(prompt_data) * 100)
                    console.print(f"  • {category}: {count} prompts ({percentage:.1f}%)")
            
            # Display timeout statistics
            if timeout_stats['total'] > 0:
                console.print(f"\n[bold cyan]Timeout Statistics:[/bold cyan]")
                console.print(f"  • Average: {timeout_stats['avg']:.1f} seconds")
                console.print(f"  • Range: {timeout_stats['min']}-{timeout_stats['max']} seconds")
            
            # Display top tags
            if tag_stats:
                console.print(f"\n[bold cyan]Most Common Tags:[/bold cyan]")
                sorted_tags = sorted(tag_stats.items(), key=lambda x: x[1], reverse=True)
                for tag, count in sorted_tags[:5]:  # Show top 5 tags
                    console.print(f"  • {tag}: {count} prompts")
        
        # Prompt complexity distribution (old code for backward compatibility)
        complexity_distribution = {}
        for prompt in prompt_data:
            # prompt is a PromptData object, access its attributes directly
            metadata = getattr(prompt, 'metadata', None)
            if metadata:
                complexity = getattr(metadata, 'complexity', 'unknown')
                complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
        
        if complexity_distribution:
            # This section is now handled by the detailed prompt statistics above
            pass
        
        # MCP targets
        mcp_targets = set()
        for prompt in prompt_data:
            target = getattr(prompt, 'mcp_target', None)
            if target:
                mcp_targets.add(target)
        
        if mcp_targets:
            console.print(f"\n[bold]MCP Targets ({len(mcp_targets)}):[/bold]")
            for target in mcp_targets:
                console.print(f"  • {target}")
        
    except Exception as e:
        console.print(f"❌ [red]Error getting statistics: {e}[/red]")
        # Fallback to showing basic prompt info
        try:
            prompt_data_dict = engine.prompt_loader.load_all_prompts()
            prompt_data = list(prompt_data_dict.values())
            console.print(f"\n[yellow]Showing basic prompt information only:[/yellow]")
            console.print(f"Total Prompts: {len(prompt_data)}")
        except Exception as fallback_e:
            console.print(f"❌ [red]Error loading prompts: {fallback_e}[/red]")
        console.print(f"❌ [red]Failed to get statistics: {e}[/red]")
        sys.exit(1)


@main.command("post-processing")
@click.option("--output", "-o", default="reports/", help="Output directory (default: reports/)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress and results")
@click.option("--agent", "-a", type=click.Choice(['claude', 'opencode']), help="Filter by agent type (claude or opencode)")
def post_processing(
    output: str,
    verbose: bool,
    agent: Optional[str]
):
    """Process InfluxDB monitoring data and generate evaluation metrics.
    
    This command processes all evaluation sessions from InfluxDB and generates
    individual session reports with metrics in JSON format.
    """
    console.print("[bold blue]🔄 MCP Evaluation Post-Processing[/bold blue]\n")
    
    try:
        from .post_processor import PostProcessor
        
        # Initialize post processor
        processor = PostProcessor(output_dir=output)
        
        if verbose:
            console.print(f"[cyan]Output Directory:[/cyan] {output}")
            console.print(f"[cyan]InfluxDB URL:[/cyan] {processor.config['INFLUXDB_URL']}")
            console.print(f"[cyan]Database:[/cyan] {processor.config['INFLUXDB_BUCKET']}")
            if agent:
                console.print(f"[cyan]Agent Filter:[/cyan] {agent}")
            console.print()
        
        # Run the processing with agent filtering
        if agent:
            console.print(f"[bold]🚀 Processing InfluxDB monitoring data for {agent} sessions...[/bold]")
            if agent == 'claude':
                # Extract and process only Claude sessions
                claude_sessions = processor.extract_claude_sessions()
                console.print(f"📊 Found {len(claude_sessions)} Claude sessions to process")
                
                session_results = []
                for i, session in enumerate(claude_sessions, 1):
                    session_result = processor.process_single_session(session, i)
                    session_results.append(session_result)
                
                results = {"sessions": session_results, "total": len(claude_sessions)}
                
            elif agent == 'opencode':
                # Extract and process only OpenCode sessions
                opencode_sessions = processor._extract_opencode_sessions()
                console.print(f"📊 Found {len(opencode_sessions)} OpenCode sessions to process")
                
                session_results = []
                for i, session in enumerate(opencode_sessions, 1):
                    session_result = processor.process_single_session(session, i)
                    session_results.append(session_result)
                
                results = {"sessions": session_results, "total": len(opencode_sessions)}
        else:
            console.print("[bold]🚀 Processing all InfluxDB monitoring data...[/bold]")
            results = processor.process_all()
        
        # Close the connection
        processor.close()
        
        console.print(f"\n[bold green]✅ Post-processing completed successfully![/bold green]")
        console.print(f"[cyan]Reports generated in:[/cyan] {output}")
        
    except Exception as e:
        console.print(f"[red]❌ Post-processing failed: {e}[/red]")
        logger.exception("Post-processing failed")
        sys.exit(1)


# Helper functions
def display_single_result(result, show_session: bool = True):
    """Display a single evaluation result."""
    status_color = "green" if result.success else "red"
    status_icon = "✅" if result.success else "❌"
    
    panel_title = f"{status_icon} {result.agent_type.title()} Evaluation"
    if show_session:
        panel_title += f" - Prompt {result.prompt_id}"
    
    content = []
    
    if show_session:
        content.append(f"[cyan]Session ID:[/cyan] {result.base_session_id}")
    
    content.append(f"[cyan]Success:[/cyan] [{status_color}]{result.success}[/{status_color}]")
    
    if result.response:
        # Show full response without truncation
        response = result.response
        content.append(f"[cyan]Response:[/cyan] {response}")
    
    if result.cost_usd > 0:
        content.append(f"[cyan]Cost:[/cyan] ${result.cost_usd:.4f}")
    
    if result.duration_ms > 0:
        duration_sec = result.duration_ms / 1000.0
        content.append(f"[cyan]Duration:[/cyan] {duration_sec:.1f}s")
    
    if result.error_message:
        content.append(f"[red]Error:[/red] {result.error_message}")
    
    console.print(Panel("\n".join(content), title=panel_title))


def display_comparative_result(result):
    """Display a comparative evaluation result."""
    console.print(f"[bold]Comparative Evaluation - Prompt {result.prompt_id}[/bold]")
    console.print(f"[cyan]Session ID:[/cyan] {result.base_session_id}")
    
    if result.claude_result:
        console.print("\n[bold blue]Claude Code Result:[/bold blue]")
        display_single_result(result.claude_result, show_session=False)
    
    if result.opencode_result:
        console.print("\n[bold green]OpenCode Result:[/bold green]")
        display_single_result(result.opencode_result, show_session=False)
    
    # Comparison summary
    if result.claude_result and result.opencode_result:
        console.print("\n[bold yellow]Comparison Summary:[/bold yellow]")
        
        claude_success = result.claude_result.success
        opencode_success = result.opencode_result.success
        
        if claude_success and opencode_success:
            console.print("✅ Both agents succeeded")
        elif claude_success:
            console.print("⚠️  Only Claude Code succeeded")
        elif opencode_success:
            console.print("⚠️  Only OpenCode succeeded")
        else:
            console.print("❌ Both agents failed")
        
        if result.claude_result.cost_usd > 0:
            console.print(f"💰 Claude Code cost: ${result.claude_result.cost_usd:.4f}")


def display_multi_model_result(result):
    """Display a multi-model evaluation result."""
    console.print(f"[bold]Multi-Model Evaluation - Prompt {result.prompt_id}[/bold]")
    console.print(f"[cyan]Base Session ID:[/cyan] {result.base_session_id}")
    console.print(f"[cyan]Total Cost:[/cyan] ${result.total_cost_usd:.4f}")
    console.print(f"[cyan]Total Duration:[/cyan] {result.total_duration_ms}ms")
    console.print()
    
    # Display Claude model results
    if result.claude_results:
        console.print("[bold blue]Claude Model Results:[/bold blue]")
        
        claude_table = Table(show_header=True, header_style="bold blue")
        claude_table.add_column("Model", style="cyan")
        claude_table.add_column("Status", justify="center")
        claude_table.add_column("Cost", justify="right")
        claude_table.add_column("Duration", justify="right")
        claude_table.add_column("Response Preview", style="dim")
        
        for model, model_result in result.claude_results.items():
            status = "✅ Success" if model_result.success else "❌ Failed"
            cost = f"${model_result.cost_usd:.4f}"
            duration = f"{model_result.duration_ms}ms"
            preview = model_result.response
            
            claude_table.add_row(model, status, cost, duration, preview)
        
        console.print(claude_table)
        console.print()
    
    # Display OpenCode model results
    if result.opencode_results:
        console.print("[bold green]OpenCode Model Results:[/bold green]")
        
        opencode_table = Table(show_header=True, header_style="bold green")
        opencode_table.add_column("Model", style="cyan")
        opencode_table.add_column("Status", justify="center")
        opencode_table.add_column("Cost", justify="right")
        opencode_table.add_column("Duration", justify="right")
        opencode_table.add_column("Response Preview", style="dim")
        
        for model, model_result in result.opencode_results.items():
            status = "✅ Success" if model_result.success else "❌ Failed"
            cost = f"${model_result.cost_usd:.4f}"
            duration = f"{model_result.duration_ms}ms"
            preview = model_result.response
            
            opencode_table.add_row(model, status, cost, duration, preview)
        
        console.print(opencode_table)
        console.print()
    
    # Summary comparison across models
    all_results = list(result.claude_results.values()) + list(result.opencode_results.values())
    successful_results = [r for r in all_results if r.success]
    
    if successful_results:
        console.print("[bold yellow]Model Performance Summary:[/bold yellow]")
        
        # Find fastest and cheapest
        fastest = min(successful_results, key=lambda x: x.duration_ms)
        cheapest = min(successful_results, key=lambda x: x.cost_usd) if any(r.cost_usd > 0 for r in successful_results) else None
        
        console.print(f"🏃 Fastest: {fastest.agent_type} ({_get_model_name(fastest, result)}) - {fastest.duration_ms}ms")
        if cheapest and cheapest.cost_usd > 0:
            console.print(f"💰 Cheapest: {cheapest.agent_type} ({_get_model_name(cheapest, result)}) - ${cheapest.cost_usd:.4f}")
        
        success_rate = len(successful_results) / len(all_results) * 100
        console.print(f"📊 Success Rate: {success_rate:.1f}% ({len(successful_results)}/{len(all_results)})")


def _get_model_name(result, multi_result):
    """Helper to get model name from result."""
    # Check Claude models
    for model, model_result in multi_result.claude_results.items():
        if model_result == result:
            return model
    
    # Check OpenCode models
    for model, model_result in multi_result.opencode_results.items():
        if model_result == result:
            return model
    
    return "unknown"



if __name__ == "__main__":
    main()
