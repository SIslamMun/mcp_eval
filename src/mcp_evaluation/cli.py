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
from .unified_post_processing import UnifiedPostProcessor


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
@click.option("--backend", "-b", type=click.Choice(["influxdb", "sqlite"]), default="influxdb", help="Database backend (default: influxdb)")
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
    backend: str,
    continue_session: bool,
    skip_permissions: bool
):
    """Execute evaluation for a specific prompt."""
    console.print(f"[bold blue]Running evaluation for prompt {prompt_id}[/bold blue]\n")
    
    # Initialize engine
    if config:
        engine = EvaluationEngine(config_path=config)
    else:
        # Create engine with backend override
        from .evaluation_engine import EvaluationConfig
        config_obj = EvaluationConfig(backend=backend)
        engine = EvaluationEngine(config=config_obj)

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
@click.option("--output", "-o", default="evaluation_results", help="Output file prefix")
@click.option("--format", "-f", type=click.Choice(["json", "csv"]), default="json", help="Output format")
@click.option("--config", "-c", help="Configuration file path")
def export(output: str, format: str, config: Optional[str]):
    """Export evaluation results."""
    console.print("[bold blue]Exporting evaluation results[/bold blue]\n")
    
    # Initialize engine
    if config:
        engine = EvaluationEngine(config_path=config)
    else:
        engine = EvaluationEngine()
    
    output_file = f"{output}.{format}"
    
    try:
        exported_file = engine.export_evaluation_results(output_file, format)
        console.print(f"✅ [green]Results exported to: {exported_file}[/green]")
        
        # Show statistics
        stats = engine.get_evaluation_statistics()
        console.print(f"📊 Exported {stats['total_sessions']} sessions")
        
    except Exception as e:
        console.print(f"❌ [red]Export failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--agent", "-a", type=click.Choice(["claude", "opencode", "both"]), default="both", help="Agent to use")
@click.option("--claude-model", default="sonnet", help="Claude model")
@click.option("--opencode-model", default="github-copilot/claude-3.5-sonnet", help="OpenCode model")
@click.option("--config", "-c", help="Configuration file path")
@click.option("--backend", "-b", type=click.Choice(["influxdb", "sqlite"]), default="influxdb", help="Database backend (default: influxdb)")
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
@click.option("--backend", "-b", type=click.Choice(["influxdb", "sqlite"]), default="influxdb", help="Database backend (default: influxdb)")
def stats(config: Optional[str], backend: str):
    """Display evaluation statistics."""
    console.print("[bold blue]Evaluation Statistics[/bold blue]\n")
    
    # Initialize engine
    if config:
        engine = EvaluationEngine(config_path=config)
    else:
        # Create engine with backend override
        from .evaluation_engine import EvaluationConfig
        config_obj = EvaluationConfig(backend=backend)
        engine = EvaluationEngine(config=config_obj)
        engine = EvaluationEngine()
    
    try:
        stats = engine.get_evaluation_statistics()
        
        # Create statistics table
        table = Table(title="Session Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Sessions", str(stats["total_sessions"]))
        table.add_row("Comparative Sessions", str(stats["total_comparative_sessions"]))
        table.add_row("Total Prompts", str(stats["total_prompts"]))
        table.add_row("Database Size (MB)", f"{stats['database_size_mb']:.2f}")
        table.add_row("Recent Sessions (24h)", str(stats["recent_sessions_24h"]))
        table.add_row("Average Cost (USD)", f"${stats['average_cost_usd']:.4f}")
        
        console.print(table)
        
        # Agent distribution
        if stats.get("agent_distribution"):
            console.print("\n[bold]Agent Distribution:[/bold]")
            for agent, count in stats["agent_distribution"].items():
                console.print(f"  • {agent}: {count} sessions")
        
        # Success rates
        if stats.get("success_rates"):
            console.print("\n[bold]Success Rates:[/bold]")
            for agent, rate_data in stats["success_rates"].items():
                rate = rate_data["rate"] * 100
                console.print(f"  • {agent}: {rate:.1f}% ({rate_data['success']}/{rate_data['total']})")
        
        # Prompt complexity
        if stats.get("prompt_complexity_distribution"):
            console.print("\n[bold]Prompt Complexity Distribution:[/bold]")
            for complexity, count in stats["prompt_complexity_distribution"].items():
                console.print(f"  • {complexity}: {count} prompts")
        
        # MCP targets
        if stats.get("mcp_targets"):
            console.print(f"\n[bold]MCP Targets ({len(stats['mcp_targets'])}):[/bold]")
            for target in stats["mcp_targets"]:
                console.print(f"  • {target}")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to get statistics: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("prompt_ids", nargs=-1, type=int)
@click.option("--config", "-c", help="Configuration file path")
def results(prompt_ids: List[int], config: Optional[str]):
    """Display results for specific prompts."""
    # Initialize engine
    if config:
        engine = EvaluationEngine(config_path=config)
    else:
        engine = EvaluationEngine()
    
    if not prompt_ids:
        console.print("[red]Please specify at least one prompt ID[/red]")
        sys.exit(1)
    
    for prompt_id in prompt_ids:
        console.print(f"\n[bold blue]Results for Prompt {prompt_id}[/bold blue]")
        
        try:
            results = engine.get_evaluation_results(prompt_id)
            
            if not results:
                console.print(f"[yellow]No results found for prompt {prompt_id}[/yellow]")
                continue
            
            # Group by base session ID for comparative display
            session_groups = {}
            for result in results:
                if result.base_session_id not in session_groups:
                    session_groups[result.base_session_id] = []
                session_groups[result.base_session_id].append(result)
            
            for base_session_id, session_results in session_groups.items():
                console.print(f"\n[cyan]Session: {base_session_id}[/cyan]")
                
                for result in session_results:
                    display_single_result(result, show_session=False)
                    
        except Exception as e:
            console.print(f"❌ [red]Error getting results for prompt {prompt_id}: {e}[/red]")


@main.command("post-processing")
@click.option("--output", "-o", default="reports/", help="Output directory (default: reports/)")
@click.option("--format", "-f", type=click.Choice(["csv", "json"]), default="csv", help="Report format (default: csv)")
@click.option("--backend", "-b", type=click.Choice(["influxdb", "sqlite"]), default="influxdb", help="Database backend (default: influxdb)")
@click.option("--summary", is_flag=True, help="Show summary statistics only")
# Analysis modes (mutually exclusive)
@click.option("--all", "mode_all", is_flag=True, help="Process all sessions with detailed analysis")
@click.option("--prompt", "mode_prompt", type=int, help="Process sessions for specific prompt ID")
@click.option("--prompts", "mode_prompts", help="Process sessions for multiple prompts (comma-separated)")
@click.option("--agent", "mode_agent", type=click.Choice(["claude", "opencode"]), help="Process sessions for specific agent")
@click.option("--prompt-agent", "mode_prompt_agent", help="Process sessions for prompt+agent (format: 'prompt_id:agent_type')")
# Output modes
@click.option("--with-logs", is_flag=True, help="Include detailed timeline logs (comprehensive)")
@click.option("--comprehensive", is_flag=True, help="Generate comprehensive report with all analysis")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress and results")
@click.option("--config", "-c", help="Configuration file path")
def post_processing(
    output: str,
    format: str, 
    backend: str,
    summary: bool,
    mode_all: bool,
    mode_prompt: Optional[int],
    mode_prompts: Optional[str],
    mode_agent: Optional[str],
    mode_prompt_agent: Optional[str],
    with_logs: bool,
    comprehensive: bool,
    verbose: bool,
    config: Optional[str]
):
    """Unified post-processing engine for MCP evaluation reports.
    
    Supports both fast CSV-only reports and comprehensive analysis with detailed timelines.
    """
    console.print("[bold blue]� MCP Evaluation Post-Processing Engine[/bold blue]\n")
    
    try:
        # Determine output mode - default is CSV-only unless --with-logs specified
        if not with_logs:
            # Default behavior based on analysis mode
            if mode_all or mode_prompt is not None or mode_prompts or mode_agent or mode_prompt_agent or comprehensive:
                with_logs = True  # Advanced analysis modes default to detailed logs
                console.print("[cyan]💡 Advanced analysis mode detected - enabling detailed logs by default[/cyan]")
            else:
                console.print("[cyan]💡 Default mode - generating CSV report only[/cyan]")
        
        # Show summary statistics if requested
        if summary:
            from .unified_post_processing import UnifiedPostProcessor
            engine = UnifiedPostProcessor(
                backend=backend,
                output_dir=output,
                config_path=config
            )
            
            console.print("[bold]� Data Summary Statistics:[/bold]")
            stats = engine.generate_summary_statistics(verbose=verbose)
            
            if "error" in stats:
                console.print(f"❌ [red]Error: {stats['error']}[/red]")
                return
            
            # Display statistics table
            stats_table = Table(title="Evaluation Data Summary")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            stats_table.add_row("Total Records", str(stats['total_sessions']))
            stats_table.add_row("Successful Records", str(stats['successful_sessions']))
            stats_table.add_row("Success Rate", f"{stats['success_rate']:.2f}%")
            stats_table.add_row("Database Backend", stats['database_backend'])
            stats_table.add_row("Total Cost (USD)", f"${stats['total_cost_usd']:.4f}")
            stats_table.add_row("Average Cost (USD)", f"${stats['average_cost_usd']:.4f}")
            stats_table.add_row("Average Execution Time", f"{stats['average_execution_time']:.2f}s")
            
            console.print(stats_table)
            
            # Agent distribution
            if stats['agent_distribution']:
                console.print("\n[bold]Agent Distribution:[/bold]")
                for agent, count in stats['agent_distribution'].items():
                    console.print(f"  • {agent}: {count} sessions")
            
            # Prompt distribution
            if stats['prompt_distribution']:
                console.print("\n[bold]Prompt Distribution:[/bold]")
                for prompt_id, count in sorted(stats['prompt_distribution'].items()):
                    console.print(f"  • Prompt {prompt_id}: {count} sessions")
            return
        
        # Choose processing mode
        if not with_logs:
            # Use CSV-only post-processing engine (default)
            console.print("[bold cyan]📊 CSV-Only Mode: Fast report generation[/bold cyan]")
            result = _run_csv_processing(
                output, format, backend, config, verbose
            )
        else:
            # Use advanced post-processing with detailed analysis
            console.print("[bold green]🔍 Advanced Mode: Detailed analysis with timelines[/bold green]")
            result = _run_advanced_processing(
                output, backend, mode_all, mode_prompt, mode_prompts, 
                mode_agent, mode_prompt_agent,
                comprehensive, verbose, config
            )
            
            # If advanced processing fails, suggest default mode fallback
            if result and result.get("error"):
                console.print(f"[yellow]⚠️  Advanced analysis failed: {result['error']}[/yellow]")
                console.print("[cyan]💡 Tip: Use default mode for fast CSV reports that always work[/cyan]")
                console.print("[cyan]    Example: uv run python -m mcp_evaluation post-processing --prompt 1[/cyan]")
                return result
        
        # Show completion message
        if result and not result.get("error"):
            console.print(f"\n[bold green]✅ Post-processing completed successfully![/bold green]")
        elif result and result.get("error"):
            console.print(f"[red]❌ Post-processing failed: {result['error']}[/red]")
        
        # Usage tips
        console.print(f"\n[bold yellow]💡 Usage Examples:[/bold yellow]")
        console.print("  # Fast CSV reports (default):")
        console.print("  uv run python -m mcp_evaluation post-processing --agent claude")
        console.print("  # Detailed analysis with logs:")
        console.print("  uv run python -m mcp_evaluation post-processing --with-logs --prompt 1 --verbose")
        console.print("  # Comprehensive analysis:")
        console.print("  uv run python -m mcp_evaluation post-processing --all --comprehensive --verbose")
        
    except Exception as e:
        console.print(f"[red]❌ Post-processing failed: {e}[/red]")
        logger.exception("Post-processing failed")
        sys.exit(1)


def _run_csv_processing(
    output: str,
    format: str, 
    backend: str,
    config: Optional[str],
    verbose: bool
) -> Dict[str, Any]:
    """Run CSV-only post-processing using the unified engine."""
    from .unified_post_processing import UnifiedPostProcessor
    
    # Initialize unified post-processing engine
    engine = UnifiedPostProcessor(
        backend=backend,
        output_dir=output,
        config_path=config
    )
    
    # Show applied filters
    if verbose:
        console.print("[bold]🔍 Applied Filters:[/bold]")
        console.print(f"  • Database Backend: {backend}")
        console.print(f"  • Output Format: {format}")
        console.print(f"  • Output Directory: {output}")
        console.print(f"  • Mode: CSV Reports Only")
        console.print()
    
    # Generate report using unified engine
    console.print("[bold]🚀 Generating CSV Report...[/bold]")
    
    generated_files = engine.generate_csv_report(
        output_format=format,
        verbose=verbose
    )
    
    # Display results
    console.print("\n[bold green]✅ CSV Report Generation Complete![/bold green]")
    
    results_table = Table(title="Generated Files")
    results_table.add_column("Type", style="cyan")
    results_table.add_column("Location", style="green")
    
    for file_type, path in generated_files.items():
        results_table.add_row(file_type.upper(), str(path))
    
    console.print(results_table)
    
    # Show quick preview of CSV if generated
    if 'csv' in generated_files and verbose:
        console.print(f"\n[bold]📄 CSV Report Preview:[/bold]")
        
        try:
            # Read first few lines of CSV for preview
            import csv as csv_module
            with open(generated_files['csv'], 'r', encoding='utf-8') as f:
                reader = csv_module.reader(f)
                headers = next(reader)
                rows = []
                for i, row in enumerate(reader):
                    if i >= 3:  # Show first 3 data rows
                        break
                    rows.append(row)
            
            preview_table = Table(show_lines=True)
            for header in headers[:8]:  # Show first 8 columns
                preview_table.add_column(header, style="dim")
            
            for row in rows:
                preview_table.add_row(*[str(cell) for cell in row[:8]])
            
            console.print(preview_table)
            console.print(f"[dim]... showing first 3 rows and 8 columns[/dim]")
        except Exception as e:
            console.print(f"[dim]Could not show preview: {e}[/dim]")
    
    return {"generated_files": generated_files, "success": True}


def _run_advanced_processing(
    output: str,
    backend: str,
    mode_all: bool,
    mode_prompt: Optional[int],
    mode_prompts: Optional[str],
    mode_agent: Optional[str],
    mode_prompt_agent: Optional[str],
    comprehensive: bool,
    verbose: bool,
    config: Optional[str]
) -> Dict[str, Any]:
    """Run advanced post-processing with detailed analysis and timelines."""
    try:
        from .unified_post_processing import UnifiedPostProcessor
        
        # Initialize unified post-processor
        processor = UnifiedPostProcessor(
            backend=backend,
            output_dir=output,
            config_path=config
        )
        
        # Count active mode flags - prioritize explicit analysis modes
        explicit_modes = [mode_all, bool(mode_prompt), bool(mode_prompts), bool(mode_agent), bool(mode_prompt_agent)]
        active_explicit_modes = sum(explicit_modes)
        
        result = {}
        
        if active_explicit_modes == 0:
            # No explicit analysis mode - default to processing all
            if verbose:
                console.print("[cyan]🔄 No specific mode selected - processing all sessions with detailed analysis...[/cyan]")
            result = processor.process_all_sessions(verbose=verbose)
            
        elif active_explicit_modes > 1:
            console.print("[red]❌ Please specify only one analysis mode[/red]")
            return {"error": "Multiple analysis modes specified"}
        
        else:
            # Handle explicit analysis modes
            if mode_all:
                if verbose:
                    console.print("[cyan]🔄 Processing all sessions with detailed analysis...[/cyan]")
                result = processor.process_all_sessions(verbose=verbose)
                
            elif mode_prompt is not None:
                if verbose:
                    console.print(f"[cyan]🔄 Processing sessions for prompt {mode_prompt}...[/cyan]")
                result = processor.process_by_prompt(mode_prompt, verbose=verbose)
                
            elif mode_prompts:
                try:
                    prompt_list = [int(p.strip()) for p in mode_prompts.split(',')]
                    if verbose:
                        console.print(f"[cyan]🔄 Processing sessions for prompts {prompt_list}...[/cyan]")
                    result = processor.process_by_prompts(prompt_list, verbose=verbose)
                except ValueError:
                    console.print(f"[red]❌ Invalid prompts format: {mode_prompts}[/red]")
                    return {"error": "Invalid prompts format"}
                
            elif mode_agent:
                if verbose:
                    console.print(f"[cyan]🔄 Processing sessions for agent {mode_agent}...[/cyan]")
                result = processor.process_by_agent(mode_agent, verbose=verbose)
                
            elif mode_prompt_agent:
                try:
                    parts = mode_prompt_agent.split(':')
                    if len(parts) != 2:
                        raise ValueError("Invalid format")
                    prompt_id = int(parts[0])
                    agent_type = parts[1]
                    if agent_type not in ['claude', 'opencode']:
                        raise ValueError("Invalid agent type")
                    
                    if verbose:
                        console.print(f"[cyan]🔄 Processing sessions for prompt {prompt_id} + agent {agent_type}...[/cyan]")
                    result = processor.process_by_prompt_and_agent(prompt_id, agent_type, verbose=verbose)
                except ValueError:
                    console.print(f"[red]❌ Invalid prompt-agent format: {mode_prompt_agent}[/red]")
                    console.print("Use format: 'prompt_id:agent_type' (e.g., '1:claude' or '2:opencode')")
                    return {"error": "Invalid prompt-agent format"}
        
        # Generate comprehensive report if requested
        if comprehensive:
            if verbose:
                console.print("[cyan]🔄 Generating comprehensive report...[/cyan]")
            
            # Extract filters from mode parameters for comprehensive report
            comp_filter_agent = mode_agent
            comp_filter_prompt = None
            
            if mode_prompt is not None:
                comp_filter_prompt = [mode_prompt]
            elif mode_prompts:
                try:
                    comp_filter_prompt = [int(p.strip()) for p in mode_prompts.split(',')]
                except ValueError:
                    pass
            elif mode_prompt_agent:
                try:
                    parts = mode_prompt_agent.split(':')
                    comp_filter_prompt = [int(parts[0])]
                    comp_filter_agent = parts[1]
                except ValueError:
                    pass
            
            comprehensive_result = processor.generate_comprehensive_report(
                filter_agent=comp_filter_agent,
                filter_prompt=comp_filter_prompt,
                verbose=verbose
            )
            
            if "error" not in comprehensive_result:
                console.print("[bold green]✅ Comprehensive report generated successfully![/bold green]")
                console.print(f"[cyan]Report Directory:[/cyan] {comprehensive_result['report_directory']}")
                console.print(f"[cyan]Report File:[/cyan] {comprehensive_result['report_file']}")
                result["comprehensive_report"] = comprehensive_result
            else:
                console.print(f"[red]❌ Comprehensive report failed: {comprehensive_result['error']}[/red]")
        
        # Show result summary if not already shown
        if result and not result.get("error") and not verbose:
            console.print(f"\n[bold green]✅ Advanced Analysis Complete![/bold green]")
            console.print(f"[cyan]Total Sessions Processed:[/cyan] {result.get('total_sessions', 0)}")
            console.print(f"[cyan]Successful Analyses:[/cyan] {result.get('successful', 0)}")
            console.print(f"[cyan]Failed Analyses:[/cyan] {result.get('failed', 0)}")
            
            # Show generated files
            if "results" in result and result["results"]:
                timeline_count = sum(1 for r in result["results"] if "timeline_file" in r and r["timeline_file"])
                metrics_count = sum(1 for r in result["results"] if "metrics_file" in r)
                
                console.print(f"\n[bold]📄 Generated Files:[/bold]")
                console.print(f"  • Timeline Files: {timeline_count}")
                console.print(f"  • Metrics Files: {metrics_count}")
                console.print(f"  • Output Directory: {output}")
        
        return result
        
    except ImportError as e:
        console.print(f"[red]❌ Failed to import advanced post-processing module: {e}[/red]")
        console.print("[dim]Make sure all dependencies are installed[/dim]")
        return {"error": f"Import error: {e}"}
    except Exception as e:
        console.print(f"[red]❌ Advanced post-processing failed: {e}[/red]")
        logger.exception("Advanced post-processing failed")
        return {"error": str(e)}


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
