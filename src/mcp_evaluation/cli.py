"""
Command-line interface for MCP evaluation infrastructure.
"""

import json
import sys
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from rich.progress import track

from . import EvaluationEngine, MarkdownPromptLoader, SessionManager
from .evaluation_engine import EvaluationConfig


console = Console()


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
@click.argument("prompt_id", type=int)
@click.option("--agent", "-a", type=click.Choice(["claude", "opencode", "both"]), default="both", help="Agent to use")
@click.option("--claude-model", default="sonnet", help="Claude model")
@click.option("--opencode-model", default="github-copilot/claude-3.5-sonnet", help="OpenCode model")
@click.option("--config", "-c", help="Configuration file path")
@click.option("--backend", "-b", type=click.Choice(["influxdb", "sqlite"]), default="influxdb", help="Database backend (default: influxdb)")
@click.option("--continue-session", is_flag=True, help="Continue previous session")
@click.option("--skip-permissions", is_flag=True, help="Skip Claude permissions (for sandboxes only)")
def run(
    prompt_id: int,
    agent: str,
    claude_model: str,
    opencode_model: str,
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
    
    try:
        if agent == "both":
            # Comparative evaluation
            claude_config = {
                "type": "claude",
                "model": claude_model,
                "continue_session": continue_session,
                "dangerously_skip_permissions": skip_permissions
            }
            
            opencode_config = {
                "type": "opencode",
                "model": opencode_model,
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
            # Single agent evaluation
            agent_config = {
                "type": agent,
                "model": claude_model if agent == "claude" else opencode_model,
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


if __name__ == "__main__":
    main()
