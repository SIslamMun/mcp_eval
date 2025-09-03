"""
Main evaluation engine that orchestrates prompt execution,
agent management, and session tracking for MCP evaluations.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from pydantic import BaseModel
import yaml
from rich.console import Console

from .unified_agent import UnifiedAgent, AgentConfig, AgentResponse
from .jsonl_prompt_loader import UnifiedPromptLoader, PromptData
from .session_manager import SessionManager, SessionData, ComparativeSessionData

logger = logging.getLogger(__name__)
console = Console()

# Thread-local storage for progress updates
thread_local = threading.local()


class EvaluationConfig(BaseModel):
    """Configuration for evaluation execution."""
    
    # Agent configurations
    claude_config: Dict[str, Any] = {
        "type": "claude",
        "model": "sonnet",
        "output_format": "json"
    }
    opencode_config: Dict[str, Any] = {
        "type": "opencode", 
        "model": "github-copilot/claude-3.5-sonnet",
        "enable_logs": True
    }
    
    # Evaluation settings
    prompts_dir: str = "prompts"
    db_path: str = "evaluation_sessions.db"
    max_retries: int = 3
    enable_logging: bool = True
    session_id_format: str = "eval_prompt{prompt_id:03d}_{timestamp}"
    
    # Database backend selection
    backend: str = "influxdb"  # Default to InfluxDB, can be "influxdb" or "sqlite"
    
    # InfluxDB configuration (default enabled)
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "mcp-evaluation-token"
    influxdb_org: str = "mcp-evaluation"
    influxdb_bucket: str = "evaluation-sessions"
    
    # Agent-specific overrides
    claude_allowed_tools: Optional[List[str]] = None
    claude_debug_mode: bool = False
    opencode_enable_logs: bool = True


class EvaluationResult(BaseModel):
    """Result of a single evaluation."""
    
    prompt_id: int
    base_session_id: str
    agent_type: str
    success: bool
    response: str
    session_id: Optional[str]
    cost_usd: float
    duration_ms: int
    tokens: Dict[str, Any]
    error_message: Optional[str]
    timestamp: int


class ComparativeEvaluationResult(BaseModel):
    """Result of comparative evaluation between agents."""
    
    prompt_id: int
    base_session_id: str
    claude_result: Optional[EvaluationResult]
    opencode_result: Optional[EvaluationResult]
    success: bool
    timestamp: int


class MultiModelEvaluationResult(BaseModel):
    """Result of multi-model evaluation with multiple instances per agent type."""
    
    prompt_id: int
    base_session_id: str
    claude_results: Dict[str, EvaluationResult]  # model_name -> result
    opencode_results: Dict[str, EvaluationResult]  # model_name -> result
    success: bool
    total_cost_usd: float
    total_duration_ms: int
    timestamp: int


class EvaluationEngine:
    """
    Core evaluation engine supporting both Claude Code and OpenCode
    with unified session management and comparative analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[EvaluationConfig] = None):
        """
        Initialize evaluation engine.
        
        Args:
            config_path: Path to YAML configuration file
            config: Direct configuration object
        """
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = EvaluationConfig()
            
        self.prompt_loader = UnifiedPromptLoader(self.config.prompts_dir)
        
        # Create session manager based on backend configuration
        if self.config.backend == "sqlite":
            from .session_manager import SQLiteSessionManager
            self.session_manager = SQLiteSessionManager(self.config.db_path)
        else:  # influxdb (default)
            from .session_manager import InfluxDBSessionManager
            try:
                self.session_manager = InfluxDBSessionManager(
                    db_path=self.config.db_path,
                    influxdb_url=self.config.influxdb_url,
                    influxdb_token=self.config.influxdb_token,
                    influxdb_org=self.config.influxdb_org,
                    influxdb_bucket=self.config.influxdb_bucket
                )
            except Exception as e:
                logger.warning(f"Failed to initialize InfluxDB, falling back to SQLite: {e}")
                from .session_manager import SQLiteSessionManager
                self.session_manager = SQLiteSessionManager(self.config.db_path)
    
    def _load_config(self, config_path: str) -> EvaluationConfig:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            
        return EvaluationConfig(**config_data)
    
    def execute_evaluation(
        self,
        prompt_id: int,
        agent_config: Dict[str, Any],
        base_session_id: Optional[str] = None
    ) -> EvaluationResult:
        """
        Execute single agent evaluation.
        
        Args:
            prompt_id: Prompt identifier
            agent_config: Agent configuration dictionary
            base_session_id: Base session ID (generated if None)
            
        Returns:
            EvaluationResult with execution details
        """
        # Load prompt
        try:
            prompt = self.prompt_loader.load_prompt(prompt_id)
        except FileNotFoundError as e:
            return EvaluationResult(
                prompt_id=prompt_id,
                base_session_id=base_session_id or f"error_{int(time.time())}",
                agent_type=agent_config.get("type", "unknown"),
                success=False,
                response="",
                session_id=None,
                cost_usd=0.0,
                duration_ms=0,
                tokens={},
                error_message=str(e),
                timestamp=int(time.time())
            )
        
        # Generate base session ID if not provided
        if base_session_id is None:
            base_session_id = self.session_manager.generate_base_session_id(prompt_id)
        
        # Create unified agent
        agent = UnifiedAgent(
            agent_type=agent_config["type"],
            model=agent_config.get("model")
        )
        
        # Execute with session management
        start_time = time.time()
        try:
            result = agent.execute_with_session_management(
                prompt=prompt.content,
                base_session_id=base_session_id,
                continue_conversation=agent_config.get("continue_session", False),
                **{k: v for k, v in agent_config.items() if k not in ["type", "model", "continue_session"]}
            )
            duration_ms = int((time.time() - start_time) * 1000)
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            result = AgentResponse(
                response="",
                agent=agent_config["type"],
                model=agent_config.get("model", "unknown"),
                success=False,
                error_message=str(e)
            )
        
        # Store session data
        timestamp = int(time.time())
        session_data = SessionData(
            session_id=result.session_id or base_session_id,
            prompt_id=prompt_id,
            agent_type=agent_config["type"],
            success=result.success,
            response_data={
                "response": result.response,
                "cost_usd": result.cost_usd,
                "duration_ms": result.duration_ms,
                "tokens": result.tokens,
                "agent_config": agent_config,
                "prompt": prompt.model_dump()
            },
            error_message=result.error_message,
            execution_time=result.duration_ms / 1000.0 if result.duration_ms else None,
            cost_usd=result.cost_usd,
            tokens_used=result.tokens.get("total", 0) if result.tokens else None,
            created_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),  # Always set completed_at
            timestamp=timestamp,
            total_calls=result.total_calls,
            tool_calls=result.tool_calls,
            tools_used=result.tools_used
        )
        
        self.session_manager.store_session(session_data)
        
        # Create evaluation result
        return EvaluationResult(
            prompt_id=prompt_id,
            base_session_id=base_session_id,
            agent_type=agent_config["type"],
            success=result.success,
            response=result.response,
            session_id=result.session_id,
            cost_usd=result.cost_usd,
            duration_ms=duration_ms,
            tokens=result.tokens,
            error_message=result.error_message,
            timestamp=session_data.timestamp
        )
    
    def execute_comparative_evaluation(
        self,
        prompt_id: int,
        claude_config: Optional[Dict[str, Any]] = None,
        opencode_config: Optional[Dict[str, Any]] = None
    ) -> ComparativeEvaluationResult:
        """
        Execute comparative evaluation with both agents using SAME session ID.
        
        Args:
            prompt_id: Prompt identifier
            claude_config: Claude configuration (uses default if None)
            opencode_config: OpenCode configuration (uses default if None)
            
        Returns:
            ComparativeEvaluationResult with both agent results
        """
        # Use default configurations if not provided
        if claude_config is None:
            claude_config = self.config.claude_config.copy()
        if opencode_config is None:
            opencode_config = self.config.opencode_config.copy()
            
        # Load prompt
        try:
            prompt = self.prompt_loader.load_prompt(prompt_id)
        except FileNotFoundError as e:
            timestamp = int(time.time())
            return ComparativeEvaluationResult(
                prompt_id=prompt_id,
                base_session_id=f"error_{timestamp}",
                claude_result=None,
                opencode_result=None,
                success=False,
                timestamp=timestamp
            )
        
        # Generate SAME base session ID for both agents
        timestamp = int(time.time())
        base_session_id = f"eval_prompt{prompt_id:03d}_{timestamp}"
        
        # Execute Claude Code
        claude_result = self.execute_evaluation(
            prompt_id=prompt_id,
            agent_config=claude_config,
            base_session_id=base_session_id
        )
        
        # Execute OpenCode with SAME base session ID
        opencode_result = self.execute_evaluation(
            prompt_id=prompt_id,
            agent_config=opencode_config,
            base_session_id=base_session_id
        )
        
        # Store comparative results
        comparative_data = ComparativeSessionData(
            base_session_id=base_session_id,
            prompt_id=prompt_id,
            prompt=prompt.model_dump(),
            timestamp=timestamp,
            claude_result=claude_result.model_dump() if claude_result.success else None,
            opencode_result=opencode_result.model_dump() if opencode_result.success else None
        )
        
        self.session_manager.store_comparative_session(comparative_data)
        
        return ComparativeEvaluationResult(
            prompt_id=prompt_id,
            base_session_id=base_session_id,
            claude_result=claude_result,
            opencode_result=opencode_result,
            success=claude_result.success or opencode_result.success,
            timestamp=timestamp
        )
    
    def execute_multi_model_evaluation(
        self,
        prompt_id: int,
        claude_models: List[str] = None,
        opencode_models: List[str] = None,
        continue_session: bool = False,
        skip_permissions: bool = False
    ) -> MultiModelEvaluationResult:
        """
        Execute evaluation with multiple model instances for the same agent type.
        
        Args:
            prompt_id: Prompt identifier
            claude_models: List of Claude models to test
            opencode_models: List of OpenCode models to test
            continue_session: Whether to continue previous session
            skip_permissions: Skip Claude permissions (for sandboxes only)
            
        Returns:
            MultiModelEvaluationResult with results for all model instances
        """
        # Generate base session ID for all instances
        timestamp = int(time.time())
        base_session_id = f"multi_eval_prompt{prompt_id:03d}_{timestamp}"
        
        claude_results = {}
        opencode_results = {}
        total_cost = 0.0
        total_duration = 0
        
        # Prepare all models for parallel execution
        all_tasks = []
        
        if claude_models:
            console.print(f"\n[bold blue]🚀 Starting Claude evaluation with {len(claude_models)} models (parallel)...[/bold blue]")
            for i, model in enumerate(claude_models, 1):
                all_tasks.append(("claude", model, i, len(claude_models)))
        
        if opencode_models:
            console.print(f"\n[bold green]� Starting OpenCode evaluation with {len(opencode_models)} models (parallel)...[/bold green]")
            for i, model in enumerate(opencode_models, 1):
                all_tasks.append(("opencode", model, i, len(opencode_models)))
        
        # Execute all models in parallel
        if all_tasks:
            max_workers = min(len(all_tasks), 4)  # Limit concurrent executions to 4
            console.print(f"[dim]⚡ Running {len(all_tasks)} models with {max_workers} parallel workers...[/dim]\n")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_info = {}
                for agent_type, model, model_index, total_models in all_tasks:
                    future = executor.submit(
                        self._execute_single_model,
                        agent_type, model, prompt_id, base_session_id,
                        continue_session, skip_permissions,
                        model_index, total_models
                    )
                    future_to_info[future] = (agent_type, model)
                
                # Process completed tasks
                for future in as_completed(future_to_info):
                    agent_type, original_model = future_to_info[future]
                    
                    try:
                        model, result = future.result()
                        
                        if agent_type == "claude":
                            claude_results[model] = result
                        else:
                            opencode_results[model] = result
                            
                        total_cost += result.cost_usd
                        total_duration += result.duration_ms
                        
                    except Exception as e:
                        logger.error(f"Error processing {agent_type} model {original_model}: {e}")
        
        # Check overall success
        success = any(r.success for r in claude_results.values()) or any(r.success for r in opencode_results.values())
        
        # Summary
        total_models = len(claude_results) + len(opencode_results)
        successful_models = sum(1 for r in claude_results.values() if r.success) + sum(1 for r in opencode_results.values() if r.success)
        
        if total_models > 0:
            console.print(f"\n[bold]📊 Parallel Execution Summary:[/bold]")
            console.print(f"  ✅ Successful models: {successful_models}/{total_models}")
            console.print(f"  ⚡ Parallel speedup achieved!")
            if total_cost > 0:
                console.print(f"  💰 Total cost: ${total_cost:.4f}")
        
        logger.info(f"Multi-model evaluation completed. Claude: {len(claude_results)}, OpenCode: {len(opencode_results)}")
        
        return MultiModelEvaluationResult(
            prompt_id=prompt_id,
            base_session_id=base_session_id,
            claude_results=claude_results,
            opencode_results=opencode_results,
            success=success,
            total_cost_usd=total_cost,
            total_duration_ms=total_duration,
            timestamp=int(time.time())
        )
    
    def _execute_single_model(
        self, 
        agent_type: str, 
        model: str, 
        prompt_id: int, 
        base_session_id: str,
        continue_session: bool = False,
        skip_permissions: bool = False,
        model_index: int = 1,
        total_models: int = 1
    ) -> tuple[str, EvaluationResult]:
        """
        Execute evaluation for a single model. Used for parallel execution.
        
        Returns:
            Tuple of (model_name, EvaluationResult)
        """
        try:
            # Progress update
            display_model = model.replace('github-copilot/', '')
            console.print(f"[{'blue' if agent_type == 'claude' else 'green'}]📍 Processing {agent_type.title()} model {model_index}/{total_models}: {display_model}[/{'blue' if agent_type == 'claude' else 'green'}]")
            
            if agent_type == "claude":
                config = {
                    "type": "claude",
                    "model": model,
                    "continue_session": continue_session,
                    "output_format": "json",
                    "dangerously_skip_permissions": skip_permissions
                }
            else:  # opencode
                config = {
                    "type": "opencode",
                    "model": model,
                    "continue_session": continue_session,
                    "enable_logs": True
                }
            
            model_session_id = f"{base_session_id}_{agent_type}_{model.replace('/', '_').replace(':', '_')}"
            
            start_time = time.time()
            result = self.execute_evaluation(
                prompt_id=prompt_id,
                agent_config=config,
                base_session_id=model_session_id
            )
            duration = int((time.time() - start_time) * 1000)
            
            # Update result with actual duration
            result.duration_ms = duration
            
            status = "✅ Success" if result.success else "❌ Failed"
            console.print(f"[{'blue' if agent_type == 'claude' else 'green'}]{status} {agent_type.title()} {display_model}: {result.duration_ms}ms[/{'blue' if agent_type == 'claude' else 'green'}]")
            
            return model, result
            
        except Exception as e:
            logger.error(f"Failed to execute {agent_type} with model {model}: {e}")
            display_model = model.replace('github-copilot/', '')
            console.print(f"[red]❌ Failed {agent_type.title()} {display_model}: {str(e)[:50]}...[/red]")
            
            # Create a failed result
            failed_result = EvaluationResult(
                prompt_id=prompt_id,
                base_session_id=f"{base_session_id}_{agent_type}_{model}",
                agent_type=agent_type,
                success=False,
                response="",
                session_id=None,
                cost_usd=0.0,
                duration_ms=0,
                tokens={},
                error_message=str(e),
                timestamp=int(time.time())
            )
            return model, failed_result
    
    def execute_batch_evaluation(
        self,
        prompt_ids: List[int],
        agent_config: Dict[str, Any],
        continue_sessions: bool = False
    ) -> List[EvaluationResult]:
        """
        Execute batch evaluation for multiple prompts.
        
        Args:
            prompt_ids: List of prompt identifiers
            agent_config: Agent configuration
            continue_sessions: Whether to continue sessions between prompts
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        base_session_id = None
        
        for prompt_id in prompt_ids:
            if not continue_sessions or base_session_id is None:
                base_session_id = self.session_manager.generate_base_session_id(prompt_id)
            
            # Update config for session continuation
            current_config = agent_config.copy()
            if continue_sessions and len(results) > 0:
                current_config["continue_session"] = True
            
            result = self.execute_evaluation(
                prompt_id=prompt_id,
                agent_config=current_config,
                base_session_id=base_session_id
            )
            
            results.append(result)
            
            # Use same session for continuation
            if continue_sessions:
                base_session_id = result.base_session_id
        
        return results
    
    def execute_batch_comparative_evaluation(
        self,
        prompt_ids: List[int],
        claude_config: Optional[Dict[str, Any]] = None,
        opencode_config: Optional[Dict[str, Any]] = None
    ) -> List[ComparativeEvaluationResult]:
        """
        Execute batch comparative evaluation for multiple prompts.
        
        Args:
            prompt_ids: List of prompt identifiers
            claude_config: Claude configuration
            opencode_config: OpenCode configuration
            
        Returns:
            List of ComparativeEvaluationResult objects
        """
        results = []
        
        for prompt_id in prompt_ids:
            result = self.execute_comparative_evaluation(
                prompt_id=prompt_id,
                claude_config=claude_config,
                opencode_config=opencode_config
            )
            results.append(result)
            
        return results
    
    def get_evaluation_results(self, prompt_id: int) -> List[EvaluationResult]:
        """
        Get all evaluation results for a specific prompt.
        
        Args:
            prompt_id: Prompt identifier
            
        Returns:
            List of EvaluationResult objects
        """
        sessions = self.session_manager.get_sessions_by_prompt_id(prompt_id)
        results = []
        
        for session in sessions:
            results.append(EvaluationResult(
                prompt_id=session.prompt_id,
                base_session_id=session.base_session_id,
                agent_type=session.agent_type,
                success=session.result.success,
                response=session.result.response,
                session_id=session.result.session_id,
                cost_usd=session.result.cost_usd,
                duration_ms=session.result.duration_ms,
                tokens=session.result.tokens,
                error_message=session.result.error_message,
                timestamp=session.timestamp
            ))
        
        return results
    
    def get_comparative_results(self, base_session_id: str) -> Optional[ComparativeEvaluationResult]:
        """
        Get comparative results by base session ID.
        
        Args:
            base_session_id: Base session identifier
            
        Returns:
            ComparativeEvaluationResult if found, None otherwise
        """
        comparative_data = self.session_manager.get_comparative_results_by_base_session_id(base_session_id)
        if not comparative_data:
            return None
        
        # Convert to EvaluationResult objects
        claude_result = None
        if comparative_data.claude_result:
            claude_result = EvaluationResult(**comparative_data.claude_result)
            
        opencode_result = None
        if comparative_data.opencode_result:
            opencode_result = EvaluationResult(**comparative_data.opencode_result)
        
        return ComparativeEvaluationResult(
            prompt_id=comparative_data.prompt_id,
            base_session_id=comparative_data.base_session_id,
            claude_result=claude_result,
            opencode_result=opencode_result,
            success=(claude_result and claude_result.success) or (opencode_result and opencode_result.success),
            timestamp=comparative_data.timestamp
        )
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive evaluation statistics.
        
        Returns:
            Dictionary with evaluation statistics
        """
        base_stats = self.session_manager.get_session_statistics()
        
        # Add prompt-specific statistics
        all_prompts = self.prompt_loader.load_all_prompts()
        
        complexity_stats = {"low": 0, "medium": 0, "high": 0}
        mcp_targets = set()
        
        for prompt in all_prompts.values():
            complexity_stats[prompt.metadata.complexity] += 1
            mcp_targets.update(prompt.metadata.target_mcp)
        
        base_stats.update({
            "total_prompts": len(all_prompts),
            "prompt_complexity_distribution": complexity_stats,
            "unique_mcp_targets": len(mcp_targets),
            "mcp_targets": sorted(list(mcp_targets))
        })
        
        return base_stats
    
    def export_evaluation_results(self, output_file: str, format: str = "json") -> str:
        """
        Export all evaluation results.
        
        Args:
            output_file: Output file path
            format: Export format ("json" or "csv")
            
        Returns:
            Path to exported file
        """
        return self.session_manager.export_sessions(output_file, format)
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate evaluation setup and report issues.
        
        Returns:
            Validation report
        """
        report = {
            "setup_valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check prompts directory
        prompts_report = self.prompt_loader.validate_prompt_directory()
        if prompts_report["errors"]:
            report["setup_valid"] = False
            report["issues"].extend(prompts_report["errors"])
        
        if prompts_report["total_files"] == 0:
            report["warnings"].append("No prompt files found")
        
        if prompts_report["duplicated_ids"]:
            report["issues"].append(f"Duplicate prompt IDs: {prompts_report['duplicated_ids']}")
            report["setup_valid"] = False
        
        # Check database
        try:
            stats = self.session_manager.get_session_statistics()
            report["database_accessible"] = True
            report["existing_sessions"] = stats["total_sessions"]
        except Exception as e:
            report["setup_valid"] = False
            report["issues"].append(f"Database error: {e}")
        
        # Test agent availability
        try:
            # Test Claude Code
            claude_agent = UnifiedAgent("claude", "sonnet")
            claude_test = claude_agent.execute("Test message", AgentConfig(
                type="claude", 
                model="sonnet", 
                output_format="text"
            ))
            report["claude_available"] = claude_test.success
            if not claude_test.success:
                report["warnings"].append(f"Claude Code test failed: {claude_test.error_message}")
        except Exception as e:
            report["claude_available"] = False
            report["warnings"].append(f"Claude Code unavailable: {e}")
        
        try:
            # Test OpenCode
            opencode_agent = UnifiedAgent("opencode", "github-copilot/claude-3.5-sonnet")
            opencode_test = opencode_agent.execute("Test message", AgentConfig(
                type="opencode",
                model="github-copilot/claude-3.5-sonnet"
            ))
            report["opencode_available"] = opencode_test.success
            if not opencode_test.success:
                report["warnings"].append(f"OpenCode test failed: {opencode_test.error_message}")
        except Exception as e:
            report["opencode_available"] = False
            report["warnings"].append(f"OpenCode unavailable: {e}")
        
        # Recommendations
        if prompts_report["total_files"] < 10:
            report["recommendations"].append("Consider adding more test prompts for comprehensive evaluation")
            
        if not report.get("claude_available") and not report.get("opencode_available"):
            report["setup_valid"] = False
            report["issues"].append("Neither Claude Code nor OpenCode is available")
        
        return report
