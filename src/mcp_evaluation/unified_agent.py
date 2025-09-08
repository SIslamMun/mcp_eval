"""
Unified Agent implementation supporting multiple AI agents
with consistent interface and session management.
"""

import json
import os
import subprocess
import time
import uuid
import hashlib
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentCapabilities(BaseModel):
    """Define capabilities and configuration for different agent types."""
    
    command_template: List[str]
    command_options: Dict[str, str] = {}
    session_management: Dict[str, Any]
    output_parsing: Dict[str, Any]
    storage_paths: List[str]
    supported_models: List[str]
    default_model: Optional[str] = None
    response_patterns: List[str]
    tool_patterns: List[str]
    content_extraction: Dict[str, Any]


class AgentResponse(BaseModel):
    """Standardized response format for all agents."""
    
    response: str
    session_id: Optional[str] = None
    original_session_id: Optional[str] = None
    cost_usd: float = 0.0
    duration_ms: int = 0
    tokens: Dict[str, Any] = {}
    agent: str
    model: str
    raw_output: str = ""
    stderr: str = ""
    success: bool = True
    error_message: Optional[str] = None
    
    # Tool usage metrics
    total_calls: int = 0
    tool_calls: int = 0
    tools_used: Dict[str, int] = {}


class AgentConfig(BaseModel):
    """Dynamic configuration for agent execution."""
    
    type: str
    model: Optional[str] = None
    continue_session: bool = False
    session_id: Optional[str] = None
    resume_session_id: Optional[str] = None
    
    # Dynamic options that can be applied to any agent
    output_format: str = "json"
    allowed_tools: Optional[List[str]] = None
    debug_mode: bool = False
    dangerously_skip_permissions: bool = False
    enable_logs: bool = False
    agent_name: Optional[str] = None
    
    # Content extraction parameters
    content_keywords: List[str] = []
    tool_names: List[str] = []
    response_filters: List[str] = []


class UnifiedAgent:
    """
    Unified agent interface supporting multiple AI agents dynamically
    with consistent session management and response normalization.
    """
    
    @staticmethod
    def get_available_opencode_models() -> List[str]:
        """
        Dynamically detect available OpenCode models from the system.
        
        Returns:
            List of available OpenCode models
        """
        try:
            result = subprocess.run(
                ["opencode", "models"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                models = []
                for line in result.stdout.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('opencode/') and not line.startswith('ollama/'):
                        # Filter out local/ollama models, keep github-copilot and opencode models
                        if line.startswith('github-copilot/') or line.startswith('opencode/'):
                            models.append(line)
                
                if models:
                    return models
                    
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Could not detect OpenCode models: {e}")
        
        # Fallback to commonly available models
        return [
            "github-copilot/claude-3.5-sonnet",
            "github-copilot/gpt-4o", 
            "github-copilot/claude-3.7-sonnet",
            "opencode/grok-code"
        ]
    
    @staticmethod
    def suggest_models(agent_type: str, user_preference: Optional[str] = None) -> List[str]:
        """
        Suggest appropriate models based on agent type and user preference.
        
        Args:
            agent_type: "claude" or "opencode"
            user_preference: Optional user preference like "fast", "accurate", "cheap"
            
        Returns:
            Ordered list of suggested models
        """
        if agent_type == "claude":
            claude_models = ["sonnet", "haiku", "opus"]
            if user_preference == "fast":
                return ["haiku", "sonnet", "opus"]
            elif user_preference == "accurate":
                return ["opus", "sonnet", "haiku"]
            elif user_preference == "cheap":
                return ["haiku", "sonnet", "opus"]
            else:
                return claude_models
                
        elif agent_type == "opencode":
            available_models = UnifiedAgent.get_available_opencode_models()
            
            # Prioritize based on preference
            if user_preference == "fast":
                priority_order = ["gpt-4o", "claude-3.5-sonnet", "gemini", "claude-3.7"]
            elif user_preference == "accurate":
                priority_order = ["claude-opus", "gpt-5", "claude-3.7", "claude-3.5-sonnet"]
            elif user_preference == "cheap":
                priority_order = ["gemini", "claude-3.5-sonnet", "gpt-4o"]
            else:
                priority_order = ["claude-3.5-sonnet", "gpt-4o", "claude-3.7", "gemini"]
            
            # Sort available models by priority
            suggested = []
            for priority in priority_order:
                for model in available_models:
                    if priority in model and model not in suggested:
                        suggested.append(model)
            
            # Add remaining models
            for model in available_models:
                if model not in suggested:
                    suggested.append(model)
                    
            return suggested[:5]  # Return top 5 suggestions
            
        return []
    
    # Dynamic agent configurations
    AGENT_CONFIGS = {
        "claude": AgentCapabilities(
            command_template=["claude", "--print"],
            command_options={
                "model_flag": "--model",
                "session_id_flag": "--session-id", 
                "resume_flag": "--resume",
                "continue_flag": "--continue",
                "output_format_flag": "--output-format",
                "tools_flag": "--allowed-tools",
                "debug_flag": "--debug",
                "skip_permissions_flag": "--dangerously-skip-permissions"
            },
            session_management={
                "supports_session_id": True,
                "supports_resume": True,
                "supports_continue": True,
                "uuid_format_required": True,
                "session_flag": "--session-id",
                "resume_flag": "--resume",
                "continue_flag": "--continue"
            },
            output_parsing={
                "json_response": True,
                "result_key": "result",
                "session_key": "session_id",
                "cost_key": "total_cost_usd",
                "duration_key": "duration_ms",
                "tokens_key": "usage"
            },
            storage_paths=[],
            supported_models=["sonnet", "haiku", "opus"],
            default_model="sonnet",
            response_patterns=[r'session=([a-zA-Z0-9_-]+)', r'session ([a-zA-Z0-9_-]+)', r'ses_[a-zA-Z0-9_-]+'],
            tool_patterns=[],
            content_extraction={}
        ),
        "opencode": AgentCapabilities(
            command_template=["opencode", "run"],
            command_options={
                "model_flag": "-m",
                "session_id_flag": "-s",
                "continue_flag": "-c",
                "logs_flag": "--print-logs"
            },
            session_management={
                "supports_session_id": True,
                "supports_resume": False,
                "supports_continue": True,
                "uuid_format_required": False,
                "session_flag": "-s",
                "continue_flag": "-c",
                "logs_flag": "--print-logs"
            },
            output_parsing={
                "json_response": False,
                "extract_from_storage": True
            },
            storage_paths=[
                "{home}/.local/share/opencode/storage",
                "{home}/.local/share/opencode/project/global/storage/session"
            ],
            supported_models=[],  # Will be populated dynamically
            default_model="github-copilot/claude-3.5-sonnet",
            response_patterns=[r'session=([a-zA-Z0-9_-]+)', r'session ([a-zA-Z0-9_-]+)', r'ses_[a-zA-Z0-9_-]+'],
            tool_patterns=["text", "tool", "step-start", "step-finish"],
            content_extraction={}
        )
    }
    
    def __init__(self, agent_type: str, model: Optional[str] = None):
        """
        Initialize unified agent dynamically.
        
        Args:
            agent_type: Type of agent (must be in AGENT_CONFIGS)
            model: Model to use (agent-specific format)
        """
        if agent_type not in self.AGENT_CONFIGS:
            available_types = list(self.AGENT_CONFIGS.keys())
            raise ValueError(f"Unsupported agent type: {agent_type}. Available: {available_types}")
            
        self.agent_type = agent_type
        self.capabilities = self.AGENT_CONFIGS[agent_type]
        
        # Dynamically populate OpenCode models if not already set
        if agent_type == "opencode" and not self.capabilities.supported_models:
            self.capabilities.supported_models = self.get_available_opencode_models()
            logger.info(f"Detected {len(self.capabilities.supported_models)} OpenCode models")
        
        self.model = model or getattr(self.capabilities, 'default_model', None)
        
        # Validate model is supported
        if self.model and self.capabilities.supported_models and self.model not in self.capabilities.supported_models:
            suggestions = self.suggest_models(agent_type)
            logger.warning(f"Model '{self.model}' not in supported models. Available: {self.capabilities.supported_models[:5]}")
            logger.info(f"Suggested models for {agent_type}: {suggestions}")
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported models for this agent.
        
        Returns:
            List of supported model names
        """
        if self.agent_type == "opencode" and not self.capabilities.supported_models:
            self.capabilities.supported_models = self.get_available_opencode_models()
        
        return self.capabilities.supported_models
    
    def validate_model(self, model: str) -> bool:
        """
        Validate if a model is supported by this agent.
        
        Args:
            model: Model name to validate
            
        Returns:
            True if model is supported, False otherwise
        """
        supported = self.get_supported_models()
        return model in supported if supported else True  # Allow any model if detection failed
        
    def execute(self, prompt: str, config: Optional[AgentConfig] = None) -> AgentResponse:
        """
        Execute prompt with unified interface dynamically.
        
        Args:
            prompt: The prompt to execute
            config: Agent configuration options
            
        Returns:
            Standardized AgentResponse
        """
        if config is None:
            config = AgentConfig(type=self.agent_type, model=self.model)
            
        try:
            return self._execute_dynamic(prompt, config)
        except Exception as e:
            return AgentResponse(
                response="",
                agent=self.agent_type,
                model=self.model or "unknown",
                success=False,
                error_message=str(e)
            )
    
    def _execute_dynamic(self, prompt: str, config: AgentConfig) -> AgentResponse:
        """Execute prompt using dynamic configuration."""
        capabilities = self.capabilities
        
        # Validate model before execution
        if config.model and not self.validate_model(config.model):
            logger.error(f"Invalid model '{config.model}' for {self.agent_type}")
            return AgentResponse(
                response="",
                agent=self.agent_type,
                model=config.model or "unknown",
                success=False,
                error_message=f"Model '{config.model}' is not supported for {self.agent_type}"
            )
        
        # Build command dynamically
        cmd = capabilities.command_template.copy()
        
        # Model selection
        model = config.model or self.model
        if model and capabilities.command_options.get("model_flag"):
            model_flag = capabilities.command_options["model_flag"]
            cmd.extend([model_flag, model])
                
        # Session management - let OpenCode create its own sessions
        session_mgmt = capabilities.session_management
        if self.agent_type == "claude":
            # Only Claude supports custom session IDs
            if config.session_id and session_mgmt.get("supports_session_id"):
                session_flag = capabilities.command_options.get("session_id_flag")
                if session_flag:
                    session_id = config.session_id
                    if session_mgmt.get("uuid_format_required"):
                        session_id = self._ensure_uuid_format(session_id)
                    cmd.extend([session_flag, session_id])
            elif config.resume_session_id and session_mgmt.get("supports_resume"):
                resume_flag = capabilities.command_options.get("resume_flag")
                if resume_flag:
                    resume_id = config.resume_session_id
                    if session_mgmt.get("uuid_format_required"):
                        resume_id = self._ensure_uuid_format(resume_id)
                    cmd.extend([resume_flag, resume_id])
            elif config.continue_session and session_mgmt.get("supports_continue"):
                continue_flag = capabilities.command_options.get("continue_flag")
                if continue_flag:
                    cmd.append(continue_flag)
        # For OpenCode, let it create its own session - don't force session IDs
            
        # Additional flags
        if config.output_format and capabilities.command_options.get("output_format_flag"):
            cmd.extend([capabilities.command_options["output_format_flag"], config.output_format])
        if config.allowed_tools and capabilities.command_options.get("tools_flag"):
            cmd.extend([capabilities.command_options["tools_flag"], ",".join(config.allowed_tools)])
        if config.debug_mode and capabilities.command_options.get("debug_flag"):
            cmd.append(capabilities.command_options["debug_flag"])
        
        # For Claude, always skip permissions to enable tool usage
        if self.agent_type == "claude" and capabilities.command_options.get("skip_permissions_flag"):
            cmd.append(capabilities.command_options["skip_permissions_flag"])
        elif config.dangerously_skip_permissions and capabilities.command_options.get("skip_permissions_flag"):
            cmd.append(capabilities.command_options["skip_permissions_flag"])
            
        if config.enable_logs and capabilities.command_options.get("logs_flag"):
            cmd.append(capabilities.command_options["logs_flag"])
            
        cmd.append(prompt)
        
        # Execute command with timeout and measure execution time
        import time
        start_time = time.time()
        try:
            logger.info(f"Executing command: {' '.join(cmd[:3])} ... (model: {model})")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True
            )
            actual_duration_ms = int((time.time() - start_time) * 1000)
        except Exception as e:
            actual_duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Execution failed: {e}")
            return AgentResponse(
                response="",
                agent=self.agent_type,
                model=model or "unknown", 
                success=False,
                error_message=f"Execution failed: {str(e)}",
                duration_ms=actual_duration_ms
            )
        
        # Parse response dynamically
        return self._parse_response_dynamic(result, config, model, actual_duration_ms)
    
    def _parse_response_dynamic(self, result, config: AgentConfig, model: Optional[str], actual_duration_ms: int) -> AgentResponse:
        """Parse response using dynamic configuration."""
        capabilities = self.capabilities
        output_config = capabilities.output_parsing
        
        # Extract session ID using dynamic patterns
        session_id = config.session_id
        extracted_session_id = None
        
        import re
        # Look for OpenCode session pattern in logs
        if self.agent_type == "opencode":
            opencode_session_match = re.search(r'session=([a-zA-Z0-9_]+)', result.stderr)
            if opencode_session_match:
                extracted_session_id = opencode_session_match.group(1)
            else:
                # Look for session creation pattern
                session_create_match = re.search(r'session=([a-zA-Z0-9_]+)', result.stderr)
                if session_create_match:
                    extracted_session_id = session_create_match.group(1)
        else:
            # Claude and other agents
            for pattern in capabilities.response_patterns:
                match = re.search(pattern, result.stderr)
                if match:
                    if pattern.startswith("ses_"):
                        extracted_session_id = match.group(0)
                    else:
                        extracted_session_id = match.group(1)
                    break
                
        if extracted_session_id and not session_id:
            session_id = extracted_session_id
        
        # Parse output based on capabilities
        response_text = None
        cost_usd = 0.0
        duration_ms = actual_duration_ms  # Use actual measured duration as default
        tokens = {}
        raw_json_output = None  # Store the full JSON for tool extraction
        
        if output_config.get("json_response") and result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                raw_json_output = result.stdout  # Store the full JSON
                response_text = data.get(output_config["result_key"], "")
                session_id = data.get(output_config["session_key"], session_id)
                cost_usd = data.get(output_config["cost_key"], 0.0)
                # For Claude, use reported duration if available, otherwise use actual measured time
                reported_duration = data.get(output_config["duration_key"], 0)
                duration_ms = reported_duration if reported_duration > 0 else actual_duration_ms
                tokens = data.get(output_config["tokens_key"], {})
            except json.JSONDecodeError:
                pass
        
        # If JSON parsing failed or not applicable, try other methods
        if not response_text:
            if output_config.get("extract_from_storage") and extracted_session_id:
                response_text = self._extract_from_storage_dynamic(extracted_session_id, config)
            elif not response_text:
                response_text = self._filter_stdout_dynamic(result.stdout, config)
        
        # Fallback message
        if not response_text:
            response_text = f"{self.agent_type.title()} execution completed successfully but response content not accessible in current mode."
        
        # Extract tool usage metrics using raw JSON if available, otherwise use response text
        tool_input = raw_json_output if raw_json_output else response_text
        total_calls, tool_calls, tools_used = self._extract_tool_usage(tool_input, result.stdout, result.stderr)
        
        return AgentResponse(
            response=response_text,
            session_id=session_id,
            original_session_id=config.session_id or config.resume_session_id,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            tokens=tokens,
            agent=self.agent_type,
            model=model or "unknown",
            raw_output=result.stdout,
            stderr=result.stderr,
            success=result.returncode == 0,
            error_message=result.stderr if result.returncode != 0 else None,
            total_calls=total_calls,
            tool_calls=tool_calls,
            tools_used=tools_used
        )
    
    def _extract_tool_usage(self, response_text: str, stdout: str, stderr: str) -> tuple[int, int, Dict[str, int]]:
        """
        Extract tool usage information from agent output dynamically.
        This method analyzes agent responses to detect MCP tool usage without hardcoded patterns.
        Returns: (total_calls, tool_calls, tools_used_dict)
        """
        import re
        import json
        
        tools_used = {}
        tool_calls = 0
        total_calls = 0
        
        try:
            # For Claude: Parse JSON response from raw_output
            if response_text.strip().startswith('{'):
                data = json.loads(response_text)
                total_calls = data.get('num_turns', 0)
                usage = data.get('usage', {})
                server_tool_use = usage.get('server_tool_use', {})
                
                # Add any server tool use (generic approach)
                for tool_type, count in server_tool_use.items():
                    if isinstance(count, int) and count > 0:
                        tools_used[tool_type] = count
                        tool_calls += count
                
                # Estimate tool calls from num_turns
                if total_calls > 1:
                    estimated_tool_calls = total_calls - 1
                    tool_calls = max(tool_calls, estimated_tool_calls)
                
                # Generic pattern-based tool detection from result text
                result_text = data.get('result', '')
                detected_tools = self._detect_tools_dynamically(result_text)
                
                # Add detected tools to tools_used
                for tool_name in detected_tools:
                    if tool_name in tools_used:
                        tools_used[tool_name] += 1
                    else:
                        tools_used[tool_name] = 1
                
                # Update tool call count if we detected specific tools
                if detected_tools and tool_calls == 0:
                    tool_calls = len(detected_tools)
                elif detected_tools:
                    tool_calls = max(tool_calls, len(detected_tools))
                    
            else:
                # For OpenCode and other agents: Parse stderr for tool usage
                if stderr:
                    # Remove ANSI color codes for clean parsing
                    clean_stderr = re.sub(r'\x1b\[[0-9;]*m', '', stderr)
                    
                    # Dynamic tool detection from stderr patterns
                    tool_pattern = r'\|\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+'
                    tools_found = re.findall(tool_pattern, clean_stderr)
                    
                    # Count each tool usage
                    for tool in tools_found:
                        if tool in tools_used:
                            tools_used[tool] += 1
                        else:
                            tools_used[tool] = 1
                        tool_calls += 1
                    
                    total_calls = tool_calls
                
                # Fallback: Generic tool detection from all text
                if tool_calls == 0:
                    all_text = f"{response_text} {stdout} {stderr}"
                    detected_tools = self._detect_tools_dynamically(all_text)
                    
                    # Count detected tools
                    for tool_name in detected_tools:
                        tools_used[tool_name] = tools_used.get(tool_name, 0) + 1
                        tool_calls += 1
                    
                    # Estimate total API calls (tool calls + at least 1 main API call)
                    total_calls = max(tool_calls + 1, 1)
                
        except Exception as e:
            logger.debug(f"Error extracting tool usage: {e}")
            # Fallback to basic counting
            total_calls = 1
            tool_calls = 0
            tools_used = {}
        
        return total_calls, tool_calls, tools_used
    
    def _detect_tools_dynamically(self, text: str) -> set:
        """
        Dynamically detect MCP tool usage from text without hardcoded patterns.
        Uses generic patterns that work across different MCP implementations.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Set of detected tool names
        """
        import re
        
        detected_tools = set()
        
        # Generic patterns for common MCP tool categories
        generic_patterns = {
            # Pattern for explicit tool mentions (Tool: name, Tool name:, etc.)
            'explicit_tool': r'(?:Tool|tool)[\s:]+([a-zA-Z_][a-zA-Z0-9_]*)',
            
            # Pattern for function-like calls (function_name(), tool_name())
            'function_call': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)',
            
            # Pattern for action verbs indicating tool usage
            'action_pattern': r'(?:executing|calling|using|invoking|running)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            
            # Pattern for MCP server references
            'mcp_reference': r'(?:mcp|MCP)[\s-]*([a-zA-Z_][a-zA-Z0-9_]*)',
            
            # Pattern for command-like structures
            'command_pattern': r'`([a-zA-Z_][a-zA-Z0-9_]*)`',
            
            # Pattern for step-based execution (common in OpenCode)
            'step_pattern': r'step[\s-]*([a-zA-Z_][a-zA-Z0-9_]*)',
        }
        
        # Apply generic patterns to detect tools
        for pattern_name, pattern in generic_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Filter out common non-tool words
                if self._is_likely_tool_name(match):
                    detected_tools.add(match.lower())
        
        # Additional heuristics for specific agent types
        if self.agent_type == "claude":
            # Claude-specific tool detection patterns
            claude_tools = self._detect_claude_tools(text)
            detected_tools.update(claude_tools)
        elif self.agent_type == "opencode":
            # OpenCode-specific tool detection patterns  
            opencode_tools = self._detect_opencode_tools(text)
            detected_tools.update(opencode_tools)
        
        return detected_tools
    
    def _is_likely_tool_name(self, name: str) -> bool:
        """
        Heuristic to determine if a detected name is likely a tool name.
        
        Args:
            name: Potential tool name
            
        Returns:
            True if name appears to be a tool name
        """
        # Skip common words that are not tools
        common_non_tools = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'upon', 'against',
            'within', 'without', 'since', 'until', 'while', 'because', 'although',
            'if', 'when', 'where', 'how', 'why', 'what', 'who', 'which', 'this',
            'that', 'these', 'those', 'a', 'an', 'some', 'any', 'all', 'each',
            'every', 'no', 'none', 'one', 'two', 'three', 'first', 'second',
            'last', 'next', 'previous', 'new', 'old', 'big', 'small', 'good',
            'bad', 'right', 'wrong', 'true', 'false', 'yes', 'no', 'maybe',
            'can', 'could', 'will', 'would', 'should', 'must', 'may', 'might',
            'do', 'does', 'did', 'done', 'have', 'has', 'had', 'be', 'is', 
            'are', 'was', 'were', 'been', 'being', 'get', 'got', 'go', 'went',
            'gone', 'come', 'came', 'see', 'saw', 'seen', 'know', 'knew', 'known',
            'take', 'took', 'taken', 'give', 'gave', 'given', 'make', 'made',
            'say', 'said', 'tell', 'told', 'think', 'thought', 'find', 'found',
            'use', 'used', 'work', 'worked', 'call', 'called', 'try', 'tried',
            'ask', 'asked', 'need', 'needed', 'want', 'wanted', 'look', 'looked',
            'seem', 'seemed', 'feel', 'felt', 'leave', 'left', 'put', 'keep',
            'kept', 'let', 'begin', 'began', 'begun', 'help', 'helped', 'show',
            'showed', 'shown', 'hear', 'heard', 'play', 'played', 'run', 'ran',
            'move', 'moved', 'live', 'lived', 'believe', 'believed', 'hold',
            'held', 'bring', 'brought', 'happen', 'happened', 'write', 'wrote',
            'written', 'provide', 'provided', 'sit', 'sat', 'stand', 'stood',
            'lose', 'lost', 'pay', 'paid', 'meet', 'met', 'include', 'included',
            'continue', 'continued', 'set', 'learn', 'learned', 'change', 'changed',
            'lead', 'led', 'understand', 'understood', 'watch', 'watched',
            'follow', 'followed', 'stop', 'stopped', 'create', 'created',
            'speak', 'spoke', 'spoken', 'read', 'open', 'opened', 'close',
            'closed', 'build', 'built', 'spend', 'spent', 'cut', 'grow', 'grew',
            'grown', 'fall', 'fell', 'fallen', 'reach', 'reached', 'kill',
            'killed', 'remain', 'remained', 'suggest', 'suggested', 'raise',
            'raised', 'pass', 'passed', 'sell', 'sold', 'require', 'required',
            'report', 'reported', 'decide', 'decided', 'pull', 'pulled'
        }
        
        name_lower = name.lower().strip()
        
        # Skip if it's a common non-tool word
        if name_lower in common_non_tools:
            return False
        
        # Skip if too short (likely not a meaningful tool name)
        if len(name_lower) < 3:
            return False
        
        # Skip if it's all numbers
        if name_lower.isdigit():
            return False
        
        # Tool names typically contain underscores, are compound words, or end with common suffixes
        tool_indicators = [
            '_',  # Contains underscore (common in function names)
            'mcp',  # Contains mcp
            'tool',  # Contains tool
            'api',  # Contains api
            'file',  # File operations
            'dir',  # Directory operations  
            'search',  # Search operations
            'get',  # Getter operations
            'set',  # Setter operations
            'create',  # Create operations
            'delete',  # Delete operations
            'update',  # Update operations
            'list',  # List operations
            'run',  # Run operations
            'execute',  # Execute operations
            'fetch',  # Fetch operations
            'load',  # Load operations
            'save',  # Save operations
            'read',  # Read operations
            'write',  # Write operations
        ]
        
        # Check if name contains any tool indicators
        for indicator in tool_indicators:
            if indicator in name_lower:
                return True
        
        # If name is CamelCase or contains capital letters (common in API names)
        if any(c.isupper() for c in name) and len(name) > 3:
            return True
        
        # If name follows common naming patterns
        if len(name_lower) > 5 and any(pattern in name_lower for pattern in ['handler', 'service', 'client', 'manager', 'processor']):
            return True
        
        return False
    
    def _detect_claude_tools(self, text: str) -> set:
        """
        Detect Claude-specific tool usage patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of detected Claude tool names
        """
        import re
        
        detected = set()
        
        # Claude-specific patterns
        claude_patterns = [
            r'Tool\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:',  # Tool name:
            r'Using\s+tool\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # Using tool name
            r'Calling\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+tool',  # Calling name tool
            r'```(\w+)',  # Code blocks often indicate tool usage
            r'<tool_use>.*?<tool_name>([^<]+)</tool_name>',  # XML-style tool calls
        ]
        
        for pattern in claude_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if self._is_likely_tool_name(match):
                    detected.add(match.lower())
        
        return detected
    
    def _detect_opencode_tools(self, text: str) -> set:
        """
        Detect OpenCode-specific tool usage patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of detected OpenCode tool names
        """
        import re
        
        detected = set()
        
        # OpenCode-specific patterns
        opencode_patterns = [
            r'\|\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+',  # | tool_name format
            r'step-start.*?([a-zA-Z_][a-zA-Z0-9_]*)',  # step-start patterns
            r'step-finish.*?([a-zA-Z_][a-zA-Z0-9_]*)',  # step-finish patterns
            r'executing\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # executing tool_name
            r'tool\.([a-zA-Z_][a-zA-Z0-9_]*)',  # tool.name patterns
        ]
        
        for pattern in opencode_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if self._is_likely_tool_name(match):
                    detected.add(match.lower())
        
        return detected
    
    def _filter_stdout_dynamic(self, stdout: str, config: AgentConfig) -> Optional[str]:
        """Filter stdout content dynamically based on configuration."""
        response_text = stdout.strip()
        
        # Dynamic filtering based on config
        filter_patterns = config.response_filters or ["created session", "session=", "info ", "debug "]
        
        if response_text and not any(pattern in response_text.lower() for pattern in filter_patterns):
            lines = response_text.split('\n')
            content_lines = []
            
            for line in lines:
                if not any(pattern in line.lower() for pattern in filter_patterns):
                    content_lines.append(line)
            
            if content_lines:
                filtered_response = '\n'.join(content_lines).strip()
                if filtered_response:  # Any non-empty content
                    return filtered_response
        
        return None
    
    def _extract_from_storage_dynamic(self, session_id: str, config: AgentConfig) -> Optional[str]:
        """Extract response from storage using dynamic configuration."""
        try:
            capabilities = self.capabilities
            storage_paths = capabilities.storage_paths
            
            # Expand storage paths dynamically
            expanded_paths = []
            for path_template in storage_paths:
                expanded_path = path_template.format(home=str(Path.home()))
                expanded_paths.append(Path(expanded_path))
            
            # Find recent messages dynamically
            all_messages = []
            for base_path in expanded_paths:
                if base_path.exists():
                    message_files = list(base_path.glob("message/*/msg_*.json"))
                    all_messages.extend(message_files)
            
            # Sort by modification time
            recent_messages = []
            for msg_file in all_messages:
                try:
                    with open(msg_file, 'r') as f:
                        msg_data = json.load(f)
                    
                    if msg_data.get('role') == 'assistant':
                        mtime = msg_file.stat().st_mtime
                        recent_messages.append((mtime, msg_file, msg_data))
                        
                except (json.JSONDecodeError, IOError):
                    continue
            
            recent_messages.sort(key=lambda x: x[0], reverse=True)
            
            # Try ALL recent messages - no limits
            for mtime, msg_file, msg_data in recent_messages:
                msg_id = msg_data.get('id')
                if not msg_id:
                    continue
                
                # Look for parts dynamically
                for base_path in expanded_paths:
                    parts_path = base_path / "part" / msg_id
                    if parts_path.exists():
                        response = self._extract_parts_dynamic(parts_path, config)
                        if response:
                            return response
            
            return None
            
        except Exception as e:
            return None
    
    def _extract_parts_dynamic(self, parts_path: Path, config: AgentConfig) -> Optional[str]:
        """Extract content from parts directory dynamically."""
        try:
            capabilities = self.capabilities
            extraction_config = capabilities.content_extraction
            tool_patterns = capabilities.tool_patterns
            
            content_keywords = config.content_keywords or extraction_config.get("keywords", [])
            content_indicators = config.content_keywords or extraction_config.get("content_indicators", [])  # Use same as keywords if not specified
            
            text_content = []
            tool_results = []
            
            for part_file in parts_path.glob("prt_*.json"):
                try:
                    with open(part_file, 'r') as f:
                        part_data = json.load(f)
                    
                    part_type = part_data.get('type')
                    
                    # Extract text content dynamically
                    if part_type == 'text' and 'text' in part_data:
                        text_content.append(part_data['text'])
                    
                    # Extract tool results dynamically
                    elif part_type in tool_patterns:
                        tool_result = self._extract_tool_content_dynamic(part_data, content_keywords, content_indicators)
                        if tool_result:
                            tool_results.append(tool_result)
                            
                except (json.JSONDecodeError, IOError) as e:
                    continue
            
            # Combine results dynamically
            if text_content:
                response = '\n'.join(text_content)
                if tool_results:
                    response += '\n\n' + '\n'.join(tool_results)
                return response
            elif tool_results:
                return '\n'.join(tool_results)
            
            return None
            
        except Exception as e:
            return None
    
    def _extract_tool_content_dynamic(self, part_data: dict, keywords: List[str], indicators: List[str]) -> Optional[str]:
        """Extract tool content dynamically based on keywords and indicators."""
        try:
            tool_name = part_data.get('tool', '')
            state = part_data.get('state', {})
            
            if state.get('status') == 'completed' and 'output' in state:
                output = state['output']
                
                # If keywords/indicators provided, use them for filtering
                if keywords or indicators:
                    if isinstance(output, str):
                        if any(indicator in output for indicator in indicators) or any(keyword in output.lower() for keyword in keywords):
                            return f"Tool {tool_name}: {output}"
                else:
                    # No filtering - extract any meaningful content
                    if isinstance(output, str) and output.strip():
                        return f"Tool {tool_name}: {output}"
                
                # Try JSON parsing for structured data
                try:
                    output_data = json.loads(output)
                    if isinstance(output_data, dict):
                        extracted = self._extract_structured_content_dynamic(output_data, keywords, indicators)
                        if extracted:
                            return f"Tool {tool_name}: {extracted}"
                except json.JSONDecodeError:
                    pass
            
            return None
            
        except Exception as e:
            return None
    
    def _extract_structured_content_dynamic(self, data: dict, keywords: List[str], indicators: List[str]) -> Optional[str]:
        """Extract structured content dynamically."""
        try:
            # Look for content with keywords/indicators if provided
            if keywords or indicators:
                if 'content' in data and isinstance(data['content'], list):
                    for item in data['content']:
                        if isinstance(item, dict) and 'text' in item:
                            text = item['text']
                            if (keywords and any(keyword in text.lower() for keyword in keywords)) or \
                               (indicators and any(indicator in text for indicator in indicators)):
                                return text
            else:
                # No keywords - extract any meaningful content
                if 'content' in data and isinstance(data['content'], list):
                    for item in data['content']:
                        if isinstance(item, dict) and 'text' in item:
                            text = item['text']
                            if text.strip():  # Any non-empty text
                                return text
            
            # Extract any structured data dynamically
            result_parts = []
            for key, value in data.items():
                if key == 'content':  # Already processed above
                    continue
                    
                # If keywords provided, filter by them
                if keywords and not any(keyword in key.lower() for keyword in keywords):
                    continue
                    
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (str, int, float)) and str(subvalue).strip():
                            result_parts.append(f"{subkey}: {subvalue}")
                elif isinstance(value, (str, int, float)) and str(value).strip():
                    result_parts.append(f"{key}: {value}")
            
            if result_parts:
                return ', '.join(result_parts)  # No limit - return complete output
                
            return None
            
        except Exception:
            return None
    
    def _ensure_uuid_format(self, base_session_id: str) -> str:
        """
        Ensure consistent UUID format for Claude Code sessions.
        
        Args:
            base_session_id: Base session identifier
            
        Returns:
            Valid UUID string
        """
        try:
            # If already a valid UUID, use it
            uuid.UUID(base_session_id)
            return base_session_id
        except ValueError:
            # Convert base_session_id to deterministic UUID
            namespace = uuid.UUID('12345678-1234-5678-9abc-123456789abc')
            return str(uuid.uuid5(namespace, base_session_id))
    
    def _ensure_opencode_session(self, session_id: str) -> None:
        """
        Create OpenCode session file if it doesn't exist.
        
        Args:
            session_id: Session identifier for OpenCode
        """
        # Get project-specific path
        cwd = os.getcwd().replace('/', '-').lstrip('-')
        session_dir = Path.home() / ".local/share/opencode/project" / cwd / "storage/session/info"
        session_file = session_dir / f"{session_id}.json"
        
        if not session_file.exists():
            session_dir.mkdir(parents=True, exist_ok=True)
            session_data = {
                "id": session_id,
                "version": "0.5.29",
                "title": f"Evaluation session {session_id}",
                "time": {
                    "created": int(time.time() * 1000),
                    "updated": int(time.time() * 1000)
                }
            }
            session_file.write_text(json.dumps(session_data, indent=2))
    
    def execute_with_session_management(
        self,
        prompt: str,
        base_session_id: Optional[str] = None,
        continue_conversation: bool = False,
        **kwargs
    ) -> AgentResponse:
        """
        High-level session management for both agents with SAME session ID strategy.
        
        Args:
            prompt: The prompt to execute
            base_session_id: Base session ID (same for both agents)
            continue_conversation: Whether to continue existing conversation
            **kwargs: Additional agent-specific parameters
            
        Returns:
            AgentResponse with session tracking
        """
        # Remove timeout from kwargs if present
        kwargs.pop("timeout", None)  # Remove timeout
        model = kwargs.pop("model", self.model)
        
        config = AgentConfig(
            type=self.agent_type,
            model=model,
            **kwargs
        )
        
        capabilities = self.capabilities
        session_mgmt = capabilities.session_management
        
        if session_mgmt.get("supports_resume") and continue_conversation and base_session_id:
            # Resume existing session using the base_session_id
            config.resume_session_id = base_session_id
        elif session_mgmt.get("supports_session_id") and base_session_id:
            # Use base_session_id for new session
            config.session_id = base_session_id
        elif session_mgmt.get("supports_continue"):
            # Use continue session flag
            config.continue_session = True
                
        return self.execute(prompt, config)
    
    def generate_session_id(self, prompt_id: int, timestamp: Optional[int] = None) -> str:
        """
        Generate consistent session ID format for evaluation.
        
        Args:
            prompt_id: Prompt identifier
            timestamp: Unix timestamp (defaults to current time)
            
        Returns:
            Formatted session ID: eval_prompt{id:03d}_{timestamp}
        """
        if timestamp is None:
            timestamp = int(time.time())
        return f"eval_prompt{prompt_id:03d}_{timestamp}"


def create_agent(agent_type: str, model: Optional[str] = None) -> UnifiedAgent:
    """
    Factory function to create unified agents.
    
    Args:
        agent_type: "claude" or "opencode"
        model: Model to use
        
    Returns:
        Configured UnifiedAgent instance
    """
    return UnifiedAgent(agent_type, model)


# Predefined agent configurations
CLAUDE_SONNET = lambda: create_agent("claude", "sonnet")
OPENCODE_CLAUDE = lambda: create_agent("opencode", "github-copilot/claude-3.5-sonnet")
