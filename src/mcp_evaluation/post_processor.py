"""
MCP Evaluation Post Processor

This module reads monitoring data from hooks/plugins in InfluxDB and generates reports.
Post processor for reading MCP evaluation monitoring data.

It works with the actual data structure: claude_code_events and opencode_events.
"""

import json
import logging
import re
from typing import Optional
import re
import hashlib
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# InfluxDB client
try:
    from influxdb_client import InfluxDBClient
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_influxdb_config() -> Dict[str, str]:
    """Load InfluxDB configuration from unified .env file."""
    config = {}
    
    # Check for unified .env file in project root
    unified_env = Path(".env")
    if unified_env.exists():
        with open(unified_env, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    config[key] = value.strip('"\'')
    
    return {
        "INFLUXDB_URL": config.get("INFLUXDB_URL", "http://localhost:8086"),
        "INFLUXDB_TOKEN": config.get("INFLUXDB_TOKEN", "admin-token"),
        "INFLUXDB_ORG": config.get("INFLUXDB_ORG", "admin-org"),
        "INFLUXDB_BUCKET": config.get("INFLUXDB_BUCKET", "evaluation-sessions"),
        # Post-processing specific configuration
        "OUTPUT_DIR": config.get("POST_PROCESSING_OUTPUT_DIR", "reports"),
        "TIME_WINDOW_HOURS": int(config.get("POST_PROCESSING_TIME_WINDOW_HOURS", "24")),
        "SESSION_LOOKUP_DAYS": int(config.get("POST_PROCESSING_SESSION_LOOKUP_DAYS", "30")),
        "MIN_EXECUTION_TIME_SECONDS": float(config.get("POST_PROCESSING_MIN_EXECUTION_TIME_SECONDS", "0.1")),
        "SOURCE_APP": config.get("POST_PROCESSING_SOURCE_APP", "mcp_evaluation"),
        "CLAUDE_MEASUREMENT": config.get("POST_PROCESSING_CLAUDE_MEASUREMENT", "claude_code_events"),
        "OPENCODE_MEASUREMENT": config.get("POST_PROCESSING_OPENCODE_MEASUREMENT", "opencode_events"),
        # Agent event types
        "OPENCODE_TOOL_START": config.get("POST_PROCESSING_OPENCODE_TOOL_START", "tool.execute.before"),
        "OPENCODE_TOOL_END": config.get("POST_PROCESSING_OPENCODE_TOOL_END", "tool.execute.after"),
        "CLAUDE_TOOL_START": config.get("POST_PROCESSING_CLAUDE_TOOL_START", "PreToolUse"),
        "CLAUDE_TOOL_END": config.get("POST_PROCESSING_CLAUDE_TOOL_END", "PostToolUse"),
        # Error message templates
        "ERROR_NO_TOOLS": config.get("POST_PROCESSING_ERROR_NO_TOOLS", "No tool calls executed - session completed without using any tools"),
        "ERROR_INCOMPLETE_TOOLS": config.get("POST_PROCESSING_ERROR_INCOMPLETE_TOOLS", "Tool calls initiated but no tools completed successfully"),
        "ERROR_GENERAL_FAILURE": config.get("POST_PROCESSING_ERROR_GENERAL_FAILURE", "Session failed despite tool execution")
    }


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for a session."""
    number: Optional[int] = None  # Sequential record number
    prompt: Optional[str] = None  # Prompt ID used
    session_id: str = ""  # Unique session identifier
    agent_type: str = ""  # Agent used (claude/opencode)
    model: Optional[str] = None  # Specific model used
    success: bool = False  # Evaluation success status
    execution_time: float = 0.0  # Duration in seconds
    number_of_calls: int = 0  # Total API/system calls
    number_of_tool_calls: int = 0  # MCP tool invocations
    tools_used: List[str] = None  # Tools and call counts
    cost_usd: Optional[float] = None  # Cost in USD (Claude only)
    response_length: int = 0  # Response text length
    created_at: str = ""  # Session start timestamp
    completed_at: str = ""  # Session end timestamp
    logfile: str = ""  # Path to communication log
    error_message: Optional[str] = None  # Error details if failed

    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []


@dataclass
class MonitoringSession:
    """Represents a monitoring session from hooks/plugins."""
    session_id: str
    agent_type: str
    start_time: datetime
    end_time: datetime
    events: List[Dict[str, Any]]
    tools_used: List[str]
    event_count: int


def safe_json_loads(data):
    """Safely load JSON data."""
    if isinstance(data, str):
        try:
            return json.loads(data)
        except:
            return {}
    return data if isinstance(data, dict) else {}


class PostProcessor:
    """Simple processor for InfluxDB monitoring data."""
    
    # Fallback defaults (used only if .env values fail to load)
    FALLBACK_TIME_WINDOW_HOURS = 24
    FALLBACK_SESSION_LOOKUP_DAYS = 30
    FALLBACK_MIN_EXECUTION_TIME_SECONDS = 0.1
    FALLBACK_OUTPUT_DIR = "reports/"
    FALLBACK_SOURCE_APP = "mcp_evaluation"
    FALLBACK_CLAUDE_MEASUREMENT = "claude_code_events"
    FALLBACK_OPENCODE_MEASUREMENT = "opencode_events"
    FALLBACK_OPENCODE_TOOL_START = "tool.execute.before"
    FALLBACK_OPENCODE_TOOL_END = "tool.execute.after"
    FALLBACK_CLAUDE_TOOL_START = "PreToolUse"
    FALLBACK_CLAUDE_TOOL_END = "PostToolUse"
    
    # Dynamic keyword detection patterns (not hardcoded to specific prompts)
    USER_INTERACTION_PATTERNS = [
        'show', 'get', 'tell', 'what', 'how', 'display', 'information', 'help'
    ]
    
    SUCCESS_INDICATORS = [
        '✅', 'success', 'completed successfully', 'operation completed',
        'status": "success"', 'finished', 'done', '"success":true'
    ]
    
    ERROR_INDICATORS = [
        '❌', 'error', 'failed', 'exception', 'traceback',
        'status": "error"', 'status": "failed"', '"success":false'
    ]
    
    @staticmethod
    def extract_prompt_id_from_content(content: str) -> Optional[int]:
        """
        Extract prompt ID from injected HTML comment in content.
        
        Args:
            content: Content that may contain injected prompt ID
            
        Returns:
            Prompt ID if found, None otherwise
        """
        if not content:
            return None
            
        # Look for HTML comment with prompt ID: <!-- EVAL_PROMPT_ID:123 -->
        pattern = r'<!--\s*EVAL_PROMPT_ID:(\d+)\s*-->'
        match = re.search(pattern, content)
        
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
                
        return None
    
    @staticmethod
    def extract_model_from_content(content: str) -> Optional[str]:
        """
        Extract model name from injected HTML comment in content.
        
        Args:
            content: Content that may contain injected model name
            
        Returns:
            Model name if found, None otherwise
        """
        if not content:
            return None
            
        # Look for HTML comment with model name: <!-- EVAL_MODEL:github-copilot/claude-3.5-sonnet -->
        # Updated pattern to handle model names with slashes, dashes, dots, and numbers
        pattern = r'<!--\s*EVAL_MODEL:([^>\s]+)\s*-->'
        match = re.search(pattern, content)
        
        if match:
            return match.group(1).strip()
                
        return None
    
    def __init__(self, output_dir: str = None, time_window_hours: int = None):
        """Initialize the processor."""
        if not INFLUXDB_AVAILABLE:
            raise ImportError("InfluxDB client not available. Install with: pip install influxdb-client")
        
        # Load config first
        self.config = load_influxdb_config()
        
        # Use provided values, then config values, then fallbacks
        self.output_dir = Path(output_dir or self.config.get("OUTPUT_DIR", self.FALLBACK_OUTPUT_DIR))
        self.output_dir.mkdir(exist_ok=True)
        
        # Use provided time window or config value or fallback
        self.time_window_hours = time_window_hours or self.config.get("TIME_WINDOW_HOURS", self.FALLBACK_TIME_WINDOW_HOURS)
        
        # Connect to InfluxDB
        self.client = InfluxDBClient(
            url=self.config["INFLUXDB_URL"],
            token=self.config["INFLUXDB_TOKEN"],
            org=self.config["INFLUXDB_ORG"]
        )
        self.query_api = self.client.query_api()
        
        print(f"✅ Connected to InfluxDB at {self.config['INFLUXDB_URL']}")
        print(f"📊 Bucket: {self.config['INFLUXDB_BUCKET']}")
        print(f"🏢 Org: {self.config['INFLUXDB_ORG']}")
        print(f"⏰ Time window: {self.time_window_hours} hours")

    def extract_claude_sessions(self) -> List[MonitoringSession]:
        """Extract Claude monitoring sessions."""
        print("🔍 Extracting Claude sessions...")
        
        # Query to get all unique session IDs
        sessions_query = f'''
        from(bucket: "{self.config['INFLUXDB_BUCKET']}")
          |> range(start: -{self.config.get("SESSION_LOOKUP_DAYS", self.FALLBACK_SESSION_LOOKUP_DAYS)}d)
          |> filter(fn: (r) => r._measurement == "{self.config.get("CLAUDE_MEASUREMENT", self.FALLBACK_CLAUDE_MEASUREMENT)}")
          |> filter(fn: (r) => r.source_app == "{self.config.get("SOURCE_APP", self.FALLBACK_SOURCE_APP)}")
          |> keep(columns: ["session_id"])
          |> distinct(column: "session_id")
        '''
        
        sessions_result = self.query_api.query(org=self.config['INFLUXDB_ORG'], query=sessions_query)
        session_ids = []
        for table in sessions_result:
            for record in table.records:
                session_id = record.values.get('session_id')
                if session_id and session_id not in session_ids:
                    session_ids.append(session_id)
        
        print(f"Found {len(session_ids)} Claude sessions")
        
        all_sessions = []
        
        # Process each session
        for session_id in session_ids:
            session_data = self.get_session_data(session_id)
            if session_data:
                all_sessions.append(session_data)
        
        return all_sessions
    
    def get_session_data(self, session_id: str) -> Optional[MonitoringSession]:
        """Get complete data for a specific session."""
        
        # Query to get all events for this specific session
        query = f'''
        from(bucket: "{self.config['INFLUXDB_BUCKET']}")
          |> range(start: -{self.config.get("SESSION_LOOKUP_DAYS", self.FALLBACK_SESSION_LOOKUP_DAYS)}d)
          |> filter(fn: (r) => r._measurement == "{self.config.get("CLAUDE_MEASUREMENT", self.FALLBACK_CLAUDE_MEASUREMENT)}")
          |> filter(fn: (r) => r.session_id == "{session_id}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        '''
        
        result = self.query_api.query(org=self.config['INFLUXDB_ORG'], query=query)
        
        events = []
        tools_used = set()
        start_time = None
        end_time = None
        
        for table in result:
            for record in table.records:
                time = record.get_time()
                hook_event_type = record.values.get('hook_event_type', 'unknown')
                
                # Track time range
                if start_time is None or time < start_time:
                    start_time = time
                if end_time is None or time > end_time:
                    end_time = time
                
                # Get all field values for this timestamp/event
                payload_data = record.values.get('payload', '')
                chat_data = record.values.get('chat_data', '')
                summary_data = record.values.get('summary', '')
                
                # Parse payload if available
                payload = {}
                if payload_data:
                    try:
                        payload = json.loads(payload_data)
                        tool_name = payload.get('tool_name', '')
                        if tool_name:
                            tools_used.add(tool_name)
                    except:
                        pass
                
                # Create event with all available data
                event = {
                    'timestamp': time,
                    'type': hook_event_type,
                    'tool_name': payload.get('tool_name', '') if payload else '',
                    'payload': payload,
                    'payload_data': payload_data,
                    'chat_data': chat_data,
                    'summary_data': summary_data
                }
                events.append(event)
        
        if not events:
            return None
        
        return MonitoringSession(
            session_id=session_id,
            agent_type='claude',
            start_time=start_time or datetime.now(),
            end_time=end_time or datetime.now(),
            events=events,
            tools_used=list(tools_used),
            event_count=len(events)
        )

    def get_session_directory(self, agent_type: str, session_id: str) -> Path:
        """Create and return the directory path for a specific session."""
        session_dir = self.output_dir / agent_type / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def calculate_session_metrics(self, session: MonitoringSession, log_content: str, record_number: int) -> EvaluationMetrics:
        """Calculate comprehensive metrics for a single session using InfluxDB data."""
        
        metrics = EvaluationMetrics()
        
        # 1. Sequential record number
        metrics.number = record_number
        
        # 2. Prompt ID - extract from session attributes or content
        prompt_id = None
        if hasattr(session, 'prompt_id') and session.prompt_id:
            prompt_id = session.prompt_id
        else:
            # Try to extract prompt ID from injected content in messages
            for event in session.events:
                # Check different content locations based on agent type
                content_sources = []
                
                # For message content (OpenCode)
                message_content = event.get('message_content', {})
                if isinstance(message_content, dict):
                    part_info = message_content.get('properties', {}).get('part', {})
                    # Try both 'text' and 'content' fields
                    part_content = part_info.get('text', '') or part_info.get('content', '')
                    if part_content:
                        content_sources.append(part_content)
                
                # For direct content fields 
                if event.get('content'):
                    content_sources.append(str(event['content']))
                
                # For prompt field in events
                if event.get('prompt'):
                    content_sources.append(str(event['prompt']))
                
                # For Claude-specific fields
                if event.get('payload_data'):
                    content_sources.append(str(event['payload_data']))
                if event.get('chat_data'):
                    content_sources.append(str(event['chat_data']))
                if event.get('summary_data'):
                    content_sources.append(str(event['summary_data']))
                
                # For Claude payload content
                if event.get('payload') and isinstance(event['payload'], dict):
                    # Check various payload fields that might contain prompt content
                    for key, value in event['payload'].items():
                        if isinstance(value, str) and value:
                            content_sources.append(value)
                
                # Try to extract prompt ID from any content source
                for content in content_sources:
                    if isinstance(content, str):
                        extracted_id = self.extract_prompt_id_from_content(content)
                        if extracted_id:
                            prompt_id = extracted_id
                            print(f"   🎯 Extracted prompt ID {prompt_id} from content")
                            break
                
                if prompt_id:
                    break
        
        metrics.prompt = str(prompt_id) if prompt_id else None
        
        # 3. Session ID - direct from InfluxDB
        metrics.session_id = session.session_id
        
        # 4. Agent type - direct from InfluxDB
        metrics.agent_type = session.agent_type
        
        # 5. Model - extract from InfluxDB events or injected content
        model_name = None
        if hasattr(session, 'model_info') and session.model_info:
            model_name = session.model_info
        elif hasattr(session, 'model') and session.model:
            model_name = session.model
        else:
            # Try to extract model name from injected content in messages
            for event in session.events:
                # Check different content locations based on agent type
                content_sources = []
                
                # For OpenCode message content
                message_content = event.get('message_content', {})
                if isinstance(message_content, dict):
                    part_info = message_content.get('properties', {}).get('part', {})
                    # Try both 'text' and 'content' fields
                    part_content = part_info.get('text', '') or part_info.get('content', '')
                    if part_content:
                        content_sources.append(part_content)
                
                # For Claude-specific fields
                if event.get('payload_data'):
                    content_sources.append(str(event['payload_data']))
                if event.get('chat_data'):
                    content_sources.append(str(event['chat_data']))
                if event.get('summary_data'):
                    content_sources.append(str(event['summary_data']))
                
                # For Claude payload content
                if event.get('payload') and isinstance(event['payload'], dict):
                    for key, value in event['payload'].items():
                        if isinstance(value, str) and value:
                            content_sources.append(value)
                
                # Try to extract model name from any content source
                for content in content_sources:
                    if isinstance(content, str):
                        extracted_model = self.extract_model_from_content(content)
                        if extracted_model:
                            model_name = extracted_model
                            print(f"   🎯 Extracted model {model_name} from content")
                            break
                
                if model_name:
                    break
        
        metrics.model = model_name
        
        # 6. Success - extract from InfluxDB events
        if hasattr(session, 'success_status'):
            metrics.success = session.success_status
        elif hasattr(session, 'success'):
            metrics.success = session.success
        else:
            # Fallback: check if any tool executions completed successfully
            # Use configurable event types for different agents
            opencode_end = self.config.get("OPENCODE_TOOL_END", self.FALLBACK_OPENCODE_TOOL_END)
            claude_end = self.config.get("CLAUDE_TOOL_END", self.FALLBACK_CLAUDE_TOOL_END)
            metrics.success = any(
                opencode_end in event.get('type', '') or 
                event.get('type', '') == claude_end
                for event in session.events
            )
        
        # 7. Execution time - calculate based on actual event timestamps
        tool_start_time = None
        tool_end_time = None
        message_start_time = None
        
        # Get actual first and last event timestamps for total session time
        sorted_events = sorted(session.events, key=lambda x: x.get('timestamp', session.start_time))
        if sorted_events:
            actual_start_time = sorted_events[0].get('timestamp', session.start_time)
            actual_end_time = sorted_events[-1].get('timestamp', session.end_time)
        else:
            actual_start_time = session.start_time
            actual_end_time = session.end_time
        
        # Find tool execution window for reference
        for event in sorted_events:
            event_type = event.get('type', '')
            
            # Track tool execution events specifically
            if event_type == 'tool.execute.before':
                if tool_start_time is None:
                    tool_start_time = event.get('timestamp')
            elif event_type == 'tool.execute.after':
                tool_end_time = event.get('timestamp')
            
            # Also check for tool timing in message parts
            try:
                message_content = event.get('message_content', {})
                part_info = message_content.get('properties', {}).get('part', {})
                
                if part_info.get('type') == 'tool':
                    state = part_info.get('state', {})
                    # Tool start (pending status)
                    if state.get('status') == 'pending':
                        if tool_start_time is None:
                            tool_start_time = event.get('timestamp')
                    # Tool completion (completed status or with result)
                    elif (state.get('status') in ['completed', 'success'] or 
                          state.get('result') is not None):
                        tool_end_time = event.get('timestamp')
            except:
                pass
                
            # Track first meaningful user interaction
            if (message_start_time is None and 
                event_type in ['message.part.updated', 'chat.message'] and
                any(keyword in str(event).lower() for keyword in self.USER_INTERACTION_PATTERNS)):
                message_start_time = event.get('timestamp')
        
        # Use total session execution time (from first to last event)
        metrics.execution_time = (actual_end_time - actual_start_time).total_seconds()
        print(f"   📊 Total session execution time: {metrics.execution_time:.3f}s")
        print(f"   📅 Session span: {actual_start_time} → {actual_end_time}")
        
        # Also show tool execution window for reference if available
        if tool_start_time and tool_end_time:
            tool_duration = (tool_end_time - tool_start_time).total_seconds()
            print(f"   🔧 Tool execution window: {tool_duration:.3f}s")
        
        # Ensure minimum reasonable time from config
        min_exec_time = self.config.get("MIN_EXECUTION_TIME_SECONDS", self.FALLBACK_MIN_EXECUTION_TIME_SECONDS)
        if metrics.execution_time < min_exec_time:
            metrics.execution_time = min_exec_time
        
        # 8. Number of tool calls - total tool invocations (MCP and regular tools)
        if hasattr(session, 'tool_calls_count') and session.tool_calls_count:
            metrics.number_of_tool_calls = session.tool_calls_count
        elif hasattr(session, 'tool_calls') and session.tool_calls:
            metrics.number_of_tool_calls = session.tool_calls
        else:
            # Count from events - look for tool execution start events
            # For OpenCode: configurable tool start events
            # For Claude: configurable tool start events
            opencode_start = self.config.get("OPENCODE_TOOL_START", self.FALLBACK_OPENCODE_TOOL_START)
            claude_start = self.config.get("CLAUDE_TOOL_START", self.FALLBACK_CLAUDE_TOOL_START)
            metrics.number_of_tool_calls = len([
                e for e in session.events 
                if e.get('type') == opencode_start or e.get('type') == claude_start
            ])
        
        # 9. Number of calls - same as tool calls (total tool invocations)
        metrics.number_of_calls = metrics.number_of_tool_calls
        
        # 10. Tools used - actual tools from InfluxDB
        if session.tools_used:
            # Count occurrences of each tool
            tool_counts = {}
            for event in session.events:
                if 'tool.execute' in event.get('type', ''):
                    tool_name = (
                        event.get('tool_data', {}).get('tool') or
                        event.get('payload', {}).get('toolName') or
                        'unknown_tool'
                    )
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            
            if tool_counts:
                metrics.tools_used = [f"{tool}:{count}" for tool, count in tool_counts.items()]
            else:
                metrics.tools_used = list(session.tools_used)  # Basic list
        else:
            metrics.tools_used = []
        
        # 11. Cost USD - Claude only, not available in current data
        metrics.cost_usd = None
        
        # 12. Response length - extract from InfluxDB
        if hasattr(session, 'response_length'):
            metrics.response_length = session.response_length
        else:
            # Extract from message content in events
            total_length = 0
            max_response_length = 0
            
            for event in session.events:
                # Check different content locations based on agent type
                content_sources = []
                
                # For OpenCode message content
                message_content = event.get('message_content', {})
                if isinstance(message_content, dict):
                    part_info = message_content.get('properties', {}).get('part', {})
                    # Try both 'text' and 'content' fields
                    part_content = part_info.get('text', '') or part_info.get('content', '')
                    if part_content:
                        content_sources.append(part_content)
                
                # For Claude-specific fields
                if event.get('payload_data'):
                    content_sources.append(str(event['payload_data']))
                if event.get('chat_data'):
                    content_sources.append(str(event['chat_data']))
                if event.get('summary_data'):
                    content_sources.append(str(event['summary_data']))
                
                # For Claude payload content
                if event.get('payload') and isinstance(event['payload'], dict):
                    for key, value in event['payload'].items():
                        if isinstance(value, str) and value:
                            content_sources.append(value)
                
                # Find the longest content (final response)
                for content in content_sources:
                    if isinstance(content, str) and len(content) > max_response_length:
                        # Skip prompt injection content when calculating response length
                        if not content.startswith('<!-- EVAL_PROMPT_ID:'):
                            max_response_length = len(content)
            
            metrics.response_length = max_response_length
        
        # 13. Created at - session start
        metrics.created_at = session.start_time.isoformat()
        
        # 14. Completed at - session end  
        metrics.completed_at = session.end_time.isoformat()
        
        # 15. Log file path - absolute path
        log_path = self.output_dir / f"{session.agent_type}_{session.session_id}_monitoring.log"
        metrics.logfile = str(log_path.absolute())
        
        # 16. Error message - extract from InfluxDB if available
        if hasattr(session, 'error_message') and session.error_message:
            metrics.error_message = session.error_message
        else:
            # Check events for explicit errors
            error_events = [
                e for e in session.events 
                if 'error' in e.get('type', '').lower() or e.get('payload', {}).get('error')
            ]
            if error_events:
                metrics.error_message = str(error_events[0].get('payload', {}).get('error', 'Unknown error'))
            elif not metrics.success:
                # Provide descriptive error message for failed sessions without explicit errors
                if metrics.number_of_tool_calls == 0:
                    metrics.error_message = self.config.get("ERROR_NO_TOOLS", 
                        "No tool calls executed - session completed without using any tools")
                elif len(session.tools_used) == 0:
                    metrics.error_message = self.config.get("ERROR_INCOMPLETE_TOOLS", 
                        "Tool calls initiated but no tools completed successfully")
                else:
                    metrics.error_message = self.config.get("ERROR_GENERAL_FAILURE", 
                        "Session failed despite tool execution")
            else:
                metrics.error_message = None
        
        return metrics
    
    def _extract_prompt_info(self, log_content: str) -> Optional[str]:
        """Extract prompt information from log - leave blank if not directly available as ID."""
        # User specified: "dont create any fake id leave it blank"
        # Prompt text exists in InfluxDB but no prompt ID field exists, so leave blank
        return None
    
    def _count_api_calls(self, log_content: str) -> int:
        """Count API/system calls from log events."""
        api_call_patterns = ['PreToolUse', 'PostToolUse', 'UserPromptSubmit']
        count = 0
        for line in log_content.split('\n'):
            if 'Type:' in line:
                for pattern in api_call_patterns:
                    if pattern in line:
                        count += 1
                        break
        return count
    
    def _calculate_response_length(self, log_content: str) -> int:
        """Calculate total response length."""
        tool_response_pattern = r'tool_response: \[.*?\]'
        matches = re.findall(tool_response_pattern, log_content, re.DOTALL)
        
        total_length = 0
        for match in matches:
            try:
                json_match = re.search(r'\[.*\]', match, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group(0))
                    for item in response_data:
                        if isinstance(item, dict) and 'text' in item:
                            total_length += len(item['text'])
            except:
                total_length += len(match)
        
        return total_length
    
    def _determine_success(self, log_content: str) -> bool:
        """Determine if session was successful."""
        log_lower = log_content.lower()
        
        # Check for explicit error indicators first
        for indicator in self.ERROR_INDICATORS:
            if indicator.lower() in log_lower:
                return False
        
        # Check for success indicators
        for indicator in self.SUCCESS_INDICATORS:
            if indicator.lower() in log_lower:
                return True
            if indicator in log_lower:
                return True
        
        # If tool responses are present and no errors, consider successful
        if 'tool_response:' in log_content and 'content' in log_content:
            return True
        
        return False
    
    def _extract_error_message(self, log_content: str) -> Optional[str]:
        """Extract error message if present."""
        error_patterns = [
            r'error[:\s]+(.+)',
            r'failed[:\s]+(.+)',
            r'exception[:\s]+(.+)'
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, log_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_model_info(self, agent_type: str) -> Optional[str]:
        """Extract model information - only if available in InfluxDB data."""
        # Model info is not captured in current hook data, so return None
        return None
    
    def _estimate_claude_cost(self, response_length: int, api_calls: int) -> Optional[float]:
        """Estimate Claude API cost - only if cost data is available in InfluxDB."""
        # Cost data is not captured in current hook data, so return None
        return None

    def _extract_opencode_sessions(self) -> List[MonitoringSession]:
        """Extract ALL OpenCode data and parse for evaluation metrics."""
        print("🔍 Reading ALL OpenCode data from InfluxDB...")
        try:
            # Get data from configurable time window
            query = f'''
                from(bucket: "{self.config['INFLUXDB_BUCKET']}")
                |> range(start: -{self.time_window_hours}h)
                |> filter(fn: (r) => r._measurement == "{self.config.get("OPENCODE_MEASUREMENT", self.FALLBACK_OPENCODE_MEASUREMENT)}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: false)
            '''
            
            print(f"🔍 Query: {query}")  # Debug the query
            
            result = self.query_api.query(org=self.config['INFLUXDB_ORG'], query=query)
            
            # Collect ALL raw data first
            all_events = []
            unique_session_ids = set()
            unique_event_types = set()
            
            for table in result:
                for record in table.records:
                    event_data = {
                        'timestamp': record.get_time(),
                        'payload_raw': record.values.get('payload', '{}'),
                        'tool_data_raw': record.values.get('tool_data', '{}'),
                        'message_content_raw': record.values.get('message_content', '{}'),
                        'sessionId_raw': record.values.get('sessionId', ''),
                        'call_id': record.values.get('call_id', ''),
                        'event_type': record.values.get('event_type', ''),
                        'source_app': record.values.get('source_app', ''),
                        'tool_name': record.values.get('tool_name', ''),
                        'session_id': record.values.get('session_id', ''),  # Add this field
                    }
                    all_events.append(event_data)
                    
                    # Collect unique values for analysis
                    unique_event_types.add(event_data.get('event_type', ''))
                    
                    # Try to extract session IDs from various fields
                    payload = safe_json_loads(event_data['payload_raw'])
                    if payload.get('event', {}).get('properties', {}).get('sessionID'):
                        unique_session_ids.add(payload['event']['properties']['sessionID'])
                    if event_data.get('sessionId_raw'):
                        unique_session_ids.add(event_data['sessionId_raw'])
                    if event_data.get('session_id'):
                        unique_session_ids.add(event_data['session_id'])
            
            print(f"   📊 Total OpenCode events found: {len(all_events)}")
            print(f"   📊 Unique event types: {sorted(unique_event_types)}")
            print(f"   📊 Unique session IDs found: {len(unique_session_ids)}")
            for sid in sorted(unique_session_ids):
                print(f"      - {sid}")
            
            # Let's examine the first few events to understand the data structure
            print(f"   🔍 Sample events (first 3):")
            for i, event in enumerate(all_events[:3]):
                print(f"      Event {i+1}:")
                print(f"         timestamp: {event['timestamp']}")
                print(f"         event_type: {event['event_type']}")
                print(f"         sessionId_raw: {event['sessionId_raw']}")
                print(f"         session_id: {event['session_id']}")
                payload = safe_json_loads(event['payload_raw'])
                session_from_payload = payload.get('event', {}).get('properties', {}).get('sessionID', 'none')
                print(f"         sessionID from payload: {session_from_payload}")
                print()
            
            # Now let's analyze the session distribution
            print(f"   🔍 Analyzing session distribution...")
            
            # Parse and group by actual OpenCode session IDs first
            sessions_data = {}
            evaluation_sessions = []  # Track evaluation sessions specifically
            
            # First pass: Group events by actual OpenCode session IDs
            print("   🔍 Grouping events by actual OpenCode session IDs...")
            
            session_events = {}  # session_id -> list of events
            
            for event_data in all_events:
                try:
                    # Parse JSON fields
                    payload = safe_json_loads(event_data['payload_raw'])
                    tool_data = safe_json_loads(event_data['tool_data_raw'])
                    message_content = safe_json_loads(event_data['message_content_raw'])
                    
                    # Extract session ID from the event
                    session_id = None
                    
                    # Priority order for session ID extraction
                    session_sources = [
                        # 1. Direct sessionId fields
                        ('payload.sessionId', payload.get('sessionId')),
                        ('raw.sessionId', event_data.get('sessionId_raw')),
                        
                        # 2. Event properties sessionID (capital ID) - most reliable for OpenCode
                        ('event.properties.sessionID', payload.get('event', {}).get('properties', {}).get('sessionID')),
                        
                        # 3. Message content sessionID 
                        ('message.properties.sessionID', message_content.get('properties', {}).get('part', {}).get('sessionID')),
                        
                        # 4. Tool data sessionId
                        ('tool_data.sessionId', tool_data.get('sessionId')),
                    ]
                    
                    # Find first valid session ID
                    for source, sid in session_sources:
                        if sid and sid not in ['unknown', 'startup', '']:
                            session_id = sid
                            break
                    
                    if session_id:
                        if session_id not in session_events:
                            session_events[session_id] = []
                        session_events[session_id].append(event_data)
                
                except Exception as e:
                    print(f"   ⚠️  Error parsing event: {e}")
                    continue
            
            print(f"   📊 Found {len(session_events)} distinct OpenCode sessions")
            
            # Second pass: Identify evaluation sessions (those with meaningful interactions)
            import datetime
            recent_cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=self.time_window_hours)
            
            for session_id, events in session_events.items():
                # Check if this session has evaluation-worthy content
                has_user_message = False
                has_tool_execution = False
                session_start = None
                session_end = None
                
                print(f"   🔍 Analyzing session {session_id} with {len(events)} events")
                
                # First, let's see what unique event types we have
                event_types = set()
                for event_data in events:
                    event_types.add(event_data.get('event_type', ''))
                print(f"      📊 Unique event types: {sorted(event_types)}")
                
                for event_data in events:
                    timestamp = event_data['timestamp']
                    if session_start is None or timestamp < session_start:
                        session_start = timestamp
                    if session_end is None or timestamp > session_end:
                        session_end = timestamp
                    
                    event_type = event_data.get('event_type', '')
                    
                    # Check for user messages with meaningful content
                    if ('message.part.updated' in event_type or
                        'chat.message' in event_type):
                        content_str = str(event_data).lower()
                        if any(keyword in content_str for keyword in self.USER_INTERACTION_PATTERNS):
                            has_user_message = True
                            print(f"      ✅ User message found: {event_type}")
                    
                    # Check for tool execution - look for actual tool events AND tool data within messages
                    if (event_type in ['tool.execute.before', 'tool.execute.after'] or
                        'tool.execute' in event_type or
                        event_data.get('tool_name')):
                        has_tool_execution = True
                        tool_name = event_data.get('tool_name', 'unknown_tool')
                        print(f"      ✅ Tool execution found: {event_type} - {tool_name}")
                        break
                    
                    # Also check within message content for tool information
                    try:
                        payload = safe_json_loads(event_data['payload_raw'])
                        message_content = safe_json_loads(event_data['message_content_raw'])
                        
                        # Check if message contains tool execution info
                        part_info = message_content.get('properties', {}).get('part', {})
                        if (part_info.get('type') == 'tool' or 
                            part_info.get('tool') or
                            part_info.get('callID')):
                            has_tool_execution = True
                            tool_name = part_info.get('tool', 'unknown_tool')
                            print(f"      ✅ Tool execution found in message: {tool_name} - {part_info.get('callID', 'no_call_id')}")
                            break
                    except:
                        pass
                
                # Only include sessions that are recent and have both user interaction and tool usage
                if (session_start and session_start > recent_cutoff and 
                    has_user_message and has_tool_execution):
                    
                    print(f"   🎯 Evaluation session identified: {session_id}")
                    print(f"      📅 Duration: {session_start} → {session_end}")
                    print(f"      📝 User message: {has_user_message}, 🔧 Tool execution: {has_tool_execution}")
                    
                    evaluation_sessions.append(session_id)
                    
                    # Process this session
                    for event_data in events:
                        try:
                            # Parse JSON fields
                            payload = safe_json_loads(event_data['payload_raw'])
                            tool_data = safe_json_loads(event_data['tool_data_raw'])
                            message_content = safe_json_loads(event_data['message_content_raw'])
                            
                            # Initialize session if not exists
                            if session_id not in sessions_data:
                                sessions_data[session_id] = {
                                    'session_id': session_id,
                                    'agent_type': 'opencode',
                                    'start_time': session_start,
                                    'end_time': session_end,
                                    'events': [],
                                    'tools_used': set(),
                                    'event_count': 0,
                                }
                            
                            # Update session data
                            session = sessions_data[session_id]
                            
                            # Update session end_time with latest event timestamp
                            event_timestamp = event_data['timestamp']
                            if session['end_time'] is None or event_timestamp > session['end_time']:
                                session['end_time'] = event_timestamp
                            
                            # Extract detailed information from each event
                            event_info = {
                                'timestamp': event_data['timestamp'],
                                'type': payload.get('event', {}).get('type', event_data.get('event_type', 'unknown')),
                                'tool_data': tool_data,
                                'message_content': message_content,
                                'payload': payload
                            }
                            session['events'].append(event_info)
                            
                            # Extract specific data based on event type
                            event_type = event_info['type']
                            
                            # Tool usage detection - look for actual tool events AND embedded tool data
                            if event_type in ['tool.execute.before', 'tool.execute.after']:
                                tool_name = (tool_data.get('tool') or 
                                           payload.get('toolName') or 
                                           event_data.get('tool_name') or 
                                           'unknown_tool')
                                session['tools_used'].add(tool_name)
                                
                                # Count tool calls properly
                                if 'tool.execute.before' in event_type:
                                    session['tool_calls'] = session.get('tool_calls', 0) + 1
                            
                            # Also check for tool info embedded in message content
                            part_info = message_content.get('properties', {}).get('part', {})
                            if part_info.get('type') == 'tool' and part_info.get('tool'):
                                tool_name = part_info.get('tool')
                                session['tools_used'].add(tool_name)
                                
                                # Count tool start events (pending status)
                                if part_info.get('state', {}).get('status') == 'pending':
                                    session['tool_calls'] = session.get('tool_calls', 0) + 1
                                    print(f"   🔧 Tool detected: {tool_name} (pending) - total calls: {session['tool_calls']}")
                                elif part_info.get('state', {}).get('status') == 'completed':
                                    print(f"   ✅ Tool completed: {tool_name}")
                                    session['success'] = True
                            
                            # Count events
                            session['event_count'] += 1
                            
                            # Model information
                            if payload.get('event', {}).get('properties', {}).get('model'):
                                session['model'] = payload['event']['properties']['model']
                            elif message_content.get('properties', {}).get('model'):
                                session['model'] = message_content['properties']['model']
                            
                            # Success detection (if tool execution completed)
                            if (event_type == 'tool.execute.after' or
                                (part_info.get('type') == 'tool' and 
                                 part_info.get('state', {}).get('status') in ['completed', 'success'])):
                                session['success'] = True
                            
                        except Exception as e:
                            print(f"   ⚠️  Error processing event in session {session_id}: {e}")
                            continue
                else:
                    # Debug why sessions are being skipped
                    print(f"   ⏭️  Skipping session {session_id}:")
                    print(f"      📅 Recent: {session_start and session_start > recent_cutoff}")
                    print(f"      📝 User message: {has_user_message}")
                    print(f"      🔧 Tool execution: {has_tool_execution}")
                    print(f"      ⏰ Start time: {session_start}")
            
            # Convert sessions_data to list format
            all_sessions = []
            for session_id, session_data in sessions_data.items():
                if session_data:
                    # Convert set to list for JSON serialization
                    session_data['tools_used'] = list(session_data['tools_used'])
                    
                    # Remove fields not expected by MonitoringSession
                    session_fields = session_data.copy()
                    tool_calls_count = session_fields.pop('tool_calls', 0)  # Save tool_calls count
                    model_info = session_fields.pop('model', None)         # Save model info
                    success_status = session_fields.pop('success', False)  # Save success status
                    
                    session = MonitoringSession(**session_fields)
                    
                    # Add custom attributes after creation
                    session.tool_calls_count = tool_calls_count
                    session.model_info = model_info
                    session.success_status = success_status
                    
                    all_sessions.append(session)
            
            print(f"   📊 Total unique sessions: {len(all_sessions)}")
            print(f"   🎯 Evaluation sessions found: {len(evaluation_sessions)}")
            for session_id in evaluation_sessions:
                print(f"      - {session_id}")
            
            print(f"   Found {len(all_sessions)} OpenCode sessions")
            return all_sessions
        
        except Exception as e:
            print(f"   ❌ Error extracting OpenCode sessions: {e}")
            return []

    def create_session_logs(self, sessions: List[MonitoringSession]) -> List[Path]:
        """Create detailed log files for each session."""
        print("📝 Creating session log files...")
        
        log_files = []
        
        for session in sessions:
            # Create session-specific directory
            session_dir = self.get_session_directory(session.agent_type, session.session_id)
            log_file = session_dir / "monitoring.log"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=== MCP Monitoring Session Log ===\n")
                f.write(f"Session ID: {session.session_id}\n")
                f.write(f"Agent Type: {session.agent_type}\n")
                f.write(f"Start Time: {session.start_time}\n")
                f.write(f"End Time: {session.end_time}\n")
                f.write(f"Duration: {(session.end_time - session.start_time).total_seconds():.2f} seconds\n")
                f.write(f"Event Count: {session.event_count}\n")
                f.write(f"Tools Used: {', '.join(session.tools_used) if session.tools_used else 'None'}\n")
                f.write("=" * 60 + "\n\n")
                
                # Write events timeline
                f.write("Event Timeline:\n")
                f.write("-" * 40 + "\n")
                
                for i, event in enumerate(session.events, 1):
                    f.write(f"\n[Event {i}] {event['timestamp']}\n")
                    f.write(f"Type: {event['type']}\n")
                    
                    if event.get('tool_name'):
                        f.write(f"Tool: {event['tool_name']}\n")
                    
                    # Dump ALL data from InfluxDB
                    f.write(f"=== COMPLETE INFLUXDB DATA ===\n")
                    
                    # Show payload data
                    payload = event.get('payload', {})
                    if payload:
                        f.write(f"--- PAYLOAD DATA ---\n")
                        for key, value in payload.items():
                            f.write(f"{key}: {value}\n")
                    
                    # Show chat_data if available
                    if 'chat_data_data' in event:
                        f.write(f"--- CHAT DATA ---\n")
                        f.write(f"chat_data: {event['chat_data_data']}\n")
                    
                    # Show summary if available  
                    if 'summary_data' in event:
                        f.write(f"--- SUMMARY DATA ---\n")
                        f.write(f"summary: {event['summary_data']}\n")
                    
                    # Show any other data fields
                    for key, value in event.items():
                        if key not in ['timestamp', 'type', 'tool_name', 'payload', 'chat_data_data', 'summary_data']:
                            f.write(f"--- {key.upper()} ---\n")
                            f.write(f"{key}: {value}\n")
                    
                    f.write(f"=== END INFLUXDB DATA ===\n")
                    
                    # Add specific details based on agent type (legacy format for compatibility)
                    if session.agent_type == 'claude':
                        if event['type'] == 'PreToolUse':
                            f.write(f"Tool Input: {payload.get('tool_input', {})}\n")
                        elif event['type'] == 'PostToolUse':
                            tool_response = payload.get('tool_response', [])
                            if tool_response:
                                # Get full tool response without truncation
                                response_text = tool_response[0].get('text', '')
                                f.write(f"Tool Response: {response_text}\n")
                        elif event['type'] == 'UserPromptSubmit':
                            f.write(f"Prompt: {payload.get('prompt', '')}\n")
                    
                    elif session.agent_type == 'opencode':
                        if event['type'] == 'tool_execution':
                            f.write(f"Tool Data: {event.get('value', '')[:100]}...\n")
                    
                    f.write("-" * 40 + "\n")
            
            log_files.append(log_file)
            print(f"   Created: {log_file}")
        
        return log_files

    def process_single_session(self, session: MonitoringSession, session_number: int) -> Dict[str, Any]:
        """Process a single session independently with its own log and metrics."""
        print(f"📊 Processing session {session_number}: {session.session_id} ({session.agent_type})")
        
        # Create session-specific directory
        session_dir = self.get_session_directory(session.agent_type, session.session_id)
        
        # Create monitoring log for this session
        log_file = session_dir / "monitoring.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=== MCP Monitoring Session Log ===\n")
            f.write(f"Session ID: {session.session_id}\n")
            f.write(f"Agent Type: {session.agent_type}\n")
            f.write(f"Start Time: {session.start_time}\n")
            f.write(f"End Time: {session.end_time}\n")
            f.write(f"Duration: {(session.end_time - session.start_time).total_seconds():.2f} seconds\n")
            f.write(f"Event Count: {session.event_count}\n")
            f.write(f"Tools Used: {', '.join(session.tools_used)}\n")
            f.write("=" * 56 + "\n\n")
            f.write("Event Timeline:\n")
            f.write("-" * 40 + "\n\n")
            
            # Write events for this session
            for i, event in enumerate(session.events, 1):
                f.write(f"[Event {i}] {event['timestamp']}\n")
                f.write(f"Type: {event.get('type', 'unknown')}\n")
                f.write("=== COMPLETE INFLUXDB DATA ===\n")
                f.write("--- TOOL_DATA ---\n")
                f.write(f"tool_data: {event.get('tool_data', {})}\n")
                f.write("--- MESSAGE_CONTENT ---\n")
                f.write(f"message_content: {event.get('message_content', {})}\n")
                f.write("=== END INFLUXDB DATA ===\n")
                f.write("-" * 40 + "\n\n")
        
        print(f"   📝 Created log: {log_file}")
        
        # Read the log content and calculate metrics for this session
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        metrics = self.calculate_session_metrics(session, log_content, session_number)
        
        # Save individual evaluation metrics file for this session
        metrics_file = session_dir / "evaluation_metrics.json"
        metrics_data = {
            'generated_at': datetime.now().isoformat(),
            'session_metrics': asdict(metrics)
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"   📊 Created metrics: {metrics_file}")
        
        # Print session summary
        print(f"   ✅ Session {session_number} complete:")
        print(f"      📍 Agent: {metrics.agent_type} | Model: {metrics.model}")
        print(f"      🕒 Duration: {metrics.execution_time:.2f}s | Tool calls: {metrics.number_of_tool_calls}")
        print(f"      📈 Success: {metrics.success}")
        
        return {
            'session_id': session.session_id,
            'agent_type': session.agent_type,
            'session_dir': session_dir,
            'log_file': log_file,
            'metrics_file': metrics_file,
            'metrics': metrics
        }

    def process_all(self) -> Dict[str, Any]:
        """Process all monitoring data and generate reports."""
        print("🚀 Processing all InfluxDB monitoring data...")
        
        # Extract all sessions
        claude_sessions = self.extract_claude_sessions()
        opencode_sessions = self._extract_opencode_sessions()
        all_sessions = claude_sessions + opencode_sessions
        
        print(f"📊 Found {len(all_sessions)} total sessions to process individually")
        
        # Process each session independently
        session_results = []
        metrics_list = []
        
        for i, session in enumerate(all_sessions, 1):
            session_result = self.process_single_session(session, i)
            session_results.append(session_result)
            metrics_list.append(session_result['metrics'])
        
        # Print metrics summary
        print("\n=== EVALUATION METRICS SUMMARY ===")
        for metric in metrics_list:
            print(f"Session {metric.number}: {metric.session_id}")
            print(f"  Agent: {metric.agent_type} | Model: {metric.model}")
            print(f"  Prompt: {metric.prompt} | Success: {metric.success}")
            print(f"  Duration: {metric.execution_time:.2f}s | Tool calls: {metric.number_of_tool_calls}")
            if metric.cost_usd:
                print(f"  Cost: ${metric.cost_usd}")
            print(f"  Response length: {metric.response_length} chars")
            if metric.error_message:
                print(f"  Error: {metric.error_message}")
            print()
        
        # Create overall results
        results = {
            'session_results': session_results,
            'total_sessions': len(all_sessions),
            'claude_sessions': len(claude_sessions),
            'opencode_sessions': len(opencode_sessions),
            'metrics': metrics_list
        }
        
        print(f"✅ Processing complete!")
        print(f"   Total sessions: {len(all_sessions)}")
        print(f"   Claude sessions: {len(claude_sessions)}")
        print(f"   OpenCode sessions: {len(opencode_sessions)}")
        print(f"   Individual session reports created: {len(session_results)}")
        print(f"   Output directory: {self.output_dir}")
        
        return results
        print(f"   Output directory: {self.output_dir}")
    
    def export_to_csv(self, output_path: Optional[str] = None, agent_filter: Optional[str] = None) -> str:
        """
        Export all evaluation metrics to a comprehensive CSV file.
        
        Args:
            output_path: Custom output path for CSV file. If None, uses default reports directory
            agent_filter: Filter by agent type ('claude' or 'opencode'). If None, exports all
            
        Returns:
            Path to the generated CSV file
        """
        print("📊 Exporting evaluation metrics to CSV...")
        
        # Determine output file path
        if output_path:
            output_file = Path(output_path)
            if output_file.is_dir():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"evaluation_metrics_{timestamp}.csv"
                if agent_filter:
                    filename = f"evaluation_metrics_{agent_filter}_{timestamp}.csv"
                output_file = output_file / filename
        else:
            # Use default reports directory
            reports_dir = Path(self.output_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_metrics_{timestamp}.csv"
            if agent_filter:
                filename = f"evaluation_metrics_{agent_filter}_{timestamp}.csv"
            output_file = reports_dir / filename
        
        # Extract sessions based on filter
        if agent_filter == 'claude':
            sessions = self.extract_claude_sessions()
            print(f"📋 Exporting {len(sessions)} Claude sessions to CSV...")
        elif agent_filter == 'opencode':
            sessions = self._extract_opencode_sessions()
            print(f"📋 Exporting {len(sessions)} OpenCode sessions to CSV...")
        else:
            claude_sessions = self.extract_claude_sessions()
            opencode_sessions = self._extract_opencode_sessions()
            sessions = claude_sessions + opencode_sessions
            print(f"📋 Exporting {len(sessions)} total sessions ({len(claude_sessions)} Claude, {len(opencode_sessions)} OpenCode) to CSV...")
        
        # Process sessions and collect metrics
        metrics_list = []
        for i, session in enumerate(sessions, 1):
            try:
                # Get log content for this session
                log_file = Path(self.output_dir) / f"{session.agent_type}_{session.session_id}_monitoring.log"
                log_content = ""
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                
                # Calculate metrics for this session
                metrics = self.calculate_session_metrics(session, log_content, i)
                metrics_list.append(metrics)
                
                print(f"   ✅ Session {i}: {session.session_id} ({session.agent_type})")
                
            except Exception as e:
                print(f"   ❌ Error processing session {session.session_id}: {e}")
                continue
        
        # Define CSV columns (matching evaluation_metrics.json structure)
        csv_columns = [
            'number', 'prompt', 'session_id', 'agent_type', 'model', 'success',
            'execution_time', 'number_of_calls', 'number_of_tool_calls', 'tools_used',
            'cost_usd', 'response_length', 'created_at', 'completed_at', 'logfile', 'error_message'
        ]
        
        # Write CSV file
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                
                for metrics in metrics_list:
                    # Convert metrics to dict and prepare for CSV
                    row = asdict(metrics)
                    
                    # Handle list fields (convert to string)
                    if row['tools_used']:
                        row['tools_used'] = '; '.join(row['tools_used'])
                    else:
                        row['tools_used'] = ''
                    
                    # Handle None values
                    for key, value in row.items():
                        if value is None:
                            row[key] = ''
                    
                    writer.writerow(row)
            
            print(f"✅ CSV export completed successfully!")
            print(f"   📄 File: {output_file}")
            print(f"   📊 Records: {len(metrics_list)}")
            
            return str(output_file)
            
        except Exception as e:
            print(f"❌ Error writing CSV file: {e}")
            raise
    
    def close(self):
        """Close InfluxDB connection."""
        if hasattr(self, 'client'):
            self.client.close()


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process InfluxDB monitoring data')
    parser.add_argument('--output-dir', default=None, 
                       help=f'Output directory for reports (default: from .env or {PostProcessor.FALLBACK_OUTPUT_DIR})')
    parser.add_argument('--time-window', type=int, default=24,
                       help='Time window in hours to query (default: 24)')
    parser.add_argument('--min-execution-time', type=float, default=0.1,
                       help='Minimum execution time in seconds (default: 0.1)')
    
    args = parser.parse_args()
    
    # Create processor with configuration from .env and command line overrides
    processor = PostProcessor(
        output_dir=args.output_dir,
        time_window_hours=args.time_window
    )
    try:
        results = processor.process_all()
        return results
    finally:
        processor.close()


if __name__ == "__main__":
    main()
