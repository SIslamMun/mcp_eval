"""
Unified Post-Processing Engine for MCP Evaluation

This module consolidates all post-processing functionality into a single file:
- CSV report generation from InfluxDB
- Timeline log creation with full response content
- Comprehensive data analysis and metrics calculation
- All filtering and analysis capabilities
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import csv
from dataclasses import dataclass, asdict

try:
    from session_manager import create_session_manager
except ImportError:
    try:
        from .session_manager import create_session_manager
    except ImportError:
        print("Warning: Could not import session manager")
        create_session_manager = None

try:
    from .utils import extract_model_from_config, extract_model_from_influxdb_values, safe_json_loads, calculate_execution_time_from_timestamps
except ImportError:
    try:
        from utils import extract_model_from_config, extract_model_from_influxdb_values, safe_json_loads, calculate_execution_time_from_timestamps
    except ImportError:
        print("Warning: Could not import utility functions")
        def extract_model_from_config(config, agent_type="unknown"):
            return "unknown"
        def extract_model_from_influxdb_values(values):
            return "unknown"
        def safe_json_loads(data):
            return {}
        def calculate_execution_time_from_timestamps(start, end):
            return 0.0

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRecord:
    """Represents a single evaluation session record."""
    number: int
    prompt: int
    session_id: str
    agent_type: str
    model: str
    success: bool
    execution_time: float
    number_of_calls: int
    number_of_tool_calls: int
    tools_used: str
    cost_usd: float
    response_length: int
    created_at: str
    completed_at: str
    logfile: str
    error_message: str


@dataclass
class SessionEvent:
    """Represents a timestamped event in a session."""
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]


@dataclass 
class SessionMetrics:
    """Comprehensive session metrics and analysis."""
    session_id: str
    prompt_id: int
    agent_type: str
    model: str
    success: bool
    execution_time: float
    cost_usd: float
    total_calls: int
    tool_calls: int
    tools_used: Dict[str, int]
    response_length: int
    created_at: str
    completed_at: str
    events_count: int
    timeline_file: Optional[str] = None
    error_message: Optional[str] = None


class UnifiedPostProcessor:
    """
    Unified post-processing engine that handles all MCP evaluation analysis.
    
    Features:
    - CSV report generation with full data from InfluxDB
    - Timeline log creation with complete response content
    - Comprehensive metrics calculation and analysis
    - Flexible filtering by prompt, agent, date range
    - Support for both fast CSV-only and detailed analysis modes
    """
    
    def __init__(self, backend: str = "influxdb", output_dir: str = "reports/", config_path: Optional[str] = None, prompts_dir: str = "prompts"):
        """
        Initialize the unified post-processor.
        
        Args:
            backend: Database backend ("influxdb" or "sqlite")
            output_dir: Output directory for reports and logs
            config_path: Optional configuration file path
            prompts_dir: Directory containing prompts (default: "prompts")
        """
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config_path = config_path
        self.prompts_dir = prompts_dir
        
        # Cache for prompt content (loaded once, reused for all timeline logs)
        self._prompts_cache = None
        
        # Initialize session manager
        self._init_session_manager()
        
        logger.info(f"UnifiedPostProcessor initialized with {backend} backend")

    def _init_session_manager(self):
        """Initialize the session manager based on backend choice."""
        if not create_session_manager:
            raise ImportError("Session manager not available")
            
        # Load configuration if provided
        config = {}
        if self.config_path:
            try:
                import yaml
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if 'session_manager' in config:
                        config = config['session_manager']
            except Exception as e:
                logger.warning(f"Could not load config from {self.config_path}: {e}")
        
        # Set default values
        if self.backend == "influxdb":
            config.setdefault("influxdb_url", "http://localhost:8086")
            config.setdefault("influxdb_token", "mcp-evaluation-token") 
            config.setdefault("influxdb_org", "mcp-evaluation")
            config.setdefault("influxdb_bucket", "evaluation-sessions")
        
        try:
            self.session_manager = create_session_manager(self.backend, **config)
            logger.info(f"✅ Connected to {self.backend} backend")
        except Exception as e:
            logger.error(f"Failed to connect to {self.backend}: {e}")
            raise

    def generate_csv_report(
        self, 
        output_format: str = "csv",
        filter_agent: Optional[str] = None,
        filter_prompt: Optional[Union[int, List[int]]] = None,
        verbose: bool = False
    ) -> Dict[str, Path]:
        """
        Generate CSV report with all evaluation data from InfluxDB.
        
        Args:
            output_format: Output format ("csv" or "json")
            filter_agent: Filter by agent type ("claude" or "opencode")
            filter_prompt: Filter by prompt ID(s)
            verbose: Enable verbose output
            
        Returns:
            Dictionary mapping report type to file path
        """
        if verbose:
            print(f"🚀 Generating {output_format.upper()} report from InfluxDB...")
            
        # Get all session data
        sessions = self._get_all_sessions_from_influxdb(
            filter_agent=filter_agent,
            filter_prompt=filter_prompt
        )
        
        if verbose:
            print(f"📊 Found {len(sessions)} sessions to process")
            
        # Convert to evaluation records
        records = []
        for i, session_data in enumerate(sessions, 1):
            record = self._convert_session_to_record(session_data, i)
            records.append(record)
            
        # Generate output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "csv":
            output_file = self.output_dir / f"evaluation_report_{timestamp}.csv"
            self._write_csv_report(records, output_file)
        else:
            output_file = self.output_dir / f"evaluation_report_{timestamp}.json"
            self._write_json_report(records, output_file)
            
        # Create metadata file
        metadata_file = self.output_dir / "metadata.json"
        self._write_metadata(records, metadata_file, filter_agent, filter_prompt)
        
        return {
            output_format: output_file,
            "metadata": metadata_file
        }

    def analyze_session_with_timeline(
        self,
        session_id: str,
        verbose: bool = False
    ) -> SessionMetrics:
        """
        Analyze a single session and create timeline log with full content.
        
        Args:
            session_id: Session ID to analyze
            verbose: Enable verbose output
            
        Returns:
            Comprehensive session metrics
        """
        if verbose:
            print(f"📊 Analyzing session {session_id}...")
            
        # Extract events from InfluxDB
        events = self._extract_influxdb_events(session_id)
        
        # Get session metadata
        session_data = self._get_session_metadata(session_id)
        
        # Calculate metrics
        metrics = self._calculate_session_metrics(session_data, events)
        
        # Generate timeline log with full content
        timeline_file = self._generate_timeline_log(session_id, events, session_data)
        metrics.timeline_file = str(timeline_file)
        
        if verbose:
            success_icon = "✅" if metrics.success else "❌"
            print(f"   {success_icon} {metrics.agent_type} | Prompt {metrics.prompt_id} | {metrics.execution_time:.1f}s | {metrics.tool_calls} tools")
            
        return metrics

    def process_by_agent(
        self, 
        agent_type: str,
        generate_timelines: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Process all sessions for a specific agent."""
        if verbose:
            print(f"🔄 Processing sessions for agent {agent_type}...")
            
        sessions = self._get_sessions_by_filter(agent_type=agent_type)
        return self._process_session_batch(sessions, generate_timelines, verbose, f"agent {agent_type}")

    def process_by_prompt(
        self,
        prompt_id: int, 
        generate_timelines: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Process all sessions for a specific prompt."""
        if verbose:
            print(f"🔄 Processing sessions for prompt {prompt_id}...")
            
        sessions = self._get_sessions_by_filter(prompt_id=prompt_id)
        return self._process_session_batch(sessions, generate_timelines, verbose, f"prompt {prompt_id}")

    def process_by_prompts(
        self,
        prompt_ids: List[int],
        generate_timelines: bool = True, 
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Process all sessions for multiple prompts."""
        if verbose:
            print(f"🔄 Processing sessions for prompts {prompt_ids}...")
            
        sessions = self._get_sessions_by_filter(prompt_ids=prompt_ids)
        return self._process_session_batch(sessions, generate_timelines, verbose, f"prompts {prompt_ids}")

    def process_by_prompt_and_agent(
        self,
        prompt_id: int,
        agent_type: str,
        generate_timelines: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Process all sessions for specific prompt and agent combination."""
        if verbose:
            print(f"🔄 Processing sessions for prompt {prompt_id} + agent {agent_type}...")
            
        sessions = self._get_sessions_by_filter(prompt_id=prompt_id, agent_type=agent_type)
        return self._process_session_batch(sessions, generate_timelines, verbose, f"prompt {prompt_id} + agent {agent_type}")

    def process_all_sessions(
        self,
        generate_timelines: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Process all available sessions."""
        if verbose:
            print("🔄 Processing all sessions with detailed analysis...")
            
        sessions = self._get_all_session_ids()
        return self._process_session_batch(sessions, generate_timelines, verbose, "all sessions")

    def generate_summary_statistics(self, verbose: bool = False) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        if verbose:
            print("📊 Generating summary statistics...")
            
        # Get all sessions 
        sessions = self._get_all_sessions_from_influxdb()
        
        total_sessions = len(sessions)
        successful_sessions = sum(1 for s in sessions if s.get('success', False))
        success_rate = (successful_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        # Agent distribution
        agent_distribution = {}
        for session in sessions:
            agent = session.get('agent_type', 'unknown')
            agent_distribution[agent] = agent_distribution.get(agent, 0) + 1
            
        # Prompt distribution
        prompt_distribution = {}
        for session in sessions:
            prompt = session.get('prompt_id', 'unknown')
            prompt_distribution[prompt] = prompt_distribution.get(prompt, 0) + 1
            
        # Cost analysis
        total_cost = sum(session.get('cost_usd', 0) for session in sessions)
        avg_cost = total_cost / total_sessions if total_sessions > 0 else 0
        
        # Execution time analysis
        execution_times = [session.get('execution_time', 0) for session in sessions if session.get('execution_time')]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": success_rate,
            "agent_distribution": agent_distribution,
            "prompt_distribution": prompt_distribution,
            "total_cost_usd": total_cost,
            "average_cost_usd": avg_cost,
            "average_execution_time": avg_execution_time,
            "database_backend": self.backend
        }

    # Private methods for data extraction and processing
    
    def _get_all_sessions_from_influxdb(
        self,
        filter_agent: Optional[str] = None,
        filter_prompt: Optional[Union[int, List[int]]] = None
    ) -> List[Dict[str, Any]]:
        """Get all session data from InfluxDB with optional filtering."""
        try:
            # Build query with filters
            filters = ['r._measurement == "evaluation_session"']
            
            if filter_agent:
                filters.append(f'r.agent_type == "{filter_agent}"')
                
            if filter_prompt:
                if isinstance(filter_prompt, int):
                    filters.append(f'r.prompt_id == "{filter_prompt}"')
                elif isinstance(filter_prompt, list):
                    prompt_filter = " or ".join([f'r.prompt_id == "{pid}"' for pid in filter_prompt])
                    filters.append(f'({prompt_filter})')
                    
            query = f'''
                from(bucket: "{self.session_manager.influxdb_bucket}")
                |> range(start: -365d)
                |> filter(fn: (r) => {" and ".join(filters)})
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: false)
            '''
            
            result = self.session_manager.query_api.query(org=self.session_manager.influxdb_org, query=query)
            
            sessions = []
            for table in result:
                for record in table.records:
                    session_data = {
                        'session_id': record.values.get('session_id', ''),
                        'prompt_id': record.values.get('prompt_id', 0),
                        'agent_type': record.values.get('agent_type', ''),
                        'success': record.values.get('success', False),
                        'execution_time': record.values.get('execution_time', 0),
                        'cost_usd': record.values.get('cost_usd', 0),
                        'created_at': record.values.get('created_at', ''),
                        'completed_at': record.values.get('completed_at', ''),
                        'total_calls': record.values.get('total_calls', 0),
                        'tool_calls': record.values.get('tool_calls', 0),
                        'tools_used': record.values.get('tools_used', ''),
                        'response_data': record.values.get('response_data', ''),
                        'agent_config': record.values.get('agent_config', ''),
                        'error_message': record.values.get('error_message', ''),
                        '_time': record.get_time()
                    }
                    sessions.append(session_data)
                    
            return sessions
            
        except Exception as e:
            logger.error(f"Error querying InfluxDB: {e}")
            return []

    def _extract_influxdb_events(self, session_id: str) -> List[SessionEvent]:
        """Extract detailed events directly from InfluxDB with full content."""
        events = []
        
        try:
            # Query for all events related to this session
            query = f'''
                from(bucket: "{self.session_manager.influxdb_bucket}")
                |> range(start: -365d)
                |> filter(fn: (r) => r._measurement == "evaluation_session")
                |> filter(fn: (r) => r.session_id == "{session_id}")
                |> sort(columns: ["_time"])
            '''
            
            result = self.session_manager.query_api.query(org=self.session_manager.influxdb_org, query=query)
            
            # Process session metadata events
            for table in result:
                for record in table.records:
                    timestamp = record.get_time()
                    field = record.get_field()
                    value = record.get_value()
                    values = record.values
                    
                    # Create event based on field type
                    if field == "response_data" and value:
                        try:
                            response_data = safe_json_loads(value) if isinstance(value, str) else value
                            events.append(SessionEvent(
                                timestamp=timestamp,
                                event_type='response_generated',
                                data=response_data
                            ))
                        except:
                            events.append(SessionEvent(
                                timestamp=timestamp,
                                event_type='response_recorded',
                                data={'raw_response': str(value)}
                            ))
                    
                    # Handle tool execution data - comprehensive parsing
                    elif field in ['tool_calls', 'tools_used'] and value:
                        try:
                            tools_data = safe_json_loads(value) if isinstance(value, str) else value
                            if isinstance(tools_data, dict):
                                for tool_name, count in tools_data.items():
                                    events.append(SessionEvent(
                                        timestamp=timestamp,
                                        event_type='tool_execution',
                                        data={
                                            'tool_name': tool_name,
                                            'status': 'completed',
                                            'execution_count': count,
                                            'source_field': field
                                        }
                                    ))
                            else:
                                events.append(SessionEvent(
                                    timestamp=timestamp,
                                    event_type='tool_summary',
                                    data={
                                        'tools_summary': str(value),
                                        'source_field': field
                                    }
                                ))
                        except Exception as e:
                            events.append(SessionEvent(
                                timestamp=timestamp,
                                event_type='data_parse_error',
                                data={
                                    'field': field,
                                    'error': str(e),
                                    'raw_data': str(value)
                                }
                            ))
                    
                    # Capture ALL other session fields as events
                    elif value is not None and str(value).strip() and field not in ['created_at']:
                        events.append(SessionEvent(
                            timestamp=timestamp,
                            event_type='session_data',
                            data={
                                'field_name': field,
                                'field_value': value,
                                'data_type': type(value).__name__
                            }
                        ))
                    
                    elif field == "created_at":
                        events.append(SessionEvent(
                            timestamp=timestamp,
                            event_type='session_created',
                            data={
                                'session_id': session_id,
                                'prompt_id': values.get('prompt_id', 0),
                                'agent_type': values.get('agent_type', ''),
                                'model': extract_model_from_influxdb_values(values)
                            }
                        ))
            
            # Query for individual tool execution events if available
            tool_query = f'''
                from(bucket: "{self.session_manager.influxdb_bucket}")
                |> range(start: -365d)
                |> filter(fn: (r) => r._measurement == "tool_execution")
                |> filter(fn: (r) => r.session_id == "{session_id}")
                |> sort(columns: ["_time"])
            '''
            
            try:
                tool_result = self.session_manager.query_api.query(org=self.session_manager.influxdb_org, query=tool_query)
                
                for table in tool_result:
                    for record in table.records:
                        timestamp = record.get_time()
                        field = record.get_field()
                        value = record.get_value()
                        values = record.values
                        
                        if field == "tool_output":
                            events.append(SessionEvent(
                                timestamp=timestamp,
                                event_type='tool_execution',
                                data={
                                    'tool_name': values.get('tool_name', 'unknown'),
                                    'status': 'completed',
                                    'output': value  # Full output, no truncation
                                }
                            ))
            except:
                pass  # Tool execution events might not exist
                
        except Exception as e:
            logger.error(f"Error extracting events from InfluxDB: {e}")
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        return events

    def _get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Get session metadata from the session manager."""
        try:
            # Query session data
            sessions = self._get_all_sessions_from_influxdb()
            for session in sessions:
                if session.get('session_id') == session_id:
                    return session
            return {}
        except Exception as e:
            logger.error(f"Error getting session metadata: {e}")
            return {}

    def _calculate_session_metrics(self, session_data: Dict[str, Any], events: List[SessionEvent]) -> SessionMetrics:
        """Calculate comprehensive metrics for a session."""
        # Extract basic metrics
        session_id = session_data.get('session_id', '')
        prompt_id = session_data.get('prompt_id', 0)
        agent_type = session_data.get('agent_type', '')
        success = session_data.get('success', False)
        execution_time = float(session_data.get('execution_time', 0))
        cost_usd = float(session_data.get('cost_usd', 0))
        total_calls = int(session_data.get('total_calls', 0))
        tool_calls = int(session_data.get('tool_calls', 0))
        created_at = session_data.get('created_at', '')
        completed_at = session_data.get('completed_at', '')
        error_message = session_data.get('error_message', '')
        
        # Extract model information
        agent_config = safe_json_loads(session_data.get('agent_config', '{}'))
        model = extract_model_from_config(agent_config, agent_type)
        
        # Parse tools used
        tools_used_raw = session_data.get('tools_used', '')
        tools_used = {}
        if tools_used_raw:
            try:
                # Parse format like "tool1:count1,tool2:count2"
                for tool_count in tools_used_raw.split(','):
                    if ':' in tool_count:
                        tool, count = tool_count.strip().split(':', 1)
                        tools_used[tool.strip()] = int(count.strip())
            except:
                tools_used = {'unknown': 1}
        
        # Calculate response length from events
        response_length = 0
        for event in events:
            if event.event_type in ['response_generated', 'response_recorded']:
                response_text = event.data.get('text', '') or str(event.data.get('response', ''))
                response_length += len(response_text)
        
        return SessionMetrics(
            session_id=session_id,
            prompt_id=prompt_id,
            agent_type=agent_type,
            model=model,
            success=success,
            execution_time=execution_time,
            cost_usd=cost_usd,
            total_calls=total_calls,
            tool_calls=tool_calls,
            tools_used=tools_used,
            response_length=response_length,
            created_at=created_at,
            completed_at=completed_at,
            events_count=len(events),
            error_message=error_message
        )

    def _get_prompt_content(self, prompt_id: int) -> str:
        """Get prompt content from cache or load if needed."""
        # Load prompts cache if not already loaded
        if self._prompts_cache is None:
            try:
                from .jsonl_prompt_loader import UnifiedPromptLoader
                loader = UnifiedPromptLoader(self.prompts_dir)
                self._prompts_cache = loader.load_all_prompts()
            except Exception as e:
                logger.warning(f"Could not load prompts cache: {e}")
                self._prompts_cache = {}
        
        # Get prompt content from cache
        if prompt_id in self._prompts_cache:
            content = self._prompts_cache[prompt_id].content
            # Return full prompt content - no truncation
            return content
        else:
            return f"Prompt {prompt_id} not found in cache"

    def _generate_timeline_log(
        self,
        session_id: str,
        events: List[SessionEvent],
        session_data: Dict[str, Any]
    ) -> Path:
        """Generate conversational flow timeline log."""
        # Extract session information
        prompt_id = session_data.get('prompt_id', 0)
        agent_type = session_data.get('agent_type', 'unknown')
        agent_config = safe_json_loads(session_data.get('agent_config', '{}'))
        model = extract_model_from_config(agent_config, agent_type)
        success = session_data.get('success', False)
        execution_time = float(session_data.get('execution_time', 0))
        cost_usd = float(session_data.get('cost_usd', 0))
        
        # Convert prompt_id to int if it's a string
        try:
            prompt_id = int(prompt_id)
        except (ValueError, TypeError):
            pass
        
        # Create organized folder structure: reports/prompt_X/logs/
        prompt_folder = self.output_dir / f"prompt_{prompt_id}"
        logs_folder = prompt_folder / "logs"
        logs_folder.mkdir(parents=True, exist_ok=True)
        
        # Place log file in the organized structure
        log_file = logs_folder / f"{session_id}_timeline.log"

        prompt_content = self._get_prompt_content(prompt_id)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=== MCP Evaluation Session Log ===\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Prompt ID: {prompt_id}\n")
            f.write(f"Agent Type: {agent_type}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Success: {success}\n")
            f.write(f"Execution Time: {execution_time:.2f}s\n")
            f.write(f"Cost: ${cost_usd:.4f}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            
            # User Prompt
            f.write("User Prompt:\n")
            f.write(f"{prompt_content}\n\n")
            f.write("=" * 60 + "\n\n")
            
            # Extract conversational flow from events
            response_content = None
            tool_calls = []
            
            # Parse events to extract the conversational flow
            for event in events:
                if event.event_type == 'response_generated':
                    response_content = event.data.get('text') or str(event.data.get('response', ''))
                elif event.event_type == 'tool_execution':
                    tool_name = event.data.get('tool_name', 'unknown')
                    if tool_name != 'unknown':
                        tool_calls.append(tool_name)
            
            # Generate conversational log
            if response_content:
                # Extract tool responses from the response content if available
                tool_responses = {}
                response_lines = response_content.strip().split('\n')
                
                # Parse tool responses from content
                for line in response_lines:
                    for tool_name in tool_calls:
                        tool_key = tool_name.replace('_', '_').lower()
                        if f'tool {tool_key}:' in line.lower() or f'tool {tool_name}:' in line.lower():
                            # Extract the tool response part
                            tool_start = line.lower().find(f'tool ')
                            if tool_start >= 0:
                                tool_responses[tool_name] = line[tool_start:].strip()
                
                # Write conversational flow - show actual agent reasoning  
                f.write("LLM Call: Initial reasoning and tool execution\n")
                f.write("LLM Response:\n")
                # Show the beginning of the actual response as initial reasoning
                if response_lines and len(response_lines[0]) > 10:
                    f.write(f"{response_lines[0]}\n\n")
                else:
                    f.write("Processing the request...\n\n")
                
                # Tool calls and responses
                for i, tool_name in enumerate(tool_calls, 1):
                    f.write(f"Tool Call #{i}: {tool_name}\n")
                    f.write("Tool Response: ")
                    
                    # Use actual tool response if available, otherwise show status
                    if tool_name in tool_responses:
                        # Show complete tool response - no truncation
                        tool_resp = tool_responses[tool_name]
                        f.write(f"{tool_resp}\n\n")
                    else:
                        # Use tool status from events if available
                        tool_status = "executed successfully"
                        for event in events:
                            if event.event_type == 'tool_execution' and event.data.get('tool_name') == tool_name:
                                status = event.data.get('status', 'completed')
                                tool_status = f"{status}"
                                break
                        f.write(f"Tool {tool_status}\n\n")
                
                # Final LLM response - use the complete response content
                if len(tool_calls) > 0:
                    f.write("LLM Call: Final response with results\n")
                    f.write("LLM Response:\n")
                    f.write(f"{response_content.strip()}\n\n")
                else:
                    # If no tools, this is the direct response
                    f.write("LLM Call: Direct response\n")
                    f.write("LLM Response:\n")
                    f.write(f"{response_content.strip()}\n\n")
                
                # End the session authentically
                f.write("=" * 60 + "\n")
                f.write(f"Session completed: {len(tool_calls)} tools used, {len(response_content)} chars response\n")
            
            else:
                # Fallback if no response content found
                f.write("LLM Call: Request processing\n")
                f.write("LLM Response: [No response data available in database]\n\n")
                
                # Show tool executions from events if available
                for i, tool_name in enumerate(tool_calls, 1):
                    f.write(f"Tool Call #{i}: {tool_name}\n")
                    # Get actual tool status from events
                    tool_status = "status unknown"
                    for event in events:
                        if event.event_type == 'tool_execution' and event.data.get('tool_name') == tool_name:
                            status = event.data.get('status', 'unknown')
                            count = event.data.get('execution_count', 1)
                            tool_status = f"{status} (executed {count} time{'s' if count != 1 else ''})"
                            break
                    f.write(f"Tool Response: {tool_status}\n\n")
                
                f.write("=" * 60 + "\n")
                f.write(f"Session incomplete: {len(tool_calls)} tools detected, no response data\n")
        
        return log_file

    def _convert_session_to_record(self, session_data: Dict[str, Any], number: int) -> EvaluationRecord:
        """Convert session data to evaluation record for CSV output."""
        # Extract model information
        agent_config = safe_json_loads(session_data.get('agent_config', '{}'))
        model = extract_model_from_config(agent_config, session_data.get('agent_type', ''))
        
        # Parse tools used for display
        tools_used_raw = session_data.get('tools_used', '')
        
        # Calculate response length
        response_data = safe_json_loads(session_data.get('response_data', '{}'))
        response_length = len(str(response_data.get('response', '')))
        
        return EvaluationRecord(
            number=number,
            prompt=int(session_data.get('prompt_id', 0)),
            session_id=session_data.get('session_id', ''),
            agent_type=session_data.get('agent_type', ''),
            model=model,
            success=session_data.get('success', False),
            execution_time=float(session_data.get('execution_time', 0)),
            number_of_calls=int(session_data.get('total_calls', 0)),
            number_of_tool_calls=int(session_data.get('tool_calls', 0)),
            tools_used=tools_used_raw,
            cost_usd=float(session_data.get('cost_usd', 0)),
            response_length=response_length,
            created_at=session_data.get('created_at', ''),
            completed_at=session_data.get('completed_at', ''),
            logfile=f"logs/{session_data.get('session_id', '')}.log",
            error_message=session_data.get('error_message', '')
        )

    def _write_csv_report(self, records: List[EvaluationRecord], output_file: Path):
        """Write evaluation records to CSV file."""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if records:
                fieldnames = list(asdict(records[0]).keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for record in records:
                    writer.writerow(asdict(record))

    def _write_json_report(self, records: List[EvaluationRecord], output_file: Path):
        """Write evaluation records to JSON file."""
        data = [asdict(record) for record in records]
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=2)

    def _write_metadata(self, records: List[EvaluationRecord], metadata_file: Path, filter_agent: Optional[str], filter_prompt: Optional[Union[int, List[int]]]):
        """Write metadata about the report."""
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_records": len(records),
            "backend": self.backend,
            "filters": {
                "agent": filter_agent,
                "prompt": filter_prompt
            },
            "summary": {
                "successful": sum(1 for r in records if r.success),
                "failed": sum(1 for r in records if not r.success),
                "total_cost": sum(r.cost_usd for r in records),
                "avg_execution_time": sum(r.execution_time for r in records) / len(records) if records else 0
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_sessions_by_filter(
        self, 
        agent_type: Optional[str] = None,
        prompt_id: Optional[int] = None,
        prompt_ids: Optional[List[int]] = None
    ) -> List[str]:
        """Get session IDs matching the given filters."""
        try:
            # Build filter conditions
            filters = ['r._measurement == "evaluation_session"']
            
            if agent_type:
                filters.append(f'r.agent_type == "{agent_type}"')
            if prompt_id is not None:
                filters.append(f'r.prompt_id == "{prompt_id}"')
            elif prompt_ids:
                prompt_filter = " or ".join([f'r.prompt_id == "{pid}"' for pid in prompt_ids])
                filters.append(f'({prompt_filter})')
                
            query = f'''
                from(bucket: "{self.session_manager.influxdb_bucket}")
                |> range(start: -365d)
                |> filter(fn: (r) => {" and ".join(filters)})
                |> distinct(column: "session_id")
            '''
            
            result = self.session_manager.query_api.query(org=self.session_manager.influxdb_org, query=query)
            
            session_ids = []
            for table in result:
                for record in table.records:
                    session_id = record.values.get('session_id')
                    if session_id and session_id not in session_ids:
                        session_ids.append(session_id)
                        
            print(f"🗄️  Found {len(session_ids)} sessions for {filters}")
            return session_ids
            
        except Exception as e:
            logger.error(f"Error getting sessions by filter: {e}")
            return []

    def _get_all_session_ids(self) -> List[str]:
        """Get all session IDs from InfluxDB."""
        try:
            query = f'''
                from(bucket: "{self.session_manager.influxdb_bucket}")
                |> range(start: -365d)
                |> filter(fn: (r) => r._measurement == "evaluation_session")
                |> distinct(column: "session_id")
            '''
            
            result = self.session_manager.query_api.query(org=self.session_manager.influxdb_org, query=query)
            
            session_ids = []
            for table in result:
                for record in table.records:
                    session_id = record.values.get('session_id')
                    if session_id and session_id not in session_ids:
                        session_ids.append(session_id)
            
            print(f"🗄️  Found {len(session_ids)} sessions in database")
            return session_ids
            
        except Exception as e:
            logger.error(f"Error getting all session IDs: {e}")
            return []

    def _process_session_batch(
        self,
        session_ids: List[str],
        generate_timelines: bool,
        verbose: bool,
        description: str
    ) -> Dict[str, Any]:
        """Process a batch of sessions and return results."""
        results = []
        successful = 0
        failed = 0
        
        for i, session_id in enumerate(session_ids, 1):
            try:
                if verbose:
                    print(f"📊 Processing session {i}/{len(session_ids)}: {session_id}")
                
                if generate_timelines:
                    metrics = self.analyze_session_with_timeline(session_id, verbose)
                    results.append(asdict(metrics))
                    
                    if metrics.success:
                        successful += 1
                    else:
                        failed += 1
                else:
                    # Just get basic metrics without timeline
                    session_data = self._get_session_metadata(session_id)
                    if session_data:
                        results.append(session_data)
                        if session_data.get('success'):
                            successful += 1
                        else:
                            failed += 1
                            
            except Exception as e:
                logger.error(f"Error processing session {session_id}: {e}")
                failed += 1
        
        summary = {
            "total_sessions": len(session_ids),
            "successful": successful, 
            "failed": failed,
            "results": results,
            "description": description,
            "output_directory": str(self.output_dir)
        }
        
        if verbose:
            print(f"\n📈 **Batch Analysis Summary ({description}):**")
            print(f"   Total Sessions: {len(session_ids)}")
            print(f"   Successful: {successful}")
            print(f"   Failed: {failed}")
            print(f"   Output Directory: {self.output_dir}")
            
        return summary
