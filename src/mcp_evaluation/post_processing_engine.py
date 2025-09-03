"""
Post-processing engine for MCP evaluation data analysis and reporting.

Extracts evaluation data from InfluxDB/SQLite and generates comprehensive reports
including CSV files and communication logs.
"""

import json
import csv
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from pydantic import BaseModel
import re

from .session_manager import create_session_manager, SessionData

logger = logging.getLogger(__name__)


class EvaluationRecord(BaseModel):
    """Structured evaluation record for analysis."""
    number: int
    prompt_id: int
    session_id: str
    agent_type: str
    model: str
    success: bool
    execution_time: float
    cost_usd: float
    total_calls: int
    tool_calls: int
    tools_used: Dict[str, int]  # tool_name: call_count
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    response_length: int = 0
    communication_log: str = ""
    error_message: Optional[str] = None


class ReportMetadata(BaseModel):
    """Metadata for generated reports."""
    generated_at: datetime
    total_records: int
    database_backend: str
    filters_applied: Dict[str, Any]
    output_directory: str
    report_version: str = "1.0"


class CommunicationLogExtractor:
    """Extract and format communication logs from evaluation data."""
    
    @staticmethod
    def extract_tool_usage(response_data: Dict[str, Any]) -> Tuple[int, int, Dict[str, int]]:
        """
        Extract tool usage information from response data.
        Returns: (total_calls, tool_calls, tools_used_dict)
        """
        total_calls = 0
        tool_calls = 0
        tools_used = {}
        
        if not response_data:
            return total_calls, tool_calls, tools_used
        
        # Look for tool usage patterns in response
        response_text = str(response_data.get("response", ""))
        
        # Count tool mentions (basic pattern matching)
        tool_patterns = {
            "bash": r"Tool bash:|bash:|`bash\s",
            "read_file": r"Tool read_file:|read_file:|reading file",
            "write_file": r"Tool write_file:|write_file:|writing file",
            "list_dir": r"Tool list_dir:|list_dir:|listing directory",
            "MCP": r"Tool MCP:|MCP:|mcp server",
            "opencode": r"opencode|github-copilot",
            "claude": r"claude|anthropic"
        }
        
        for tool_name, pattern in tool_patterns.items():
            matches = len(re.findall(pattern, response_text, re.IGNORECASE))
            if matches > 0:
                tools_used[tool_name] = matches
                tool_calls += matches
        
        # Estimate total calls (tool calls + API calls)
        total_calls = tool_calls + 1  # At least one API call for the response
        
        # Look for additional call indicators
        if "request" in response_text.lower():
            total_calls += response_text.lower().count("request")
        
        return total_calls, tool_calls, tools_used
    
    @staticmethod
    def format_communication_log(session_data: Dict[str, Any]) -> str:
        """Format session data into a readable communication log."""
        log_lines = []
        
        # Header
        log_lines.append(f"=== MCP Evaluation Communication Log ===")
        log_lines.append(f"Session ID: {session_data.get('session_id', 'Unknown')}")
        log_lines.append(f"Prompt ID: {session_data.get('prompt_id', 'Unknown')}")
        log_lines.append(f"Agent: {session_data.get('agent_type', 'Unknown')}")
        log_lines.append(f"Created: {session_data.get('created_at', 'Unknown')}")
        log_lines.append(f"Status: {session_data.get('status', 'Unknown')}")
        log_lines.append("")
        
        # Response data
        response_data = session_data.get('response_data', {})
        if response_data:
            log_lines.append("=== Evaluation Response ===")
            if isinstance(response_data, dict):
                response = response_data.get('response', str(response_data))
            else:
                response = str(response_data)
            
            log_lines.append(response)
            log_lines.append("")
        
        # Error information
        if session_data.get('error_message'):
            log_lines.append("=== Error Information ===")
            log_lines.append(session_data['error_message'])
            log_lines.append("")
        
        # Metadata
        metadata = session_data.get('metadata', {})
        if metadata:
            log_lines.append("=== Session Metadata ===")
            for key, value in metadata.items():
                log_lines.append(f"{key}: {value}")
            log_lines.append("")
        
        log_lines.append(f"=== End of Communication Log ===")
        
        return "\n".join(log_lines)


class MetricsExtractor:
    """Extract and calculate key metrics from evaluation data."""
    
    @staticmethod
    def extract_model_from_config(agent_config: Dict[str, Any], agent_type: str) -> str:
        """Extract model name from agent configuration."""
        if not agent_config:
            return "unknown"
        
        # Try different configuration patterns
        model = agent_config.get("model", "")
        if model:
            return model
        
        # Default models based on agent type
        if agent_type == "claude":
            return "sonnet"  # Default Claude model
        elif agent_type == "opencode":
            return "github-copilot/claude-3.5-sonnet"  # Default OpenCode model
        
        return "unknown"
    
    @staticmethod
    def calculate_execution_time(created_at: Optional[str], completed_at: Optional[str]) -> float:
        """Calculate execution time from timestamps."""
        if not created_at or not completed_at:
            return 0.0
        
        try:
            if isinstance(created_at, str):
                created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created = created_at
                
            if isinstance(completed_at, str):
                completed = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
            else:
                completed = completed_at
                
            delta = completed - created
            return delta.total_seconds()
        except Exception as e:
            logger.warning(f"Failed to calculate execution time: {e}")
            return 0.0


class ReportGenerator:
    """Generate various report formats from evaluation data."""
    
    def __init__(self, output_dir: str = "reports/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create logs subdirectory
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
    
    def generate_csv_report(
        self, 
        records: List[EvaluationRecord], 
        filename: Optional[str] = None
    ) -> str:
        """Generate CSV report from evaluation records."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.csv"
        
        csv_path = self.output_dir / filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header row
            writer.writerow([
                'number',
                'prompt',
                'session_id',
                'agent_type',
                'model',
                'success',
                'execution_time',
                'number_of_calls',
                'number_of_tool_calls',
                'tools_used',
                'cost_usd',
                'response_length',
                'created_at',
                'completed_at',
                'logfile',
                'error_message'
            ])
            
            # Data rows
            for record in records:
                # Format tools_used as "tool1:count1,tool2:count2"
                tools_str = ",".join([f"{tool}:{count}" for tool, count in record.tools_used.items()])
                
                # Generate logfile name
                logfile = f"{record.session_id}.log"
                
                writer.writerow([
                    record.number,
                    record.prompt_id,
                    record.session_id,
                    record.agent_type,
                    record.model,
                    record.success,
                    f"{record.execution_time:.2f}",
                    record.total_calls,
                    record.tool_calls,
                    tools_str or "none",
                    f"{record.cost_usd:.4f}",
                    record.response_length,
                    record.created_at.isoformat() if record.created_at else "",
                    record.completed_at.isoformat() if record.completed_at else "",
                    f"logs/{logfile}",
                    record.error_message or ""
                ])
        
        return str(csv_path)
    
    def generate_log_files(self, records: List[EvaluationRecord], session_data_map: Dict[str, Dict]) -> int:
        """Generate individual log files for each session."""
        generated_count = 0
        
        for record in records:
            session_data = session_data_map.get(record.session_id)
            if not session_data:
                continue
            
            log_filename = f"{record.session_id}.log"
            log_path = self.logs_dir / log_filename
            
            communication_log = CommunicationLogExtractor.format_communication_log(session_data)
            
            with open(log_path, 'w', encoding='utf-8') as logfile:
                logfile.write(communication_log)
            
            generated_count += 1
        
        return generated_count
    
    def generate_metadata_file(self, metadata: ReportMetadata) -> str:
        """Generate metadata JSON file for the report."""
        metadata_path = self.output_dir / "metadata.json"
        
        metadata_dict = metadata.dict()
        metadata_dict['generated_at'] = metadata_dict['generated_at'].isoformat()
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2)
        
        return str(metadata_path)


class PostProcessingEngine:
    """Main engine for extracting and processing evaluation data."""
    
    def __init__(
        self,
        backend: str = "influxdb",
        output_dir: str = "reports/",
        config_path: Optional[str] = None
    ):
        self.backend = backend
        self.output_dir = output_dir
        self.config_path = config_path
        
        # Initialize session manager
        self.session_manager = create_session_manager(backend=backend)
        
        # Initialize components
        self.report_generator = ReportGenerator(output_dir)
        self.metrics_extractor = MetricsExtractor()
        self.log_extractor = CommunicationLogExtractor()
        
        logger.info(f"Initialized PostProcessingEngine with {backend} backend")
    
    def extract_evaluation_data(
        self,
        filter_agent: Optional[str] = None,
        filter_prompt: Optional[List[int]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Tuple[List[EvaluationRecord], Dict[str, Dict]]:
        """
        Extract evaluation data from the database with optional filters.
        Returns: (records_list, session_data_map)
        """
        logger.info("Extracting evaluation data from database...")
        
        # Get all sessions from the session manager
        if hasattr(self.session_manager, 'get_all_sessions'):
            sessions_data = self.session_manager.get_all_sessions()
        else:
            # Fallback: get sessions by querying statistics and then individual sessions
            sessions_data = self._get_sessions_fallback()
        
        records = []
        session_data_map = {}
        record_number = 1
        
        for session_data in sessions_data:
            # Apply filters
            if filter_agent and session_data.get('agent_type') != filter_agent:
                continue
            
            if filter_prompt:
                prompt_id = session_data.get('prompt_id')
                if prompt_id not in filter_prompt:
                    continue
            
            # Date filtering (if timestamps are available)
            if date_from or date_to:
                created_at_str = session_data.get('created_at')
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        if date_from and created_at < date_from:
                            continue
                        if date_to and created_at > date_to:
                            continue
                    except Exception:
                        # Skip if date parsing fails
                        continue
            
            # Extract metrics
            session_id = session_data.get('session_id', f'unknown_{record_number}')
            prompt_id = session_data.get('prompt_id', 0)
            agent_type = session_data.get('agent_type', 'unknown')
            
            # Extract model from agent config
            agent_config = session_data.get('agent_config', {})
            if isinstance(agent_config, str):
                try:
                    agent_config = json.loads(agent_config)
                except:
                    agent_config = {}
            
            model = self.metrics_extractor.extract_model_from_config(agent_config, agent_type)
            
            # Success status
            success = session_data.get('status') == 'completed' or session_data.get('success', False)
            
            # Execution time
            execution_time = session_data.get('execution_time', 0.0)
            if not execution_time:
                execution_time = self.metrics_extractor.calculate_execution_time(
                    session_data.get('created_at'),
                    session_data.get('completed_at')
                )
            
            # Cost
            cost_usd = 0.0
            metadata = session_data.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            cost_usd = metadata.get('cost_usd', 0.0) or session_data.get('cost_usd', 0.0)
            
            # Extract tool usage
            response_data = session_data.get('response_data', {})
            if isinstance(response_data, str):
                try:
                    response_data = json.loads(response_data)
                except:
                    response_data = {}
            
            # Get tool usage from database first (preferred), fallback to log parsing
            total_calls = session_data.get('total_calls', 0)
            tool_calls = session_data.get('tool_calls', 0)
            tools_used = session_data.get('tools_used', {})
            
            # If no data in database, try log extraction as fallback
            if not total_calls and not tool_calls and not tools_used:
                total_calls, tool_calls, tools_used = self.log_extractor.extract_tool_usage(response_data)
            
            # Response length
            response_length = 0
            if response_data:
                response_text = str(response_data.get('response', str(response_data)))
                response_length = len(response_text)
            
            # Timestamps
            created_at = None
            completed_at = None
            if session_data.get('created_at'):
                try:
                    created_at = datetime.fromisoformat(session_data['created_at'].replace('Z', '+00:00'))
                except:
                    pass
            
            if session_data.get('completed_at'):
                try:
                    completed_at = datetime.fromisoformat(session_data['completed_at'].replace('Z', '+00:00'))
                except:
                    pass
            
            # Create record
            record = EvaluationRecord(
                number=record_number,
                prompt_id=prompt_id,
                session_id=session_id,
                agent_type=agent_type,
                model=model,
                success=success,
                execution_time=execution_time,
                cost_usd=cost_usd,
                total_calls=total_calls,
                tool_calls=tool_calls,
                tools_used=tools_used,
                created_at=created_at,
                completed_at=completed_at,
                response_length=response_length,
                error_message=session_data.get('error_message')
            )
            
            records.append(record)
            session_data_map[session_id] = session_data
            record_number += 1
        
        logger.info(f"Extracted {len(records)} evaluation records")
        return records, session_data_map
    
    def _get_sessions_fallback(self) -> List[Dict[str, Any]]:
        """Fallback method to get sessions when get_all_sessions is not available."""
        sessions = []
        
        try:
            # Get statistics to find session info
            stats = self.session_manager.get_session_statistics()
            
            # Try to get sessions by querying the database directly
            if hasattr(self.session_manager, 'query_api') and self.backend == 'influxdb':
                # InfluxDB query to get all sessions
                query = f'''
                    from(bucket: "{self.session_manager.influxdb_bucket}")
                    |> range(start: -365d)
                    |> filter(fn: (r) => r._measurement == "evaluation_session")
                    |> filter(fn: (r) => r._field == "stored")
                    |> group(columns: ["session_id"])
                    |> last()
                '''
                
                result = self.session_manager.query_api.query(org=self.session_manager.influxdb_org, query=query)
                
                for table in result:
                    for record in table.records:
                        session_id = record.values.get("session_id")
                        if session_id:
                            # Get full session details
                            session_data = self.session_manager.get_session(session_id)
                            if session_data:
                                sessions.append(session_data)
            
            elif hasattr(self.session_manager, 'db_path') and self.backend == 'sqlite':
                # SQLite query to get all sessions
                import sqlite3
                with sqlite3.connect(self.session_manager.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT session_id, prompt_id, agent_type, status, created_at, completed_at,
                               error_message, response_data, metadata, agent_config
                        FROM evaluation_sessions
                        ORDER BY created_at DESC
                    """)
                    
                    for row in cursor.fetchall():
                        session_data = {
                            'session_id': row[0],
                            'prompt_id': row[1],
                            'agent_type': row[2],
                            'status': row[3],
                            'created_at': row[4],
                            'completed_at': row[5],
                            'error_message': row[6],
                            'response_data': row[7],
                            'metadata': row[8],
                            'agent_config': row[9]
                        }
                        sessions.append(session_data)
        
        except Exception as e:
            logger.warning(f"Failed to retrieve sessions using fallback method: {e}")
        
        return sessions
    
    def generate_report(
        self,
        output_format: str = "csv",
        include_logs: bool = True,
        filter_agent: Optional[str] = None,
        filter_prompt: Optional[List[int]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive evaluation report.
        Returns dictionary with paths to generated files.
        """
        logger.info("Starting report generation...")
        
        # Extract data
        records, session_data_map = self.extract_evaluation_data(
            filter_agent=filter_agent,
            filter_prompt=filter_prompt,
            date_from=date_from,
            date_to=date_to
        )
        
        if not records:
            raise ValueError("No evaluation data found matching the specified filters")
        
        generated_files = {}
        
        # Generate CSV report
        if output_format == "csv":
            csv_path = self.report_generator.generate_csv_report(records)
            generated_files['csv'] = csv_path
            logger.info(f"Generated CSV report: {csv_path}")
        
        # Generate log files if requested
        if include_logs:
            log_count = self.report_generator.generate_log_files(records, session_data_map)
            generated_files['logs'] = f"{log_count} log files in {self.report_generator.logs_dir}"
            logger.info(f"Generated {log_count} communication log files")
        
        # Generate metadata
        filters_applied = {
            'agent': filter_agent,
            'prompts': filter_prompt,
            'date_from': date_from.isoformat() if date_from else None,
            'date_to': date_to.isoformat() if date_to else None
        }
        
        metadata = ReportMetadata(
            generated_at=datetime.now(),
            total_records=len(records),
            database_backend=self.backend,
            filters_applied=filters_applied,
            output_directory=str(self.output_dir)
        )
        
        metadata_path = self.report_generator.generate_metadata_file(metadata)
        generated_files['metadata'] = metadata_path
        
        logger.info("Report generation completed successfully")
        return generated_files
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about available data."""
        try:
            records, _ = self.extract_evaluation_data()
            
            if not records:
                return {"error": "No evaluation data found"}
            
            # Calculate statistics
            total_records = len(records)
            successful_records = sum(1 for r in records if r.success)
            success_rate = (successful_records / total_records) * 100 if total_records > 0 else 0
            
            # Agent distribution
            agent_distribution = {}
            for record in records:
                agent_distribution[record.agent_type] = agent_distribution.get(record.agent_type, 0) + 1
            
            # Prompt distribution
            prompt_distribution = {}
            for record in records:
                prompt_distribution[record.prompt_id] = prompt_distribution.get(record.prompt_id, 0) + 1
            
            # Cost statistics (for records with cost > 0)
            cost_records = [r.cost_usd for r in records if r.cost_usd > 0]
            total_cost = sum(cost_records) if cost_records else 0
            avg_cost = total_cost / len(cost_records) if cost_records else 0
            
            # Time statistics
            time_records = [r.execution_time for r in records if r.execution_time > 0]
            avg_time = sum(time_records) / len(time_records) if time_records else 0
            
            return {
                "total_records": total_records,
                "successful_records": successful_records,
                "success_rate": round(success_rate, 2),
                "agent_distribution": agent_distribution,
                "prompt_distribution": prompt_distribution,
                "total_cost": round(total_cost, 4),
                "average_cost": round(avg_cost, 4),
                "average_execution_time": round(avg_time, 2),
                "database_backend": self.backend
            }
        
        except Exception as e:
            logger.error(f"Failed to get summary statistics: {e}")
            return {"error": str(e)}
