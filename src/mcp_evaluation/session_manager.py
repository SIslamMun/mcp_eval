"""Session management for MCP evaluations using InfluxDB or SQLite."""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from pydantic import BaseModel

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_session_manager(backend: str = "influxdb", **kwargs):
    """
    Factory function to create the appropriate session manager.
    
    Args:
        backend: "influxdb" or "sqlite"
        **kwargs: Configuration parameters for the session manager
    
    Returns:
        SessionManager instance (InfluxDB or SQLite based)
    """
    if backend == "sqlite":
        return SQLiteSessionManager(kwargs.get("db_path", "evaluation_sessions.db"))
    elif backend == "influxdb":
        if not INFLUXDB_AVAILABLE:
            logger.warning("InfluxDB not available, falling back to SQLite")
            return SQLiteSessionManager(kwargs.get("db_path", "evaluation_sessions.db"))
        return InfluxDBSessionManager(
            db_path=kwargs.get("db_path", "evaluation_sessions.db"),
            influxdb_url=kwargs.get("influxdb_url", "http://localhost:8086"),
            influxdb_token=kwargs.get("influxdb_token", "mcp-evaluation-token"),
            influxdb_org=kwargs.get("influxdb_org", "mcp-evaluation"),
            influxdb_bucket=kwargs.get("influxdb_bucket", "evaluation-sessions")
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


class SessionData(BaseModel):
    """Data structure for individual session results."""
    session_id: str
    prompt_id: int
    agent_type: str
    success: bool
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    cost_usd: Optional[float] = None
    tokens_used: Optional[int] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    timestamp: Optional[int] = None
    
    # Tool usage metrics
    total_calls: Optional[int] = None
    tool_calls: Optional[int] = None
    tools_used: Optional[Dict[str, int]] = None


class ComparativeSessionData(BaseModel):
    """Data structure for comparative evaluation results."""
    base_session_id: str
    prompt_id: int
    claude_result: Optional[Dict[str, Any]] = None
    opencode_result: Optional[Dict[str, Any]] = None
    comparative_analysis: Optional[Dict[str, Any]] = None
    prompt: Optional[Dict[str, Any]] = None
    timestamp: Optional[int] = None


class SQLiteSessionManager:
    """Simple SQLite-based session manager for basic functionality."""
    
    def __init__(self, db_path: str):
        """Initialize with SQLite database."""
        import sqlite3
        import time
        self.db_path = Path(db_path)
        self._ensure_db_exists()
        
    def _ensure_db_exists(self):
        """Create database and tables if they don't exist."""
        import sqlite3
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    prompt_id INTEGER NOT NULL,
                    agent_type TEXT NOT NULL,
                    agent_config TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    response_data TEXT,
                    metadata TEXT,
                    total_calls INTEGER DEFAULT 0,
                    tool_calls INTEGER DEFAULT 0,
                    tools_used TEXT
                )
            """)
            
            # Add the new columns to existing tables if they don't exist
            try:
                conn.execute("ALTER TABLE evaluation_sessions ADD COLUMN total_calls INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE evaluation_sessions ADD COLUMN tool_calls INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE evaluation_sessions ADD COLUMN tools_used TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comparative_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_session_id TEXT UNIQUE NOT NULL,
                    prompt_id INTEGER NOT NULL,
                    claude_result TEXT,
                    opencode_result TEXT,
                    comparative_analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            
    def create_session(self, session_id: str, prompt_id: int, agent_type: str, 
                      agent_config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new evaluation session."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO evaluation_sessions 
                (session_id, prompt_id, agent_type, agent_config, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id, prompt_id, agent_type,
                json.dumps(agent_config), json.dumps(metadata or {})
            ))
        logger.info(f"Created session {session_id} for prompt {prompt_id} with agent {agent_type}")
        return session_id
        
    def start_session(self, session_id: str) -> None:
        """Mark session as started."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE evaluation_sessions SET status = 'running' WHERE session_id = ?
            """, (session_id,))
        
    def complete_session(self, session_id: str, response_data: Dict[str, Any]) -> None:
        """Mark session as completed."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE evaluation_sessions 
                SET status = 'completed', response_data = ? WHERE session_id = ?
            """, (json.dumps(response_data), session_id))
        
    def fail_session(self, session_id: str, error_message: str) -> None:
        """Mark session as failed."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE evaluation_sessions 
                SET status = 'failed', error_message = ? WHERE session_id = ?
            """, (error_message, session_id))
            
    def record_metric(self, session_id: str, metric_name: str, metric_value: float) -> None:
        """Record a performance metric."""
        logger.debug(f"Metric {metric_name}={metric_value} for session {session_id} (SQLite doesn't store metrics separately)")
        
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM evaluation_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            session = dict(row)
            if session.get('agent_config'):
                session['agent_config'] = json.loads(session['agent_config'])
            if session.get('metadata'):
                session['metadata'] = json.loads(session['metadata'] or '{}')
            if session.get('response_data'):
                session['response_data'] = json.loads(session['response_data'])
            return session
            
    def get_sessions_by_prompt(self, prompt_id: int) -> List[Dict[str, Any]]:
        """Get all sessions for a prompt."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM evaluation_sessions WHERE prompt_id = ?", (prompt_id,))
            sessions = []
            for row in cursor.fetchall():
                session = dict(row)
                if session.get('agent_config'):
                    session['agent_config'] = json.loads(session['agent_config'])
                if session.get('metadata'):
                    session['metadata'] = json.loads(session['metadata'] or '{}')
                if session.get('response_data'):
                    session['response_data'] = json.loads(session['response_data'])
                sessions.append(session)
            return sessions
            
    def get_session_metrics(self, session_id: str) -> List[Dict[str, Any]]:
        """Get metrics for a session."""
        return []  # SQLite version doesn't store metrics separately
        
    def get_comparative_analysis(self, prompt_id: int) -> Dict[str, Any]:
        """Get comparative analysis."""
        sessions = self.get_sessions_by_prompt(prompt_id)
        return {
            "prompt_id": prompt_id,
            "total_sessions": len(sessions),
            "sessions": sessions,
            "analysis": "Basic SQLite analysis"
        }
        
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM evaluation_sessions")
            total = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT agent_type, COUNT(*) FROM evaluation_sessions GROUP BY agent_type")
            agents = dict(cursor.fetchall())
            
            cursor = conn.execute("SELECT status, COUNT(*) FROM evaluation_sessions GROUP BY status")
            status = dict(cursor.fetchall())
            
        return {
            "total_sessions": total,
            "total_comparative_sessions": 0,
            "total_prompts": 0,
            "agent_distribution": agents,
            "status_distribution": status,
            "success_rate": 0.0,
            "success_rates": {},
            "average_metrics": {},
            "average_cost_usd": 0.0,
            "recent_sessions_24h": total,
            "prompt_complexity_distribution": {},
            "mcp_targets": [],
            "database_type": "sqlite",
            "database_size_mb": float(self.db_path.stat().st_size / 1024 / 1024) if self.db_path.exists() else 0.0
        }
        
    def generate_base_session_id(self, prompt_id: int, timestamp: Optional[int] = None) -> str:
        """Generate base session ID."""
        import time
        if timestamp is None:
            timestamp = int(time.time())
        return f"eval_prompt{prompt_id:03d}_{timestamp}"
        
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days."""
        pass
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions from SQLite database."""
        import sqlite3
        sessions = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT session_id, prompt_id, agent_type, status, created_at, completed_at,
                           error_message, response_data, metadata, agent_config, total_calls, tool_calls, tools_used
                    FROM evaluation_sessions
                    ORDER BY created_at DESC
                """)
                
                for row in cursor.fetchall():
                    # Parse tools_used JSON if present
                    tools_used = {}
                    if row[12]:  # tools_used column
                        try:
                            tools_used = json.loads(row[12])
                        except:
                            tools_used = {}
                    
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
                        'agent_config': row[9],
                        'total_calls': row[10] or 0,
                        'tool_calls': row[11] or 0,
                        'tools_used': tools_used,
                        'success': row[3] == 'completed'
                    }
                    sessions.append(session_data)
                    
        except Exception as e:
            logger.error(f"Failed to get all sessions: {e}")
        
        return sessions
        
    def store_session(self, session_data: SessionData) -> None:
        """Store session data."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            # Serialize tools_used dict as JSON
            tools_used_json = json.dumps(session_data.tools_used) if session_data.tools_used else None
            
            conn.execute("""
                INSERT OR REPLACE INTO evaluation_sessions 
                (session_id, prompt_id, agent_type, agent_config, status, error_message, response_data, metadata, total_calls, tool_calls, tools_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_data.session_id,
                session_data.prompt_id,
                session_data.agent_type,
                json.dumps({"type": session_data.agent_type}),
                "completed" if session_data.success else "failed",
                session_data.error_message,
                json.dumps(session_data.response_data) if session_data.response_data else None,
                json.dumps({"execution_time": session_data.execution_time, "cost_usd": session_data.cost_usd}),
                session_data.total_calls or 0,
                session_data.tool_calls or 0,
                tools_used_json
            ))
        logger.info(f"Stored session {session_data.session_id}")
        
    def store_comparative_session(self, comparative_data: "ComparativeSessionData") -> None:
        """Store comparative session data."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO comparative_sessions 
                (base_session_id, prompt_id, claude_result, opencode_result, comparative_analysis, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                comparative_data.base_session_id,
                comparative_data.prompt_id,
                json.dumps(comparative_data.claude_result) if comparative_data.claude_result else None,
                json.dumps(comparative_data.opencode_result) if comparative_data.opencode_result else None,
                json.dumps(comparative_data.comparative_analysis) if comparative_data.comparative_analysis else None
            ))
        logger.info(f"Stored comparative session {comparative_data.base_session_id}")
        
    def close(self):
        """Close connection."""
        pass


class InfluxDBSessionManager:
    """InfluxDB-based session manager for advanced time-series analytics."""
    
    def __init__(self, db_path: str = None, 
                 influxdb_url: str = "http://localhost:8086",
                 influxdb_token: str = "mcp-evaluation-token",
                 influxdb_org: str = "mcp-evaluation", 
                 influxdb_bucket: str = "evaluation-sessions"):
        """Initialize session manager with InfluxDB."""
        if not INFLUXDB_AVAILABLE:
            raise ImportError("InfluxDB client not available. Install with: uv pip install influxdb-client")
            
        self.influxdb_url = influxdb_url
        self.influxdb_token = influxdb_token
        self.influxdb_org = influxdb_org
        self.influxdb_bucket = influxdb_bucket
        
        # Initialize InfluxDB client
        try:
            self.client = InfluxDBClient(
                url=self.influxdb_url,
                token=self.influxdb_token,
                org=self.influxdb_org
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            
            # Test connection
            self._test_connection()
            logger.info(f"Connected to InfluxDB at {self.influxdb_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            raise
        
    def _test_connection(self):
        """Test InfluxDB connection."""
        try:
            # Simple health check
            health = self.client.health()
            if health.status != "pass":
                raise Exception(f"InfluxDB health check failed: {health.status}")
        except Exception as e:
            raise Exception(f"InfluxDB connection test failed: {e}")
            
    def create_session(self, session_id: str, prompt_id: int, agent_type: str, 
                      agent_config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new evaluation session."""
        try:
            # Create session point
            point = Point("evaluation_session") \
                .tag("session_id", session_id) \
                .tag("prompt_id", str(prompt_id)) \
                .tag("agent_type", agent_type) \
                .tag("status", "pending") \
                .field("agent_config", json.dumps(agent_config)) \
                .field("metadata", json.dumps(metadata or {})) \
                .field("created", True) \
                .time(datetime.now(timezone.utc), WritePrecision.NS)
                
            self.write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=point)
            
            logger.info(f"Created session {session_id} for prompt {prompt_id} with agent {agent_type}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            raise
        
    def start_session(self, session_id: str) -> None:
        """Mark session as started."""
        try:
            point = Point("evaluation_session") \
                .tag("session_id", session_id) \
                .tag("status", "running") \
                .field("started", True) \
                .time(datetime.now(timezone.utc), WritePrecision.NS)
                
            self.write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=point)
            
            logger.info(f"Started session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to start session {session_id}: {e}")
            raise
        
    def complete_session(self, session_id: str, response_data: Dict[str, Any]) -> None:
        """Mark session as completed with response data."""
        try:
            point = Point("evaluation_session") \
                .tag("session_id", session_id) \
                .tag("status", "completed") \
                .field("response_data", json.dumps(response_data)) \
                .field("completed", True) \
                .time(datetime.now(timezone.utc), WritePrecision.NS)
                
            self.write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=point)
            
            logger.info(f"Completed session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to complete session {session_id}: {e}")
            raise
        
    def fail_session(self, session_id: str, error_message: str) -> None:
        """Mark session as failed with error message."""
        try:
            point = Point("evaluation_session") \
                .tag("session_id", session_id) \
                .tag("status", "failed") \
                .field("error_message", error_message) \
                .field("failed", True) \
                .time(datetime.now(timezone.utc), WritePrecision.NS)
                
            self.write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=point)
            
            logger.error(f"Failed session {session_id}: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to record session failure {session_id}: {e}")
            raise
        
    def record_metric(self, session_id: str, metric_name: str, metric_value: float) -> None:
        """Record a performance metric for a session."""
        try:
            point = Point("performance_metrics") \
                .tag("session_id", session_id) \
                .tag("metric_name", metric_name) \
                .field("value", metric_value) \
                .time(datetime.now(timezone.utc), WritePrecision.NS)
                
            self.write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=point)
            
            logger.debug(f"Recorded metric {metric_name}={metric_value} for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name} for session {session_id}: {e}")
            raise
        
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details by ID."""
        try:
            # Query for ALL fields for this session to reconstruct complete data
            query = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "evaluation_session")
                |> filter(fn: (r) => r.session_id == "{session_id}")
                |> sort(columns: ["_time"], desc: true)
            '''
            
            result = self.query_api.query(org=self.influxdb_org, query=query)
            
            if not result:
                return None
            
            # Collect all field values for this session
            session_fields = {}
            session_metadata = {}
            
            for table in result:
                for record in table.records:
                    field_name = record.get_field()
                    field_value = record.get_value()
                    
                    # Store the most recent value for each field
                    if field_name not in session_fields:
                        session_fields[field_name] = field_value
                        
                        # Capture common metadata from any record
                        if not session_metadata:
                            session_metadata = {
                                "session_id": session_id,
                                "prompt_id": record.values.get("prompt_id", "0"),
                                "agent_type": record.values.get("agent_type", "unknown"),
                                "status": record.values.get("status", "unknown"),
                                "created_at": record.get_time().isoformat()
                            }
            
            if not session_fields:
                return None
            
            # Parse complex fields
            tools_used = {}
            if "tools_used" in session_fields:
                try:
                    tools_used = json.loads(session_fields["tools_used"]) if session_fields["tools_used"] else {}
                except (json.JSONDecodeError, TypeError):
                    tools_used = {}
            
            response_data = None
            if "response_data" in session_fields:
                try:
                    response_data = json.loads(session_fields["response_data"]) if session_fields["response_data"] else None
                except (json.JSONDecodeError, TypeError):
                    response_data = session_fields["response_data"]
            
            agent_config = {}
            if "agent_config" in session_fields:
                try:
                    agent_config = json.loads(session_fields["agent_config"]) if session_fields["agent_config"] else {}
                except (json.JSONDecodeError, TypeError):
                    agent_config = {}
            
            metadata = {}
            if "metadata" in session_fields:
                try:
                    metadata = json.loads(session_fields["metadata"]) if session_fields["metadata"] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            
            # Reconstruct complete session data
            session = {
                "session_id": session_id,
                "prompt_id": int(session_metadata.get("prompt_id", 0)),
                "agent_type": session_metadata.get("agent_type", "unknown"),
                "status": session_metadata.get("status", "unknown"),
                "created_at": session_metadata.get("created_at"),
                "agent_config": agent_config,
                "metadata": metadata,
                "response_data": response_data,
                "error_message": session_fields.get("error_message", ""),
                "success": session_fields.get("success", False),
                "execution_time": float(session_fields.get("execution_time", 0.0)),
                "cost_usd": float(session_fields.get("cost_usd", 0.0)),
                "tokens_used": int(session_fields.get("tokens_used", 0)),
                "total_calls": int(session_fields.get("total_calls", 0)),
                "tool_calls": int(session_fields.get("tool_calls", 0)),
                "tools_used": tools_used
            }
                
            return session
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
            
    def get_sessions_by_prompt(self, prompt_id: int) -> List[Dict[str, Any]]:
        """Get all sessions for a specific prompt."""
        try:
            query = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "evaluation_session")
                |> filter(fn: (r) => r.prompt_id == "{prompt_id}")
                |> filter(fn: (r) => r._field == "created" or r._field == "completed" or r._field == "failed")
                |> sort(columns: ["_time"], desc: true)
            '''
            
            result = self.query_api.query(org=self.influxdb_org, query=query)
            sessions = []
            
            for table in result:
                for record in table.records:
                    session = {
                        "session_id": record.values.get("session_id"),
                        "prompt_id": prompt_id,
                        "agent_type": record.values.get("agent_type"),
                        "status": record.values.get("status"),
                        "created_at": record.get_time().isoformat(),
                        "agent_config": json.loads(record.values.get("agent_config", "{}")),
                        "metadata": json.loads(record.values.get("metadata", "{}")),
                        "response_data": None,
                        "error_message": record.values.get("error_message")
                    }
                    
                    if record.values.get("response_data"):
                        session["response_data"] = json.loads(record.values.get("response_data"))
                        
                    sessions.append(session)
                    
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get sessions for prompt {prompt_id}: {e}")
            return []
            
    def get_session_metrics(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all metrics for a session."""
        try:
            query = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "performance_metrics")
                |> filter(fn: (r) => r.session_id == "{session_id}")
                |> sort(columns: ["_time"], desc: false)
            '''
            
            result = self.query_api.query(org=self.influxdb_org, query=query)
            metrics = []
            
            for table in result:
                for record in table.records:
                    metrics.append({
                        "session_id": session_id,
                        "metric_name": record.values.get("metric_name"),
                        "metric_value": record.get_value(),
                        "recorded_at": record.get_time().isoformat()
                    })
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics for session {session_id}: {e}")
            return []
            
    def get_comparative_analysis(self, prompt_id: int) -> Dict[str, Any]:
        """Get comparative analysis for all agents on a specific prompt."""
        sessions = self.get_sessions_by_prompt(prompt_id)
        
        if not sessions:
            return {"prompt_id": prompt_id, "sessions": [], "analysis": "No sessions found"}
            
        analysis = {
            "prompt_id": prompt_id,
            "total_sessions": len(sessions),
            "agents": {},
            "status_summary": {},
            "performance_metrics": {}
        }
        
        # Group by agent type
        for session in sessions:
            agent_type = session['agent_type']
            status = session['status']
            
            if agent_type not in analysis['agents']:
                analysis['agents'][agent_type] = []
            analysis['agents'][agent_type].append(session)
            
            # Status summary
            if status not in analysis['status_summary']:
                analysis['status_summary'][status] = 0
            analysis['status_summary'][status] += 1
            
            # Get metrics for this session
            metrics = self.get_session_metrics(session['session_id'])
            if metrics:
                if agent_type not in analysis['performance_metrics']:
                    analysis['performance_metrics'][agent_type] = {}
                    
                for metric in metrics:
                    metric_name = metric['metric_name']
                    metric_value = metric['metric_value']
                    
                    if metric_name not in analysis['performance_metrics'][agent_type]:
                        analysis['performance_metrics'][agent_type][metric_name] = []
                    analysis['performance_metrics'][agent_type][metric_name].append(metric_value)
        
        return analysis
        
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days."""
        # InfluxDB has built-in retention policies, but we can implement custom cleanup
        try:
            cutoff_time = datetime.now(timezone.utc).replace(microsecond=0) - \
                         datetime.timedelta(days=days_old)
            
            # For now, just log the cleanup request
            # In a real implementation, you might use InfluxDB's delete API
            logger.info(f"Cleanup requested for sessions older than {days_old} days (before {cutoff_time})")
            logger.info("Note: InfluxDB retention policies handle automatic cleanup")
            
            return 0  # Return 0 since we're relying on retention policies
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
            
    def store_session(self, session_data: SessionData) -> None:
        """Store session data in InfluxDB."""
        try:
            point = Point("evaluation_session") \
                .tag("session_id", session_data.session_id) \
                .tag("prompt_id", str(session_data.prompt_id)) \
                .tag("agent_type", session_data.agent_type) \
                .tag("status", "completed" if session_data.success else "failed") \
                .field("success", session_data.success) \
                .field("response_data", json.dumps(session_data.response_data) if session_data.response_data else "") \
                .field("error_message", session_data.error_message or "") \
                .field("execution_time", session_data.execution_time or 0.0) \
                .field("cost_usd", session_data.cost_usd or 0.0) \
                .field("tokens_used", session_data.tokens_used or 0) \
                .field("total_calls", session_data.total_calls or 0) \
                .field("tool_calls", session_data.tool_calls or 0) \
                .field("tools_used", json.dumps(session_data.tools_used) if session_data.tools_used else "{}") \
                .field("stored", True) \
                .time(datetime.now(timezone.utc), WritePrecision.NS)
                
            self.write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=point)
            
            logger.info(f"Stored session {session_data.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store session {session_data.session_id}: {e}")
            raise
            
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics and analytics."""
        try:
            # Get total sessions count
            query_total = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "evaluation_session")
                |> filter(fn: (r) => r._field == "stored")
                |> count()
            '''
            
            result_total = self.query_api.query(org=self.influxdb_org, query=query_total)
            total_sessions = 0
            if result_total:
                # Sum all tables since InfluxDB returns one table per session
                for table in result_total:
                    for record in table.records:
                        total_sessions += record.get_value()
            
            # Get agent distribution
            query_agents = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "evaluation_session")
                |> filter(fn: (r) => r._field == "stored")
                |> group(columns: ["agent_type"])
                |> count()
            '''
            
            result_agents = self.query_api.query(org=self.influxdb_org, query=query_agents)
            agent_distribution = {}
            if result_agents:
                for table in result_agents:
                    for record in table.records:
                        agent_type = record.values.get("agent_type", "unknown")
                        count = record.get_value()
                        agent_distribution[agent_type] = count
            
            # Get success rates by status
            query_status = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "evaluation_session")
                |> filter(fn: (r) => r._field == "success")
                |> group(columns: ["status"])
                |> count()
            '''
            
            result_status = self.query_api.query(org=self.influxdb_org, query=query_status)
            status_distribution = {}
            if result_status:
                for table in result_status:
                    for record in table.records:
                        status = record.values.get("status", "unknown")
                        count = record.get_value()
                        status_distribution[status] = count
            
            # Calculate success rate
            completed = status_distribution.get("completed", 0)
            failed = status_distribution.get("failed", 0)
            total_finished = completed + failed
            success_rate = (completed / total_finished * 100) if total_finished > 0 else 0
            
            # Get average cost
            query_cost = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "evaluation_session")
                |> filter(fn: (r) => r._field == "cost_usd")
                |> filter(fn: (r) => r._value > 0.0)
                |> mean()
            '''
            
            result_cost = self.query_api.query(org=self.influxdb_org, query=query_cost)
            avg_cost = 0.0
            if result_cost and result_cost[0].records:
                avg_cost = result_cost[0].records[0].get_value() or 0.0
            
            # Estimate database size by counting total data points
            query_size = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "evaluation_session")
                |> count()
            '''
            
            result_size = self.query_api.query(org=self.influxdb_org, query=query_size)
            total_data_points = 0
            if result_size:
                for table in result_size:
                    for record in table.records:
                        total_data_points += record.get_value()
            
            # Count comparative sessions
            query_comparative = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "comparative_session")
                |> filter(fn: (r) => r._field == "created")
                |> count()
            '''
            
            result_comparative = self.query_api.query(org=self.influxdb_org, query=query_comparative)
            total_comparative_sessions = 0
            if result_comparative:
                for table in result_comparative:
                    for record in table.records:
                        total_comparative_sessions += record.get_value()
            
            # Rough size estimation: ~1KB per data point average
            estimated_size_mb = (total_data_points * 1.0) / 1024
            
            return {
                "total_sessions": total_sessions,
                "total_comparative_sessions": total_comparative_sessions,
                "total_prompts": 0,  # Will be filled by evaluation engine
                "agent_distribution": agent_distribution,
                "status_distribution": status_distribution,
                "success_rate": success_rate,
                "success_rates": {},  # Will be calculated by evaluation engine
                "average_metrics": {},  # TODO: Implement performance metrics queries
                "average_cost_usd": avg_cost,
                "recent_sessions_24h": total_sessions,  # Simplified for now
                "prompt_complexity_distribution": {},  # Will be filled by evaluation engine
                "mcp_targets": [],  # Will be filled by evaluation engine
                "database_type": "influxdb",
                "database_size_mb": round(estimated_size_mb, 3),
                "bucket": self.influxdb_bucket,
                "organization": self.influxdb_org,
                "total_data_points": total_data_points
            }
            
        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            return {
                "total_sessions": 0,
                "total_comparative_sessions": 0,
                "total_prompts": 0,
                "agent_distribution": {},
                "status_distribution": {},
                "success_rate": 0.0,
                "success_rates": {},
                "average_metrics": {},
                "average_cost_usd": 0.0,
                "recent_sessions_24h": 0,
                "prompt_complexity_distribution": {},
                "mcp_targets": [],
                "database_type": "influxdb",
                "database_size_mb": 0.0,
                "error": str(e)
            }
            
    def generate_base_session_id(self, prompt_id: int, timestamp: Optional[int] = None) -> str:
        """
        Generate consistent base session ID format for evaluation.
        
        Args:
            prompt_id: Prompt identifier
            timestamp: Unix timestamp (defaults to current time)
            
        Returns:
            Formatted base session ID: eval_prompt{id:03d}_{timestamp}
        """
        import time
        if timestamp is None:
            timestamp = int(time.time())
        return f"eval_prompt{prompt_id:03d}_{timestamp}"
            
    def store_comparative_session(self, comparative_data: "ComparativeSessionData") -> None:
        """Store comparative session data in InfluxDB."""
        try:
            point = Point("comparative_session") \
                .tag("base_session_id", comparative_data.base_session_id) \
                .tag("prompt_id", str(comparative_data.prompt_id)) \
                .field("claude_result", json.dumps(comparative_data.claude_result) if comparative_data.claude_result else "") \
                .field("opencode_result", json.dumps(comparative_data.opencode_result) if comparative_data.opencode_result else "") \
                .field("comparative_analysis", json.dumps(comparative_data.comparative_analysis) if comparative_data.comparative_analysis else "") \
                .field("created", True) \
                .time(datetime.now(timezone.utc), WritePrecision.NS)
                
            self.write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=point)
            
            logger.info(f"Stored comparative session {comparative_data.base_session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store comparative session {comparative_data.base_session_id}: {e}")
            raise
            
    def close(self):
        """Close InfluxDB connection."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("Closed InfluxDB connection")
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions from InfluxDB."""
        sessions = []
        
        try:
            # Query to get all unique sessions
            query = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: -365d)
                |> filter(fn: (r) => r._measurement == "evaluation_session")
                |> filter(fn: (r) => r._field == "stored")
                |> group(columns: ["session_id"])
                |> last()
            '''
            
            result = self.query_api.query(org=self.influxdb_org, query=query)
            
            for table in result:
                for record in table.records:
                    session_id = record.values.get("session_id")
                    if session_id:
                        # Get full session details
                        session_data = self.get_session(session_id)
                        if session_data:
                            sessions.append(session_data)
                            
        except Exception as e:
            logger.error(f"Failed to get all sessions from InfluxDB: {e}")
        
        return sessions


# Compatibility alias - use the factory function by default
SessionManager = create_session_manager
