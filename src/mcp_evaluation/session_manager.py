"""Session management for MCP evaluations with no persistent storage."""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def create_session_manager(**kwargs):
    """
    Factory function to create in-memory session manager only.
    
    Args:
        **kwargs: Configuration parameters (ignored)
    
    Returns:
        InMemorySessionManager instance
    """
    # Always use in-memory storage - no persistent data storage
    return InMemorySessionManager()


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


class InMemorySessionManager:
    """In-memory session manager with no persistent storage."""
    
    def __init__(self):
        """Initialize session manager with in-memory storage."""
        self.sessions = {}  # Dict[str, SessionData]
        self.comparative_sessions = {}  # Dict[str, ComparativeSessionData]
        logger.info("Initialized in-memory session manager (no persistent storage)")
        
    def generate_base_session_id(self, prompt_id: int) -> str:
        """Generate base session ID for comparative evaluation."""
        timestamp = int(datetime.now(timezone.utc).timestamp())
        return f"eval_prompt{prompt_id:03d}_{timestamp}"
        
    def create_session(self, session_id: str, prompt_id: int, agent_type: str, 
                      agent_config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new evaluation session."""
        logger.info(f"Created session {session_id} for prompt {prompt_id} with agent {agent_type}")
        return session_id
        
    def start_session(self, session_id: str) -> None:
        """Mark session as started."""
        logger.info(f"Started session {session_id}")
        
    def complete_session(self, session_id: str, response_data: Dict[str, Any]) -> None:
        """Mark session as completed with response data."""
        logger.info(f"Completed session {session_id}")
        
    def fail_session(self, session_id: str, error_message: str) -> None:
        """Mark session as failed with error message."""
        logger.error(f"Failed session {session_id}: {error_message}")
        
    def record_metric(self, session_id: str, metric_name: str, metric_value: float) -> None:
        """Record a performance metric for a session."""
        logger.debug(f"Recorded metric {metric_name}={metric_value} for session {session_id}")
        
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details by ID."""
        session_data = self.sessions.get(session_id)
        if session_data:
            return session_data.dict()
        return None
    
    def get_sessions_by_prompt_id(self, prompt_id: int) -> List[Dict[str, Any]]:
        """Get all sessions for a prompt ID."""
        matching_sessions = []
        for session_data in self.sessions.values():
            if session_data.prompt_id == prompt_id:
                matching_sessions.append(session_data.dict())
        return matching_sessions
    
    def store_session(self, session_data: SessionData) -> None:
        """Store session data in memory."""
        self.sessions[session_data.session_id] = session_data
        logger.info(f"Stored session {session_data.session_id} in memory")
            
    def store_comparative_session(self, comparative_data: ComparativeSessionData) -> None:
        """Store comparative session data in memory."""
        self.comparative_sessions[comparative_data.base_session_id] = comparative_data
        logger.info(f"Stored comparative session {comparative_data.base_session_id} in memory")
    
    def get_comparative_results_by_base_session_id(self, base_session_id: str) -> Optional[Dict[str, Any]]:
        """Get comparative results by base session ID."""
        comparative_data = self.comparative_sessions.get(base_session_id)
        if comparative_data:
            return comparative_data.dict()
        return None
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        total_sessions = len(self.sessions)
        comparative_sessions = len(self.comparative_sessions)
        
        # Calculate recent sessions (last 24 hours)
        yesterday_timestamp = int(datetime.now(timezone.utc).timestamp()) - 86400
        recent_sessions = sum(1 for session in self.sessions.values() 
                            if session.timestamp and session.timestamp > yesterday_timestamp)
        
        # Calculate average cost
        costs = [session.cost_usd for session in self.sessions.values() if session.cost_usd is not None]
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        
        return {
            "total_sessions": total_sessions,
            "total_comparative_sessions": comparative_sessions,
            "comparative_sessions": comparative_sessions,
            "recent_sessions": recent_sessions,
            "recent_sessions_24h": recent_sessions,
            "average_cost_usd": avg_cost,
            "database_size_mb": 0.0,  # No persistent storage
            "total_prompts": 0,  # Will be filled by evaluation engine
            "agent_distribution": {},  # Will be filled by evaluation engine
            "status_distribution": {},  # Will be filled by evaluation engine
            "success_rate": 0.0,  # Will be filled by evaluation engine
            "database_type": "in-memory"
        }
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all stored sessions."""
        return [session.dict() for session in self.sessions.values()]


# Compatibility alias
SessionManager = create_session_manager
