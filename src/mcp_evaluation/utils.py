"""
Shared utilities for MCP evaluation system.
Contains common functions used across multiple modules.
"""

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def extract_model_from_config(agent_config: Dict[str, Any], agent_type: str = "unknown") -> str:
    """
    Extract model name from agent configuration.
    
    This is the unified version used across all modules to ensure consistency.
    """
    if not agent_config:
        # Use default models based on agent type
        if agent_type == "claude":
            return "claude-3-5-sonnet-20241022"
        elif agent_type == "opencode":
            return "github-copilot/claude-3.5-sonnet"
        return "unknown"
    
    # Try different configuration patterns
    model = agent_config.get("model", "")
    if model:
        return model
    
    # Check for nested model configs
    if "config" in agent_config:
        nested_model = agent_config["config"].get("model", "")
        if nested_model:
            return nested_model
    
    # Default models based on agent type
    if agent_type == "claude":
        return "claude-3-5-sonnet-20241022"
    elif agent_type == "opencode":
        return "github-copilot/claude-3.5-sonnet"
    
    return "unknown"


def safe_json_loads(data: Any) -> Dict[str, Any]:
    """
    Safely load JSON data, returning empty dict if parsing fails.
    """
    if isinstance(data, dict):
        return data
    
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return {}
    
    return {}


def calculate_execution_time_from_timestamps(
    created_at: Optional[str], 
    completed_at: Optional[str]
) -> float:
    """Calculate execution time from timestamp strings."""
    if not created_at or not completed_at:
        return 0.0
    
    try:
        from datetime import datetime
        
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


def extract_model_from_influxdb_values(values: Dict[str, Any]) -> str:
    """Extract model name from InfluxDB record values."""
    
    # Try direct model field
    if 'model' in values:
        return values['model']
    
    # Try to parse from agent_config
    agent_config = values.get('agent_config')
    if agent_config:
        parsed_config = safe_json_loads(agent_config)
        agent_type = values.get('agent_type', 'unknown')
        return extract_model_from_config(parsed_config, agent_type)
    
    # Default based on agent type
    agent_type = values.get('agent_type', 'unknown')
    return extract_model_from_config({}, agent_type)
