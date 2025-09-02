"""MCP Evaluation Infrastructure - Core package"""

__version__ = "0.1.0"

from .unified_agent import UnifiedAgent
from .evaluation_engine import EvaluationEngine
from .session_manager import SessionManager
from .prompt_loader import MarkdownPromptLoader

__all__ = [
    "UnifiedAgent",
    "EvaluationEngine", 
    "SessionManager",
    "MarkdownPromptLoader",
]
