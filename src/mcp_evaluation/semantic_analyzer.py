"""
Semantic Analysis Engine for MCP Evaluation System

This module provides intelligent semantic analysis of evaluation results using Claude
to assess task completion quality, response appropriateness, and provide actionable insights
beyond mechanical success detection.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .unified_agent import UnifiedAgent, AgentConfig
from .post_processor import EvaluationMetrics, MonitoringSession

logger = logging.getLogger(__name__)


@dataclass
class TaskComprehensionScore:
    """Assessment of how well the agent understood the prompt."""
    understood_correctly: bool
    interpretation_accuracy: float  # 0.0-1.0
    context_awareness: float  # 0.0-1.0
    missing_requirements: List[str]


@dataclass
class ApproachQualityScore:
    """Quality of the agent's problem-solving approach."""
    logical_coherence: float  # 0.0-1.0
    methodology_appropriateness: float  # 0.0-1.0
    execution_strategy: str
    alternative_approaches: List[str]


@dataclass
class ToolEffectivenessScore:
    """Assessment of tool selection and usage."""
    appropriate_selection: bool
    usage_efficiency: float  # 0.0-1.0
    missing_tools: List[str]
    unnecessary_tools: List[str]
    execution_quality: float  # 0.0-1.0


@dataclass
class ResponseCompletenessScore:
    """Completeness and quality of agent response."""
    completeness: float  # 0.0-1.0
    accuracy: float  # 0.0-1.0
    clarity: float  # 0.0-1.0
    actionability: float  # 0.0-1.0
    user_value: float  # 0.0-1.0


@dataclass
class SemanticAnalysis:
    """Comprehensive semantic analysis results for a single session."""
    # Core Assessment
    session_id: str
    agent_type: str
    semantic_success: bool  # True if task actually completed despite technical issues
    confidence_score: float  # 0.0-1.0 confidence in semantic assessment
    quality_score: float  # 0.0-1.0 overall response quality
    
    # Detailed Evaluation
    task_comprehension: TaskComprehensionScore
    approach_quality: ApproachQualityScore
    tool_effectiveness: ToolEffectivenessScore
    response_completeness: ResponseCompletenessScore
    
    # Insights
    failure_root_cause: Optional[str]
    improvement_suggestions: List[str]
    false_negative_detected: bool
    
    # Metadata
    analysis_timestamp: str
    analysis_cost_usd: float
    analysis_model: str


@dataclass
class ComparativeSemanticAnalysis:
    """Comparative analysis between Claude and OpenCode results."""
    prompt_id: str
    better_performer: str  # "claude", "opencode", "tie"
    performance_gap: float  # Quantified difference (0.0-1.0)
    
    # Agent-specific insights
    claude_strengths: List[str]
    opencode_strengths: List[str]
    complementary_insights: List[str]
    
    # Quality comparison
    claude_quality_score: float
    opencode_quality_score: float
    approach_differences: List[str]
    
    # Metadata
    analysis_timestamp: str
    analysis_cost_usd: float


@dataclass
class BatchSemanticInsights:
    """Pattern analysis across multiple evaluation sessions."""
    total_sessions_analyzed: int
    semantic_success_rate: float
    technical_success_rate: float
    false_negative_rate: float
    average_quality_score: float
    
    # Pattern insights
    top_failure_types: List[str]
    tool_usage_patterns: Dict[str, float]
    agent_performance_comparison: Dict[str, float]
    improvement_opportunities: List[str]
    
    # Cost tracking
    total_analysis_cost: float
    analysis_timestamp: str


class AnalysisPrompts:
    """Structured prompts for different types of semantic analysis."""
    
    SINGLE_SESSION_ANALYSIS = """
You are an expert evaluator analyzing MCP (Model Context Protocol) evaluation sessions. Your task is to provide comprehensive semantic analysis that goes beyond technical success/failure to assess actual task completion quality.

**Session Context:**
- Session ID: {session_id}
- Agent Type: {agent_type}
- Model: {model}
- Prompt: {prompt_text}
- Technical Success: {technical_success}
- Execution Time: {execution_time}s
- Tools Used: {tools_used}
- Tool Calls: {tool_calls}
- Response Length: {response_length} chars

**Agent Response/Output:**
{agent_response}

**Communication Log Summary:**
{communication_log_summary}

**Tool Execution Details:**
{tool_execution_details}

**Analysis Instructions:**
Please provide a comprehensive semantic analysis addressing these key questions:

1. **Task Completion Assessment:** Did the agent actually complete the requested task, regardless of technical status?
2. **Task Understanding:** How well did the agent understand the prompt requirements and context?
3. **Approach Quality:** Was the problem-solving approach logical, appropriate, and well-executed?
4. **Tool Usage:** Were the right tools selected and used effectively?
5. **Response Quality:** How complete, accurate, clear, and actionable is the response?
6. **Failure Analysis:** If there was a failure, what was the root cause and type?
7. **False Negative Detection:** Is this a case where the task succeeded but was marked as failed?

**Required Output Format:**
You MUST respond with ONLY a valid JSON object. No explanatory text, no markdown formatting, no backticks, no code blocks. Start your response immediately with the opening brace {{ and end with the closing brace }}.

{{
  "semantic_success": true,
  "confidence_score": 0.85,
  "quality_score": 0.78,
  "task_comprehension": {{
    "understood_correctly": true,
    "interpretation_accuracy": 0.9,
    "context_awareness": 0.8,
    "missing_requirements": ["example requirement"]
  }},
  "approach_quality": {{
    "logical_coherence": 0.85,
    "methodology_appropriateness": 0.9,
    "execution_strategy": "systematic approach",
    "alternative_approaches": ["alternative method"]
  }},
  "tool_effectiveness": {{
    "appropriate_selection": true,
    "usage_efficiency": 0.8,
    "missing_tools": ["tool name"],
    "unnecessary_tools": ["tool name"],
    "execution_quality": 0.85
  }},
  "response_completeness": {{
    "completeness": 0.9,
    "accuracy": 0.95,
    "clarity": 0.8,
    "actionability": 0.85,
    "user_value": 0.88
  }},
  "failure_root_cause": null,
  "improvement_suggestions": ["suggestion 1", "suggestion 2"],
  "false_negative_detected": false,
  "reasoning": "Detailed reasoning for the analysis"
}}

IMPORTANT: Your response must be valid JSON only. Do not include any explanatory text before or after the JSON.
"""

    COMPARATIVE_ANALYSIS = """
You are comparing two MCP evaluation agent responses to the same prompt. Analyze their approaches, quality, and effectiveness.

**Prompt:** {prompt_text}

**Claude Response:**
- Technical Success: {claude_success}
- Execution Time: {claude_time}s
- Cost: ${claude_cost}
- Tools Used: {claude_tools}
- Response: {claude_response}

**OpenCode Response:**  
- Technical Success: {opencode_success}
- Execution Time: {opencode_time}s
- Tools Used: {opencode_tools}
- Response: {opencode_response}

**Analysis Instructions:**
Compare these responses across multiple dimensions:

1. **Overall Performance:** Which agent performed better and by how much?
2. **Approach Differences:** How did their problem-solving strategies differ?
3. **Tool Usage:** Compare tool selection and usage effectiveness
4. **Response Quality:** Compare completeness, accuracy, and usefulness
5. **Efficiency:** Consider time, cost, and resource usage
6. **Strengths:** What are the unique strengths of each agent?
7. **Complementary Value:** Do the responses provide different valuable insights?

**Required Output Format:**
Respond ONLY with a valid JSON object (no markdown code blocks):

{{
  "better_performer": "claude",
  "performance_gap": 0.25,
  "claude_strengths": ["strength 1", "strength 2"],
  "opencode_strengths": ["strength 1", "strength 2"],
  "complementary_insights": ["insight 1", "insight 2"],
  "claude_quality_score": 0.85,
  "opencode_quality_score": 0.70,
  "approach_differences": ["difference 1", "difference 2"],
  "reasoning": "Detailed comparison reasoning"
}}

Provide balanced, objective analysis focusing on actionable insights.
"""

    BATCH_ANALYSIS = """
You are analyzing patterns across multiple MCP evaluation sessions to identify systemic insights and optimization opportunities.

**Batch Summary:**
- Total Sessions: {total_sessions}
- Technical Success Rate: {technical_success_rate}%
- Average Execution Time: {avg_execution_time}s
- Most Used Tools: {top_tools}
- Agent Distribution: {agent_distribution}

**Session Details:**
{session_summaries}

**Analysis Instructions:**
Identify systemic patterns and provide strategic insights:

1. **Success Patterns:** What characteristics correlate with success/failure?
2. **Tool Effectiveness:** Which tools are most/least effective?
3. **Agent Comparison:** How do Claude and OpenCode compare systematically?
4. **Quality Trends:** What patterns emerge in response quality?
5. **Optimization Opportunities:** Where are the biggest improvement opportunities?
6. **False Negative Patterns:** Are there systematic false negatives?

**Required Output Format:**
Respond ONLY with a valid JSON object (no markdown code blocks):

{{
  "semantic_success_rate": 0.85,
  "false_negative_rate": 0.15,
  "average_quality_score": 0.78,
  "top_failure_types": ["timeout", "incomplete"],
  "tool_usage_patterns": {{
    "tool_name": 0.8
  }},
  "agent_performance_comparison": {{
    "claude": 0.85,
    "opencode": 0.78
  }},
  "improvement_opportunities": ["opportunity 1", "opportunity 2"],
  "key_insights": ["insight 1", "insight 2"],
  "recommendations": ["recommendation 1", "recommendation 2"]
}}
"""


class SemanticAnalysisEngine:
    """
    Intelligent post-evaluation analysis using Claude for semantic understanding.
    """
    
    def __init__(self, claude_model: str = "sonnet", config: Optional[Dict] = None):
        """
        Initialize semantic analysis engine.
        
        Args:
            claude_model: Claude model to use for analysis ('haiku', 'sonnet', 'opus')
            config: Optional configuration dictionary
        """
        self.claude_model = claude_model
        self.config = config or self._load_default_config()
        
        # Create separate Claude client for analysis
        self.analysis_client = UnifiedAgent("claude", claude_model)
        
        logger.info(f"Initialized SemanticAnalysisEngine with model: {claude_model}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default semantic analysis configuration."""
        return {
            "confidence_threshold": 0.7,
            "max_cost_per_session": 0.05,
            "batch_size": 5,
            "enable_caching": True,
            "analysis_timeout": 60.0
        }
    
    def analyze_session_semantics(self, 
                                 session_metrics: EvaluationMetrics,
                                 session_data: MonitoringSession,
                                 prompt_context: str) -> SemanticAnalysis:
        """
        Perform comprehensive semantic analysis of individual session.
        
        Args:
            session_metrics: Technical evaluation metrics
            session_data: Raw monitoring session data
            prompt_context: Original prompt text
            
        Returns:
            SemanticAnalysis with comprehensive assessment
        """
        logger.info(f"Starting semantic analysis for session: {session_metrics.session_id}")
        
        start_time = time.time()
        
        # Prepare analysis context
        analysis_context = self._prepare_session_context(
            session_metrics, session_data, prompt_context
        )
        
        # Execute semantic analysis
        try:
            print(f"[DEBUG] Preparing analysis context for session: {session_metrics.session_id}")
            analysis_context = self._prepare_session_context(
                session_metrics, session_data, prompt_context
            )
            print(f"[DEBUG] Analysis context keys: {list(analysis_context.keys())}")
            
            print(f"[DEBUG] Attempting to format analysis prompt...")
            analysis_prompt = AnalysisPrompts.SINGLE_SESSION_ANALYSIS.format(**analysis_context)
            print(f"[DEBUG] Analysis prompt formatted successfully, length: {len(analysis_prompt)}")
            
            # Get analysis from Claude
            agent_config = AgentConfig(
                type="claude",
                model=self.claude_model,
                output_format="json",
                dangerously_skip_permissions=True
            )
            
            print(f"[DEBUG] About to call UnifiedAgent with config: {agent_config}")
            response = self.analysis_client.execute(analysis_prompt, agent_config)
            print(f"[DEBUG] UnifiedAgent response received")
            print(f"[DEBUG] Response success: {response.success}")
            print(f"[DEBUG] Response type: {type(response)}")
            
            if hasattr(response, 'error_message'):
                print(f"[DEBUG] Error message: {response.error_message}")
            if hasattr(response, 'response'):
                print(f"[DEBUG] Response field type: {type(response.response)}")
                print(f"[DEBUG] Response field length: {len(response.response) if response.response else 'None'}")
                print(f"[DEBUG] Response field content: {repr(response.response[:200] if response.response else 'None')}")
            
            if not response.success:
                print(f"[ERROR] Claude API call failed: {response.error_message}")
                raise Exception(f"Analysis failed: {response.error_message}")
            
            print(f"[DEBUG] Claude API success: {response.success}")
            print(f"[DEBUG] Raw response type: {type(response.response)}")
            print(f"[DEBUG] Raw response length: {len(response.response) if response.response else 'None'}")
            
            # Parse analysis results - handle potential formatting issues
            response_text = response.response.strip()
            
            # Debug: log the raw response for troubleshooting
            logger.debug(f"Raw semantic analysis response ({len(response_text)} chars): {response_text}")
            print(f"[DEBUG] Claude response length: {len(response_text)}")
            print(f"[DEBUG] Claude response content: {repr(response_text)}")
            
            # Check if response seems complete
            if not response_text:
                raise Exception("Empty response from Claude")
            
            if len(response_text) < 20:
                logger.warning(f"Suspiciously short response: '{response_text}'")
                print(f"[WARNING] Very short response: {repr(response_text)}")
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Clean up common formatting issues
            response_text = response_text.strip()
            
            # Try to extract JSON if wrapped in other text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end+1]
                logger.debug(f"Extracted JSON object ({len(json_text)} chars): {json_text}")
                
                try:
                    analysis_data = json.loads(json_text)
                    logger.debug("Successfully parsed JSON")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    logger.error(f"Problematic JSON: {repr(json_text)}")
                    # Try to fix common JSON issues
                    fixed_json = json_text.replace('\n', '').replace('\r', '')
                    try:
                        analysis_data = json.loads(fixed_json)
                        logger.debug("Fixed JSON by removing newlines")
                    except json.JSONDecodeError as e2:
                        logger.error(f"Still can't parse JSON after fixes: {e2}")
                        # Try one more time with aggressive cleaning
                        try:
                            clean_json = re.sub(r'\s+', ' ', json_text)
                            clean_json = re.sub(r',\s*}', '}', clean_json)
                            clean_json = re.sub(r',\s*]', ']', clean_json)
                            analysis_data = json.loads(clean_json)
                            logger.debug("Fixed JSON with aggressive cleaning")
                        except json.JSONDecodeError as e3:
                            logger.error(f"Final JSON parsing attempt failed: {e3}")
                            raise Exception(f"Failed to parse JSON response after multiple attempts: {str(e)}")
            else:
                logger.warning(f"No JSON braces found in response: {repr(response_text)}")
                try:
                    analysis_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Direct JSON decode error: {e}")
                    logger.error(f"Full response: {repr(response_text)}")
                    raise Exception(f"Failed to parse response as JSON: {str(e)}")
            
            analysis_cost = response.cost_usd or 0.0
            
            # Create semantic analysis object
            semantic_analysis = self._create_semantic_analysis(
                session_metrics, analysis_data, analysis_cost
            )
            
            duration = time.time() - start_time
            logger.info(f"Semantic analysis completed in {duration:.2f}s, cost: ${analysis_cost:.4f}")
            
            return semantic_analysis
            
        except Exception as e:
            print(f"[ERROR] Semantic analysis exception: {e}")
            print(f"[ERROR] Exception type: {type(e)}")
            logger.error(f"Semantic analysis failed for {session_metrics.session_id}: {e}")
            # Return default analysis on failure
            return self._create_fallback_analysis(session_metrics, str(e))
    
    def analyze_comparative_semantics(self, 
                                    claude_result: EvaluationMetrics,
                                    opencode_result: EvaluationMetrics, 
                                    prompt_context: str) -> ComparativeSemanticAnalysis:
        """
        Compare semantic quality between Claude and OpenCode results.
        
        Args:
            claude_result: Claude evaluation metrics
            opencode_result: OpenCode evaluation metrics
            prompt_context: Original prompt text
            
        Returns:
            ComparativeSemanticAnalysis with detailed comparison
        """
        logger.info(f"Starting comparative semantic analysis for prompt: {claude_result.prompt}")
        
        start_time = time.time()
        
        # Prepare comparative context
        comparison_context = self._prepare_comparative_context(
            claude_result, opencode_result, prompt_context
        )
        
        try:
            analysis_prompt = AnalysisPrompts.COMPARATIVE_ANALYSIS.format(**comparison_context)
            
            # Get comparative analysis from Claude
            agent_config = AgentConfig(
                type="claude",
                model=self.claude_model,
                output_format="json",
                dangerously_skip_permissions=True
            )
            
            response = self.analysis_client.execute(analysis_prompt, agent_config)
            
            if not response.success:
                raise Exception(f"Comparative analysis failed: {response.error_message}")
            
            # Parse analysis results - handle potential formatting issues
            response_text = response.response.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Try to extract JSON if wrapped in other text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end+1]
                analysis_data = json.loads(json_text)
            else:
                analysis_data = json.loads(response_text)
            
            analysis_cost = response.cost_usd or 0.0
            
            # Create comparative analysis object
            comparative_analysis = ComparativeSemanticAnalysis(
                prompt_id=claude_result.prompt or "unknown",
                better_performer=analysis_data.get("better_performer", "tie"),
                performance_gap=analysis_data.get("performance_gap", 0.0),
                claude_strengths=analysis_data.get("claude_strengths", []),
                opencode_strengths=analysis_data.get("opencode_strengths", []),
                complementary_insights=analysis_data.get("complementary_insights", []),
                claude_quality_score=analysis_data.get("claude_quality_score", 0.0),
                opencode_quality_score=analysis_data.get("opencode_quality_score", 0.0),
                approach_differences=analysis_data.get("approach_differences", []),
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                analysis_cost_usd=analysis_cost
            )
            
            duration = time.time() - start_time
            logger.info(f"Comparative analysis completed in {duration:.2f}s, cost: ${analysis_cost:.4f}")
            
            return comparative_analysis
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            # Return default analysis on failure
            return self._create_fallback_comparative_analysis(claude_result, opencode_result, str(e))
    
    def analyze_batch_patterns(self, 
                             session_results: List[EvaluationMetrics]) -> BatchSemanticInsights:
        """
        Identify patterns across multiple evaluation sessions.
        
        Args:
            session_results: List of evaluation metrics to analyze
            
        Returns:
            BatchSemanticInsights with pattern analysis
        """
        logger.info(f"Starting batch pattern analysis for {len(session_results)} sessions")
        
        start_time = time.time()
        
        # Prepare batch context
        batch_context = self._prepare_batch_context(session_results)
        
        try:
            analysis_prompt = AnalysisPrompts.BATCH_ANALYSIS.format(**batch_context)
            
            # Get batch analysis from Claude
            agent_config = AgentConfig(
                type="claude",
                model=self.claude_model,
                output_format="json",
                dangerously_skip_permissions=True
            )
            
            response = self.analysis_client.execute(analysis_prompt, agent_config)
            
            if not response.success:
                raise Exception(f"Batch analysis failed: {response.error_message}")
            
            # Parse analysis results - handle potential formatting issues
            response_text = response.response.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Try to extract JSON if wrapped in other text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end+1]
                analysis_data = json.loads(json_text)
            else:
                analysis_data = json.loads(response_text)
            
            analysis_cost = response.cost_usd or 0.0
            
            # Create batch insights object
            batch_insights = BatchSemanticInsights(
                total_sessions_analyzed=len(session_results),
                semantic_success_rate=analysis_data.get("semantic_success_rate", 0.0),
                technical_success_rate=batch_context["technical_success_rate"],
                false_negative_rate=analysis_data.get("false_negative_rate", 0.0),
                average_quality_score=analysis_data.get("average_quality_score", 0.0),
                top_failure_types=analysis_data.get("top_failure_types", []),
                tool_usage_patterns=analysis_data.get("tool_usage_patterns", {}),
                agent_performance_comparison=analysis_data.get("agent_performance_comparison", {}),
                improvement_opportunities=analysis_data.get("improvement_opportunities", []),
                total_analysis_cost=analysis_cost,
                analysis_timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            duration = time.time() - start_time
            logger.info(f"Batch analysis completed in {duration:.2f}s, cost: ${analysis_cost:.4f}")
            
            return batch_insights
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            # Return default analysis on failure
            return self._create_fallback_batch_analysis(session_results, str(e))
    
    def _prepare_session_context(self, 
                               session_metrics: EvaluationMetrics,
                               session_data: MonitoringSession,
                               prompt_context: str) -> Dict[str, Any]:
        """Prepare context for single session analysis."""
        return {
            "session_id": session_metrics.session_id,
            "agent_type": session_metrics.agent_type,
            "model": session_metrics.model or "unknown",
            "prompt_text": prompt_context,
            "technical_success": session_metrics.success,
            "execution_time": session_metrics.execution_time,
            "tools_used": "; ".join(session_metrics.tools_used) if session_metrics.tools_used else "None",
            "tool_calls": session_metrics.number_of_tool_calls,
            "response_length": session_metrics.response_length,
            "agent_response": self._extract_agent_response(session_data, session_metrics),
            "communication_log_summary": self._summarize_communication_log(session_data),
            "tool_execution_details": self._extract_tool_execution_details(session_data)
        }
    
    def _prepare_comparative_context(self,
                                   claude_result: EvaluationMetrics,
                                   opencode_result: EvaluationMetrics,
                                   prompt_context: str) -> Dict[str, Any]:
        """Prepare context for comparative analysis."""
        return {
            "prompt_text": prompt_context,
            "claude_success": claude_result.success,
            "claude_time": claude_result.execution_time,
            "claude_cost": claude_result.cost_usd or 0.0,
            "claude_tools": "; ".join(claude_result.tools_used) if claude_result.tools_used else "None",
            "claude_response": f"Response length: {claude_result.response_length} chars",
            "opencode_success": opencode_result.success,
            "opencode_time": opencode_result.execution_time,
            "opencode_tools": "; ".join(opencode_result.tools_used) if opencode_result.tools_used else "None",
            "opencode_response": f"Response length: {opencode_result.response_length} chars"
        }
    
    def _prepare_batch_context(self, session_results: List[EvaluationMetrics]) -> Dict[str, Any]:
        """Prepare context for batch analysis."""
        total_sessions = len(session_results)
        successful_sessions = sum(1 for s in session_results if s.success)
        technical_success_rate = (successful_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        avg_execution_time = sum(s.execution_time for s in session_results) / total_sessions if total_sessions > 0 else 0
        
        # Count tool usage
        tool_counts = {}
        for session in session_results:
            if session.tools_used:
                for tool in session.tools_used:
                    tool_name = tool.split(':')[0] if ':' in tool else tool
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Agent distribution
        agent_counts = {}
        for session in session_results:
            agent_counts[session.agent_type] = agent_counts.get(session.agent_type, 0) + 1
        
        # Session summaries (first 10 for analysis)
        session_summaries = []
        for i, session in enumerate(session_results[:10]):
            summary = f"Session {i+1}: {session.agent_type}, Success: {session.success}, Tools: {len(session.tools_used or [])}, Time: {session.execution_time:.2f}s"
            session_summaries.append(summary)
        
        return {
            "total_sessions": total_sessions,
            "technical_success_rate": technical_success_rate,
            "avg_execution_time": avg_execution_time,
            "top_tools": ", ".join([f"{tool}({count})" for tool, count in top_tools]),
            "agent_distribution": agent_counts,
            "session_summaries": "\n".join(session_summaries)
        }
    
    def _extract_agent_response(self, session_data: MonitoringSession, session_metrics: EvaluationMetrics = None) -> str:
        """Extract agent response from session data."""
        # First try to get response content from evaluation metrics if available
        if session_metrics and hasattr(session_metrics, 'response_content') and session_metrics.response_content:
            response_content = session_metrics.response_content
            # Truncate very long responses for analysis
            if len(response_content) > 2000:
                return response_content[:2000] + "... [truncated]"
            return response_content
        
        # Fallback to original logic if no response content available
        if hasattr(session_data, 'events') and session_data.events:
            # Try to find response in events
            for event in reversed(session_data.events):  # Check latest events first
                if isinstance(event, dict):
                    # Look for response content in various fields
                    for field in ['response', 'content', 'text', 'message']:
                        if field in event and event[field]:
                            return str(event[field])[:1000]  # Limit length
        
        # If no content found but we have metrics, show the discrepancy
        if session_metrics:
            response_length = getattr(session_metrics, 'response_length', 0)
            if response_length > 0:
                return f"Response generated ({response_length} characters) but content not accessible for analysis"
            else:
                return "No response content found"
        
        return f"Response length: {getattr(session_data, 'response_length', 0)} characters"
    
    def _summarize_communication_log(self, session_data: MonitoringSession) -> str:
        """Create summary of communication log."""
        if hasattr(session_data, 'events') and session_data.events:
            event_types = [event.get('type', 'unknown') for event in session_data.events if isinstance(event, dict)]
            event_summary = f"Events: {len(event_types)} total, Types: {', '.join(set(event_types))}"
            return event_summary
        
        return f"Session duration: {getattr(session_data, 'event_count', 0)} events"
    
    def _extract_tool_execution_details(self, session_data: MonitoringSession) -> str:
        """Extract tool execution details from session data."""
        if hasattr(session_data, 'tools_used') and session_data.tools_used:
            tools_summary = f"Tools used: {', '.join(session_data.tools_used)}"
            return tools_summary
        
        return "No tool execution details available"
    
    def _create_semantic_analysis(self, 
                                session_metrics: EvaluationMetrics,
                                analysis_data: Dict[str, Any],
                                analysis_cost: float) -> SemanticAnalysis:
        """Create SemanticAnalysis object from analysis results."""
        return SemanticAnalysis(
            session_id=session_metrics.session_id,
            agent_type=session_metrics.agent_type,
            semantic_success=analysis_data.get("semantic_success", False),
            confidence_score=analysis_data.get("confidence_score", 0.0),
            quality_score=analysis_data.get("quality_score", 0.0),
            task_comprehension=TaskComprehensionScore(
                understood_correctly=analysis_data.get("task_comprehension", {}).get("understood_correctly", False),
                interpretation_accuracy=analysis_data.get("task_comprehension", {}).get("interpretation_accuracy", 0.0),
                context_awareness=analysis_data.get("task_comprehension", {}).get("context_awareness", 0.0),
                missing_requirements=analysis_data.get("task_comprehension", {}).get("missing_requirements", [])
            ),
            approach_quality=ApproachQualityScore(
                logical_coherence=analysis_data.get("approach_quality", {}).get("logical_coherence", 0.0),
                methodology_appropriateness=analysis_data.get("approach_quality", {}).get("methodology_appropriateness", 0.0),
                execution_strategy=analysis_data.get("approach_quality", {}).get("execution_strategy", "unknown"),
                alternative_approaches=analysis_data.get("approach_quality", {}).get("alternative_approaches", [])
            ),
            tool_effectiveness=ToolEffectivenessScore(
                appropriate_selection=analysis_data.get("tool_effectiveness", {}).get("appropriate_selection", False),
                usage_efficiency=analysis_data.get("tool_effectiveness", {}).get("usage_efficiency", 0.0),
                missing_tools=analysis_data.get("tool_effectiveness", {}).get("missing_tools", []),
                unnecessary_tools=analysis_data.get("tool_effectiveness", {}).get("unnecessary_tools", []),
                execution_quality=analysis_data.get("tool_effectiveness", {}).get("execution_quality", 0.0)
            ),
            response_completeness=ResponseCompletenessScore(
                completeness=analysis_data.get("response_completeness", {}).get("completeness", 0.0),
                accuracy=analysis_data.get("response_completeness", {}).get("accuracy", 0.0),
                clarity=analysis_data.get("response_completeness", {}).get("clarity", 0.0),
                actionability=analysis_data.get("response_completeness", {}).get("actionability", 0.0),
                user_value=analysis_data.get("response_completeness", {}).get("user_value", 0.0)
            ),
            failure_root_cause=analysis_data.get("failure_root_cause"),
            improvement_suggestions=analysis_data.get("improvement_suggestions", []),
            false_negative_detected=analysis_data.get("false_negative_detected", False),
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            analysis_cost_usd=analysis_cost,
            analysis_model=self.claude_model
        )
    
    def _create_fallback_analysis(self, session_metrics: EvaluationMetrics, error_msg: str) -> SemanticAnalysis:
        """Create fallback analysis when semantic analysis fails."""
        return SemanticAnalysis(
            session_id=session_metrics.session_id,
            agent_type=session_metrics.agent_type,
            semantic_success=session_metrics.success,  # Fall back to technical success
            confidence_score=0.0,
            quality_score=0.5 if session_metrics.success else 0.0,
            task_comprehension=TaskComprehensionScore(
                understood_correctly=session_metrics.success,
                interpretation_accuracy=0.5,
                context_awareness=0.5,
                missing_requirements=[]
            ),
            approach_quality=ApproachQualityScore(
                logical_coherence=0.5,
                methodology_appropriateness=0.5,
                execution_strategy="unknown",
                alternative_approaches=[]
            ),
            tool_effectiveness=ToolEffectivenessScore(
                appropriate_selection=len(session_metrics.tools_used or []) > 0,
                usage_efficiency=0.5,
                missing_tools=[],
                unnecessary_tools=[],
                execution_quality=0.5
            ),
            response_completeness=ResponseCompletenessScore(
                completeness=0.5,
                accuracy=0.5,
                clarity=0.5,
                actionability=0.5,
                user_value=0.5
            ),
            failure_root_cause=f"Semantic analysis failed: {error_msg}" if not session_metrics.success else None,
            improvement_suggestions=[f"Semantic analysis unavailable due to: {error_msg}"],
            false_negative_detected=False,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            analysis_cost_usd=0.0,
            analysis_model=self.claude_model
        )
    
    def _create_fallback_comparative_analysis(self, 
                                           claude_result: EvaluationMetrics,
                                           opencode_result: EvaluationMetrics, 
                                           error_msg: str) -> ComparativeSemanticAnalysis:
        """Create fallback comparative analysis when analysis fails."""
        # Basic comparison based on technical success
        if claude_result.success and not opencode_result.success:
            better_performer = "claude"
            performance_gap = 0.5
        elif opencode_result.success and not claude_result.success:
            better_performer = "opencode"
            performance_gap = 0.5
        elif claude_result.success and opencode_result.success:
            better_performer = "tie"
            performance_gap = 0.0
        else:
            better_performer = "tie"
            performance_gap = 0.0
        
        return ComparativeSemanticAnalysis(
            prompt_id=claude_result.prompt or "unknown",
            better_performer=better_performer,
            performance_gap=performance_gap,
            claude_strengths=[f"Technical success: {claude_result.success}"],
            opencode_strengths=[f"Technical success: {opencode_result.success}"],
            complementary_insights=[f"Comparative analysis failed: {error_msg}"],
            claude_quality_score=0.5 if claude_result.success else 0.0,
            opencode_quality_score=0.5 if opencode_result.success else 0.0,
            approach_differences=["Analysis unavailable"],
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            analysis_cost_usd=0.0
        )
    
    def _create_fallback_batch_analysis(self, 
                                      session_results: List[EvaluationMetrics], 
                                      error_msg: str) -> BatchSemanticInsights:
        """Create fallback batch analysis when analysis fails."""
        total_sessions = len(session_results)
        successful_sessions = sum(1 for s in session_results if s.success)
        technical_success_rate = (successful_sessions / total_sessions) if total_sessions > 0 else 0.0
        
        return BatchSemanticInsights(
            total_sessions_analyzed=total_sessions,
            semantic_success_rate=technical_success_rate,  # Fall back to technical rate
            technical_success_rate=technical_success_rate,
            false_negative_rate=0.0,
            average_quality_score=0.5,
            top_failure_types=[f"Batch analysis failed: {error_msg}"],
            tool_usage_patterns={},
            agent_performance_comparison={},
            improvement_opportunities=[f"Enable semantic analysis: {error_msg}"],
            total_analysis_cost=0.0,
            analysis_timestamp=datetime.now(timezone.utc).isoformat()
        )