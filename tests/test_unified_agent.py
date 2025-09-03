"""
Test unified agent functionality and session management.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from mcp_evaluation.unified_agent import UnifiedAgent, AgentConfig, AgentResponse


class TestUnifiedAgent:
    """Test the UnifiedAgent class."""
    
    def test_agent_initialization(self):
        """Test agent initialization with different types."""
        # Claude agent
        claude_agent = UnifiedAgent("claude", "sonnet")
        assert claude_agent.agent_type == "claude"
        assert claude_agent.model == "sonnet"
        
        # OpenCode agent
        opencode_agent = UnifiedAgent("opencode", "github-copilot/claude-3.5-sonnet")
        assert opencode_agent.agent_type == "opencode"
        assert opencode_agent.model == "github-copilot/claude-3.5-sonnet"
        
        # Invalid agent type
        with pytest.raises(ValueError, match="Unsupported agent type"):
            UnifiedAgent("invalid", "model")
    
    def test_agent_config_validation(self):
        """Test agent configuration validation."""
        # Valid config
        config = AgentConfig(type="claude", model="sonnet")
        assert config.type == "claude"
        assert config.model == "sonnet"
        assert config.timeout is None or config.timeout == 60  # timeout is optional
        
        # Invalid complexity (if we had validation)
        config = AgentConfig(type="opencode", model="github-copilot/gpt-4o")
        assert config.type == "opencode"
    
    @patch('subprocess.run')
    def test_claude_execution_success(self, mock_subprocess):
        """Test successful Claude Code execution."""
        # Mock successful JSON response
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '{"result": "Hello World", "session_id": "test-123", "total_cost_usd": 0.01, "duration_ms": 1500, "usage": {"input_tokens": 10, "output_tokens": 5}}'
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        agent = UnifiedAgent("claude", "sonnet")
        config = AgentConfig(type="claude", model="sonnet", output_format="json")
        
        result = agent.execute("Hello", config)
        
        assert result.success is True
        assert result.response == "Hello World"
        assert result.session_id == "test-123"
        assert result.cost_usd == 0.01
        assert result.duration_ms == 1500
        assert result.agent == "claude"
        
        # Verify command construction
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        assert "claude" in cmd
        assert "--print" in cmd
        assert "--model" in cmd and "sonnet" in cmd
        assert "--output-format" in cmd and "json" in cmd
    
    @patch('subprocess.run')
    def test_claude_execution_failure(self, mock_subprocess):
        """Test Claude Code execution failure."""
        # Mock failed execution
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Model not available"
        mock_subprocess.return_value = mock_result
        
        agent = UnifiedAgent("claude", "sonnet")
        config = AgentConfig(type="claude", model="sonnet")
        
        result = agent.execute("Hello", config)
        
        assert result.success is False
        assert result.error_message == "Error: Model not available"
        assert result.agent == "claude"
    
    @patch('subprocess.run')
    def test_opencode_execution_success(self, mock_subprocess):
        """Test successful OpenCode execution."""
        # Mock successful response
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello from OpenCode"
        mock_result.stderr = "INFO session=ses_abc123 created"
        mock_subprocess.return_value = mock_result
        
        agent = UnifiedAgent("opencode", "github-copilot/claude-3.5-sonnet")
        config = AgentConfig(type="opencode", model="github-copilot/claude-3.5-sonnet")
        
        result = agent.execute("Hello", config)
        
        assert result.success is True
        assert result.response == "Hello from OpenCode"
        assert result.session_id == "ses_abc123"
        assert result.agent == "opencode"
        
        # Verify command construction
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        assert "opencode" in cmd
        assert "run" in cmd
        assert "-m" in cmd
        assert "github-copilot/claude-3.5-sonnet" in cmd
    
    def test_session_id_generation(self):
        """Test session ID generation."""
        agent = UnifiedAgent("claude", "sonnet")
        
        # Test with specific timestamp
        session_id = agent.generate_session_id(prompt_id=1, timestamp=1700000000)
        assert session_id == "eval_prompt001_1700000000"
        
        # Test with current timestamp
        session_id = agent.generate_session_id(prompt_id=42)
        assert session_id.startswith("eval_prompt042_")
        assert len(session_id.split("_")) == 3
    
    def test_uuid_format_conversion(self):
        """Test UUID format conversion for Claude."""
        agent = UnifiedAgent("claude", "sonnet")
        
        # Valid UUID should remain unchanged
        valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
        result = agent._ensure_uuid_format(valid_uuid)
        assert result == valid_uuid
        
        # Invalid UUID should be converted deterministically
        base_id = "eval_prompt001_1700000000"
        result1 = agent._ensure_uuid_format(base_id)
        result2 = agent._ensure_uuid_format(base_id)
        
        # Should be consistent
        assert result1 == result2
        # Should be valid UUID format
        assert len(result1) == 36
        assert result1.count("-") == 4
    
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_opencode_session_creation(self, mock_mkdir, mock_exists, mock_write):
        """Test OpenCode session file creation."""
        agent = UnifiedAgent("opencode", "github-copilot/claude-3.5-sonnet")
        
        # Mock file doesn't exist
        mock_exists.return_value = False
        
        # Create session
        agent._ensure_opencode_session("test_session_123")
        
        # Verify directory creation
        mock_mkdir.assert_called_once()
        
        # Verify file writing
        mock_write.assert_called_once()
        written_content = mock_write.call_args[0][0]
        session_data = eval(written_content.replace('null', 'None').replace('true', 'True').replace('false', 'False'))
        
        assert '"id": "test_session_123"' in written_content
        assert '"version": "0.5.29"' in written_content
    
    @patch('subprocess.run')
    def test_session_management_execution(self, mock_subprocess):
        """Test high-level session management execution."""
        # Mock Claude response
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '{"result": "Session test", "session_id": "uuid-123"}'
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        agent = UnifiedAgent("claude", "sonnet")
        
        # Test with base session ID
        result = agent.execute_with_session_management(
            prompt="Test session",
            base_session_id="eval_prompt001_1700000000"
        )
        
        assert result.success is True
        assert result.response == "Session test"
        
        # Verify session ID was used in command
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        assert "--session-id" in cmd


class TestAgentResponse:
    """Test AgentResponse model."""
    
    def test_response_creation(self):
        """Test response creation and validation."""
        response = AgentResponse(
            response="Test response",
            session_id="test-123",
            agent="claude",
            model="sonnet"
        )
        
        assert response.response == "Test response"
        assert response.session_id == "test-123"
        assert response.agent == "claude"
        assert response.model == "sonnet"
        assert response.success is True  # default
        assert response.cost_usd == 0.0  # default
    
    def test_response_with_error(self):
        """Test response with error information."""
        response = AgentResponse(
            response="",
            agent="opencode",
            model="gpt-4o",
            success=False,
            error_message="Connection timeout"
        )
        
        assert response.success is False
        assert response.error_message == "Connection timeout"
        assert response.response == ""


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary configuration directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


class TestIntegration:
    """Integration tests for agent functionality."""
    
    @patch('subprocess.run')
    def test_comparative_execution_workflow(self, mock_subprocess):
        """Test workflow that would be used in comparative evaluation."""
        # Mock responses for both agents
        def side_effect(cmd, **kwargs):
            if "claude" in cmd:
                result = Mock()
                result.returncode = 0
                result.stdout = '{"result": "Claude response", "session_id": "claude-123"}'
                result.stderr = ""
                return result
            elif "opencode" in cmd:
                result = Mock()
                result.returncode = 0
                result.stdout = "OpenCode response"
                result.stderr = "INFO session=opencode-123"
                return result
        
        mock_subprocess.side_effect = side_effect
        
        # Create agents
        claude_agent = UnifiedAgent("claude", "sonnet")
        opencode_agent = UnifiedAgent("opencode", "github-copilot/claude-3.5-sonnet")
        
        # Same base session ID
        base_session_id = "eval_prompt001_1700000000"
        prompt = "Compare agent responses"
        
        # Execute with both agents
        claude_result = claude_agent.execute_with_session_management(
            prompt=prompt,
            base_session_id=base_session_id
        )
        
        opencode_result = opencode_agent.execute_with_session_management(
            prompt=prompt,
            base_session_id=base_session_id
        )
        
        # Verify both succeeded
        assert claude_result.success is True
        assert opencode_result.success is True
        
        # Verify responses
        assert claude_result.response == "Claude response"
        assert opencode_result.response == "OpenCode response"
        
        # Verify session tracking
        assert claude_result.original_session_id == base_session_id
        assert opencode_result.session_id == "opencode-123"
        
        # Verify both agents were called
        assert mock_subprocess.call_count == 2
