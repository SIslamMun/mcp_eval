# Session Management Test Results

## ✅ Both Agents Support Custom Session Management

### Claude Code Test Results

#### Test 1: Custom Session ID Creation
```bash
$ EVAL_UUID="937db35e-c5d9-4570-a2c6-1fe734ce27ae"
$ claude --print --output-format json --session-id "$EVAL_UUID" "Hello, remember: My favorite color is blue."

Response:
{
  "result": "Hello! I've noted that your favorite color is blue. I'm ready to help you with any software engineering tasks in the iowarp-hooks repository. What would you like to work on?",
  "session_id": "937db35e-c5d9-4570-a2c6-1fe734ce27ae",
  "total_cost_usd": 0.0105507,
  "usage": {"input_tokens": 4, "output_tokens": 43},
  "num_turns": 1
}
```

#### Test 2: Session Resumption (Context Preserved)
```bash
$ claude --print --output-format json --resume "937db35e-c5d9-4570-a2c6-1fe734ce27ae" "What is my favorite color?"

Response:
{
  "result": "Blue",
  "session_id": "15d98761-eae3-467b-a92d-4ee06bb0d52b",  # New session ID, but context preserved
  "total_cost_usd": 0.0062346,
  "usage": {"input_tokens": 4, "output_tokens": 4},
  "num_turns": 3  # Shows conversation continuity
}
```

#### Test 3: Session ID Reuse Limitation
```bash
$ claude --print --session-id "937db35e-c5d9-4570-a2c6-1fe734ce27ae" "Another question"

Error: Session ID 937db35e-c5d9-4570-a2c6-1fe734ce27ae is already in use.
```

### OpenCode Test Results

#### Test 1: Custom Session ID with Pre-created Session File
```bash
$ mkdir -p ~/.local/share/opencode/project/$(pwd | sed 's/\//-/g')/storage/session/info/
$ cat > ~/.local/share/opencode/project/$(pwd | sed 's/\//-/g')/storage/session/info/ses_custom_test_session_001.json << 'EOF'
{
  "id": "ses_custom_test_session_001",
  "version": "0.5.29",
  "title": "Custom test session",
  "time": {
    "created": $(date +%s000),
    "updated": $(date +%s000)
  }
}
EOF

$ opencode run -m "github-copilot/claude-3.5-sonnet" -s "ses_custom_test_session_001" "What is 2+2?"
4
```

#### Test 2: Session Continuity
```bash
$ opencode run -m "github-copilot/claude-3.5-sonnet" -s "ses_custom_test_session_001" "What was my previous mathematical question?"
You asked "What is 2+2?"
```

## 🎯 Implementation Strategy for Evaluation Engine

### Unified Session Management Approach - SAME Session IDs

```python
class SessionManager:
    def __init__(self):
        self.active_sessions = {}
        
    def create_evaluation_session(self, base_session_id: str, agent_type: str) -> str:
        """Create session using SAME base_session_id for both agents"""
        if agent_type == "claude":
            # Convert base_session_id to deterministic UUID for Claude Code
            session_uuid = self._ensure_uuid_format(base_session_id)
            self.active_sessions[base_session_id] = {
                "agent": "claude",
                "session_uuid": session_uuid,
                "base_session_id": base_session_id,  # SAME for both agents
                "original_id": base_session_id
            }
            return session_uuid
            
        elif agent_type == "opencode":
            # Create session file using SAME base_session_id
            self._ensure_opencode_session(base_session_id)
            self.active_sessions[base_session_id] = {
                "agent": "opencode", 
                "session_id": base_session_id,  # Uses base_session_id directly
                "base_session_id": base_session_id,  # SAME for both agents
                "original_id": base_session_id
            }
            return base_session_id
            
    def continue_evaluation_session(self, base_session_id: str, agent_type: str) -> str:
        """Continue session using SAME base_session_id"""
        if base_session_id not in self.active_sessions:
            return self.create_evaluation_session(base_session_id, agent_type)
            
        session_info = self.active_sessions[base_session_id]
        
        if agent_type == "claude":
            # Return UUID for --resume, but based on SAME base_session_id
            return session_info["session_uuid"]
        else:
            # Return SAME base_session_id for OpenCode
            return session_info["session_id"]
            
    def _ensure_uuid_format(self, base_session_id: str) -> str:
        """Convert base_session_id to deterministic UUID for Claude"""
        import uuid, hashlib
        try:
            # If already UUID, use it
            uuid.UUID(base_session_id)
            return base_session_id
        except ValueError:
            # Create deterministic UUID from SAME base_session_id
            namespace = uuid.UUID('12345678-1234-5678-9abc-123456789abc')
            return str(uuid.uuid5(namespace, base_session_id))
```

### Evaluation Execution Pattern - SAME IDs

```python
# Generate SAME session ID for both agents
prompt_id = 1
timestamp = int(time.time())
base_session_id = f"eval_prompt{prompt_id:03d}_{timestamp}"  # e.g., "eval_prompt001_1733123456"

# Claude Code execution with SAME base_session_id
claude_agent = UnifiedAgent("claude", "sonnet")
claude_result = claude_agent.execute_with_session_management(
    prompt="Evaluate this MCP server implementation",
    base_session_id=base_session_id,  # SAME ID
    continue_conversation=False
)

# OpenCode execution with SAME base_session_id  
opencode_agent = UnifiedAgent("opencode", "github-copilot/claude-3.5-sonnet")
opencode_result = opencode_agent.execute_with_session_management(
    prompt="Evaluate this MCP server implementation",  # SAME prompt
    base_session_id=base_session_id,  # SAME ID
    continue_conversation=False
)

# Results have SAME base_session_id, different agent-internal session IDs
print(f"Base Session ID (SAME): {base_session_id}")
print(f"Claude Internal Session: {claude_result['session_id']}")  # UUID format
print(f"OpenCode Internal Session: {opencode_result['session_id']}")  # Same as base_session_id
```

### Multi-turn Evaluation with SAME Session

```python
# First prompt - both agents use SAME base_session_id
base_session_id = "eval_prompt001_1733123456"

# Both agents receive same initial prompt
initial_prompt = "Analyze this HDF5 file structure"
claude_result1 = claude_agent.execute_with_session_management(initial_prompt, base_session_id, False)
opencode_result1 = opencode_agent.execute_with_session_management(initial_prompt, base_session_id, False)

# Follow-up prompt - both agents continue SAME session
followup_prompt = "What optimization recommendations do you have?"
claude_result2 = claude_agent.execute_with_session_management(followup_prompt, base_session_id, True)
opencode_result2 = opencode_agent.execute_with_session_management(followup_prompt, base_session_id, True)

# Both agents maintain context from their respective first interactions
```

## 🎉 Summary

**✅ Claude Code**: 
- Supports `--session-id <uuid>` for custom session creation
- Supports `--resume <session-id>` for context continuation
- Requires valid UUID format for session IDs
- Cannot reuse session IDs for creation (but can resume)

**✅ OpenCode**:
- Supports `-s <session-id>` for custom sessions
- Requires pre-existing session files with JSON structure
- Can reuse session IDs across multiple executions
- Session files enable persistent conversation history

**🚀 Ready for Implementation**: Both agents now have reliable custom session management for the MCP evaluation infrastructure!
