# Claude Code vs OpenCode Command Comparison

## Command Similarity Analysis

### ✅ **Similar/Compatible Commands**

| Functionality | Claude Code | OpenCode | Status |
|---------------|-------------|----------|---------|
| **Basic Execution** | `claude --print "prompt"` | `opencode run -m "model" "prompt"` | ✅ Both work |
| **Model Selection** | `claude --model sonnet "prompt"` | `opencode run -m "github-copilot/claude-3.5-sonnet" "prompt"` | ✅ Both work |
| **Continue Session** | `claude --continue "prompt"` | `opencode run -c "prompt"` | ✅ Both work |
| **JSON Output** | `claude --output-format json "prompt"` | N/A | ⚠️ Claude only |
| **Debug Mode** | `claude --debug api "prompt"` | `opencode run --print-logs "prompt"` | ✅ Both work |
| **Session Tracking** | Auto session IDs in JSON | Auto session IDs in logs | ✅ Both work |

### ❌ **Different/Incompatible Commands**

| Functionality | Claude Code | OpenCode | Notes |
|---------------|-------------|----------|-------|
| **Tool Control** | `--allowed-tools "Bash,Edit"` | N/A | Claude only |
| **Custom Session ID** | `--session-id <uuid>` (✅ works, requires UUID) | `-s <session-id>` (✅ works with setup) | Both support custom sessions with different requirements |
| **Session Resumption** | `--resume <session-id>` (✅ maintains context) | `-s <session-id>` (✅ continues session) | Claude uses resume, OpenCode reuses session file |
| **Temperature Control** | N/A | N/A | Neither has direct control |
| **MCP Configuration** | `--mcp-config` | Config file only | Different approaches |
| **Agent Selection** | N/A | `--agent` (fails) | OpenCode only, broken |

## 🎯 **OpenCode Custom Session ID - SUCCESS!**

**Discovery**: OpenCode custom session IDs work if you create the session file first!

### Working Approach:
```bash
# 1. Create session file with proper structure
mkdir -p ~/.local/share/opencode/project/$(pwd | sed 's/\//-/g')/storage/session/info/
cat > ~/.local/share/opencode/project/$(pwd | sed 's/\//-/g')/storage/session/info/ses_custom_test_001.json << 'EOF'
{
  "id": "ses_custom_test_001", 
  "version": "0.5.29",
  "title": "Custom evaluation session",
  "time": {
    "created": $(date +%s000),
    "updated": $(date +%s000)
  }
}
EOF

# 2. Use the custom session
opencode run -m "github-copilot/claude-3.5-sonnet" -s "ses_custom_test_001" "First message"
opencode run -m "github-copilot/claude-3.5-sonnet" -s "ses_custom_test_001" "Follow-up message"
```

### Test Results:
```bash
# First message
$ opencode run -m "github-copilot/claude-3.5-sonnet" -s "ses_custom_test_session_001" "What is 2+2?"
4

# Session continuity test  
$ opencode run -m "github-copilot/claude-3.5-sonnet" -s "ses_custom_test_session_001" "What was my previous mathematical question?"
You asked "What is 2+2?"
```

**Status**: ✅ **Both agents now support custom session management!**

## 🎯 **Claude Code Session Management - SUCCESS!**

**Discovery**: Claude Code supports custom session IDs with `--session-id` and session resumption with `--resume`.

### Claude Code Session Approach:
```bash
# 1. Create new session with custom UUID
EVAL_SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
claude --print --output-format json --session-id "$EVAL_SESSION_ID" "Starting evaluation session. My favorite color is blue."

# 2. Resume the session (creates new session ID but maintains history)
claude --print --output-format json --resume "$EVAL_SESSION_ID" "What is my favorite color?"
# Response: "Blue" (with new session_id in response)
```

### Test Results:
```bash
# Create session with UUID
$ claude --print --output-format json --session-id "937db35e-c5d9-4570-a2c6-1fe734ce27ae" "Hello, remember: My favorite color is blue."
{"result":"Hello! I've noted that your favorite color is blue...","session_id":"937db35e-c5d9-4570-a2c6-1fe734ce27ae",...}

# Resume session (maintains context)
$ claude --print --output-format json --resume "937db35e-c5d9-4570-a2c6-1fe734ce27ae" "What is my favorite color?"
{"result":"Blue","session_id":"15d98761-eae3-467b-a92d-4ee06bb0d52b",...}
```

**Key Findings:**
- ✅ `--session-id <uuid>` creates sessions with custom UUIDs (must be valid UUID format)
- ✅ `--resume <session_id>` continues existing sessions with full context
- ⚠️ Resumed sessions get new session IDs in response, but maintain conversation history
- ❌ Cannot reuse same session ID for multiple `--session-id` calls (returns "already in use" error)

**Status**: ✅ **Both agents now support reliable custom session management!**

## 🔄 **Unified Evaluation Interface Design**

Based on the analysis, here's a unified approach that works with both agents:

```python
class UnifiedAgent:
    def __init__(self, agent_type: str, model: str = None):
        self.agent_type = agent_type  # "claude" or "opencode"
        self.model = model
        
    def execute(self, prompt: str, **kwargs) -> dict:
        """Unified execution interface"""
        if self.agent_type == "claude":
            return self._execute_claude(prompt, **kwargs)
        elif self.agent_type == "opencode":
            return self._execute_opencode(prompt, **kwargs)
            
    def _execute_claude(self, prompt: str, continue_session=False, 
                       output_format="json", allowed_tools=None, session_id=None, resume_session_id=None) -> dict:
        cmd = ["claude", "--print"]
        
        if self.model:
            cmd.extend(["--model", self.model])
        if session_id:
            # Create new session with specific UUID
            cmd.extend(["--session-id", session_id])
        elif resume_session_id:
            # Resume existing session (maintains context, gets new session ID)
            cmd.extend(["--resume", resume_session_id])
        elif continue_session:
            # Continue most recent session
            cmd.append("--continue")
        if output_format:
            cmd.extend(["--output-format", output_format])
        if allowed_tools:
            cmd.extend(["--allowed-tools", ",".join(allowed_tools)])
            
        cmd.append(prompt)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if output_format == "json":
            data = json.loads(result.stdout)
            return {
                "response": data["result"],
                "session_id": data["session_id"],
                "original_session_id": session_id or resume_session_id,  # Track original session for resume
                "cost_usd": data.get("total_cost_usd", 0),
                "duration_ms": data.get("duration_ms", 0),
                "tokens": data.get("usage", {}),
                "agent": "claude",
                "model": self.model or "sonnet",
                "raw_output": result.stdout,
                "stderr": result.stderr
            }
        else:
            return {
                "response": result.stdout.strip(),
                "session_id": None,  # Not available in text mode
                "original_session_id": session_id or resume_session_id,
                "agent": "claude",
                "model": self.model or "sonnet",
                "raw_output": result.stdout,
                "stderr": result.stderr
            }
    
    def _execute_opencode(self, prompt: str, continue_session=False, 
                         enable_logs=False, session_id=None) -> dict:
        cmd = ["opencode", "run"]
        
        if self.model:
            cmd.extend(["-m", self.model])
        if session_id:
            # Create session file if it doesn't exist
            self._ensure_opencode_session(session_id)
            cmd.extend(["-s", session_id])
        if continue_session and not session_id:
            cmd.append("-c")
        if enable_logs:
            cmd.append("--print-logs")
            
        cmd.append(prompt)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Use provided session_id or extract from logs
        final_session_id = session_id
        if not final_session_id and "session=" in result.stderr:
            import re
            match = re.search(r'session=([a-zA-Z0-9_-]+)', result.stderr)
            if match:
                final_session_id = match.group(1)
        
        return {
            "response": result.stdout.strip(),
            "session_id": final_session_id,
            "cost_usd": 0,  # Not available
            "duration_ms": 0,  # Not directly available
            "tokens": {},  # Not available
            "agent": "opencode",
            "model": self.model or "unknown",
            "raw_output": result.stdout,
            "stderr": result.stderr
        }
    
    def _ensure_opencode_session(self, session_id: str):
        """Create OpenCode session file if it doesn't exist"""
        import os
        import json
        import time
        
        # Get project-specific path
        cwd = os.getcwd().replace('/', '-').lstrip('-')
        session_dir = os.path.expanduser(
            f"~/.local/share/opencode/project/{cwd}/storage/session/info/"
        )
        session_file = os.path.join(session_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            os.makedirs(session_dir, exist_ok=True)
            session_data = {
                "id": session_id,
                "version": "0.5.29",
                "title": f"Evaluation session {session_id}",
                "time": {
                    "created": int(time.time() * 1000),
                    "updated": int(time.time() * 1000)
                }
            }
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
    
    def _generate_uuid_session_id(self) -> str:
        """Generate UUID for Claude Code session IDs"""
        import uuid
        return str(uuid.uuid4())
        
    def execute_with_session_management(self, prompt: str, base_session_id: str = None, continue_conversation: bool = False) -> dict:
        """High-level session management for both agents - SAME session ID for both agents"""
        if self.agent_type == "claude":
            if continue_conversation and base_session_id:
                # Resume existing Claude session using the SAME base_session_id
                return self._execute_claude(prompt, resume_session_id=base_session_id)
            elif base_session_id:
                # Convert base_session_id to UUID format if needed, but keep it consistent
                session_uuid = self._ensure_uuid_format(base_session_id)
                return self._execute_claude(prompt, session_id=session_uuid)
            else:
                return self._execute_claude(prompt, continue_session=True)
                
        elif self.agent_type == "opencode":
            if base_session_id:
                # OpenCode uses the SAME base_session_id directly (no conversion needed)
                return self._execute_opencode(prompt, session_id=base_session_id)
            elif continue_conversation:
                return self._execute_opencode(prompt, continue_session=True)
            else:
                return self._execute_opencode(prompt)
    
    def _ensure_uuid_format(self, base_session_id: str) -> str:
        """Ensure consistent UUID format for Claude while preserving base session identity"""
        try:
            import uuid
            # If already a valid UUID, use it
            uuid.UUID(base_session_id)
            return base_session_id
        except ValueError:
            # Convert base_session_id to deterministic UUID
            import hashlib
            # Create deterministic UUID from base_session_id
            namespace = uuid.UUID('12345678-1234-5678-9abc-123456789abc')
            return str(uuid.uuid5(namespace, base_session_id))
```

## 📋 **Updated Evaluation Engine Design**

```python
class EvaluationEngine:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.session_manager = SessionManager()
        self.prompt_loader = MarkdownPromptLoader()
        
    def execute_evaluation(self, prompt_id: int, agent_config: dict) -> dict:
        """Execute SAME prompt with SAME session ID across both agents for direct comparison"""
        
        # Load prompt from markdown - SAME prompt_id for both agents
        prompt = self.prompt_loader.load_prompt(prompt_id)
        
        # Generate consistent session ID format for both agents
        import time
        timestamp = int(time.time())
        # Session ID format: eval_prompt{id}_{timestamp} - SAME for both agents
        base_session_id = f"eval_prompt{prompt_id:03d}_{timestamp}"
        
        # Create unified agent
        agent = UnifiedAgent(
            agent_type=agent_config["type"],  # "claude" or "opencode"
            model=agent_config.get("model")
        )
        
        # Execute with SAME base_session_id - agent differentiation is internal
        result = agent.execute_with_session_management(
            prompt=prompt["content"],
            base_session_id=base_session_id,
            continue_conversation=agent_config.get("continue_session", False)
        )
        
        # Store session data with consistent identification
        session_data = {
            "prompt_id": prompt_id,  # SAME across agents
            "base_session_id": base_session_id,  # SAME across agents
            "agent_type": agent_config["type"],  # "claude" or "opencode"
            "agent_session_id": result["session_id"],  # Agent-specific internal session ID
            "prompt": prompt,
            "result": result,
            "timestamp": timestamp,
            "agent_config": agent_config
        }
        
        self.session_manager.store_session(base_session_id, session_data)
        
        return session_data
    
    def execute_comparative_evaluation(self, prompt_id: int, claude_config: dict, opencode_config: dict) -> dict:
        """Execute SAME prompt with SAME session ID across BOTH agents for direct comparison"""
        
        # Load the SAME prompt
        prompt = self.prompt_loader.load_prompt(prompt_id)
        
        # Generate SAME base session ID for both agents
        import time
        timestamp = int(time.time())
        base_session_id = f"eval_prompt{prompt_id:03d}_{timestamp}"
        
        # Execute with Claude Code
        claude_agent = UnifiedAgent("claude", claude_config.get("model"))
        claude_result = claude_agent.execute_with_session_management(
            prompt=prompt["content"],
            base_session_id=base_session_id,
            continue_conversation=claude_config.get("continue_session", False)
        )
        
        # Execute with OpenCode using SAME base_session_id
        opencode_agent = UnifiedAgent("opencode", opencode_config.get("model")) 
        opencode_result = opencode_agent.execute_with_session_management(
            prompt=prompt["content"],
            base_session_id=base_session_id,
            continue_conversation=opencode_config.get("continue_session", False)
        )
        
        # Store comparative results
        comparative_data = {
            "prompt_id": prompt_id,  # SAME
            "base_session_id": base_session_id,  # SAME
            "prompt": prompt,  # SAME
            "timestamp": timestamp,
            "claude_result": {
                "agent_type": "claude",
                "agent_session_id": claude_result["session_id"],
                "response": claude_result["response"],
                "cost_usd": claude_result.get("cost_usd", 0),
                "duration_ms": claude_result.get("duration_ms", 0),
                "tokens": claude_result.get("tokens", {}),
                "raw_output": claude_result["raw_output"]
            },
            "opencode_result": {
                "agent_type": "opencode", 
                "agent_session_id": opencode_result["session_id"],
                "response": opencode_result["response"],
                "cost_usd": opencode_result.get("cost_usd", 0),
                "duration_ms": opencode_result.get("duration_ms", 0), 
                "tokens": opencode_result.get("tokens", {}),
                "raw_output": opencode_result["raw_output"]
            }
        }
        
        # Store under SAME base_session_id
        self.session_manager.store_comparative_session(base_session_id, comparative_data)
        
        return comparative_data
```

## 🔧 **Recommended Agent Configuration**

```yaml
# evaluation_config.yaml
agents:
  claude_sonnet:
    type: "claude"
    model: "sonnet"
    allowed_tools: ["Bash", "Edit", "Read", "Write"]
    
  opencode_claude:
    type: "opencode"
    model: "github-copilot/claude-3.5-sonnet"

# Example: SAME prompt_id, SAME base_session_id for both agents
evaluation_execution:
  # Both agents will use: prompt_id=001, base_session_id="eval_prompt001_1733123456"
  - prompt_id: 1
    agents: ["claude_sonnet", "opencode_claude"]
    session_strategy: "same_session_id"  # Both use same base session ID
    
  - prompt_id: 2  
    agents: ["claude_sonnet", "opencode_claude"]
    session_strategy: "same_session_id"


evaluation_settings:
  timeout_seconds: 60
  max_retries: 3
  enable_logging: true
  session_id_format: "eval_prompt{prompt_id:03d}_{timestamp}"  # SAME format for both
```

## 🎯 **Key Unified Features**

### ✅ **What Works with Both:**
1. **Basic execution**: Both can run prompts programmatically
2. **Model selection**: Both support different models
3. **Session continuation**: Both support multi-turn conversations
4. **Logging/Debug**: Both provide execution details
5. **MCP integration**: Both auto-load MCP servers
6. **Hook system**: Claude has built-in hooks, OpenCode can use plugins

### 🔄 **Normalization Strategy:**
1. **Unified response format**: Convert both outputs to common structure
2. **Session tracking**: Extract session IDs from both (JSON vs logs)
3. **Error handling**: Standardize timeout and error responses
4. **Metrics collection**: Use hooks/plugins for both agents
5. **Cost tracking**: Available for Claude, estimated for OpenCode

This unified approach allows the evaluation engine to work with both agents while leveraging their individual strengths!