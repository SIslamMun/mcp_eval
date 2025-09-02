# Unified Session ID Strategy - SAME IDs for Direct Comparison

## 🎯 Core Concept
Both Claude Code and OpenCode use **SAME prompt IDs and SAME base session IDs** for direct comparison. Agent differentiation is handled internally through the UnifiedAgent class.

## 📋 Session ID Format Standard
```
Base Session ID Format: "eval_prompt{prompt_id:03d}_{timestamp}"
Examples:
- eval_prompt001_1733123456  (Prompt 1, timestamp 1733123456)
- eval_prompt042_1733123789  (Prompt 42, timestamp 1733123789)  
- eval_prompt100_1733124000  (Prompt 100, timestamp 1733124000)
```

## 🔄 Agent Session Management

### Claude Code Approach
- **Input**: base_session_id = "eval_prompt001_1733123456"
- **Conversion**: Deterministic UUID via uuid5(namespace, base_session_id)  
- **Result**: UUID like "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
- **Usage**: `claude --session-id "a1b2c3d4-..." --resume "a1b2c3d4-..."`

### OpenCode Approach  
- **Input**: base_session_id = "eval_prompt001_1733123456"
- **Usage**: Direct usage with session file creation
- **Session File**: `~/.local/share/opencode/project/.../eval_prompt001_1733123456.json`
- **Usage**: `opencode run -s "eval_prompt001_1733123456"`

## 🧪 Practical Example

### Setup
```python
# Both agents evaluate SAME prompt with SAME session ID
prompt_id = 1
prompt_content = "Analyze this HDF5 dataset structure and suggest optimizations"
base_session_id = f"eval_prompt{prompt_id:03d}_{int(time.time())}"
# Result: "eval_prompt001_1733123456"
```

### Execution
```python
# Claude Code execution
claude_agent = UnifiedAgent("claude", "sonnet")
claude_result = claude_agent.execute_with_session_management(
    prompt=prompt_content,
    base_session_id="eval_prompt001_1733123456",  # SAME ID
    continue_conversation=False
)

# OpenCode execution  
opencode_agent = UnifiedAgent("opencode", "github-copilot/claude-3.5-sonnet")
opencode_result = opencode_agent.execute_with_session_management(
    prompt=prompt_content,  # SAME prompt
    base_session_id="eval_prompt001_1733123456",  # SAME ID  
    continue_conversation=False
)
```

### Results Structure
```python
# Claude Result
{
    "response": "The HDF5 dataset shows...",
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",  # UUID format
    "original_session_id": "eval_prompt001_1733123456",      # SAME base ID
    "agent": "claude",
    "cost_usd": 0.05,
    "duration_ms": 2500,
    "tokens": {"input": 50, "output": 120}
}

# OpenCode Result  
{
    "response": "This HDF5 structure can be optimized...", 
    "session_id": "eval_prompt001_1733123456",               # SAME as base ID
    "original_session_id": "eval_prompt001_1733123456",      # SAME base ID
    "agent": "opencode",
    "cost_usd": 0,        # Not available
    "duration_ms": 0,     # Not available  
    "tokens": {}          # Not available
}
```

## 📊 Comparative Analysis Benefits

### Direct Comparison Possible
```python
# Query results by SAME base_session_id
session_data = session_manager.get_comparative_results("eval_prompt001_1733123456")

# Compare responses directly
claude_response = session_data["claude_result"]["response"]
opencode_response = session_data["opencode_result"]["response"]

# Analyze differences
response_similarity = compare_responses(claude_response, opencode_response)
cost_difference = session_data["claude_result"]["cost_usd"] - session_data["opencode_result"]["cost_usd"]
performance_difference = analyze_performance_metrics(session_data)
```

### Multi-turn Evaluation with SAME Session
```python
base_session_id = "eval_prompt001_1733123456"

# Round 1: Both agents get same initial prompt
initial_prompt = "Analyze this dataset structure"
claude_r1 = claude_agent.execute_with_session_management(initial_prompt, base_session_id, False)
opencode_r1 = opencode_agent.execute_with_session_management(initial_prompt, base_session_id, False)

# Round 2: Both agents get same follow-up (with their respective context)
followup_prompt = "What are your top 3 optimization recommendations?"
claude_r2 = claude_agent.execute_with_session_management(followup_prompt, base_session_id, True)
opencode_r2 = opencode_agent.execute_with_session_management(followup_prompt, base_session_id, True)

# Both maintain context from Round 1, enabling fair comparison of multi-turn capabilities
```

## 🎯 Evaluation Matrix Structure

```
Evaluation Matrix: n models × m agents × p prompts

With SAME session IDs:
- prompt_001: eval_prompt001_{timestamp}
  ├── claude_sonnet: UUID converted from base_session_id  
  └── opencode_claude: base_session_id directly

- prompt_002: eval_prompt002_{timestamp}  
  ├── claude_sonnet: UUID converted from base_session_id
  └── opencode_claude: base_session_id directly

Results stored with SAME base_session_id for direct comparison
```

## ✅ Implementation Status

**Ready for implementation** with this unified session ID strategy:
- ✅ SAME prompt IDs across both agents
- ✅ SAME base session IDs for direct comparison
- ✅ Agent-specific session management (UUID conversion vs direct usage)  
- ✅ Session continuity for multi-turn evaluations
- ✅ Comparative analysis capability
- ✅ Unified result storage and retrieval

This approach enables true apples-to-apples comparison between Claude Code and OpenCode while respecting each agent's technical requirements! 🚀
