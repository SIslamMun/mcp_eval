# MCP Evaluation Infrastructure Plan

## Thoughts

This is a focused MCP evaluation system that separates evaluation execution from post-processing. The key insight is creating a session-based tracking system with strict data formats and configurable agent parameters.

The architecture follows a clean separation of concerns:
- **Evaluation Engine**: Core execution logic with one test at a time
- **Prompt Management**: Markdown files with YAML frontmatter for structured prompts
- **Agent Configuration**: Configurable models and parameters for Claude Code and OpenCode
- **Session Tracking**: Unique session IDs for full conversation retrieval


Key design principles:
- **Complexity Redefinition**: Low=1 MCP query, Medium=multiple queries same MCP, High=multiple MCPs
- **Markdown Format**: YAML frontmatter with markdown content for prompts
- **UV Management**: Modern Python dependency management
- **Evaluation Focus**: Core evaluation separated from results processing

## Pseudocode Workflow 

```python
# Core evaluation workflow (unified agents, one test at a time)
prompt = prompt_loader.load_from_markdown(prompt_id)
agent = UnifiedAgent(agent_type, model_config, mcp_allowlist, hooks_enabled=True)
session_id = uuid.generate()
result = agent.execute(prompt, session_id, **agent_specific_params)
session_manager.store_session(session_id, prompt, result, normalized_events)
# Post-processing happens separately in different module
```

## Prompt 

```
Create a Python-based MCP evaluation infrastructure using uv for dependency management with the following components:

**Reference Documentation**: The complete command testing and compatibility analysis for both agents is documented in:
- `docs/claude_arguments.md` - Comprehensive Claude Code command testing results  
- `docs/opencode_arguments.md` - Comprehensive OpenCode command testing results
- `docs/agent_comparison.md` - Unified agent comparison and session management approach

1. **UnifiedAgent** class that:
   - Supports both Claude Code and OpenCode with consistent interface (see agent_comparison.md for implementation)
   - Handles model selection for both agents (claude --model vs opencode -m, documented in respective argument files)
   - Normalizes outputs to common format with SAME session tracking across both agents
   - Manages session continuation (claude --continue/--resume vs opencode -c/-s, see session management docs)
   - Handles agent-specific features (JSON output for Claude, logs for OpenCode, tested in argument files)

2. **EvaluationEngine** class that:
   - Loads prompts from Markdown files with strict frontmatter format:
     ```markdown
     ---
     id: 1
     complexity: "low"
     target_mcp: ["mcp-name"]
     description: "Brief description"
     ---
     # Prompt Title
     
     Your prompt text here...
     ```
   - Complexity definitions: low=single MCP query, medium=multiple queries same MCP, high=multiple MCPs
   - Supports unified agent execution with both Claude Code and OpenCode
   - Executes one test at a time with agent-specific configurations

3. **SessionManager** class that:
   - Generates SAME session IDs for both agents (base_session_id format: "eval_prompt{id:03d}_{timestamp}")
   - Converts base_session_id to UUID for Claude Code (deterministic conversion)
   - Uses base_session_id directly for OpenCode (with session file creation)  
   - Tracks conversation events within SAME logical session across both agents
   - Provides query interface: get_session_by_prompt_id(), get_comparative_results_by_base_session_id()
   - Session traceability: prompt_id -> SAME base_session_id -> both agent results
   - Handles both Claude's JSON metadata and OpenCode's log-based tracking

4. **Agent Configuration** (based on tested capabilities in claude_arguments.md and opencode_arguments.md):
   - Claude Code: Configure model (sonnet/opus), output format (JSON), allowed tools, session management (--session-id/--resume)
   - OpenCode: Configure model selection (github-copilot/*, opencode/*), enable logging, session management (-s with file creation)
   - Both agents integrated with hook system for observability (documented command testing confirms MCP auto-discovery)
   - Unified response format with normalized SAME session tracking across both agents



5. **API Design - SAME Session IDs**:
   ```python
   # Unified usage with SAME session IDs for direct comparison
   engine = EvaluationEngine()
   prompt_id = 35
   prompt = engine.load_prompt(prompt_id)
   
   # Generate SAME base session ID for both agents
   import time
   base_session_id = f"eval_prompt{prompt_id:03d}_{int(time.time())}"
   
   # Claude Code execution with base_session_id
   claude_config = {"type": "claude", "model": "sonnet"}
   claude_result = engine.execute_evaluation(prompt_id, claude_config)
   # Internally uses base_session_id -> UUID conversion
   
   # OpenCode execution with SAME base_session_id
   opencode_config = {"type": "opencode", "model": "github-copilot/claude-3.5-sonnet"}
   opencode_result = engine.execute_evaluation(prompt_id, opencode_config)  
   # Internally uses base_session_id directly
   
   # Direct comparative evaluation with SAME session ID
   comparative_result = engine.execute_comparative_evaluation(
       prompt_id=prompt_id,
       claude_config=claude_config,
       opencode_config=opencode_config
   )
   
   # Retrieve comparative results by SAME base_session_id
   session_data = engine.session_manager.get_comparative_results(base_session_id)
   ```


## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Markdown      │    │   Evaluation     │    │   Session       │
│   Prompt Loader │───▶│   Engine         │───▶│   Manager       │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                    ┌──────────────────┐        ┌─────────────────┐
                    │  UnifiedAgent    │        │   Hook System   │
                    │ (Claude+OpenCode)│        │  (Observability)│
                    └──────────────────┘        └─────────────────┘

```

## Key Features

1. **Unified Agent Design**: Single interface for both Claude Code and OpenCode (implementation details in agent_comparison.md)
2. **SAME Session Tracking**: prompt_id -> SAME base_session_id -> both agent results traceability (validated in claude_arguments.md and opencode_arguments.md)
3. **Markdown Format**: YAML frontmatter with structured prompt content
4. **UV Managed**: Modern Python dependency management
5. **Validated Commands**: All agent capabilities tested and documented in respective argument files
6. **Session Management**: Both agents support custom session IDs with different technical approaches (see session management documentation)
5. **Agent Flexibility**: Support for Claude Code (rich JSON) and OpenCode (model variety)
6. **One Test at a Time**: Sequential execution with agent-specific configurations
7. **Normalized Output**: Common response format despite agent differences

## Agent Comparison Summary

## Implementation Documentation References

**Complete Testing Documentation Available:**
- **`docs/claude_arguments.md`** - Comprehensive testing of all Claude Code commands, JSON output formats, session management, hook integration, and debug capabilities
- **`docs/opencode_arguments.md`** - Comprehensive testing of all OpenCode commands, model selection, session management, logging, and plugin integration  
- **`docs/agent_comparison.md`** - Unified agent interface implementation, session management strategy, and comparative analysis approach
- **`docs/session_management_test.md`** - Detailed session ID testing results and implementation examples
- **`docs/unified_session_strategy.md`** - Complete unified session ID strategy for direct agent comparison

Project Structure:
- Use uv for dependency management (pyproject.toml)
- Markdown files with YAML frontmatter for prompts dataset
- YAML configuration for agents
- Unified agent interface supporting both Claude Code and OpenCode
- Separate evaluation 
- Comprehensive error handling and timeouts
- Unit tests for core components

Generate the complete implementation focusing on the unified agent interface first, with clear separation between evaluation execution and results processing. The system should seamlessly work with both Claude Code and OpenCode while leveraging their individual strengths (Claude's JSON output & hooks vs OpenCode's model variety).
```