# MCP Evaluation System

A simple evaluation system for testing MCP (Model Context Protocol) functionality with Claude and OpenCode agents.

## 🚀 Quick Start

### 1. Install
```bash
uv venv --clear
source .venv/bin/activate
uv sync
```

### 2. Setup InfluxDB (default database)
```bash
./setup_influxdb.sh
```

### 3. Setup Grafana Dashboard (optional)
```bash
./setup_grafana.sh
```

### 4. Run Evaluations
```bash
# Run single prompt with both agents
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions

# Run all prompts with both agents
uv run python -m mcp_evaluation run-all --agent both --skip-permissions

# View statistics
uv run python -m mcp_evaluation stats
```

## � What it does

- **Tests MCP servers** with both Claude and OpenCode agents
- **Compares agent performance** on identical tasks
- **Stores results** in InfluxDB time-series database
- **Tracks costs** and execution times
- **Generates statistics** and analytics

## � Project Structure

```
├── src/mcp_evaluation/          # Core evaluation system
│   ├── cli.py                   # Command-line interface
│   ├── evaluation_engine.py     # Main evaluation orchestration
│   ├── session_manager.py       # InfluxDB/SQLite storage
│   ├── unified_agent.py         # Claude/OpenCode interface
│   └── prompt_loader.py         # Markdown prompt parser
├── prompts/                     # Test prompts (001.md - 005.md, 999.md)
├── tests/                       # Unit tests
└── docs/                        # Documentation
```

## 🔧 Commands

### Single Evaluations
```bash
# Both agents (default models)
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions

# Claude only (default: sonnet)
uv run python -m mcp_evaluation run 1 --agent claude --skip-permissions

# OpenCode only (default: github-copilot/claude-3.5-sonnet)
uv run python -m mcp_evaluation run 1 --agent opencode

# Custom models
uv run python -m mcp_evaluation run 1 --agent claude --claude-model haiku --skip-permissions
uv run python -m mcp_evaluation run 1 --agent opencode --opencode-model "github-copilot/gpt-4o"
```

### Model Selection
```bash
# Available Claude models: sonnet, haiku, opus
uv run python -m mcp_evaluation run 1 --agent claude --claude-model haiku --skip-permissions

# Available OpenCode models:
uv run python -m mcp_evaluation run 1 --agent opencode --opencode-model "github-copilot/gpt-4o"
uv run python -m mcp_evaluation run 1 --agent opencode --opencode-model "github-copilot/claude-3.5-sonnet"
uv run python -m mcp_evaluation run 1 --agent opencode --opencode-model "opencode/llama-3.1-70b"

# Both agents with custom models (RECOMMENDED for model comparison)
uv run python -m mcp_evaluation run 1 --agent both --claude-model haiku --opencode-model "github-copilot/gpt-4o" --skip-permissions
uv run python -m mcp_evaluation run 1 --agent both --claude-model sonnet --opencode-model "opencode/llama-3.1-70b" --skip-permissions
uv run python -m mcp_evaluation run 1 --agent both --claude-model opus --opencode-model "github-copilot/claude-3.5-sonnet" --skip-permissions
```

### Batch Evaluations
```bash
# All prompts (default models)
uv run python -m mcp_evaluation run-all --agent both --skip-permissions

# All prompts with custom models
uv run python -m mcp_evaluation run-all --agent both --claude-model haiku --opencode-model "github-copilot/gpt-4o" --skip-permissions

# Specific prompts with custom models
uv run python -m mcp_evaluation batch 1 2 3 --agent both --claude-model opus --opencode-model "opencode/llama-3.1-70b" --skip-permissions
```

### Statistics & Analysis
```bash
# View statistics
uv run python -m mcp_evaluation stats

# Use SQLite instead of InfluxDB
uv run python -m mcp_evaluation stats --backend sqlite
```

## 🎯 Example Output

```
Comparative Evaluation - Prompt 1
Session ID: eval_prompt001_1756832005

Claude Code Result:
✅ Success: True
Response: ## System Hardware Discovery Summary...
Cost: $0.1351
Duration: 37.8s

OpenCode Result:  
✅ Success: True
Response: Key system information summary...
Duration: 29.7s

Comparison Summary:
✅ Both agents succeeded
💰 Claude Code cost: $0.1351
```

## � Statistics Dashboard

```
Evaluation Statistics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric                ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Total Sessions        │ 5       │
│ Comparative Sessions  │ 2       │
│ Database Size (MB)    │ 0.04    │
│ Average Cost (USD)    │ $0.1351 │
└───────────────────────┴─────────┘

Agent Distribution:
• claude: 3 sessions
• opencode: 2 sessions
```

## ⚙️ Configuration

### Model Options

| Agent | Available Models | Example Usage |
|-------|------------------|---------------|
| **Claude** | `sonnet` (default), `haiku`, `opus` | `--claude-model haiku` |
| **OpenCode** | `github-copilot/claude-3.5-sonnet` (default)<br>`github-copilot/gpt-4o`<br>`opencode/llama-3.1-70b` | `--opencode-model "github-copilot/gpt-4o"` |

### Both Agents Example
```bash
# Compare Claude Haiku vs GPT-4o
uv run python -m mcp_evaluation run 1 --agent both --claude-model haiku --opencode-model "github-copilot/gpt-4o" --skip-permissions

# Compare Claude Sonnet vs Llama 3.1
uv run python -m mcp_evaluation run 1 --agent both --claude-model sonnet --opencode-model "opencode/llama-3.1-70b" --skip-permissions
```

### Default (InfluxDB)
No configuration needed! Uses InfluxDB by default.

### SQLite Alternative
```bash
uv run python -m mcp_evaluation run 1 --backend sqlite --skip-permissions
```

### Custom Config
Create `evaluation_config_influxdb.yaml` or `evaluation_config_sqlite.yaml` to customize settings.

## 🧪 Creating Test Prompts

Create `.md` files in `prompts/` directory:

```markdown
---
id: 1
complexity: "low"
target_mcp: ["node-hardware-mcp"]
description: "Basic MCP hardware information query"
timeout: 60
expected_tools: ["MCP", "Read"]
tags: ["basic", "discovery"]
---

# Basic MCP Hardware Discovery

Please discover and use available MCP servers to gather system information.
```

## 📝 Notes

- **Claude requires `--skip-permissions`** for automated evaluation
- **OpenCode is free** to use
- **InfluxDB runs in Docker** (started by setup script)
- **Grafana dashboard available** at http://localhost:3000 (admin/admin)
- **Data persists** between runs in time-series format
- **Session IDs** link comparative evaluations

---

**Simple, fast, effective MCP testing.** 🚀
