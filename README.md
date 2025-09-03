# MCP Evaluation System

A comprehensive evaluation system for testing MCP (Model Context Protocol) functionality with Claude and OpenCode agents, featuring **parallel multi-model execution** and **unified post-processing**.

## ✨ Key Features

- **⚡ Parallel Multi-Model Execution**: Run multiple models simultaneously
- **📄 JSONL Prompt System**: Efficient prompt loading from `prompts_dataset.jsonl`
- **🔄 Real-time Progress Tracking**: Live evaluation monitoring
- **🧠 Intelligent Model Validation**: Automatic model detection with suggestions
- **💰 Cost & Performance Analytics**: Track costs, execution times, success rates
- **📊 Unified Post-Processing**: Dual-mode report generation (CSV/Advanced)
- **🛡️ Unlimited Processing Time**: No artificial timeout constraints

## 🚀 Quick Start

```bash
# Install dependencies
uv venv --clear && source .venv/bin/activate && uv sync

# Setup database
./scripts/setup_influxdb.sh

# Run evaluation
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions

# Generate reports
uv run python -m mcp_evaluation post-processing --summary
```

## 🔧 Command Reference

### Core Commands

| Command | Purpose | Basic Usage |
|---------|---------|-------------|
| `run <ID>` | Single prompt evaluation | `run 1 --agent both --skip-permissions` |
| `run-all` | All prompts evaluation | `run-all --agent both --skip-permissions` |
| `post-processing` | Generate reports | `post-processing --summary` |
| `models` | List available models | `models --agent both --preference fast` |
| `test` | Quick functionality test | `test --agent opencode` |
| `cleanup` | Clean processes | `cleanup` |

### Agent & Model Options

**Claude Models:** `sonnet` (balanced), `haiku` (fast), `opus` (accurate)
**OpenCode Models:** `github-copilot/claude-3.5-sonnet`, `github-copilot/gpt-4o`

### Key Examples

```bash
# Single evaluation
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions

# Multi-model parallel
uv run python -m mcp_evaluation run 1 --claude-models sonnet,haiku --skip-permissions

# All prompts
uv run python -m mcp_evaluation run-all --agent both --skip-permissions

# Generate reports
uv run python -m mcp_evaluation post-processing --summary
uv run python -m mcp_evaluation post-processing --agent claude --verbose

# Utilities
uv run python -m mcp_evaluation models --preference fast
uv run python -m mcp_evaluation test --agent opencode
```

## 📊 Post-Processing

Generate reports with dual processing modes:
- **📊 CSV-Only (Default)**: Fast reports
- **📝 Advanced Mode**: Reports + timeline logs (auto-enabled with filters)

### Key Options
| Option | Description | Example |
|--------|-------------|---------|
| `--summary` | Quick statistics only | `--summary` |
| `--agent` | Filter by agent | `--agent claude` |
| `--prompt` | Filter by prompt | `--prompt 1` |
| `--all` | Process all with logs | `--all` |
| `--verbose` | Detailed progress | `--verbose` |

### Examples
```bash
# Quick summary
uv run python -m mcp_evaluation post-processing --summary

# Fast CSV report (default)
uv run python -m mcp_evaluation post-processing

# Advanced analysis with logs
uv run python -m mcp_evaluation post-processing --agent claude --verbose
uv run python -m mcp_evaluation post-processing --all --verbose
```

## 🎯 Common Workflows

### Quick Start
```bash
uv run python -m mcp_evaluation test --agent opencode
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions  
uv run python -m mcp_evaluation post-processing --summary
```

### Full Evaluation
```bash
uv run python -m mcp_evaluation run-all --agent both --skip-permissions
uv run python -m mcp_evaluation post-processing --all --verbose
```

## 🔧 Agent Configuration

### Models
**Claude:** `haiku` (fast), `sonnet` (balanced), `opus` (accurate)  
**OpenCode:** `github-copilot/claude-3.5-sonnet`, `github-copilot/gpt-4o`

### Usage
```bash
# Single model
--claude-model sonnet --opencode-model "github-copilot/gpt-4o"

# Multi-model parallel
--claude-models sonnet,haiku --opencode-models "github-copilot/claude-3.5-sonnet,github-copilot/gpt-4o"
```

## 📁 File Structure

```
├── src/mcp_evaluation/          # Core system
│   ├── cli.py                   # Command interface
│   ├── evaluation_engine.py     # Evaluation orchestration
│   ├── unified_agent.py         # Agent interface
│   ├── *_post_processing.py     # Report generation
│   └── session_manager.py       # Database storage
├── prompts/                     
│   ├── prompts_dataset.jsonl    # Primary prompts (7 total)
│   └── backup_old_format/       # Backup .md files
├── scripts/                     # Setup scripts
└── tests/                       # Unit tests
```

## 📖 Help Information

```bash
# General help
uv run python -m mcp_evaluation --help

# Command-specific help
uv run python -m mcp_evaluation <command> --help
```

---

**Fast, parallel, comprehensive MCP testing with unlimited processing time.** ⚡📊🚀
