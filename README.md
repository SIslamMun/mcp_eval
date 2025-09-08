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
uv run python -m mcp_evaluation post-processing
```

## 🔧 Command Reference

### Core Commands

| Command | Purpose | Basic Usage |
|---------|---------|-------------|
| `run <ID>` | Single prompt evaluation | `run 1 --agent both --skip-permissions` |
| `run-all` | All prompts evaluation | `run-all --agent both --skip-permissions` |
| `post-processing` | Generate reports | `post-processing` |
| `stats` | View evaluation statistics | `stats` |
| `models` | List available models | `models --agent both --preference fast` |
| `test` | Quick functionality test | `test --agent opencode` |
| `cleanup` | Clean processes | `cleanup` |

### Additional Commands

| Command | Purpose | Basic Usage |
|---------|---------|-------------|
| `setup` | Initialize infrastructure | `setup` |
| `batch <IDs>` | Multiple prompt evaluation | `batch 1 2 3 --agent both` |

### Agent & Model Options

**Claude Models:** `sonnet` (balanced), `haiku` (fast), `opus` (accurate)
**OpenCode Models:** `github-copilot/claude-3.5-sonnet`, `github-copilot/gpt-4o`, `github-copilot/claude-3.7-sonnet`

### Key Examples

```bash
# Single evaluation
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions

# Multi-model parallel
uv run python -m mcp_evaluation run 1 --claude-models sonnet,haiku --skip-permissions

# All prompts
uv run python -m mcp_evaluation run-all --agent both --skip-permissions

# Generate reports
uv run python -m mcp_evaluation post-processing
uv run python -m mcp_evaluation post-processing --agent claude --verbose

# Utilities
uv run python -m mcp_evaluation stats


# Advanced commands
uv run python -m mcp_evaluation setup
uv run python -m mcp_evaluation batch 1 2 3 --agent both --skip-permissions
```

## 📊 Post-Processing

Process InfluxDB monitoring data and generate evaluation metrics with JSON reports.

### Key Options
| Option | Description | Example |
|--------|-------------|---------|
| `--output` / `-o` | Output directory | `--output reports/` |
| `--agent` / `-a` | Filter by agent | `--agent claude` |
| `--verbose` / `-v` | Detailed progress | `--verbose` |

### Examples
```bash
# Basic post-processing (all sessions)
uv run python -m mcp_evaluation post-processing

# Process with detailed output
uv run python -m mcp_evaluation post-processing --verbose

# Process only Claude sessions
uv run python -m mcp_evaluation post-processing --agent claude --verbose

# Process only OpenCode sessions  
uv run python -m mcp_evaluation post-processing --agent opencode --verbose

# Custom output directory
uv run python -m mcp_evaluation post-processing --output custom_reports/ --verbose
```

## 🎯 Common Workflows

### Quick Start
```bash
uv run python -m mcp_evaluation test --agent opencode
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions  
uv run python -m mcp_evaluation post-processing
```

### Full Evaluation
```bash
uv run python -m mcp_evaluation run-all --agent both --skip-permissions
uv run python -m mcp_evaluation post-processing --verbose
```

## 🔧 Agent Configuration

### Models
**Claude:** `haiku` (fast), `sonnet` (balanced), `opus` (accurate)  
**OpenCode:** `github-copilot/claude-3.5-sonnet`, `github-copilot/gpt-4o`, `github-copilot/claude-3.7-sonnet`

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
│   ├── post_processor.py        # Report generation
│   ├── session_manager.py       # Database storage
│   ├── jsonl_prompt_loader.py   # JSONL prompt loader
│   └── prompt_loader.py         # Markdown prompt loader
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
