# MCP Evaluation System

A comprehensive evaluation system for testing MCP (Model Context Protocol) functionality with Claude and OpenCode agents, featuring **parallel multi-model execution** for performance comparison.

## ✨ Key Features

- **⚡ Parallel Multi-Model Execution**: Run multiple models simultaneously for maximum efficiency
- **� JSONL Prompt System**: Efficient single-file prompt loading from `prompts_dataset.jsonl`
- **�🔄 Real-time Progress Tracking**: See exactly which models are processing and completing
- **🧠 Intelligent Model Validation**: Automatic invalid model detection with suggestions
- **⏱️ Timeout Protection**: 2-minute per-model timeout with graceful shutdown
- **💰 Cost & Performance Analytics**: Track costs, execution times, and success rates
- **🛠️ Robust Error Handling**: Comprehensive cleanup and troubleshooting tools
- **📊 Rich Console Output**: Color-coded progress indicators and detailed summaries

## 🚀 Quick Start

### 1. Install
```bash
uv venv --clear
source .venv/bin/activate
uv sync
```

### 2. Setup InfluxDB (default database)
```bash
./scripts/setup_influxdb.sh
```

### 3. Setup Grafana Dashboard (optional)
```bash
./scripts/setup_grafana.sh
```

### 4. Run Evaluations

**Basic Command Structure:**
```bash
uv run python -m mcp_evaluation <COMMAND> [ARGUMENTS] [OPTIONS]
```

**Quick Examples:**
```bash
# Run single prompt with both agents
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions

# Run parallel multi-model comparison 🚀 NEW!
uv run python -m mcp_evaluation run 1 --claude-models sonnet,haiku --opencode-models github-copilot/claude-3.5-sonnet,github-copilot/gpt-4o --skip-permissions

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
- **Uses JSONL prompt dataset** for efficient single-file prompt management

## � Project Structure

```
├── src/mcp_evaluation/          # Core evaluation system
│   ├── cli.py                   # Command-line interface
│   ├── evaluation_engine.py     # Main evaluation orchestration
│   ├── session_manager.py       # InfluxDB/SQLite storage
│   ├── unified_agent.py         # Claude/OpenCode interface
│   ├── prompt_loader.py         # Markdown prompt parser (fallback)
│   └── jsonl_prompt_loader.py   # JSONL prompt system (primary)
├── scripts/                     # Setup scripts (InfluxDB, Grafana)
├── prompts/                     # Prompt dataset
│   ├── prompts_dataset.jsonl    # Primary prompt source (7 prompts)
│   └── backup_old_format/       # Original .md files (backup)
├── tests/                       # Unit tests
├── docs/                        # Documentation (including FUNCTIONALITY.md)
└── grafana-mcp-evaluation-dashboard.json  # Grafana dashboard configuration
```

## 🔧 Commands

### 🆕 Parallel Multi-Model Execution (Recommended!)

**Command Template:**
```bash
uv run python -m mcp_evaluation run <PROMPT_ID> [--claude-models MODEL1,MODEL2,...] [--opencode-models MODEL1,MODEL2,...] [--skip-permissions]
```

**Examples:**
```bash
# Multiple Claude models in parallel
uv run python -m mcp_evaluation run 1 --claude-models sonnet,haiku,opus --skip-permissions

# Multiple OpenCode models in parallel
uv run python -m mcp_evaluation run 1 --opencode-models github-copilot/claude-3.5-sonnet,github-copilot/gpt-4o,github-copilot/claude-3.7-sonnet-thought

# Mixed parallel execution (both Claude and OpenCode with multiple models)
uv run python -m mcp_evaluation run 1 --claude-models sonnet,haiku --opencode-models github-copilot/claude-3.5-sonnet,github-copilot/gpt-4o --skip-permissions

# Check available models
uv run python -m mcp_evaluation models --agent claude
uv run python -m mcp_evaluation models --agent opencode
uv run python -m mcp_evaluation models  # Both agents
```

### Single Model Evaluations

**Command Template:**
```bash
uv run python -m mcp_evaluation run <PROMPT_ID> --agent <claude|opencode|both> [--claude-model <MODEL>] [--opencode-model <MODEL>] [--skip-permissions]
```

**Examples:**
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

**Command Template:**
```bash
uv run python -m mcp_evaluation run <PROMPT_ID> --agent <claude|opencode|both> [--claude-model <sonnet|haiku|opus>] [--opencode-model "<MODEL_NAME>"] [--skip-permissions]
```

**Examples:**
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

**Command Templates:**
```bash
# Run all prompts
uv run python -m mcp_evaluation run-all --agent <claude|opencode|both> [--claude-model <MODEL>] [--opencode-model <MODEL>] [--skip-permissions]

# Run specific prompts
uv run python -m mcp_evaluation batch <PROMPT_ID1> <PROMPT_ID2> ... --agent <claude|opencode|both> [OPTIONS]
```

**Examples:**
```bash
# All prompts (default models)
uv run python -m mcp_evaluation run-all --agent both --skip-permissions

# All prompts with custom models
uv run python -m mcp_evaluation run-all --agent both --claude-model haiku --opencode-model "github-copilot/gpt-4o" --skip-permissions

# Specific prompts with custom models
uv run python -m mcp_evaluation batch 1 2 3 --agent both --claude-model opus --opencode-model "opencode/llama-3.1-70b" --skip-permissions
```

### Testing & Troubleshooting

**Command Templates:**
```bash
# Quick test
uv run python -m mcp_evaluation test --agent <claude|opencode> [--timeout <SECONDS>] [--skip-permissions]

# Clean up processes
uv run python -m mcp_evaluation cleanup

# List available models
uv run python -m mcp_evaluation models [--agent <claude|opencode>]
```

**Examples:**
```bash
# Quick test with timeout
uv run python -m mcp_evaluation test --agent opencode --timeout 30
uv run python -m mcp_evaluation test --agent claude --timeout 60 --skip-permissions

# Clean up stuck processes
uv run python -m mcp_evaluation cleanup

# List available models
uv run python -m mcp_evaluation models --agent opencode
```

### Statistics & Analysis

**Command Templates:**
```bash
# View statistics
uv run python -m mcp_evaluation stats [--backend <influxdb|sqlite>]

# Export results
uv run python -m mcp_evaluation export [--format <json|csv>] [--output <FILE>]

# View specific results
uv run python -m mcp_evaluation results <PROMPT_ID> [--backend <influxdb|sqlite>]

# Generate comprehensive reports (NEW!)
uv run python -m mcp_evaluation post-processing [OPTIONS]
```

**Examples:**
```bash
# View statistics
uv run python -m mcp_evaluation stats

# Use SQLite instead of InfluxDB
uv run python -m mcp_evaluation stats --backend sqlite

# Export results to JSON
uv run python -m mcp_evaluation export --format json --output results.json

# View results for prompt 1
uv run python -m mcp_evaluation results 1

# Generate comprehensive CSV report with logs
uv run python -m mcp_evaluation post-processing

# Generate report for specific agent and prompts
uv run python -m mcp_evaluation post-processing --filter-agent opencode --filter-prompt 1,999

# Show data summary statistics
uv run python -m mcp_evaluation post-processing --summary
```

## 🎯 Example Output (Parallel Multi-Model)

```
Running evaluation for prompt 1

Multi-model instance evaluation detected

✅ Claude models to process: ['sonnet', 'haiku']
✅ OpenCode models to process: ['github-copilot/claude-3.5-sonnet', 'github-copilot/gpt-4o']

🚀 Starting Claude evaluation with 2 models (parallel)...
🚀 Starting OpenCode evaluation with 2 models (parallel)...
⚡ Running 4 models with 4 parallel workers...

📍 Processing Claude model 1/2: sonnet
📍 Processing Claude model 2/2: haiku  
📍 Processing Opencode model 1/2: claude-3.5-sonnet
📍 Processing Opencode model 2/2: gpt-4o
✅ Success Opencode claude-3.5-sonnet: 8000ms
✅ Success Claude haiku: 8200ms
✅ Success Claude sonnet: 8500ms  
✅ Success Opencode gpt-4o: 9100ms

📊 Parallel Execution Summary:
  ✅ Successful models: 4/4
  ⚡ Parallel speedup achieved!
  💰 Total cost: $0.0847

Model Performance Summary:
🏃 Fastest: opencode (github-copilot/claude-3.5-sonnet) - 8000ms
📊 Success Rate: 100.0% (4/4)
```

## � Statistics Dashboard

## � Statistics Dashboard

```
📄 Using JSONL prompt source: prompts/prompts_dataset.jsonl

Evaluation Statistics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric                ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Total Sessions        │ 52      │
│ Comparative Sessions  │ 11      │
│ Total Prompts         │ 7       │
│ Database Size (MB)    │ 0.36    │
│ Recent Sessions (24h) │ 52      │
│ Average Cost (USD)    │ $0.0913 │
└───────────────────────┴─────────┘

Agent Distribution:
  • claude: 20 sessions
  • opencode: 32 sessions

Prompt Complexity Distribution:
  • low: 4 prompts
  • medium: 2 prompts
  • high: 1 prompts

MCP Targets (1):
  • node-hardware-mcp
```

## ⚙️ Configuration

### 🆕 Multi-Model Options

| Command | Description | Example |
|---------|-------------|---------|
| `--claude-models` | Multiple Claude models (comma-separated) | `--claude-models sonnet,haiku,opus` |
| `--opencode-models` | Multiple OpenCode models (comma-separated) | `--opencode-models github-copilot/claude-3.5-sonnet,github-copilot/gpt-4o` |
| `--skip-permissions` | Skip Claude permission dialogs | `--skip-permissions` |

### Single Model Options

| Agent | Available Models | Example Usage |
|-------|------------------|---------------|
| **Claude** | `sonnet` (default), `haiku`, `opus` | `--claude-model haiku --skip-permissions` |
| **OpenCode** | `github-copilot/claude-3.5-sonnet` (default)<br>`github-copilot/gpt-4o`<br>`github-copilot/claude-3.7-sonnet-thought`<br>`github-copilot/o1-mini` | `--opencode-model "github-copilot/gpt-4o"` |

### Multi-Model Comparison Examples

**Command Template:**
```bash
uv run python -m mcp_evaluation run <PROMPT_ID> [--claude-models <MODEL1,MODEL2,...>] [--opencode-models <MODEL1,MODEL2,...>] [--skip-permissions]
```

**Examples:**
```bash
# Performance comparison across Claude models
uv run python -m mcp_evaluation run 1 --claude-models sonnet,haiku,opus --skip-permissions

# OpenCode model comparison  
uv run python -m mcp_evaluation run 1 --opencode-models github-copilot/claude-3.5-sonnet,github-copilot/gpt-4o,github-copilot/claude-3.7-sonnet-thought

# Cross-platform comparison (recommended for comprehensive evaluation)
uv run python -m mcp_evaluation run 1 --claude-models sonnet,haiku --opencode-models github-copilot/claude-3.5-sonnet,github-copilot/gpt-4o --skip-permissions
```

## 📋 Command Reference

| Command | Purpose | Template |
|---------|---------|----------|
| `run` | Execute single prompt evaluation | `run <ID> --agent <TYPE> [OPTIONS]` |
| `run-all` | Execute all available prompts | `run-all --agent <TYPE> [OPTIONS]` |
| `batch` | Execute specific prompts | `batch <ID1> <ID2> ... --agent <TYPE> [OPTIONS]` |
| `models` | List available models | `models [--agent <TYPE>]` |
| `test` | Quick functionality test | `test --agent <TYPE> [--timeout <SEC>]` |
| `cleanup` | Clean stuck processes | `cleanup` |
| `stats` | View evaluation statistics | `stats [--backend <TYPE>]` |
| `results` | View specific results | `results <ID> [--backend <TYPE>]` |
| `export` | Export evaluation data | `export [--format <TYPE>] [--output <FILE>]` |
| `setup` | Initialize infrastructure | `setup [--backend <TYPE>]` |
| `post-processing` | Generate comprehensive reports | `post-processing [OPTIONS]` |

### Common Arguments

| Argument | Description | Values | Example |
|----------|-------------|--------|---------|
| `--agent` | Choose evaluation agent | `claude`, `opencode`, `both` | `--agent both` |
| `--claude-model` | Single Claude model | `sonnet`, `haiku`, `opus` | `--claude-model haiku --skip-permissions` |
| `--opencode-model` | Single OpenCode model | `github-copilot/MODEL` | `--opencode-model "github-copilot/gpt-4o"` |
| `--claude-models` | Multiple Claude models | Comma-separated list | `--claude-models sonnet,haiku --skip-permissions` |
| `--opencode-models` | Multiple OpenCode models | Comma-separated list | `--opencode-models github-copilot/claude-3.5-sonnet,github-copilot/gpt-4o` |
| `--skip-permissions` | Skip Claude permissions | Flag (no value) | `--skip-permissions` |
| `--backend` | Database backend | `influxdb`, `sqlite` | `--backend sqlite` |
| `--timeout` | Test timeout in seconds | Integer | `--timeout 60` |
| `--continue-session` | Continue previous session | Flag (no value) | `--continue-session` |

## 📊 Post-Processing & Advanced Analytics

### 🔄 **Post-Processing Engine (NEW!)**

Generate comprehensive evaluation reports from stored database data with advanced filtering and analysis capabilities.

**Command Template:**
```bash
uv run python -m mcp_evaluation post-processing [OPTIONS]
```

**Key Features:**
- **📊 CSV Report Generation**: Structured data with all evaluation metrics
- **📝 Communication Logs**: Individual log files for each evaluation session
- **🔍 Advanced Filtering**: Filter by agent, prompts, date ranges
- **📈 Summary Statistics**: Quick data overview and distribution analysis
- **🗂️ Multiple Formats**: CSV output with JSON metadata

**Options:**
| Option | Description | Example |
|--------|-------------|---------|
| `--output`, `-o` | Output directory | `--output reports/` |
| `--format`, `-f` | Report format (csv, json) | `--format csv` |
| `--backend`, `-b` | Database backend | `--backend influxdb` |
| `--filter-agent` | Filter by agent type | `--filter-agent claude` |
| `--filter-prompt` | Filter by prompt IDs | `--filter-prompt 1,2,999` |
| `--date-from` | Filter from date | `--date-from 2025-09-01` |
| `--date-to` | Filter to date | `--date-to 2025-09-03` |
| `--no-logs` | Skip generating log files (logs included by default) | `--no-logs` |
| `--summary` | Show statistics only | `--summary` |

**Example Commands:**
```bash
# Generate full comprehensive report
uv run python -m mcp_evaluation post-processing

# Claude-only analysis for specific prompts
uv run python -m mcp_evaluation post-processing --filter-agent claude --filter-prompt 1,2,3

# Date-range analysis
uv run python -m mcp_evaluation post-processing --date-from 2025-09-01 --date-to 2025-09-03

# Quick data summary
uv run python -m mcp_evaluation post-processing --summary

# Custom output location without logs
uv run python -m mcp_evaluation post-processing --output /path/to/reports --format csv --no-logs
```

### 📄 **Report Output Structure**

```
reports/
├── evaluation_report_20250903_143022.csv    # Main CSV report
├── logs/                                    # Communication logs
│   ├── eval_prompt001_1756832722.log
│   ├── eval_prompt999_1756919874.log
│   └── ...
└── metadata.json                           # Report generation metadata
```

### 📊 **CSV Report Columns**

| Column | Description | Example |
|--------|-------------|---------|
| `number` | Sequential record number | `1, 2, 3...` |
| `prompt` | Prompt ID used | `1, 2, 999` |
| `session_id` | Unique session identifier | `eval_prompt001_1756832722` |
| `agent_type` | Agent used (claude/opencode) | `claude`, `opencode` |
| `model` | Specific model used | `sonnet`, `github-copilot/gpt-4o` |
| `success` | Evaluation success status | `True`, `False` |
| `execution_time` | Duration in seconds | `45.2`, `12.8` |
| `number_of_calls` | Total API/system calls | `5`, `12` |
| `number_of_tool_calls` | MCP tool invocations | `3`, `7` |
| `tools_used` | Tools and call counts | `bash:2,read_file:1` |
| `cost_usd` | Cost in USD (Claude only) | `0.2376`, `0.0000` |
| `response_length` | Response text length | `1024`, `2048` |
| `created_at` | Session start timestamp | `2025-09-03T12:26:42Z` |
| `completed_at` | Session end timestamp | `2025-09-03T12:27:28Z` |
| `logfile` | Path to communication log | `logs/session_id.log` |
| `error_message` | Error details if failed | `Timeout occurred` |

### 📈 **Summary Statistics Output**

```bash
# Example summary output
📊 Data Summary Statistics:
       Evaluation Data Summary       
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric                 ┃ Value    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Total Records          │ 53       │
│ Successful Records     │ 47       │
│ Success Rate           │ 88.68%   │
│ Database Backend       │ influxdb │
│ Total Cost (USD)       │ $4.25    │
│ Average Cost (USD)     │ $0.0913  │
│ Average Execution Time │ 35.2s    │
└────────────────────────┴──────────┘

Agent Distribution:
  • claude: 20 sessions
  • opencode: 33 sessions

Prompt Distribution:
  • Prompt 1: 28 sessions
  • Prompt 999: 14 sessions
  • Prompt 2: 4 sessions
```

## 📄 JSONL Prompt System

The system uses a single JSONL file for efficient prompt management:

**Primary Source:**
```
prompts/prompts_dataset.jsonl  # 7 prompts loaded automatically
```

**JSONL Format:**
```json
{"id": 1, "name": "001-low-general", "complexity": "low", "category": "general", "target_mcp": ["node-hardware-mcp"], "description": "Basic MCP discovery and hardware information query", "timeout": 60, "expected_tools": ["MCP", "Read"], "tags": ["basic", "discovery", "hardware"], "content": "# Basic MCP Hardware Discovery\n\nPlease discover and use the available MCP servers to gather system information."}
```

**Benefits:**
- ✅ **Single file** instead of multiple .md files
- ✅ **Fast loading** with automatic detection
- ✅ **Fallback support** to .md files if needed
- ✅ **Rich metadata** with complexity, categories, tags
- ✅ **7 prompts available** with IDs: [1, 2, 3, 4, 5, 999, 1000]

**Status Check:**
```bash
# System automatically shows: "📄 Using JSONL prompt source: prompts/prompts_dataset.jsonl"
uv run python -m mcp_evaluation stats
```

## 📝 Notes

- **Claude requires `--skip-permissions`** for automated evaluation
- **OpenCode is free** to use  
- **⚡ Parallel execution** automatically optimizes performance with ThreadPoolExecutor
- **Model validation** provides intelligent suggestions for invalid models
- **Timeout protection** (2 minutes per model) prevents hanging processes
- **Real-time progress** shows which models are processing and completing
- **📄 JSONL prompt system** loads 7 prompts from single dataset file
- **InfluxDB runs in Docker** (started by `scripts/setup_influxdb.sh`)
- **Grafana dashboard available** at http://localhost:3000/d/mcp-evaluation-main/mcp-evaluation-system-dashboard (admin/admin)
- **Data persists** between runs in time-series format
- **Session IDs** link comparative evaluations

### 🔧 Troubleshooting

- **Process stuck?** Use `uv run python -m mcp_evaluation cleanup`
- **Invalid models?** Use `uv run python -m mcp_evaluation models --agent [claude|opencode]`
- **Need faster testing?** Use `--skip-permissions` for Claude
- **Timeout issues?** Default 2-minute timeout per model with graceful shutdown
- **JSONL not loading?** Check for `prompts/prompts_dataset.jsonl` file (auto-detected)
- **Need detailed analysis?** Use `uv run python -m mcp_evaluation post-processing --summary` for data overview
- **Report generation failed?** Check database connection and try `--backend sqlite` as fallback

---

**Fast, parallel, comprehensive MCP testing with JSONL prompt management.** ⚡📄🚀
