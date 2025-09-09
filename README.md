# MCP Evaluation System

A comprehensive evaluation system for testing MCP (Model Context Protocol) functionality with Claude and OpenCode agents, featuring **parallel multi-model execution**, **unified post-processing**, and **interactive tree visualizations**.

## ✨ Key Features

- **⚡ Parallel Multi-Model Execution**: Run multiple models simultaneously
- **📄 JSONL Prompt System**: Efficient prompt loading from `prompts_dataset.jsonl`
- **🔄 Real-time Progress Tracking**: Live evaluation monitoring with InfluxDB
- **🌳 Interactive Tree Visualizations**: Beautiful HTML conversation trees with expand/collapse
- **🧠 Intelligent Model Validation**: Automatic model detection with suggestions
- **💰 Cost & Performance Analytics**: Track costs, execution times, success rates
- **📊 Unified Post-Processing**: Dual-mode report generation (CSV/Advanced) with user prompt extraction
- **🧠 Semantic Analysis**: AI-powered evaluation quality assessment beyond simple success/failure
- **🔍 False Negative Detection**: Identify tasks that technically failed but actually succeeded
- **🛡️ Unlimited Processing Time**: No artificial timeout constraints
- **📱 Responsive Design**: Mobile-friendly visualizations for sharing and presentations

## 🚀 Quick Start

```bash
# Install dependencies
uv venv --clear && source .venv/bin/activate && uv sync

# Setup database
./scripts/setup_influxdb.sh

# Run evaluation
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions

# Generate reports with interactive trees
uv run python -m mcp_evaluation post-processing

# View interactive visualizations
uv run python -m mcp_evaluation log-visualizer --serve --port 8080

# Generate reports with AI-powered semantic analysis
uv run python -m mcp_evaluation post-processing --semantic
```

## 🔧 Command Reference

### Core Commands

| Command | Purpose | Basic Usage |
|---------|---------|-------------|
| `run <ID>` | Single prompt evaluation | `run 1 --agent both --skip-permissions` |
| `run-all` | All prompts evaluation | `run-all --agent both --skip-permissions` |
| `post-processing` | Generate reports | `post-processing --semantic --verbose` |
| `log-visualizer` | Interactive tree visualizations | `log-visualizer --serve --port 8080` |
| `semantic-analysis` | AI-powered quality analysis | `semantic-analysis session <session_id>` |
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

## 🧠 Semantic Analysis

**AI-powered evaluation quality assessment** that goes beyond simple success/failure to understand actual task completion quality, detect false negatives, and provide actionable insights.

### Core Features
- **False Negative Detection**: Identifies sessions marked as failed but actually completed successfully
- **Quality Assessment**: Evaluates response completeness, accuracy, and usefulness 
- **Agent Comparison**: Intelligent comparison between Claude and OpenCode approaches
- **Pattern Recognition**: Identifies trends and optimization opportunities across sessions
- **Actionable Insights**: Specific improvement suggestions for prompts and system optimization

### Quick Start
```bash
# Enable semantic analysis in post-processing
uv run python -m mcp_evaluation post-processing --semantic --verbose

# Analyze a specific session
uv run python -m mcp_evaluation semantic-analysis session <session_id> --verbose

# Batch analysis with cost control
uv run python -m mcp_evaluation semantic-analysis batch --limit 10 --cost-limit 5.0
```

### Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `post-processing --semantic` | Enable AI analysis during report generation | `post-processing --semantic --verbose` |
| `semantic-analysis session <id>` | Analyze specific evaluation session | `semantic-analysis session abc123 --verbose` |
| `semantic-analysis batch` | Analyze multiple sessions with patterns | `semantic-analysis batch --agent claude --limit 20` |

### Configuration Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--semantic` | Enable semantic analysis | `false` | `--semantic` |
| `--semantic-model` | Claude model for analysis | `sonnet` | `--semantic-model haiku` |
| `--semantic-cost-limit` | Maximum analysis cost (USD) | `5.0` | `--semantic-cost-limit 10.0` |
| `--model` | Analysis model selection | `sonnet` | `--model opus` |
| `--cost-limit` | Batch analysis cost limit | `5.0` | `--cost-limit 2.0` |
| `--limit` | Max sessions to analyze | `10` | `--limit 25` |

### Environment Configuration

Add to your `.env` file:
```bash
# Semantic Analysis Configuration
SEMANTIC_ANALYSIS_ENABLED=true
SEMANTIC_ANALYSIS_MODEL=sonnet          # haiku (fast/cheap) | sonnet (balanced) | opus (best quality)
SEMANTIC_ANALYSIS_CONFIDENCE_THRESHOLD=0.7
SEMANTIC_ANALYSIS_MAX_COST_PER_SESSION=0.05
SEMANTIC_ANALYSIS_BATCH_SIZE=5
```

### Output Examples

**Enhanced CSV Export** (with `--semantic` flag):
- All standard metrics plus semantic analysis columns:
- `semantic_success`, `semantic_confidence`, `quality_score`
- `task_comprehension_score`, `tool_effectiveness_score`, `response_completeness_score`  
- `false_negative_flag`, `improvement_suggestions`, `semantic_analysis_cost`

**Session Analysis Output**:
```
🧠 Semantic Analysis Complete
Session ID: abc123
Technical Success: False
Semantic Success: True  ← False negative detected!
Quality Score: 0.85
Confidence: 0.92

💡 Improvement Suggestions:
  1. Increase tool execution timeout for complex operations
  2. Add intermediate progress reporting
  3. Implement graceful timeout handling
```

**Batch Analysis Summary**:
```
📊 Batch Analysis Complete
Sessions Analyzed: 20
Semantic Success Rate: 85%
False Negative Rate: 15%  ← Technical failures that were actual successes
Average Quality Score: 0.78

💡 Key Improvement Opportunities:
  1. Better timeout handling in prompts
  2. Clearer task specification  
  3. Tool selection guidance
```

### Use Cases

**Debugging Failed Evaluations**:
```bash
# Analyze why a session technically failed but may have actually succeeded
uv run python -m mcp_evaluation semantic-analysis session ses_xyz123 --verbose
```

**System Optimization**:
```bash
# Identify patterns and improvement opportunities
uv run python -m mcp_evaluation semantic-analysis batch --limit 50
```

**Quality Assurance**:
```bash
# Find high-quality responses regardless of technical status
uv run python -m mcp_evaluation semantic-analysis batch --agent both --cost-limit 10.0
```

**Cost Management**:
- **Haiku**: Fast, cost-effective analysis (~$0.01/session)
- **Sonnet**: Balanced quality and cost (~$0.03/session)  
- **Opus**: Highest quality analysis (~$0.08/session)

## 🌳 Interactive Tree Visualizations

Beautiful, shareable HTML conversation trees that make evaluation results easy to understand and present.

### Key Features
- **🌳 Expandable Event Trees**: Click to expand/collapse conversation events
- **🎨 Beautiful UI**: Modern design with color-coded event types
- **📱 Mobile Responsive**: Works perfectly on all devices
- **🔗 Easy Sharing**: Self-contained HTML files work anywhere
- **👁️ Complete Visibility**: Shows exact prompts with evaluation markers
- **⚡ Fast Navigation**: Jump between sessions with index page

### Quick Start
```bash
# Generate trees for all sessions
uv run python -m mcp_evaluation log-visualizer

# Start web server to view trees
uv run python -m mcp_evaluation log-visualizer --serve --port 8080

# Generate tree for specific session
uv run python -m mcp_evaluation log-visualizer --session reports/claude/session-id/
```

### Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `log-visualizer` | Generate all tree visualizations | `log-visualizer` |
| `log-visualizer --serve` | Start web server for viewing | `log-visualizer --serve --port 8080` |
| `log-visualizer --session` | Generate single session tree | `log-visualizer --session path/to/session/` |

### Generated Files

For each evaluation session, the visualizer creates:
- **`conversation_tree.html`**: Interactive tree visualization
- **Navigation index**: `reports/index.html` with overview of all sessions

### Features

**Event Types**:
- 🔧 **Tool Executions**: Green highlighting with tool details
- 🤖 **Assistant Responses**: Blue highlighting with response content  
- 📝 **Messages**: Standard events with timestamps
- 👤 **User Prompts**: Yellow highlighting showing exact input

**Interactive Elements**:
- **Expand/Collapse**: Click any event header to view details
- **Expand All Button**: Quickly open all events at once
- **Session Metadata**: View duration, event count, tools used
- **Success Indicators**: Visual status (successful/failed sessions)

### Usage Examples

**View All Sessions**:
```bash
uv run python -m mcp_evaluation log-visualizer --serve --port 8080
# Open browser to http://localhost:8080
```

**Generate and Share**:
```bash
# Generate trees
uv run python -m mcp_evaluation log-visualizer

# Share HTML files
cp reports/claude/session-id/conversation_tree.html /path/to/share/
```

**Custom Port**:
```bash
uv run python -m mcp_evaluation log-visualizer --serve --port 3000
```

### Understanding Session Status

The visualizations help distinguish between different types of sessions:

**✅ Successful MCP Evaluations**: 
- Shows evaluation prompts with `EVAL_MODEL` and `EVAL_PROMPT_ID` markers
- Contains tool executions (like CPU info requests)
- Green success indicators

**❌ Failed MCP Evaluations**:
- Sessions completed without tool usage  
- Manual conversations outside evaluation framework
- Red status indicators (failed MCP criteria, not conversation quality)

### Output Structure

```
reports/
├── index.html                           # Main navigation page
├── claude/
│   └── session-id/
│       ├── conversation_tree.html       # Interactive tree
│       ├── monitoring.log              # Raw log data
│       └── evaluation_metrics.json     # Session metrics
└── opencode/
    └── session-id/
        ├── conversation_tree.html
        ├── monitoring.log
        └── evaluation_metrics.json
```

## 📊 Post-Processing

Process InfluxDB monitoring data and generate evaluation metrics with JSON reports, CSV exports, and **interactive tree visualizations**.

### Enhanced Features

✨ **New in this version**:
- **🌳 Interactive HTML Trees**: Beautiful conversation visualizations  
- **👤 User Prompt Extraction**: Shows exact prompts sent to each agent
- **🔍 Complete Conversation Flow**: Full visibility into evaluation process
- **📱 Mobile-Friendly**: Responsive design for sharing and presentations

### CSV Export Features

Export comprehensive evaluation metrics to CSV files with the same fields as `evaluation_metrics.json`:

```bash
# Generate individual session reports + CSV export + trees
uv run python -m mcp_evaluation post-processing --csv

# Export CSV with semantic analysis and trees  
uv run python -m mcp_evaluation post-processing --semantic --csv --verbose

# CSV-only export (skip individual session reports)
uv run python -m mcp_evaluation post-processing --csv-only --csv-path ./exports/

# Generate trees and start web server
uv run python -m mcp_evaluation post-processing
uv run python -m mcp_evaluation log-visualizer --serve --port 8080

# Filter by agent and export CSV with semantic analysis
uv run python -m mcp_evaluation post-processing --agent claude --semantic --csv
uv run python -m mcp_evaluation post-processing --agent opencode --csv-only
```

### CSV Output Structure

**Standard CSV columns** from `evaluation_metrics.json`:
- `number`, `prompt`, `session_id`, `agent_type`, `model`, `success`
- `execution_time`, `number_of_calls`, `number_of_tool_calls`, `tools_used`
- `cost_usd`, `response_length`, `created_at`, `completed_at`, `logfile`, `error_message`

**Additional semantic analysis columns** (with `--semantic` flag):
- `semantic_success`, `semantic_confidence`, `quality_score`
- `task_comprehension_score`, `tool_effectiveness_score`, `response_completeness_score`
- `false_negative_flag`, `improvement_suggestions`, `semantic_analysis_cost`

### Key Options
| Option | Description | Example |
|--------|-------------|---------|
| `--output` / `-o` | Output directory | `--output reports/` |
| `--agent` / `-a` | Filter by agent | `--agent claude` |
| `--semantic` | Enable AI-powered semantic analysis | `--semantic` |
| `--semantic-model` | Claude model for analysis | `--semantic-model haiku` |
| `--verbose` / `-v` | Detailed progress | `--verbose` |

### Examples
```bash
# Basic post-processing (all sessions)
uv run python -m mcp_evaluation post-processing

# Process with semantic analysis and detailed output
uv run python -m mcp_evaluation post-processing --semantic --verbose

# Process only Claude sessions with semantic analysis
uv run python -m mcp_evaluation post-processing --agent claude --semantic --verbose

# Process only OpenCode sessions  
uv run python -m mcp_evaluation post-processing --agent opencode --verbose

# Custom output directory with semantic analysis and CSV export
uv run python -m mcp_evaluation post-processing --output custom_reports/ --semantic --csv --verbose
```

## 🎯 Common Workflows

### Quick Start
```bash
uv run python -m mcp_evaluation test --agent opencode
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions  
uv run python -m mcp_evaluation post-processing --semantic --verbose
uv run python -m mcp_evaluation log-visualizer --serve --port 8080
```

### Full Evaluation with Visualizations
```bash
uv run python -m mcp_evaluation run-all --agent both --skip-permissions
uv run python -m mcp_evaluation post-processing --semantic --csv --verbose
uv run python -m mcp_evaluation log-visualizer --serve --port 8080
```

### Advanced Analysis
```bash
# Run evaluation and analyze specific session
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions
uv run python -m mcp_evaluation semantic-analysis session <session_id> --verbose

# Batch semantic analysis for pattern identification
uv run python -m mcp_evaluation semantic-analysis batch --limit 20 --cost-limit 5.0
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
│   ├── post_processor.py        # Report generation & user prompt extraction
│   ├── log_visualizer.py        # Interactive HTML tree generator
│   ├── semantic_analyzer.py     # AI-powered semantic analysis
│   ├── session_manager.py       # Database storage
│   ├── jsonl_prompt_loader.py   # JSONL prompt loader
│   └── prompt_loader.py         # Markdown prompt loader
├── prompts/                     
│   ├── prompts_dataset.jsonl    # Primary prompts (7 total)
│   └── backup_old_format/       # Backup .md files
├── reports/                     # Generated reports
│   ├── index.html               # Tree navigation page
│   ├── claude/                  # Claude session reports
│   │   └── session-id/
│   │       ├── conversation_tree.html
│   │       ├── monitoring.log
│   │       └── evaluation_metrics.json
│   └── opencode/                # OpenCode session reports
│       └── session-id/
│           ├── conversation_tree.html
│           ├── monitoring.log
│           └── evaluation_metrics.json
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

**Fast, parallel, comprehensive MCP testing with beautiful interactive visualizations.** ⚡📊🌳🚀
