# 🎯 MCP Evaluation System - Complete Functionality Overview

## 📋 Core Functionality

### 🔧 **What It Does**
• **Tests MCP (Model Context Protocol) servers** with AI agents
• **Compares performance** between Claude and OpenCode agents
• **Stores evaluation results** in time-series database (InfluxDB/SQLite)
• **Tracks costs, timing, and success rates** for each evaluation
• **Generates comparative analytics** and statistics
• **Manages session continuity** across multiple prompts

### 🤖 **Supported Agents**
• **Claude Code** (via Anthropic API)
  - Models: `sonnet`
  - Features: Cost tracking, JSON output, session management
  - Requires: `--skip-permissions` for automation
• **OpenCode** (multiple backends)
  - Models: GitHub Copilot (Claude 3.5, GPT-4o and more)
  - Features: Free usage, detailed logging, multiple model access
  - No permission requirements

## 📥 **Input Types**

### 🗂️ **Prompt Files** (`.md` format)
```yaml
---
id: 1
complexity: "low|medium|high"
target_mcp: ["node-hardware-mcp", "other-mcp"]
description: "What this prompt tests"
timeout: 60
expected_tools: ["MCP", "Read", "Write"]
tags: ["hardware", "discovery"]
---

# Prompt Content
Instructions for the AI agent...
```

### ⚙️ **Command Parameters**
• **Prompt Selection**: Single ID (`run 1`), batch (`batch 1 2 3`), all (`run-all`)
• **Agent Selection**: `--agent claude|opencode|both`
• **Model Selection**: `--claude-model haiku`, `--opencode-model "github-copilot/gpt-4o"`
• **Backend Selection**: `--backend influxdb|sqlite`
• **Session Management**: `--continue-session`, `--continue-sessions`
• **Filters**: `--complexity low`, `--mcp-target node-hardware-mcp`

### 📁 **Configuration Files** (Optional)
• `evaluation_config_influxdb.yaml` - InfluxDB settings
• `evaluation_config_sqlite.yaml` - SQLite settings
• `opencode.json` - MCP server configuration

## 📤 **Output Types**

### 🖥️ **Terminal Output**
```
╭─────────────────────────────── ✅ Claude Evaluation - Prompt 1 ───────────────────────────────╮
│ Session ID: eval_prompt001_1756832722                                                          │
│ Success: True                                                                                  │
│ Response: ## System Hardware Discovery Summary...                                             │
│ Cost: $0.2376                                                                                 │
│ Duration: 46.2s                                                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────╯
```

### 📊 **Statistics Dashboard**
```
Evaluation Statistics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric                ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Total Sessions        │ 6       │
│ Comparative Sessions  │ 2       │
│ Database Size (MB)    │ 0.04    │
│ Average Cost (USD)    │ $0.2376 │
└───────────────────────┴─────────┘
```

### 🗄️ **Database Storage**
• **InfluxDB** (default): Time-series data with tags and fields
• **SQLite** (fallback): Relational data with JSON columns
• **Fields**: session_id, agent_type, model, success, cost, duration, response

### 📋 **Comparative Results**
```
Comparative Evaluation - Prompt 1
Session ID: eval_prompt001_1756832005

Claude Code Result:
✅ Success: True, Cost: $0.1351, Duration: 37.8s

OpenCode Result:  
✅ Success: True, Duration: 29.7s

Comparison Summary:
✅ Both agents succeeded
💰 Claude Code cost: $0.1351
```

## 🚀 **Command Categories**

### 🎯 **Single Evaluations**
```bash
# Basic single evaluations
uv run python -m mcp_evaluation run 1                                    # Prompt 1, default settings
uv run python -m mcp_evaluation run 1 --agent claude --skip-permissions  # Claude only
uv run python -m mcp_evaluation run 1 --agent opencode                   # OpenCode only
uv run python -m mcp_evaluation run 1 --agent both --skip-permissions    # Both agents (comparative)

# Custom models
uv run python -m mcp_evaluation run 1 --agent claude --claude-model haiku --skip-permissions
uv run python -m mcp_evaluation run 1 --agent opencode --opencode-model "github-copilot/gpt-4o"
uv run python -m mcp_evaluation run 1 --agent both --claude-model opus --opencode-model "opencode/llama-3.1-70b" --skip-permissions
```

### 📦 **Batch Operations**
```bash
# Run all prompts
uv run python -m mcp_evaluation run-all --agent both --skip-permissions

# Run specific prompts
uv run python -m mcp_evaluation batch 1 2 3 --agent both --skip-permissions

# With session continuity (context preserved between prompts)
uv run python -m mcp_evaluation batch 1 2 3 --continue-sessions --agent claude --skip-permissions

# Filtered by complexity or MCP target
uv run python -m mcp_evaluation run-all --complexity low --agent both --skip-permissions
uv run python -m mcp_evaluation run-all --mcp-target node-hardware-mcp --agent both --skip-permissions
```

### 📈 **Analytics & Export**
```bash
# View statistics
uv run python -m mcp_evaluation stats
uv run python -m mcp_evaluation stats --backend sqlite

# Show specific results
uv run python -m mcp_evaluation results 1 2 3

# Export data
uv run python -m mcp_evaluation export --output evaluation_results --format json
```

### 🔧 **Setup & Management**
```bash
# Initialize infrastructure
uv run python -m mcp_evaluation setup

# Setup InfluxDB (Docker)
./scripts/setup_influxdb.sh

# Validate setup
uv run python -m mcp_evaluation setup --config evaluation_config_influxdb.yaml
```

## 📋 **Data Flow**

### 🔄 **Evaluation Process**
1. **Load prompt** from `.md` file in `prompts/` directory
2. **Initialize agent** with specified model and configuration
3. **Create session** with unique ID format: `eval_prompt{ID:03d}_{timestamp}`
4. **Execute evaluation** with timeout and retry logic
5. **Capture results** (success, response, cost, timing, tool usage)
6. **Store in database** (InfluxDB time-series or SQLite relational)
7. **Display formatted output** with color-coded success/failure

### 🔗 **Session Management**
• **Individual Sessions**: Each prompt gets unique session ID
• **Comparative Sessions**: Same base session ID for both agents
• **Session Continuity**: Context preserved across multiple prompts
• **Session Format**: `eval_prompt{ID:03d}_{timestamp}`
• **UUID Conversion**: Claude converts to deterministic UUID format

## 🎯 **Use Cases**

### 🧪 **MCP Server Testing**
• Verify MCP servers work with different agents
• Test hardware discovery, file operations, API calls
• Validate MCP protocol compliance
• Regression testing for MCP updates

### ⚖️ **Agent Comparison**
• Compare Claude vs OpenCode on identical tasks
• Evaluate different models (Haiku vs GPT-4o vs Llama)
• Analyze cost vs performance trade-offs
• A/B testing for prompt effectiveness

### 📊 **Performance Analysis**
• Track success rates over time
• Monitor evaluation costs and trends
• Identify optimal model combinations
• Generate comparative performance reports

### 🔍 **Quality Assurance**
• Validate MCP integrations before deployment
• Automated evaluation pipelines
• Continuous integration testing
• Performance benchmarking

## 💾 **Storage & Persistence**

### 📈 **InfluxDB (Default)**
• **Type**: Time-series database optimized for analytics
• **Tags**: agent_type, model, complexity, mcp_target, prompt_id
• **Fields**: success, cost_usd, duration_seconds, response_length, token_count
• **Features**: Retention policies, downsampling, real-time queries
• **Setup**: Docker-based with automated initialization

### 🗃️ **SQLite (Fallback)**
• **Type**: Local file-based relational database
• **Schema**: Sessions table with JSON columns for flexible data
• **Features**: SQL queries for custom analysis, portable database file
• **Use Case**: Development, testing, offline environments

### 🔄 **Data Retention**
• All evaluation results preserved indefinitely
• Session history maintained with full context
• Cost tracking across time periods
• Comparative analysis over multiple runs
• Export capabilities for external analysis

## 🛠️ **Technical Architecture**

### 🏗️ **Core Components**
• **CLI Module** (`cli.py`): Command-line interface and argument parsing
• **Evaluation Engine** (`evaluation_engine.py`): Core orchestration logic
• **Unified Agent** (`unified_agent.py`): Agent abstraction layer
• **Session Manager** (`session_manager.py`): Database operations
• **Prompt Loader** (`prompt_loader.py`): Markdown prompt parsing

### 🔌 **Agent Integration**
• **Claude Code**: JSON API integration with cost tracking
• **OpenCode**: Multiple model support via configuration
• **Unified Interface**: Common evaluation API for both agents
• **Error Handling**: Graceful degradation and retry logic

### 🗄️ **Database Backends**
• **Factory Pattern**: Dynamic backend selection
• **InfluxDB Manager**: Time-series operations with proper tagging
• **SQLite Manager**: Relational operations with JSON flexibility
• **Automatic Fallback**: SQLite backup if InfluxDB unavailable

## 📋 **Available Models**

### 🧠 **Claude Models**
| Model | Description | Cost | Speed | Use Case |
|-------|-------------|------|-------|----------|
| `sonnet` | Claude 3.5 Sonnet (default) | Medium | Fast | General purpose |


### 🤖 **OpenCode Models**
| Model | Description | Cost | Provider |
|-------|-------------|------|----------|
| `github-copilot/claude-3.5-sonnet` | Claude 3.5 via GitHub (default) | Free | GitHub Copilot |
| `github-copilot/gpt-4o` | GPT-4o via GitHub | Free | GitHub Copilot |


## 🎛️ **Configuration Options**

### 📝 **YAML Configuration**
```yaml
# Agent configurations
claude_config:
  type: "claude"
  model: "sonnet"
  output_format: "json"
  timeout: 60
  allowed_tools: ["Bash", "Edit", "Read", "Write", "MCP"]
  dangerously_skip_permissions: true

opencode_config:
  type: "opencode" 
  model: "github-copilot/claude-3.5-sonnet"
  timeout: 60
  enable_logs: true

# Evaluation settings
prompts_dir: "prompts"
timeout_seconds: 60
max_retries: 3
session_id_format: "eval_prompt{prompt_id:03d}_{timestamp}"

# Database backend
backend: "influxdb"
influxdb_url: "http://localhost:8086"
influxdb_token: "mcp-evaluation-token"
influxdb_org: "mcp-evaluation"
influxdb_bucket: "evaluation-sessions"
```

### 🐳 **Docker Configuration**
```bash
# InfluxDB Docker setup
docker run -d \
  --name influxdb \
  -p 8086:8086 \
  -v influxdb-data:/var/lib/influxdb2 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=adminpassword \
  -e DOCKER_INFLUXDB_INIT_ORG=mcp-evaluation \
  -e DOCKER_INFLUXDB_INIT_BUCKET=evaluation-sessions \
  influxdb:2.7
```

## 🔍 **Monitoring & Analytics**

### 🌐 **InfluxDB Web Dashboard**
```bash
# Access InfluxDB UI directly in browser
http://localhost:8086

# Default credentials (from scripts/setup_influxdb.sh)
Username: admin
Password: adminpassword
Organization: mcp-evaluation
Bucket: evaluation-sessions
```

### 📊 **Dashboard Navigation**
• **Data Explorer**: Query and visualize evaluation data
• **Dashboards**: Create custom monitoring dashboards
• **Tasks**: Set up automated data processing
• **Settings**: Manage users, tokens, and buckets

### 🔍 **Direct InfluxDB Queries**
```sql
-- View all evaluation results
from(bucket: "evaluation-sessions")
  |> range(start: -7d)
  |> filter(fn: (r) => r["_measurement"] == "evaluation_results")

-- Success rate by agent
from(bucket: "evaluation-sessions")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "evaluation_results")
  |> filter(fn: (r) => r["_field"] == "success")
  |> group(columns: ["agent_type"])
  |> mean()

-- Cost analysis over time
from(bucket: "evaluation-sessions")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "evaluation_results")
  |> filter(fn: (r) => r["_field"] == "cost_usd")
  |> aggregateWindow(every: 1d, fn: sum)
```

### 📱 **Quick Dashboard Setup**
1. Open http://localhost:8086 in browser
2. Login with admin/adminpassword
3. Go to "Dashboards" → "Create Dashboard"
4. Add cells with evaluation queries
5. Save dashboard for monitoring

### 📊 **Grafana Dashboard Integration**
```bash
# Setup Grafana with InfluxDB data source
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -v grafana-data:/var/lib/grafana \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  grafana/grafana:latest

# Access Grafana Dashboard
http://localhost:3000
Username: admin
Password: admin

# Pre-configured MCP Evaluation Dashboard:
http://localhost:3000/d/mcp-evaluation-main/mcp-evaluation-system-dashboard
```

### 🔗 **InfluxDB Data Source Configuration**
```yaml
# Grafana InfluxDB Data Source Settings
URL: http://host.docker.internal:8086
Database: evaluation-sessions
User: admin
Password: adminpassword
HTTP Method: GET

# Query Language: Flux
Organization: mcp-evaluation
Default Bucket: evaluation-sessions
Token: mcp-evaluation-token
```

### 📊 **Pre-built Grafana Queries**
```sql
# Success Rate Over Time
from(bucket: "evaluation-sessions")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "evaluation_results")
  |> filter(fn: (r) => r["_field"] == "success")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)

# Cost Analysis by Agent
from(bucket: "evaluation-sessions")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "evaluation_results")
  |> filter(fn: (r) => r["_field"] == "cost_usd")
  |> group(columns: ["agent_type"])
  |> sum()

# Response Time Distribution
from(bucket: "evaluation-sessions")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "evaluation_results")
  |> filter(fn: (r) => r["_field"] == "duration_seconds")
  |> histogram(bins: [0.0, 10.0, 30.0, 60.0, 120.0, 300.0])
```

### 📊 **Built-in Statistics**
• Total sessions and comparative sessions
• Success rate by agent and model
• Average costs and duration
• Database size and recent activity
• Agent distribution and prompt complexity

### 📈 **Custom Queries**
```python
# Example InfluxDB queries
from influxdb_client import InfluxDBClient

client = InfluxDBClient(url="http://localhost:8086", token="your-token", org="mcp-evaluation")
query_api = client.query_api()

# Success rate by agent
query = '''
from(bucket: "evaluation-sessions")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "evaluation_results")
  |> group(columns: ["agent_type"])
  |> mean(column: "_value")
'''
```

### 🎯 **Performance Metrics**
• Response time percentiles
• Cost per successful evaluation
• Token efficiency ratios
• Error rate trends
• Model comparison matrices

---

**🎯 Bottom Line**: Complete MCP testing automation with AI agent comparison, comprehensive analytics, and production-ready monitoring! 🚀
