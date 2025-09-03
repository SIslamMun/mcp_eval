# 📊 Post-Processing Engine (NEW!)

**Features:** Generate comprehensive CSV reports with 16 metrics, individual communication logs, advanced filtering by agent/prompt/date, and summary statistics from InfluxDB/SQLite data.

**16 CSV Metrics:**
1. `number` - Sequential record number
2. `prompt` - Prompt ID used  
3. `session_id` - Unique session identifier
4. `agent_type` - Agent used (claude/opencode)
5. `model` - Specific model used
6. `success` - Evaluation success status
7. `execution_time` - Duration in seconds
8. `number_of_calls` - Total API/system calls
9. `number_of_tool_calls` - MCP tool invocations
10. `tools_used` - Tools and call counts
11. `cost_usd` - Cost in USD (Claude only)
12. `response_length` - Response text length
13. `created_at` - Session start timestamp
14. `completed_at` - Session end timestamp
15. `logfile` - Path to communication log
16. `error_message` - Error details if failed

**Key Commands:**
```bash
# Generate full report with logs
uv run python -m mcp_evaluation post-processing

# Filter specific data
uv run python -m mcp_evaluation post-processing --filter-agent claude --filter-prompt 1,2,3

# Quick statistics overview
uv run python -m mcp_evaluation post-processing --summary

# Date-range analysis
uv run python -m mcp_evaluation post-processing --date-from 2025-09-01 --date-to 2025-09-03

# Custom output location without logs
uv run python -m mcp_evaluation post-processing --output /path/to/reports --format csv --no-logs

# OpenCode-only analysis for specific prompts
uv run python -m mcp_evaluation post-processing --filter-agent opencode --filter-prompt 1,999

# Use SQLite backend instead of InfluxDB
uv run python -m mcp_evaluation post-processing --backend sqlite

# Generate JSON format report
uv run python -m mcp_evaluation post-processing --format json --output reports/
```

**Available Options:**
- `--output`, `-o`: Output directory (default: reports/)
- `--format`, `-f`: Report format (csv, json)
- `--backend`, `-b`: Database backend (influxdb, sqlite)
- `--filter-agent`: Filter by agent type (claude, opencode)
- `--filter-prompt`: Filter by prompt IDs (comma-separated)
- `--date-from`: Filter from date (YYYY-MM-DD)
- `--date-to`: Filter to date (YYYY-MM-DD)
- `--no-logs`: Skip generating log files
- `--summary`: Show statistics only

**Capabilities:** Exports tool analytics (bash, read_file, etc.), success rates, cost tracking, performance metrics, session timelines, and error analysis. Supports date filtering, custom output paths, and multi-database backends. Perfect for evaluation analysis and model comparison reports.

**Detailed Capabilities:**

**📊 Data Analysis & Export:**
- Generate structured CSV reports with complete evaluation metrics
- Export individual communication logs for detailed session analysis
- Create JSON metadata files with report generation details
- Support multiple output formats and custom directory paths

**🔍 Advanced Filtering & Search:**
- Filter by agent type (Claude vs OpenCode) for comparative analysis
- Select specific prompt IDs for targeted evaluation studies
- Apply date range filters for temporal analysis
- Combine multiple filters for precise data extraction

**📈 Performance Analytics:**
- Track tool usage patterns across different agents and models
- Analyze execution times and identify performance bottlenecks  
- Monitor success/failure rates and error patterns
- Calculate cost metrics and budget analysis for Claude evaluations

**🗄️ Multi-Database Support:**
- Primary InfluxDB time-series database integration
- SQLite fallback support for portable analysis
- Seamless switching between database backends
- Consistent data format across different storage systems

**📋 Report Generation:**
- Comprehensive 16-column CSV reports with all evaluation metrics
- Session-by-session communication logs for debugging
- Summary statistics with distribution analysis
- Metadata tracking for report provenance and reproducibility
