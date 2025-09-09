#!/bin/bash
# Cleanup script to remove generated files

echo "Cleaning up generated files..."

# Remove generated Python scripts
rm -f check_influxdb_data.py
rm -f cleanup_claude_sessions.py
rm -f clear_influxdb_data.py
rm -f detailed_influx_check.py
rm -f final_data_status_report.py
rm -f fixed_hook.py
rm -f log_vs_influxdb_analysis.py
rm -f verify_data_queries.py
rm -f status.md

# Remove generated utils
rm -f src/mcp_evaluation/simple_influxdb_processor.py
rm -f src/mcp_evaluation/unified_post_processing.py
rm -f src/mcp_evaluation/utils.py

# Remove reports directory
rm -rf reports/

echo "Cleanup complete!"
