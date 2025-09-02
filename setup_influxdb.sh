#!/bin/bash

# InfluxDB Setup Script for MCP Evaluation
# This script sets up InfluxDB using Docker for the MCP evaluation system

echo "🚀 Setting up InfluxDB for MCP Evaluation..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Stop existing InfluxDB container if it exists
echo "🧹 Cleaning up existing InfluxDB container..."
docker stop influxdb 2>/dev/null || true
docker rm influxdb 2>/dev/null || true

# Create InfluxDB data volume
echo "📂 Creating InfluxDB data volume..."
docker volume create influxdb-data

# Start InfluxDB container
echo "🐳 Starting InfluxDB container..."
docker run -d \
  --name influxdb \
  -p 8086:8086 \
  -v influxdb-data:/var/lib/influxdb2 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=adminpassword \
  -e DOCKER_INFLUXDB_INIT_ORG=mcp-evaluation \
  -e DOCKER_INFLUXDB_INIT_BUCKET=evaluation-sessions \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=mcp-evaluation-token \
  influxdb:2.7

# Wait for InfluxDB to be ready
echo "⏳ Waiting for InfluxDB to be ready..."
sleep 10

# Check if InfluxDB is running
if docker ps | grep -q influxdb; then
    echo "✅ InfluxDB is running successfully!"
    echo ""
    echo "📋 InfluxDB Configuration:"
    echo "   URL: http://localhost:8086"
    echo "   Username: admin"
    echo "   Password: adminpassword"
    echo "   Organization: mcp-evaluation"
    echo "   Bucket: evaluation-sessions"
    echo "   Token: mcp-evaluation-token"
    echo ""
    echo "🔧 Update your evaluation_config.yaml:"
    echo "influxdb_url: \"http://localhost:8086\""
    echo "influxdb_token: \"mcp-evaluation-token\""
    echo "influxdb_org: \"mcp-evaluation\""
    echo "influxdb_bucket: \"evaluation-sessions\""
    echo ""
    echo "🌐 Access InfluxDB UI at: http://localhost:8086"
else
    echo "❌ Failed to start InfluxDB. Check Docker logs:"
    docker logs influxdb
    exit 1
fi
