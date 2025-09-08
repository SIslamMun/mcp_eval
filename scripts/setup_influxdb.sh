#!/bin/bash

# InfluxDB Setup Script for MCP Evaluation
# This script sets up InfluxDB using Docker for the MCP evaluation system

echo "🚀 Setting up InfluxDB for MCP Evaluation..."

# Load configuration from .env file
if [ -f ".env" ]; then
    echo "📄 Loading configuration from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  .env file not found, using default values..."
    # Default values if .env is not found
    INFLUXDB_URL="http://localhost:8086"
    INFLUXDB_TOKEN="mcp-evaluation-token"
    INFLUXDB_ORG="mcp-evaluation"
    INFLUXDB_BUCKET="evaluation-sessions"
fi

# Extract port from URL (default to 8086 if not specified)
INFLUXDB_PORT=$(echo $INFLUXDB_URL | sed -n 's/.*:\([0-9]*\)$/\1/p')
if [ -z "$INFLUXDB_PORT" ]; then
    INFLUXDB_PORT=8086
fi

echo "🔧 Using configuration:"
echo "   URL: $INFLUXDB_URL"
echo "   Organization: $INFLUXDB_ORG"
echo "   Bucket: $INFLUXDB_BUCKET"
echo "   Token: $INFLUXDB_TOKEN"
echo "   Port: $INFLUXDB_PORT"

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

# Start InfluxDB container with values from .env
echo "🐳 Starting InfluxDB container..."
docker run -d \
  --name influxdb \
  -p $INFLUXDB_PORT:8086 \
  -v influxdb-data:/var/lib/influxdb2 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=adminpassword \
  -e DOCKER_INFLUXDB_INIT_ORG=$INFLUXDB_ORG \
  -e DOCKER_INFLUXDB_INIT_BUCKET=$INFLUXDB_BUCKET \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=$INFLUXDB_TOKEN \
  influxdb:2.7

# Wait for InfluxDB to be ready
echo "⏳ Waiting for InfluxDB to be ready..."
sleep 10

# Check if InfluxDB is running
if docker ps | grep -q influxdb; then
    echo "✅ InfluxDB is running successfully!"
    echo ""
    echo "📋 InfluxDB Configuration:"
    echo "   URL: $INFLUXDB_URL"
    echo "   Username: admin"
    echo "   Password: adminpassword"
    echo "   Organization: $INFLUXDB_ORG"
    echo "   Bucket: $INFLUXDB_BUCKET"
    echo "   Token: $INFLUXDB_TOKEN"
    echo ""
    echo "🔧 Configuration is automatically loaded from .env file"
    echo ""
    echo "🌐 Access InfluxDB UI at: $INFLUXDB_URL"
else
    echo "❌ Failed to start InfluxDB. Check Docker logs:"
    docker logs influxdb
    exit 1
fi
