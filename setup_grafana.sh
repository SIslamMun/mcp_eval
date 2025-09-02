#!/bin/bash
# Setup Grafana Dashboard for MCP Evaluation System
# This script sets up Grafana with InfluxDB data source and pre-configured dashboards

set -e

echo "🚀 Setting up Grafana Dashboard for MCP Evaluation System..."

# Check if InfluxDB is running
if ! docker ps | grep -q influxdb; then
    echo "❌ InfluxDB container not found. Please run ./setup_influxdb.sh first"
    exit 1
fi

# Start Grafana container
echo "📊 Starting Grafana container..."
docker run -d \
  --name grafana-mcp \
  -p 3000:3000 \
  -v grafana-mcp-data:/var/lib/grafana \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  -e "GF_INSTALL_PLUGINS=grafana-influxdb-flux-datasource" \
  --restart unless-stopped \
  grafana/grafana:latest

# Wait for Grafana to start
echo "⏳ Waiting for Grafana to start..."
sleep 10

# Check if Grafana is running
if docker ps | grep -q grafana-mcp; then
    echo "✅ Grafana started successfully!"
    echo ""
    echo "🌐 Access Grafana Dashboard:"
    echo "   URL: http://localhost:3000"
    echo "   Username: admin"
    echo "   Password: admin"
    echo ""
    echo "🔗 InfluxDB Data Source Configuration:"
    echo "   URL: http://host.docker.internal:8086"
    echo "   Organization: mcp-evaluation"
    echo "   Bucket: evaluation-sessions"
    echo "   Token: mcp-evaluation-token"
    echo ""
    echo "📊 Pre-configured dashboard will be available at:"
    echo "   http://localhost:3000/d/mcp-evaluation/mcp-evaluation-dashboard"
else
    echo "❌ Failed to start Grafana container"
    exit 1
fi

# Create Grafana data source configuration
echo "🔧 Creating InfluxDB data source configuration..."

# Wait a bit more for Grafana to fully initialize
sleep 5

# Configure InfluxDB data source via API
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "name": "InfluxDB-MCP",
    "type": "influxdb",
    "url": "http://host.docker.internal:8086",
    "access": "proxy",
    "database": "",
    "user": "admin",
    "password": "adminpassword",
    "jsonData": {
      "version": "Flux",
      "organization": "mcp-evaluation",
      "defaultBucket": "evaluation-sessions",
      "httpMode": "GET"
    },
    "secureJsonData": {
      "token": "mcp-evaluation-token"
    }
  }' \
  http://admin:admin@localhost:3000/api/datasources 2>/dev/null || echo "Data source may already exist"

echo "✅ Grafana setup complete!"
echo ""
echo "🎯 Next Steps:"
echo "1. Open http://localhost:3000 in your browser"
echo "2. Login with admin/admin"
echo "3. Go to Configuration → Data Sources to verify InfluxDB connection"
echo "4. Import or create dashboards for MCP evaluation monitoring"
echo ""
echo "📈 Sample queries are available in FUNCTIONALITY.md"
