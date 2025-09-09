#!/usr/bin/env python3
"""
Interactive HTML Tree Visualizer for MCP Evaluation Logs
Creates beautiful, shareable HTML reports with expandable conversation trees
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

class LogTreeVisualizer:
    """Generate interactive HTML tree visualizations of MCP evaluation logs."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        
    def parse_log_file(self, log_path: Path) -> Dict[str, Any]:
        """Parse a monitoring log file and extract structured data."""
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract header information
        session_info = {}
        header_section = content.split('========================================================')[0]
        
        for line in header_section.split('\n'):
            if ':' in line and not line.startswith('==='):
                key, value = line.split(':', 1)
                session_info[key.strip()] = value.strip()
        
        # Extract user prompt
        user_prompt = ""
        if "=== USER PROMPT ===" in content:
            prompt_section = content.split("=== USER PROMPT ===")[1].split("========================================================")[0]
            user_prompt = prompt_section.strip()
        
        # Extract events
        events = []
        event_sections = content.split("[Event ")
        
        for i, section in enumerate(event_sections[1:], 1):  # Skip first split part
            event_data = self._parse_event_section(section, i)
            if event_data:
                events.append(event_data)
        
        return {
            'session_info': session_info,
            'user_prompt': user_prompt,
            'events': events
        }
    
    def _parse_event_section(self, section: str, event_num: int) -> Optional[Dict[str, Any]]:
        """Parse an individual event section."""
        lines = section.split('\n')
        if not lines:
            return None
            
        # Parse event header
        header_line = lines[0]
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2})', header_line)
        type_match = re.search(r'Type: (\w+)', section)
        
        event_data = {
            'number': event_num,
            'timestamp': timestamp_match.group(1) if timestamp_match else '',
            'type': type_match.group(1) if type_match else 'Unknown',
            'content': section
        }
        
        # Extract tool information if present
        if 'Tool:' in section:
            tool_match = re.search(r'Tool: ([^\n]+)', section)
            if tool_match:
                event_data['tool'] = tool_match.group(1)
        
        # Extract response if present
        if 'Assistant Response:' in section:
            response_start = section.find('Assistant Response:')
            response_content = section[response_start:].split('=== END CONVERSATION DATA ===')[0]
            event_data['response'] = response_content.replace('Assistant Response:', '').strip()
        
        return event_data
    
    def generate_html_tree(self, session_data: Dict[str, Any], output_path: Path) -> None:
        """Generate an interactive HTML tree visualization."""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Evaluation Session: {session_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .session-meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .meta-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .meta-card .label {{
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        
        .meta-card .value {{
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .user-prompt {{
            padding: 30px;
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            margin: 20px;
            border-radius: 8px;
        }}
        
        .user-prompt h3 {{
            color: #856404;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .user-prompt pre {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #ffeaa7;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            line-height: 1.5;
        }}
        
        .conversation-tree {{
            padding: 30px;
        }}
        
        .tree-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
        }}
        
        .tree-header h3 {{
            color: #2c3e50;
            font-size: 1.5em;
        }}
        
        .expand-all {{
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.3s;
        }}
        
        .expand-all:hover {{
            background: #2980b9;
        }}
        
        .event-node {{
            margin-bottom: 15px;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .event-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.3s;
        }}
        
        .event-header:hover {{
            background: #e9ecef;
        }}
        
        .event-header.tool {{
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            border-left: 5px solid #4caf50;
        }}
        
        .event-header.response {{
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 5px solid #2196f3;
        }}
        
        .event-title {{
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .event-subtitle {{
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 3px;
        }}
        
        .event-badge {{
            background: #6c757d;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        
        .event-badge.tool {{
            background: #4caf50;
        }}
        
        .event-badge.response {{
            background: #2196f3;
        }}
        
        .event-content {{
            padding: 20px;
            background: white;
            display: none;
            border-top: 1px solid #e9ecef;
        }}
        
        .event-content.expanded {{
            display: block;
        }}
        
        .event-content pre {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            line-height: 1.4;
            border: 1px solid #dee2e6;
        }}
        
        .response-content {{
            background: #f0f8ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            margin-top: 15px;
        }}
        
        .response-content h4 {{
            color: #1976d2;
            margin-bottom: 10px;
        }}
        
        .tool-content {{
            background: #f1f8e9;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
            margin-top: 15px;
        }}
        
        .tool-content h4 {{
            color: #388e3c;
            margin-bottom: 10px;
        }}
        
        .expand-icon {{
            transition: transform 0.3s;
            font-size: 1.2em;
        }}
        
        .expand-icon.expanded {{
            transform: rotate(90deg);
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .session-meta {{
                grid-template-columns: 1fr;
            }}
            
            .tree-header {{
                flex-direction: column;
                gap: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌳 MCP Evaluation Session</h1>
            <div class="subtitle">Interactive Conversation Tree Viewer</div>
        </div>
        
        <div class="session-meta">
            <div class="meta-card">
                <div class="label">Session ID</div>
                <div class="value">{session_id}</div>
            </div>
            <div class="meta-card">
                <div class="label">Agent Type</div>
                <div class="value">{agent_type}</div>
            </div>
            <div class="meta-card">
                <div class="label">Duration</div>
                <div class="value">{duration}</div>
            </div>
            <div class="meta-card">
                <div class="label">Events</div>
                <div class="value">{event_count}</div>
            </div>
            <div class="meta-card">
                <div class="label">Tools Used</div>
                <div class="value">{tools_used}</div>
            </div>
        </div>
        
        {user_prompt_section}
        
        <div class="conversation-tree">
            <div class="tree-header">
                <h3>📋 Conversation Timeline</h3>
                <button class="expand-all" onclick="toggleAllEvents()">Expand All</button>
            </div>
            
            {events_html}
        </div>
        
        <div class="footer">
            Generated on {generated_time} | MCP Evaluation Framework
        </div>
    </div>
    
    <script>
        let allExpanded = false;
        
        function toggleEvent(eventNum) {{
            const content = document.getElementById('event-content-' + eventNum);
            const icon = document.getElementById('expand-icon-' + eventNum);
            
            if (content.classList.contains('expanded')) {{
                content.classList.remove('expanded');
                icon.classList.remove('expanded');
            }} else {{
                content.classList.add('expanded');
                icon.classList.add('expanded');
            }}
        }}
        
        function toggleAllEvents() {{
            const contents = document.querySelectorAll('.event-content');
            const icons = document.querySelectorAll('.expand-icon');
            const button = document.querySelector('.expand-all');
            
            allExpanded = !allExpanded;
            
            contents.forEach(content => {{
                if (allExpanded) {{
                    content.classList.add('expanded');
                }} else {{
                    content.classList.remove('expanded');
                }}
            }});
            
            icons.forEach(icon => {{
                if (allExpanded) {{
                    icon.classList.add('expanded');
                }} else {{
                    icon.classList.remove('expanded');
                }}
            }});
            
            button.textContent = allExpanded ? 'Collapse All' : 'Expand All';
        }}
        
        // Auto-expand first event
        document.addEventListener('DOMContentLoaded', function() {{
            const firstEvent = document.getElementById('event-content-1');
            const firstIcon = document.getElementById('expand-icon-1');
            if (firstEvent) {{
                firstEvent.classList.add('expanded');
                firstIcon.classList.add('expanded');
            }}
        }});
    </script>
</body>
</html>
        """
        
        # Prepare template variables
        session_info = session_data['session_info']
        
        session_id = session_info.get('Session ID', 'Unknown')
        agent_type = session_info.get('Agent Type', 'Unknown')
        duration = session_info.get('Duration', 'Unknown')
        event_count = session_info.get('Event Count', '0')
        tools_used = session_info.get('Tools Used', 'None')
        
        # Generate user prompt section
        user_prompt_section = ""
        if session_data['user_prompt']:
            user_prompt_section = f"""
        <div class="user-prompt">
            <h3>👤 User Prompt</h3>
            <pre>{session_data['user_prompt']}</pre>
        </div>
            """
        
        # Generate events HTML
        events_html = self._generate_events_html(session_data['events'])
        
        # Fill template
        html_content = html_template.format(
            session_id=session_id,
            agent_type=agent_type,
            duration=duration,
            event_count=event_count,
            tools_used=tools_used,
            user_prompt_section=user_prompt_section,
            events_html=events_html,
            generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_events_html(self, events: List[Dict[str, Any]]) -> str:
        """Generate HTML for the events tree."""
        events_html = ""
        
        for event in events:
            event_num = event['number']
            event_type = event['type']
            timestamp = event['timestamp']
            
            # Determine event styling
            header_class = "event-header"
            badge_class = "event-badge"
            
            if event.get('tool'):
                header_class += " tool"
                badge_class += " tool"
                event_title = f"🔧 Tool Execution: {event['tool']}"
            elif event.get('response'):
                header_class += " response"
                badge_class += " response"
                event_title = "🤖 Assistant Response"
            else:
                event_title = f"📝 {event_type}"
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%H:%M:%S")
            except:
                formatted_time = timestamp
            
            # Generate event content
            content_html = f'<pre>{event["content"]}</pre>'
            
            if event.get('response'):
                content_html += f"""
                <div class="response-content">
                    <h4>Assistant Response:</h4>
                    <div>{event['response']}</div>
                </div>
                """
            
            if event.get('tool'):
                content_html += f"""
                <div class="tool-content">
                    <h4>Tool: {event['tool']}</h4>
                    <div>Execution details in raw content above</div>
                </div>
                """
            
            events_html += f"""
            <div class="event-node">
                <div class="{header_class}" onclick="toggleEvent({event_num})">
                    <div>
                        <div class="event-title">{event_title}</div>
                        <div class="event-subtitle">{formatted_time}</div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span class="{badge_class}">Event {event_num}</span>
                        <span class="expand-icon" id="expand-icon-{event_num}">▶</span>
                    </div>
                </div>
                <div class="event-content" id="event-content-{event_num}">
                    {content_html}
                </div>
            </div>
            """
        
        return events_html
    
    def generate_session_tree(self, session_path: Path) -> Path:
        """Generate tree visualization for a single session."""
        log_file = session_path / "monitoring.log"
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        # Parse log data
        session_data = self.parse_log_file(log_file)
        
        # Generate HTML
        html_file = session_path / "conversation_tree.html"
        self.generate_html_tree(session_data, html_file)
        
        return html_file
    
    def generate_all_trees(self) -> List[Path]:
        """Generate tree visualizations for all sessions in reports directory."""
        generated_files = []
        
        for agent_dir in self.reports_dir.iterdir():
            if agent_dir.is_dir():
                for session_dir in agent_dir.iterdir():
                    if session_dir.is_dir():
                        try:
                            html_file = self.generate_session_tree(session_dir)
                            generated_files.append(html_file)
                            print(f"✅ Generated tree: {html_file}")
                        except Exception as e:
                            print(f"❌ Failed to generate tree for {session_dir}: {e}")
        
        return generated_files

def main():
    """Command line interface for the log visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate interactive HTML tree visualizations for MCP evaluation logs")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory path")
    parser.add_argument("--session", help="Specific session directory to visualize")
    parser.add_argument("--serve", action="store_true", help="Start a local web server to view the trees")
    parser.add_argument("--port", type=int, default=8080, help="Port for web server")
    
    args = parser.parse_args()
    
    visualizer = LogTreeVisualizer(args.reports_dir)
    
    if args.session:
        # Generate single session tree
        session_path = Path(args.session)
        html_file = visualizer.generate_session_tree(session_path)
        print(f"✅ Generated tree visualization: {html_file}")
    else:
        # Generate all trees
        html_files = visualizer.generate_all_trees()
        print(f"\n🎉 Generated {len(html_files)} tree visualizations!")
    
    if args.serve:
        import http.server
        import socketserver
        import webbrowser
        import os
        
        os.chdir(args.reports_dir)
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                super().end_headers()
        
        with socketserver.TCPServer(("", args.port), Handler) as httpd:
            print(f"\n🌐 Starting web server at http://localhost:{args.port}")
            print("📁 Serving files from:", os.getcwd())
            print("🔗 Open browser to view tree visualizations")
            
            # Try to open browser
            webbrowser.open(f"http://localhost:{args.port}")
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n👋 Server stopped")

if __name__ == "__main__":
    main()
