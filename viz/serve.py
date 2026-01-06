#!/usr/bin/env python3
"""Simple HTTP server for the RLVR Heartbeat visualization.

Usage:
    python viz/serve.py [trace_file] [--port PORT]

Examples:
    python viz/serve.py                           # Serve latest trace in traces/
    python viz/serve.py traces/trace_20260106.jsonl  # Serve specific trace
    python viz/serve.py --port 9000               # Custom port

Then open http://localhost:8080/viz/ in your browser.
"""

import argparse
import glob
import http.server
import json
import os
import sys
from urllib.parse import urlparse

def find_latest_trace(traces_dir="traces"):
    """Find the most recently modified trace file."""
    pattern = os.path.join(traces_dir, "*.jsonl")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def parse_args():
    parser = argparse.ArgumentParser(description="Serve RLVR Heartbeat visualization")
    parser.add_argument("trace_file", nargs="?", help="Path to trace file (default: latest in traces/)")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to serve on (default: 8080)")
    return parser.parse_args()

# Parse arguments
args = parse_args()
PORT = args.port

# Find trace file
if args.trace_file:
    TRACE_FILE = args.trace_file
else:
    TRACE_FILE = find_latest_trace()

if TRACE_FILE and not os.path.exists(TRACE_FILE):
    print(f"Error: Trace file not found: {TRACE_FILE}")
    sys.exit(1)

# Change to project root so trace files are accessible
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=".", **kwargs)

    def do_GET(self):
        # Serve /trace as the configured trace file
        parsed = urlparse(self.path)
        if parsed.path == "/trace":
            if TRACE_FILE and os.path.exists(TRACE_FILE):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(TRACE_FILE, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "No trace file configured")
            return

        # Serve /trace/info as JSON with trace file metadata
        if parsed.path == "/trace/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            info = {
                "path": TRACE_FILE,
                "exists": TRACE_FILE and os.path.exists(TRACE_FILE),
            }
            if info["exists"]:
                info["size"] = os.path.getsize(TRACE_FILE)
                info["mtime"] = os.path.getmtime(TRACE_FILE)
            self.wfile.write(json.dumps(info).encode())
            return

        return super().do_GET()

    def end_headers(self):
        # Enable CORS for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

print(f"Serving RLVR Heartbeat at http://localhost:{PORT}/viz/")
if TRACE_FILE:
    print(f"Trace file: {TRACE_FILE}")
    print(f"  Available at http://localhost:{PORT}/trace")
else:
    print("No trace file found - use 'Load trace' button in UI")
print("Press Ctrl+C to stop")

with http.server.HTTPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
