#!/usr/bin/env python3
"""Simple HTTP server for the RLVR Heartbeat visualization.

Usage:
    python viz/serve.py [port]

Then open http://localhost:8080 in your browser.
"""

import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

# Change to project root so trace files are accessible
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=".", **kwargs)

    def end_headers(self):
        # Enable CORS for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

print(f"Serving RLVR Heartbeat at http://localhost:{PORT}/viz/")
print(f"Trace files available at http://localhost:{PORT}/traces/trace.jsonl")
print("Press Ctrl+C to stop")

with http.server.HTTPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
