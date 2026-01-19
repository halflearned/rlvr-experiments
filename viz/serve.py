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
from urllib.parse import urlparse, parse_qs

def find_latest_trace(traces_dir="traces"):
    """Find the most recently modified trace file."""
    pattern = os.path.join(traces_dir, "trace_*.jsonl")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def find_rollout_file(trace_file):
    """Find the rollout file corresponding to a trace file.

    Trace files are named: traces/trace_YYYYMMDD_HHMMSS.jsonl
    Rollout files are named: traces/rollouts_YYYYMMDD_HHMMSS.jsonl
    """
    if not trace_file:
        return None

    trace_dir = os.path.dirname(trace_file) or "."
    trace_name = os.path.basename(trace_file)

    # Extract timestamp from trace filename
    # Format: trace_YYYYMMDD_HHMMSS.jsonl
    if trace_name.startswith("trace_") and trace_name.endswith(".jsonl"):
        timestamp = trace_name[6:-6]  # Extract "YYYYMMDD_HHMMSS"
        rollout_file = os.path.join(trace_dir, f"rollouts_{timestamp}.jsonl")
        if os.path.exists(rollout_file):
            return rollout_file

    # Fallback: find latest rollouts file in same directory
    pattern = os.path.join(trace_dir, "rollouts_*.jsonl")
    files = glob.glob(pattern)
    if files:
        return max(files, key=os.path.getmtime)

    return None


def lookup_rollout(rollout_file, prompt_id, version=None):
    """Look up a rollout record by prompt_id and optionally version.

    Returns the matching record or None if not found.
    If version is None, returns the first match for prompt_id.
    """
    if not rollout_file or not os.path.exists(rollout_file):
        return None

    with open(rollout_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("prompt_id") == prompt_id:
                    if version is None or record.get("version") == version:
                        return record
            except json.JSONDecodeError:
                continue

    return None

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

# Find corresponding rollout file
ROLLOUT_FILE = find_rollout_file(TRACE_FILE)

# Change to project root so trace files are accessible
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=".", **kwargs)

    def do_GET(self):
        # Serve /trace as the configured trace file
        # Query params:
        #   ?filter=metrics  - Only counter/buffer/meta events (fast initial load)
        #   ?filter=spans    - Only span events (for timeline)
        #   (no filter)      - Full trace (legacy, slow for large files)
        parsed = urlparse(self.path)
        if parsed.path == "/trace":
            if TRACE_FILE and os.path.exists(TRACE_FILE):
                query = parse_qs(parsed.query)
                filter_type = query.get("filter", [None])[0]

                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                if filter_type == "metrics":
                    # Fast path: only send counter, meta events, and downsampled buffer
                    # Buffer events are huge (can be 700MB), so we downsample them
                    # We also keep the last buffer event for accurate fates display
                    buffer_count = 0
                    buffer_sample_rate = 100  # Keep every 100th buffer event
                    last_buffer_line = None
                    output_lines = []

                    with open(TRACE_FILE, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            # Quick check without full JSON parse
                            is_span = '"type":"span"' in line or '"type": "span"' in line
                            is_buffer = '"type":"buffer"' in line or '"type": "buffer"' in line

                            if is_span:
                                continue  # Skip all spans
                            elif is_buffer:
                                buffer_count += 1
                                last_buffer_line = line  # Track last buffer for fates
                                # Downsample buffer events (they're huge due to fates accumulation)
                                if buffer_count % buffer_sample_rate == 0:
                                    self.wfile.write((line + "\n").encode())
                            else:
                                # Counter, meta events - keep all
                                self.wfile.write((line + "\n").encode())

                    # Always include the last buffer event for accurate final fates
                    if last_buffer_line and buffer_count % buffer_sample_rate != 0:
                        self.wfile.write((last_buffer_line + "\n").encode())
                elif filter_type == "spans":
                    # Only send span events (for timeline visualization)
                    # Downsample if too many
                    span_count = 0
                    max_spans = 50000  # Limit to 50k spans
                    with open(TRACE_FILE, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            if '"type":"span"' in line or '"type": "span"' in line:
                                span_count += 1
                                # Simple downsampling: keep every Nth span if too many
                                # We don't know total count upfront, so use reservoir-style
                                self.wfile.write((line + "\n").encode())
                                if span_count >= max_spans:
                                    break
                else:
                    # Full trace (legacy)
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

        # Serve /rollout?prompt_id=X&version=Y to look up a specific rollout
        if parsed.path == "/rollout":
            query = parse_qs(parsed.query)
            prompt_id = query.get("prompt_id", [None])[0]
            version_str = query.get("version", [None])[0]
            version = int(version_str) if version_str is not None else None

            if not prompt_id:
                self.send_error(400, "Missing prompt_id parameter")
                return

            record = lookup_rollout(ROLLOUT_FILE, prompt_id, version)
            if record:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(record).encode())
            else:
                self.send_error(404, f"Rollout not found for prompt_id={prompt_id}")
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
if ROLLOUT_FILE:
    print(f"Rollout file: {ROLLOUT_FILE}")
    print(f"  Available at http://localhost:{PORT}/rollout?prompt_id=...")
print("Press Ctrl+C to stop")

with http.server.HTTPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
