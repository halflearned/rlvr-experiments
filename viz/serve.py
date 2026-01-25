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

# Server-side favorites file
FAVORITES_FILE = os.path.join(os.path.dirname(__file__), ".trace_favorites.json")
# Server-side notes file (path -> note text)
NOTES_FILE = os.path.join(os.path.dirname(__file__), ".trace_notes.json")


def load_favorites():
    """Load favorites from disk."""
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, "r") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError):
            pass
    return set()


def save_favorites(favorites):
    """Save favorites to disk."""
    with open(FAVORITES_FILE, "w") as f:
        json.dump(list(favorites), f, indent=2)


def load_notes():
    """Load notes from disk. Returns dict of path -> note."""
    if os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_notes(notes):
    """Save notes to disk."""
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2)

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

# Mutable state for hot-reloading traces
class TraceState:
    trace_file = None
    rollout_file = None

# Find trace file
if args.trace_file:
    TraceState.trace_file = args.trace_file
else:
    TraceState.trace_file = find_latest_trace()

if TraceState.trace_file and not os.path.exists(TraceState.trace_file):
    print(f"Error: Trace file not found: {TraceState.trace_file}")
    sys.exit(1)

# Find corresponding rollout file
TraceState.rollout_file = find_rollout_file(TraceState.trace_file)

# Change to project root so trace files are accessible
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def list_trace_files(traces_dir="traces"):
    """List all trace files in a directory, sorted by modification time (newest first)."""
    # Look in various locations for trace files
    patterns = [
        os.path.join(traces_dir, "trace*.jsonl"),
        os.path.join(traces_dir, "*trace*.jsonl"),  # e.g., sweep_B_trace.jsonl
        os.path.join("results", "*", "traces", "trace*.jsonl"),
        os.path.join("results", "*", "traces", "*trace*.jsonl"),
        os.path.join("results", "*", "*trace*.jsonl"),  # e.g., results/sweep_traces/sweep_B_trace.jsonl
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    # Dedupe (in case multiple patterns match same file)
    files = list(set(files))
    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return files


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
            if TraceState.trace_file and os.path.exists(TraceState.trace_file):
                query = parse_qs(parsed.query)
                filter_type = query.get("filter", [None])[0]

                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                if filter_type == "metrics":
                    # Send counter and meta events (small), skip spans and buffer
                    # Buffer events are fetched separately with ?filter=buffer
                    with open(TraceState.trace_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            # Skip span and buffer events
                            if '"type":"span"' in line or '"type": "span"' in line:
                                continue
                            if '"type":"buffer"' in line or '"type": "buffer"' in line:
                                continue
                            self.wfile.write((line + "\n").encode())
                elif filter_type == "buffer":
                    # Paginated buffer events: ?filter=buffer&offset=0&limit=5000
                    offset = int(query.get("offset", [0])[0])
                    limit = int(query.get("limit", [5000])[0])
                    buffer_count = 0
                    sent_count = 0
                    with open(TraceState.trace_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            if '"type":"buffer"' in line or '"type": "buffer"' in line:
                                if buffer_count >= offset:
                                    self.wfile.write((line + "\n").encode())
                                    sent_count += 1
                                    if sent_count >= limit:
                                        break
                                buffer_count += 1
                elif filter_type == "spans":
                    # Only send span events (for timeline visualization)
                    # Supports pagination: ?filter=spans&offset=0&limit=50000
                    offset = int(query.get("offset", [0])[0])
                    limit = int(query.get("limit", [50000])[0])
                    span_count = 0
                    sent_count = 0
                    with open(TraceState.trace_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            if '"type":"span"' in line or '"type": "span"' in line:
                                if span_count >= offset:
                                    self.wfile.write((line + "\n").encode())
                                    sent_count += 1
                                    if sent_count >= limit:
                                        break
                                span_count += 1
                else:
                    # Full trace (legacy)
                    with open(TraceState.trace_file, "rb") as f:
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
                "path": TraceState.trace_file,
                "exists": TraceState.trace_file and os.path.exists(TraceState.trace_file),
            }
            if info["exists"]:
                info["size"] = os.path.getsize(TraceState.trace_file)
                info["mtime"] = os.path.getmtime(TraceState.trace_file)
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

            record = lookup_rollout(TraceState.rollout_file, prompt_id, version)
            if record:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(record).encode())
            else:
                self.send_error(404, f"Rollout not found for prompt_id={prompt_id}")
            return

        # Serve /traces to list available trace files
        if parsed.path == "/traces":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            traces = list_trace_files()
            # Include metadata for each trace
            result = []
            for path in traces[:50]:  # Limit to 50 most recent
                try:
                    stat = os.stat(path)
                    result.append({
                        "path": path,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
                except OSError:
                    continue
            self.wfile.write(json.dumps(result).encode())
            return

        # Serve /reload?path=... to switch to a different trace file
        if parsed.path == "/reload":
            query = parse_qs(parsed.query)
            new_path = query.get("path", [None])[0]

            if not new_path:
                self.send_error(400, "Missing path parameter")
                return

            if not os.path.exists(new_path):
                self.send_error(404, f"Trace file not found: {new_path}")
                return

            # Update the trace state
            TraceState.trace_file = new_path
            TraceState.rollout_file = find_rollout_file(new_path)

            print(f"[reload] Switched to: {new_path}")
            if TraceState.rollout_file:
                print(f"[reload] Rollout file: {TraceState.rollout_file}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "trace_file": TraceState.trace_file,
                "rollout_file": TraceState.rollout_file,
            }).encode())
            return

        # Serve /favorites to get/set favorite traces (server-side persistence)
        if parsed.path == "/favorites":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            favorites = load_favorites()
            self.wfile.write(json.dumps(list(favorites)).encode())
            return

        # Serve /notes?path=... to get note for a trace
        if parsed.path == "/notes":
            query = parse_qs(parsed.query)
            path = query.get("path", [None])[0]
            if not path:
                # Return all notes
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(load_notes()).encode())
            else:
                # Return note for specific path
                notes = load_notes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"note": notes.get(path, "")}).encode())
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        # POST /favorites/add?path=... to add a favorite
        if parsed.path == "/favorites/add":
            query = parse_qs(parsed.query)
            path = query.get("path", [None])[0]
            if not path:
                self.send_error(400, "Missing path parameter")
                return

            favorites = load_favorites()
            favorites.add(path)
            save_favorites(favorites)
            print(f"[favorites] Added: {path}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "favorites": list(favorites)}).encode())
            return

        # POST /favorites/remove?path=... to remove a favorite
        if parsed.path == "/favorites/remove":
            query = parse_qs(parsed.query)
            path = query.get("path", [None])[0]
            if not path:
                self.send_error(400, "Missing path parameter")
                return

            favorites = load_favorites()
            favorites.discard(path)
            save_favorites(favorites)
            print(f"[favorites] Removed: {path}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "favorites": list(favorites)}).encode())
            return

        # POST /notes to save a note for a trace
        if parsed.path == "/notes":
            query = parse_qs(parsed.query)
            path = query.get("path", [None])[0]
            if not path:
                self.send_error(400, "Missing path parameter")
                return

            # Read note content from request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else ""

            notes = load_notes()
            if body.strip():
                notes[path] = body
            else:
                notes.pop(path, None)  # Remove empty notes
            save_notes(notes)
            print(f"[notes] Updated note for: {path}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())
            return

        self.send_error(404, "Not found")

    def end_headers(self):
        # Enable CORS for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

print(f"Serving RLVR Heartbeat at http://localhost:{PORT}/viz/")
if TraceState.trace_file:
    print(f"Trace file: {TraceState.trace_file}")
    print(f"  Available at http://localhost:{PORT}/trace")
else:
    print("No trace file found - use 'Load trace' button in UI")
if TraceState.rollout_file:
    print(f"Rollout file: {TraceState.rollout_file}")
    print(f"  Available at http://localhost:{PORT}/rollout?prompt_id=...")
print("Press Ctrl+C to stop")

with http.server.HTTPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
