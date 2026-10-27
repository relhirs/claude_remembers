#!/usr/bin/env python3
"""pre-compact.py - Claude Code PreCompact hook for claude_remembers.

Fires before context compaction (manual /compact or auto-compact).
Safety net: calls write_exchange.py one final time to ensure all
messages are captured to the DB before compaction wipes context.

stop.py (async) handles normal capture, but there is a small window
where compaction could trigger before the last async stop.py finishes.
This hook closes that gap.

RULE: stdout is SACRED. Only valid JSON goes to stdout.
"""

import glob
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone


def _log_hook_error(root, hook_name, message):
    """Append a timestamped line to logs/hooks.log. Never raises."""
    try:
        log_dir = os.path.join(root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(os.path.join(log_dir, "hooks.log"), "a") as f:
            f.write(f"{ts} [{hook_name}] {message}\n")
    except Exception:
        pass


def main():
    # Read stdin (hook protocol requires it)
    sys.stdin.read()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Determine current session JSONL path from CWD
    cwd = os.getcwd()
    encoded_cwd = re.sub(r"[^a-zA-Z0-9]", "-", cwd)
    project_dir = os.path.join(os.path.expanduser("~"), ".claude", "projects", encoded_cwd)

    jsonl_files = glob.glob(os.path.join(project_dir, "*.jsonl"))
    if not jsonl_files:
        print("{}")
        return

    jsonl_path = max(jsonl_files, key=os.path.getmtime)
    session_id = os.path.splitext(os.path.basename(jsonl_path))[0]

    # Call write_exchange.py - dedup is built in, safe to call multiple times.
    # stderr goes to write_exchange.py's own log file instead of being
    # discarded, so a crash before its internal logger starts is still
    # captured somewhere.
    try:
        hostname = socket.gethostname()
        log_dir = os.path.join(root, "logs", hostname)
        os.makedirs(log_dir, exist_ok=True)
        we_log = open(os.path.join(log_dir, "write_exchange.log"), "a")
        subprocess.run(
            [sys.executable, os.path.join(root, "scripts", "write_exchange.py"),
             "--session-id", session_id,
             "--jsonl-path", jsonl_path],
            stdout=subprocess.DEVNULL,
            stderr=we_log,
        )
        we_log.close()
    except Exception as e:
        _log_hook_error(root, "pre-compact.py", f"write_exchange.py launch failed: {e!r}")

    print("{}")


if __name__ == "__main__":
    main()
