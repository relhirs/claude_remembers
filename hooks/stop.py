#!/usr/bin/env python3
"""stop.py - Claude Code Stop hook for claude_remembers.

Fires after every Claude response completes.
1. Determines current session ID and JSONL path
2. Calls write_exchange.py to capture new messages to DB
3. If backup is older than 12 hours, triggers remember_sync.py (detached)
4. Returns {} (no additional context needed)

RULE: stdout is SACRED. Only valid JSON goes to stdout.
"""

import glob
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

_BACKUP_MAX_AGE = 12 * 3600  # 12 hours in seconds


def _log_hook_error(root, hook_name, message):
    """Append a timestamped line to logs/hooks.log. Never raises.

    Covers launch-level failures (subprocess never started, or threw before
    its own internal logger could run) - distinct from write_exchange.py's
    and remember_sync.py's own per-hostname logs, which only capture
    failures *inside* those scripts once they're actually running.
    """
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

    # Determine ROOT (parent of hooks/)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Determine current session JSONL path from CWD
    cwd = os.getcwd()
    encoded_cwd = re.sub(r"[^a-zA-Z0-9]", "-", cwd)
    project_dir = os.path.join(os.path.expanduser("~"), ".claude", "projects", encoded_cwd)

    # Find most recently modified JSONL = current session
    jsonl_files = glob.glob(os.path.join(project_dir, "*.jsonl"))
    if not jsonl_files:
        print("{}")
        return

    jsonl_path = max(jsonl_files, key=os.path.getmtime)
    session_id = os.path.splitext(os.path.basename(jsonl_path))[0]

    # Call write_exchange.py - hook stdout is SACRED (JSON only), so its own
    # stdout stays suppressed, but stderr is appended to write_exchange.py's
    # own log file instead of discarded, so a crash before its internal
    # logger initializes is still captured somewhere.
    hostname = socket.gethostname()
    log_dir = os.path.join(root, "logs", hostname)
    try:
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
        _log_hook_error(root, "stop.py", f"write_exchange.py launch failed: {e!r}")

    # Auto-backup if last backup is older than 12 hours
    try:
        backup_path = os.path.join(root, "db-backup", "claude_remembers.db.bak1")
        run_backup = False
        if not os.path.exists(backup_path):
            run_backup = True
        else:
            age = time.time() - os.path.getmtime(backup_path)
            if age > _BACKUP_MAX_AGE:
                run_backup = True

        if run_backup:
            os.makedirs(log_dir, exist_ok=True)
            sync_log = open(os.path.join(log_dir, "remember_sync.log"), "a")
            subprocess.Popen(
                [sys.executable, os.path.join(root, "scripts", "remember_sync.py")],
                stdout=subprocess.DEVNULL,
                stderr=sync_log,
            )
            sync_log.close()
    except Exception as e:
        _log_hook_error(root, "stop.py", f"remember_sync.py launch failed: {e!r}")

    print("{}")


if __name__ == "__main__":
    main()
