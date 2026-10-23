#!/usr/bin/env python3
"""session-end.py - Claude Code SessionEnd hook for claude_remembers.

Fires when session ends. The `reason` field in stdin JSON tells us why:
  - "clear"             → user ran /clear (session will be replaced by new one)
  - "prompt_input_exit" → user ran /exit
  - "resume"            → switched via interactive /resume
  - "logout"            → user logged out
  - "other"             → terminal close or other

Behavior: launch scripts/summarize_session.py as a detached background
process, for EVERY reason - it generates a structured summary via Claude
Haiku (or falls back to a simple placeholder if that fails) and records
`end_reason` on the session row. hooks/user-prompt-submit.py uses that
column (not a marker-in-notes trick) to detect post-/clear recovery, with
no time window.

Always: run remember_sync.py to backup the database (detached).

RULE: stdout is SACRED. Only valid JSON goes to stdout.
"""

import glob
import json
import os
import subprocess
import sys


def _detect_session_id():
    """Auto-detect current session ID from JSONL files (same logic as stop.py)."""
    cwd = os.getcwd()
    encoded_cwd = cwd.replace("/", "-")
    project_dir = os.path.join(os.path.expanduser("~"), ".claude", "projects", encoded_cwd)
    jsonl_files = glob.glob(os.path.join(project_dir, "*.jsonl"))
    if not jsonl_files:
        return None
    jsonl_path = max(jsonl_files, key=os.path.getmtime)
    return os.path.splitext(os.path.basename(jsonl_path))[0]


def main():
    # Read stdin — parse `reason` field to distinguish /clear from /exit
    reason = "other"
    try:
        raw = sys.stdin.read()
        if raw.strip():
            hook_input = json.loads(raw)
            reason = hook_input.get("reason") or "other"
    except Exception:
        pass

    # Determine ROOT (parent of hooks/)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    session_id = _detect_session_id()
    if session_id:
        try:
            subprocess.Popen(
                [sys.executable, os.path.join(root, "scripts", "summarize_session.py"),
                 "--session-id", session_id, "--reason", reason],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass

    # Run database backup (detached - hook must return immediately)
    try:
        subprocess.Popen(
            [sys.executable, os.path.join(root, "scripts", "remember_sync.py")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass

    print("{}")


if __name__ == "__main__":
    main()
