#!/usr/bin/env python3
"""
write_decision.py - Record a locked decision in the memory database.

Writes to the `decisions` table so lookup_decision() and /remember-decide
can answer "what did we decide about X, and why" directly, instead of
relying on transcript/semantic search over full session summaries.

decision_number is a global auto-incrementing sequence across all projects
(matches the "#76"-style numbering used by remember_decide.py lookups).

Usage:
    python3 write_decision.py --project oth --description "<text>" --rationale "<text>"
    python3 write_decision.py --project oth --description "<text>"   # rationale optional

Session ID is auto-detected from the active session's JSONL file.

Exit codes: 0 = success, 1 = error
"""

import argparse
import glob
import os
import pathlib
import re
import sqlite3
import sys
from datetime import datetime, timezone

import yaml

from config_utils import resolve_db_path


def get_root_path():
    return str(pathlib.Path(__file__).resolve().parent.parent)


def load_config(root_path):
    config_path = os.path.join(root_path, "config.yaml")
    if not os.path.exists(config_path):
        print(f"ERROR: config.yaml not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def connect_db(db_path):
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def auto_detect_session_id():
    """Auto-detect current session ID from JSONL files.

    Claude Code encodes the project directory by replacing every
    non-alphanumeric character in the cwd with '-' (not just '/'),
    so e.g. '/Users/x/claude_remembers' -> '-Users-x-claude-remembers'.
    """
    cwd = os.getcwd()
    encoded_cwd = re.sub(r"[^a-zA-Z0-9]", "-", cwd)
    project_dir = os.path.join(os.path.expanduser("~"), ".claude", "projects", encoded_cwd)

    jsonl_files = glob.glob(os.path.join(project_dir, "*.jsonl"))
    if not jsonl_files:
        return None

    jsonl_path = max(jsonl_files, key=os.path.getmtime)
    return os.path.splitext(os.path.basename(jsonl_path))[0]


def write_decision(project, description, rationale, session_id, db_path):
    conn = connect_db(db_path)
    try:
        row = conn.execute(
            "SELECT prefix FROM project_registry WHERE prefix = ?",
            (project,),
        ).fetchone()
        if not row:
            print(f"ERROR: Unknown project prefix: {project}", file=sys.stderr)
            return 1

        next_num = conn.execute(
            "SELECT COALESCE(MAX(decision_number), 0) + 1 FROM decisions"
        ).fetchone()[0]

        conn.execute(
            """INSERT INTO decisions
               (project, decision_number, session_id, description, rationale, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                project,
                next_num,
                session_id,
                description,
                rationale,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        print(f"Decision #{next_num} recorded for project '{project}'.")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Record a locked decision in the memory DB")
    parser.add_argument("--project", required=True, help="Project prefix (e.g. 'oth')")
    parser.add_argument("--description", required=True, help="What was decided")
    parser.add_argument("--rationale", help="Why it was decided (optional but recommended)")
    parser.add_argument("--session-id", help=argparse.SUPPRESS)
    args = parser.parse_args()

    root_path = get_root_path()
    config = load_config(root_path)
    db_path = resolve_db_path(config)

    session_id = args.session_id or auto_detect_session_id()
    if not session_id:
        print("ERROR: No active session detected.", file=sys.stderr)
        sys.exit(1)

    exit_code = write_decision(args.project, args.description, args.rationale, session_id, db_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
