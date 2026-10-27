#!/usr/bin/env python3
"""post-compact.py - Claude Code PostCompact hook for claude_remembers.

Fires after context compaction completes.
Re-injects memory context (recent session notes, project summary)
so Claude immediately has memory after losing it to compaction.

Reuses the same DB queries as session-start.py but returns a
lighter payload (no checklist, no NEXT_SESSION.md, no gap detection).

RULE: stdout is SACRED. Only valid JSON goes to stdout.
"""

import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from config_utils import resolve_db_path


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


def normalize_for_match(s):
    """Lowercase and strip everything except letters/digits/slashes.

    Makes folder-name matching robust to however the real folder is actually
    spelled - spaces, hyphens, underscores, commas, mixed case, any symbol -
    by collapsing all of that down to one canonical form before comparing.
    """
    return re.sub(r'[^a-z0-9/]', '', s.replace('\\', '/').lower())


def main():
    # Read stdin (hook protocol requires it)
    sys.stdin.read()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        config_path = os.path.join(root, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        db_path = resolve_db_path(config)
        if not os.path.exists(db_path):
            print("{}")
            return

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000;")

        lines = []
        lines.append("## Post-Compaction Memory Context")
        lines.append("Context was just compacted. Here is what is remembered so far:")
        lines.append("")

        # Current project summary
        try:
            cwd = os.environ.get("CWD", os.getcwd())
            norm_cwd = normalize_for_match(cwd)
            mapping = config.get("jsonl_project_mapping", {})
            # "oth" matches write_exchange.py's detect_project() fallback -
            # without this, a folder with no jsonl_project_mapping entry
            # left cwd_project as None here (but still wrote project="oth"
            # when capturing messages), silently disabling the project
            # summary section below for that folder.
            cwd_project = "oth"
            for folder, prefix in mapping.items():
                if normalize_for_match(folder) in norm_cwd:
                    cwd_project = prefix
                    break
            if cwd_project:
                proj_row = conn.execute("""
                    SELECT summary FROM project_registry
                    WHERE prefix = ? AND summary IS NOT NULL AND summary != ''
                """, (cwd_project,)).fetchone()
                if proj_row:
                    lines.append("### Current Project")
                    lines.append(proj_row[0])
                    lines.append("")
        except Exception:
            pass

        # Last 5 session notes for context
        try:
            rows = conn.execute("""
                SELECT project, notes, started_at
                FROM sys_sessions
                WHERE notes IS NOT NULL AND notes != ''
                ORDER BY started_at DESC
                LIMIT 5
            """).fetchall()
            if rows:
                lines.append("### Recent Sessions")
                for project, notes, started_at in rows:
                    date = started_at[:10] if started_at else "unknown"
                    topic = ""
                    for sline in (notes or "").split("\n"):
                        sline = sline.strip()
                        if sline and not sline.startswith("Session:") and not sline.startswith("Project:") and not sline.startswith("Time:"):
                            topic = sline[:120]
                            break
                    if topic:
                        lines.append(f"- [{date}] {project}: {topic}")
                lines.append("")
        except Exception:
            pass

        conn.close()

        if lines:
            context = "\n".join(lines)
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PostCompact",
                    "additionalContext": context
                }
            }))
        else:
            print("{}")

    except Exception as e:
        _log_hook_error(root, "post-compact.py", f"context build failed, returned empty: {e!r}")
        print("{}")


if __name__ == "__main__":
    main()
