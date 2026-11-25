#!/usr/bin/env python3
"""
config_utils.py - Shared config helpers for claude_remembers.

storage.mode decides where the SQLite database lives:
  "synced" -> storage.local_db_path (kept outside the synced folder,
              since SQLite corrupts under concurrent sync writes)
  "local"  -> storage.root_path/claude_remembers.db

Every script that opens the database must resolve the path through
resolve_db_path() instead of reading storage.local_db_path directly -
that key is only present (and only required) in "synced" mode.
"""

import json
import os
import shutil
import subprocess
import sys


# Message text the harness writes on the user's behalf: subagent hand-backs,
# task notices, slash-command and `!` shell-input artifacts. They are stored
# like any other message, but they are not something the user said, so they
# are neither prompts worth measuring nor memories worth retrieving. Shared by
# the retrieval hook and the metrics layer so the two can't drift apart.
MACHINE_GENERATED_PREFIXES = (
    "<agent-message",
    "<task-notification",
    "<system-reminder",
    "<local-command",
    "caveat: the messages below",
    "<command-name",
    "<command-message",
    "<bash-input",
    "<bash-stdout",
)


def use_utf8_stdio():
    """Make stdin, stdout and stderr UTF-8, whatever the OS default is.

    On Windows, when output is piped (a slash command, a hook, CI) Python falls
    back to the legacy code page (cp1252). That page has no check mark or minus
    sign, so a plain print() crashes, and a UTF-8 question piped in is misread.
    Claude Code reads and writes UTF-8, so the command-line tools switch to it.
    Does nothing where a stream cannot be reconfigured (already replaced, closed).
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass


def is_machine_generated(text):
    """True if `text` starts with one of the harness-generated markers."""
    return (text or "").lstrip()[:60].lower().startswith(MACHINE_GENERATED_PREFIXES)


def machine_filter_sql(column="t.content"):
    """SQL fragment, starting with AND NOT, that hides harness-written messages.

    Used by the MCP search tool. It looks at the first 80 characters of `column`,
    strips leading space, tab, newline and carriage return, lowercases, and checks
    the markers above. The hook keeps its own copy of the check in
    scripts/retrieval.py, which strips spaces only; that difference is documented
    in GO_DEEPER.md and is not changed here."""
    assert all("'" not in p and "%" not in p and "_" not in p for p in MACHINE_GENERATED_PREFIXES)
    trim = f"LTRIM(SUBSTR({column}, 1, 80), ' ' || char(9) || char(10) || char(13))"
    return "AND NOT (" + " OR ".join(f"LOWER({trim}) LIKE '{p}%'" for p in MACHINE_GENERATED_PREFIXES) + ")"


def resolve_db_path(config):
    """Return the expanded, absolute path to the claude_remembers SQLite DB."""
    storage = config["storage"]
    if storage.get("mode") == "synced":
        db_path = storage["local_db_path"]
    else:
        db_path = os.path.join(storage["root_path"], "claude_remembers.db")
    return os.path.expanduser(db_path)


def resolve_project_path(cfg, project_entry):
    """Return the expanded, absolute path to a project's folder.

    `path` is the source of truth for where a project's CLAUDE.md and
    chat-files/ physically live - independent of `folder_name`, which
    keeps its separate role as the substring-match key for
    jsonl_project_mapping/detect_project. Falls back to
    root_path/folder_name for legacy entries that predate the `path`
    field, so every call site stays safe during a partial migration.

    (remember-server's MCP registration is NOT per-project - see
    ensure_mcp_user_registration() below, which registers it once at
    Claude Code's global "user" scope instead.)
    """
    path = project_entry.get("path")
    if path:
        return os.path.expanduser(path)
    root = os.path.expanduser(cfg["storage"]["root_path"])
    return os.path.join(root, project_entry["folder_name"])


def venv_python(root_path):
    """Path to the project's own Python (root/.venv), or None if not built.

    Hooks and the MCP server must run on this interpreter, not on a bare
    `python3` from PATH: Homebrew (and other package managers) re-point
    `python3` at a new version on upgrade, and the new one has none of our
    packages installed. A venv pins the packages to the project.
    """
    exe = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    path = os.path.join(root_path, ".venv", *exe.split("/"))
    return path if os.path.exists(path) else None


def ensure_venv(root_path):
    """Create root/.venv if missing and return its Python path.

    Idempotent. Built from whichever Python is running this script.
    """
    existing = venv_python(root_path)
    if existing:
        return existing
    subprocess.run([sys.executable, "-m", "venv", os.path.join(root_path, ".venv")],
                   check=True, capture_output=True, text=True, timeout=300)
    return venv_python(root_path)


def ensure_mcp_user_registration(root_path):
    """Ensure remember-server is registered at Claude Code's global
    "user" MCP scope in ~/.claude.json, so it's available in every
    directory with no per-project registration needed.

    Per-project ("local" scope) registration used to be how this worked,
    but that meant N duplicate registrations (one per project folder)
    all pointing at the same server - which Claude Code flags as a
    "conflicting scopes" diagnostic once a user-scope entry also exists.
    One user-scope entry replaces all of them. Idempotent: leaves an
    existing user-scope entry (including any custom env/args) untouched.

    Returns True if ~/.claude.json was modified.
    """
    claude_json_path = os.path.join(os.path.expanduser("~"), ".claude.json")
    if os.path.exists(claude_json_path):
        with open(claude_json_path) as f:
            claude_json = json.load(f)
    else:
        claude_json = {}

    mcp_servers = claude_json.setdefault("mcpServers", {})
    if "remember-server" in mcp_servers:
        return False

    server_path = os.path.join(root_path, "mcp", "server.py")
    python_path = venv_python(root_path) or sys.executable
    mcp_servers["remember-server"] = {
        "type": "stdio",
        "command": python_path,
        "args": ["-W", "ignore", server_path],
        "env": {"PYTHONWARNINGS": "ignore", "TORCH_CPP_LOG_LEVEL": "ERROR"},
    }
    with open(claude_json_path, "w") as f:
        json.dump(claude_json, f, indent=4)
    return True


def remove_mcp_local_registrations():
    """Strip any leftover per-project ("local" scope) remember-server
    registrations from ~/.claude.json's `projects` section.

    Pre-dates ensure_mcp_user_registration(): add-project.py and
    remember-setup.py used to write one of these per project folder,
    which conflicts with a user-scope entry once that also exists.
    Called opportunistically by scripts that touch MCP registration, so
    old installs self-heal without a separate migration step. Returns
    the list of project paths cleaned up.
    """
    claude_json_path = os.path.join(os.path.expanduser("~"), ".claude.json")
    if not os.path.exists(claude_json_path):
        return []

    with open(claude_json_path) as f:
        claude_json = json.load(f)

    cleaned = []
    for path, cfg in claude_json.get("projects", {}).items():
        if "remember-server" in cfg.get("mcpServers", {}):
            del cfg["mcpServers"]["remember-server"]
            cleaned.append(path)

    if cleaned:
        with open(claude_json_path, "w") as f:
            json.dump(claude_json, f, indent=4)

    return cleaned


def migrate_config_project_paths(cfg, config_path):
    """One-time, idempotent backfill: add `path` to any project entry
    that lacks it, defaulting to root_path/folder_name (today's
    behavior). Returns True if config.yaml was rewritten."""
    root = os.path.expanduser(cfg["storage"]["root_path"])
    changed = False
    for p in cfg.get("projects", []):
        if not p.get("path"):
            p["path"] = os.path.join(root, p["folder_name"])
            changed = True

    if changed:
        import yaml
        from datetime import datetime

        with open(config_path, "w") as f:
            f.write("# claude_remembers Configuration File\n")
            f.write("# Last updated by migrate_config_project_paths() on "
                     + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("# See config.yaml.example for documentation of all options.\n\n")
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return changed
