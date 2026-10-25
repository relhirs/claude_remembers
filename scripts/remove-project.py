#!/usr/bin/env python3
"""
remove-project.py - Remove a project from an existing claude_remembers installation.

Mirrors add-project.py in reverse: removes the config.yaml entry, the
jsonl_project_mapping entry, the project_registry row, and the MCP
registration in ~/.claude.json.

By default this does NOT touch anything already recorded in the database
(transcripts, sessions, decisions, facts) - it just stops tracking the
project going forward. Deleting that history is a separate, explicitly
confirmed step, since it's not reversible.

Usage:
    python3 scripts/remove-project.py
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def green(text):
    return f"\033[92m{text}\033[0m"

def red(text):
    return f"\033[91m{text}\033[0m"

def yellow(text):
    return f"\033[93m{text}\033[0m"

def bold(text):
    return f"\033[1m{text}\033[0m"

def ok(msg):
    print(f"  {green('OK')} {msg}")

def fail(msg):
    print(f"  {red('FAIL')} {msg}")

def warn(msg):
    print(f"  {yellow('WARN')} {msg}")

def info(msg):
    print(f"  {msg}")

def sep():
    print("  " + "-" * 60)

def ask(prompt, default=None):
    sep()
    if default:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return raw if raw else default
    return input(f"  {prompt}: ").strip()

def ask_yn(prompt, default="n"):
    sep()
    suffix = "[Y/n]" if default == "y" else "[y/N]"
    raw = input(f"  {prompt} {suffix}: ").strip().lower()
    if not raw:
        return default == "y"
    return raw in ("y", "yes")

# ---------------------------------------------------------------------------
# Load existing config
# ---------------------------------------------------------------------------

def load_config():
    """Load existing config.yaml. Returns (config_dict, config_path)."""
    script_dir = Path(__file__).resolve().parent.parent
    config_path = script_dir / "config.yaml"

    if not config_path.exists():
        fail(f"config.yaml not found at {config_path}")
        sys.exit(1)

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    return cfg, config_path


def resolve_db_path(cfg):
    if cfg["storage"]["mode"] == "synced":
        db_path = cfg["storage"]["local_db_path"]
    else:
        db_path = os.path.join(cfg["storage"]["root_path"], "claude_remembers.db")
    return os.path.expanduser(db_path)


HISTORY_TABLES = ["transcripts", "sys_sessions", "tool_results", "decisions", "facts"]


def count_history(conn, prefix):
    """Return {table: row_count} for every table that has rows under this project."""
    counts = {}
    for table in HISTORY_TABLES:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE project = ?", (prefix,)).fetchone()[0]
        except sqlite3.OperationalError:
            continue
        if n:
            counts[table] = n
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{bold('claude_remembers -- Remove Project')}")
    print("Stop tracking a project. Your conversation history is kept by default.\n")

    cfg, config_path = load_config()
    root = Path(cfg["storage"]["root_path"])

    projects = cfg.get("projects", [])
    if not projects:
        info("No projects configured.")
        return

    print(f"  Configured project(s):")
    for i, p in enumerate(projects, 1):
        print(f"    {i}. {p['prefix']:5s} {p['folder_name']:20s} {p.get('label', '')}")
    print()

    choice = ask("Which project to remove? (number or folder name)")
    if not choice:
        info("Cancelled.")
        return

    target = None
    if choice.isdigit() and 1 <= int(choice) <= len(projects):
        target = projects[int(choice) - 1]
    else:
        for p in projects:
            if p["folder_name"] == choice or p["prefix"] == choice:
                target = p
                break

    if not target:
        fail(f"No project matches '{choice}'.")
        return

    name = target["folder_name"]
    prefix = target["prefix"]
    project_dir = root / name

    db_path = resolve_db_path(cfg)
    history_counts = {}
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            history_counts = count_history(conn, prefix)
        finally:
            conn.close()

    print(f"\n  About to remove: {bold(name)} (prefix: {prefix})")
    print(f"\n  This will:")
    print(f"    - Remove it from config.yaml (projects list + jsonl_project_mapping)")
    print(f"    - Remove its project_registry row")
    print(f"    - Remove the remember-server MCP registration for {project_dir}")
    if history_counts:
        total = sum(history_counts.values())
        print(f"\n  {yellow('It will NOT delete')} the {total} existing database row(s) already")
        print(f"  recorded under prefix '{prefix}' ({', '.join(f'{t}: {c}' for t, c in history_counts.items())}).")
        print(f"  That history just becomes unreachable through this project's tracking --")
        print(f"  new activity under this folder/prefix won't be captured anymore, but")
        print(f"  nothing already said gets erased unless you opt into that below.")
    else:
        print(f"\n  No existing database history found under prefix '{prefix}'.")

    if not ask_yn("Proceed with removing project tracking?", "n"):
        info("Cancelled.")
        return

    # Step 1: config.yaml
    import yaml

    cfg["projects"] = [p for p in cfg["projects"] if p["prefix"] != prefix]
    if "jsonl_project_mapping" in cfg:
        cfg["jsonl_project_mapping"] = {
            k: v for k, v in cfg["jsonl_project_mapping"].items() if v != prefix
        }

    with open(config_path, "w") as f:
        f.write("# claude_remembers Configuration File\n")
        from datetime import datetime
        f.write("# Last updated by remove-project.py on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("# See config.yaml.example for documentation of all options.\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    ok("config.yaml updated (removed from projects + jsonl_project_mapping)")

    # Step 2: project_registry row
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DELETE FROM project_registry WHERE prefix = ?", (prefix,))
            conn.commit()
            ok("project_registry row removed")
        except Exception as e:
            warn(f"Could not update project_registry: {e}")
        finally:
            conn.close()

    # Step 3: ~/.claude.json MCP registration
    home = Path.home()
    claude_json_path = home / ".claude.json"
    project_path_str = str(project_dir)

    if claude_json_path.exists():
        with open(claude_json_path) as f:
            claude_json = json.load(f)
        projects_section = claude_json.get("projects", {})
        if project_path_str in projects_section:
            mcp_servers = projects_section[project_path_str].get("mcpServers", {})
            if "remember-server" in mcp_servers:
                del mcp_servers["remember-server"]
                projects_section[project_path_str]["mcpServers"] = mcp_servers
                claude_json["projects"] = projects_section
                with open(claude_json_path, "w") as f:
                    json.dump(claude_json, f, indent=4)
                ok("remember-server MCP registration removed")
            else:
                ok("remember-server was not registered for this project")
        else:
            ok("No ~/.claude.json entry for this project's folder")
    else:
        ok("No ~/.claude.json found - nothing to update")

    # Step 4 (opt-in, separately confirmed): delete historical DB rows
    if history_counts:
        print()
        warn("The next step is IRREVERSIBLE: it deletes the conversation history above")
        warn("from the database permanently. There is no undo once this runs.")
        if ask_yn(f"Also permanently delete all {sum(history_counts.values())} row(s) under prefix '{prefix}'?", "n"):
            confirm_text = ask(f"Type the prefix ('{prefix}') to confirm permanent deletion")
            if confirm_text == prefix:
                conn = sqlite3.connect(db_path)
                try:
                    for table in history_counts:
                        conn.execute(f"DELETE FROM {table} WHERE project = ?", (prefix,))
                    conn.commit()
                    ok(f"Deleted history for prefix '{prefix}' from: {', '.join(history_counts)}")
                finally:
                    conn.close()
            else:
                warn("Confirmation text didn't match - history NOT deleted.")
        else:
            ok("History kept (default)")

    # Step 5 (opt-in): delete the project folder
    if project_dir.exists():
        print()
        contents = list(project_dir.iterdir())
        if contents:
            info(f"  Project folder {project_dir} still contains {len(contents)} item(s)")
            for item in contents[:10]:
                info(f"    - {item.name}")
        if ask_yn(f"Delete the project folder {project_dir} too?", "n"):
            import shutil
            shutil.rmtree(project_dir)
            ok(f"Deleted {project_dir}")
        else:
            ok("Project folder kept")

    print(f"\n{'='*60}")
    print(f"  {green('Done.')} '{name}' is no longer tracked by claude_remembers.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
