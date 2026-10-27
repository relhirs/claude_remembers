#!/usr/bin/env python3
"""
add-project.py - Add a new project to an existing claude_remembers installation.

Reads your existing config.yaml and does only what's needed: config update,
folder creation, CLAUDE.md, DB registration, MCP server registration. Does
NOT re-run full setup (see remember-setup.py for first-run installation).

Usage:
    python3 scripts/add-project.py <folder-path> [--name NAME] [--prefix PFX] [--label "Label"]
        Non-interactive. Backs the /remember-add-project slash command.
        <folder-path> is the real folder this project's files live in - it's
        used as-is, nothing nested gets created. --name/--prefix/--label
        override the values otherwise derived from the folder's basename.

    python3 scripts/add-project.py
        No arguments: falls back to the original interactive prompts.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from config_utils import resolve_project_path, ensure_mcp_user_registration

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

def ask_yn(prompt, default="y"):
    sep()
    suffix = "[Y/n]" if default == "y" else "[y/N]"
    raw = input(f"  {prompt} {suffix}: ").strip().lower()
    if not raw:
        return default == "y"
    return raw in ("y", "yes")

def generate_prefix(name, existing_prefixes):
    """Generate a 2-3 char prefix from a project name, avoiding collisions."""
    parts = name.split("-")
    if len(parts) >= 2:
        candidate = "".join(p[0] for p in parts[:3])
    else:
        candidate = name[:3]

    if candidate not in existing_prefixes:
        return candidate

    for length in range(2, min(len(name.replace("-", "")), 5)):
        flat = name.replace("-", "")
        candidate = flat[:length]
        if candidate not in existing_prefixes:
            return candidate

    base = name.replace("-", "")[:2]
    for i in range(2, 100):
        candidate = f"{base}{i}"
        if candidate not in existing_prefixes:
            return candidate
    return base

# ---------------------------------------------------------------------------
# Load existing config
# ---------------------------------------------------------------------------

def load_config():
    """Load existing config.yaml. Returns (config_dict, config_path)."""
    script_dir = Path(__file__).resolve().parent.parent
    config_path = script_dir / "config.yaml"

    if not config_path.exists():
        fail(f"config.yaml not found at {config_path}")
        info("Run remember-setup.py first to create your initial installation.")
        sys.exit(1)

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    return cfg, config_path

# ---------------------------------------------------------------------------
# CLAUDE.md template
# ---------------------------------------------------------------------------

def generate_claude_md(project, cfg):
    """Generate CLAUDE.md content for a project folder."""
    root = cfg["storage"]["root_path"]
    if cfg["storage"]["mode"] == "synced":
        db_path = cfg["storage"]["local_db_path"]
    else:
        db_path = os.path.join(root, "claude_remembers.db")

    return textwrap.dedent(f"""\
        # {project['label']}

        PROJECT: {project['folder_name']}
        PREFIX: {project['prefix']}

        ## RULE #1 - USE THE MEMORY FIRST. ALWAYS.
        **You have persistent memory across sessions. USE IT before doing anything substantive.**
        - Before proposing a plan, starting a task, debugging, answering questions, or suggesting changes: **search the memory.**
        - Use `search_transcripts`, `search_semantic`, `get_recent_summaries`, `lookup_decision`, `lookup_fact`.
        - Files tell you WHAT exists. The memory tells you WHY -- strategic context, prior decisions, rejected approaches.
        - Not using the memory is the same as not having it.
        - If you have a web search MCP available, use it alongside the memory for external context.

        ## RULE #1B - NARRATE FINDINGS, DON'T WAIT TO BE ASKED
        Tool calls and their raw outputs are never captured by the memory - only your own text replies
        are. After running code or a computation that produces a real finding - a metric, a root
        cause, a decision about which approach to use - state it concisely in your reply as a matter
        of course. If you don't say it in text, it does not exist in memory.

        ## MEMORY CONNECTION
        MCP server "remember-server" provides persistent memory.
        Database: {db_path}

        ## SESSION START
        1. **Search the memory** -- query search_transcripts + get_recent_summaries for this project. This is step ONE.
        2. Call get_profile('{project['prefix']}') and get_project_state('{project['prefix']}') to load full
           working context - this merges your global preferences with any preferences set
           specifically for this project (project-specific values override global ones that
           share the same key). The session-start hook already force-injects a copy of this
           automatically every session.
        3. Hooks handle startup check and context injection automatically.

        ## TOOL ROUTING
        When you need information, use the right MCP tool:
        - "Who is [person]?" / "What do I know about..." -> lookup_fact
        - "What did we decide about..." -> lookup_decision
        - "Find conversations about..." -> search_transcripts
        - "What did we discuss about..." -> search_transcripts (or search_semantic for meaning-based)
        - "Show me session..." -> get_session
        - "Recent sessions" -> get_recent_sessions
        - "My preferences" / "My profile" -> get_profile('{project['prefix']}')
        - "Project status" -> get_project_state
        - "Memory status" / "How many sessions" -> get_status

        ## SEARCH PRIORITY
        1. Structured tables (decisions, facts) via MCP
        2. Raw transcripts via MCP search_transcripts()
        3. Semantic search via MCP search_semantic() (meaning-based)
        4. Project text files in this folder
        5. Tell user: "I don't have this information"

        ## SESSION START PROTOCOL
        When a session starts, you receive automatic context from the session-start hook
        (last session notes, NEXT_SESSION.md, recent summaries). Before responding to the user:
        1. Search the memory: search_transcripts + get_recent_summaries for project '{project['prefix']}'
        2. Review the injected context (last session notes, NEXT_SESSION.md if present)
        3. SHOW unfinished items and next-session notes to the user prominently (they must SEE and react)
        4. Output this checklist (every row must show DONE):

        | Start-Session Checklist              | Status   |
        |--------------------------------------|----------|
        | Memory searched                      | DONE     |
        | Last session notes reviewed          | DONE     |
        | Unfinished items SHOWN to user       | DONE     |
        | Next-session notes SHOWN to user     | DONE     |

        If any row cannot show DONE, stop and fix it before proceeding.

        ## SESSION END PROTOCOL
        scripts/summarize_session.py now runs automatically at session-end (generates notes + tags +
        lessons, with a safe fallback if it fails) - step 1 below is optional, your manual version
        will just get overwritten by the automatic one.
        When the user says "end session" (or similar), complete ALL steps:
        1. (Optional - automatic as of session-end) Write session notes + tags to the database:
           python3 {root}/scripts/write_session_notes.py --notes "<what was done, decisions, next steps>" --tags "<1-3 comma-separated tags>"
        2. Update project summary (if significant progress):
           python3 {root}/scripts/write_project_summary.py --prefix {project['prefix']} --summary "<current state>"
        3. Ask the user: "Anything you want Claude to know next session?"
        4. Write their answer (plus session summary) to NEXT_SESSION.md in this project folder
        5. Output this checklist (every row must show DONE):

        | End-Session Checklist                | Status   |
        |--------------------------------------|----------|
        | Session notes + tags written         | DONE     |
        | Project summary updated              | DONE     |
        | NEXT_SESSION.md written              | DONE     |

        If any row cannot show DONE, stop and fix it before proceeding.

        ## SLASH COMMANDS
        Type these in Claude Code for direct access:
        /remember-question, /remember-search, /remember-history, /remember-recap, /remember-decide,
        /remember-health, /remember-status, /remember-import, /remember-questionnaire, /remember-add-project,
        /remember-export

        ## FILE VERSIONING
        After creating or modifying any file, run:
        python3 {root}/scripts/copy_chat_file.py \\
          [filepath] --project {project['prefix']} --session $SESSION_ID
    """)

# ---------------------------------------------------------------------------
# Reusable project creation (shared with import_claude_ai.py's --new-project)
# ---------------------------------------------------------------------------

def create_project(cfg, config_path, name, prefix, label, path=None, quiet=False):
    """Create a new project: folder + CLAUDE.md + config.yaml entry +
    project_registry row + remember-server MCP registration.

    `path` is the project's physical folder - an existing folder elsewhere
    on disk, or a new one to create. Defaults to root_path/name (today's
    behavior) when omitted. CLAUDE.md/chat-files are added INTO that
    folder; an existing folder's other contents are never touched.

    Does NOT handle the "other MCP servers" offer from add-project.py's
    interactive flow - that's a UX nicety for the interactive script only,
    not essential to project creation.

    Returns the project dict {"folder_name", "prefix", "label", "path"}
    on success, raises ValueError if name/prefix collide with an existing
    project.
    """
    existing_names = {p["folder_name"] for p in cfg.get("projects", [])}
    existing_prefixes = {p["prefix"] for p in cfg.get("projects", [])}
    if name in existing_names:
        raise ValueError(f"Project folder '{name}' already exists.")
    if prefix in existing_prefixes:
        raise ValueError(f"Prefix '{prefix}' is already in use.")

    root = Path(cfg["storage"]["root_path"])
    if path is None:
        path = str(root / name)
    new_project = {"folder_name": name, "prefix": prefix, "label": label, "path": path}

    def say(msg):
        if not quiet:
            ok(msg)

    # Folder + chat-files/
    project_dir = Path(path)
    (project_dir / "chat-files").mkdir(parents=True, exist_ok=True)
    say(f"Created {project_dir}/")

    # CLAUDE.md
    claude_md_path = project_dir / "CLAUDE.md"
    if not claude_md_path.exists():
        with open(claude_md_path, "w") as f:
            f.write(generate_claude_md(new_project, cfg))
        say("CLAUDE.md written")

    # config.yaml
    import yaml

    cfg["projects"].append(new_project)
    if "jsonl_project_mapping" not in cfg:
        cfg["jsonl_project_mapping"] = {}
    cfg["jsonl_project_mapping"][name] = prefix

    with open(config_path, "w") as f:
        f.write("# claude_remembers Configuration File\n")
        f.write("# Last updated by add-project.py on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("# See config.yaml.example for documentation of all options.\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    say(f"config.yaml updated (added {name})")

    # project_registry
    if cfg["storage"]["mode"] == "synced":
        db_path = cfg["storage"]["local_db_path"]
    else:
        db_path = os.path.join(cfg["storage"]["root_path"], "claude_remembers.db")
    db_path = os.path.expanduser(db_path)

    if os.path.exists(db_path):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn = sqlite3.connect(db_path)
        try:
            # Self-healing: the `path` column may not exist yet if
            # startup_check.py's migration hasn't run in this DB.
            cols = {row[1] for row in conn.execute("PRAGMA table_info(project_registry)").fetchall()}
            if "path" not in cols:
                conn.execute("ALTER TABLE project_registry ADD COLUMN path TEXT")

            conn.execute(
                "INSERT OR IGNORE INTO project_registry (folder_name, prefix, label, status, health, registered_at, path) "
                "VALUES (?, ?, ?, 'active', 'green', ?, ?)",
                (name, prefix, label, now, path)
            )
            conn.execute(
                "UPDATE project_registry SET path = ? WHERE folder_name = ?",
                (path, name)
            )
            conn.commit()
            say(f"project_registry updated (prefix: {prefix})")
        finally:
            conn.close()

    # remember-server MCP registration - one global "user" scope entry
    # covers every project folder, so there's nothing per-project to add
    # here. This just confirms it's there (first-ever project creation on
    # a fresh install) without duplicating it if it already is.
    if ensure_mcp_user_registration(str(root)):
        say("remember-server MCP registered (user scope, applies to all projects)")

    return new_project


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(text):
    """Turn a folder basename into a valid project name: lowercase,
    spaces/underscores to hyphens, strip anything else invalid."""
    text = re.sub(r"[\s_]+", "-", text.strip().lower())
    text = re.sub(r"[^a-z0-9-]", "", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "project"


def main_noninteractive(args):
    """Non-interactive path: given a folder path, derive name/prefix/label
    and register the project with no prompts. Backs /remember-add-project."""
    cfg, config_path = load_config()

    existing_projects = cfg.get("projects", [])
    existing_names = {p["folder_name"] for p in existing_projects}
    existing_prefixes = {p["prefix"] for p in existing_projects}

    path = os.path.expanduser(args.path)
    if os.path.isfile(path):
        fail(f"'{path}' is a file, not a folder.")
        sys.exit(1)

    name = args.name or slugify(Path(path).name)
    if name in existing_names:
        fail(f"Project folder name '{name}' already exists. Pass --name to pick a different one.")
        sys.exit(1)

    prefix = args.prefix or generate_prefix(name, existing_prefixes)
    if prefix in existing_prefixes:
        fail(f"Prefix '{prefix}' is already in use. Pass --prefix to pick a different one.")
        sys.exit(1)

    label = args.label or name.replace("-", " ").title()

    real_path = os.path.realpath(path)
    existing_paths = {os.path.realpath(resolve_project_path(cfg, p)) for p in existing_projects}
    if real_path in existing_paths:
        fail(f"Another project is already registered at '{path}'.")
        sys.exit(1)

    print(f"\n{bold('claude_remembers -- Add Project')}")
    print(f"  Adding: {name} (prefix: {prefix}, label: {label})")
    print(f"  Folder: {path}\n")

    try:
        create_project(cfg, config_path, name, prefix, label, path=path)
    except ValueError as e:
        fail(str(e))
        sys.exit(1)

    print(f"\n{green('Done!')} Project '{name}' added to claude_remembers.")
    print(f"  cd {path} && claude   # the memory activates automatically\n")


def main_interactive():
    print(f"\n{bold('claude_remembers -- Add Project')}")
    print("Add a new project to your existing memory installation.\n")

    # Step 1: Load existing config
    cfg, config_path = load_config()
    root = Path(cfg["storage"]["root_path"])

    existing_projects = cfg.get("projects", [])
    existing_names = {p["folder_name"] for p in existing_projects}
    existing_prefixes = {p["prefix"] for p in existing_projects}

    print(f"  Found {len(existing_projects)} existing project(s):")
    for p in existing_projects:
        print(f"    {p['prefix']:5s} {p['folder_name']:20s} {p.get('label', '')}")
    print()

    # Step 2: Get new project details
    name_pattern = re.compile(r'^[a-z][a-z0-9-]*[a-z0-9]$|^[a-z]$')

    print("  A 'project' is one body of work (a real repo, a client, etc.). Whenever")
    print("  you later run 'claude' from inside a folder whose path contains the name")
    print("  you type below, that session is AUTOMATICALLY tagged with its prefix -")
    print("  it does not have to be this script's own folder, any real repo works.")
    print()
    print("  Pick a word from that real folder's name (e.g. 'checker' for")
    print("  '~/code/listing_checker'). Matching ignores case, spaces, hyphens,")
    print("  underscores, and any other symbol, so you don't need to match the real")
    print("  folder's exact punctuation. The prefix is effectively permanent once")
    print("  sessions are tagged with it, so pick something you'll keep.")
    print()

    while True:
        sep()
        name = input("  New project folder name: ").strip().lower()
        if not name:
            info("Cancelled.")
            return
        if not name_pattern.match(name):
            warn("Invalid. Use lowercase letters, numbers, hyphens. Must start with a letter.")
            continue
        if name in existing_names:
            warn(f"'{name}' already exists.")
            continue
        break

    prefix = generate_prefix(name, existing_prefixes)
    label = ask(f"Display name for '{name}'", name.replace("-", " ").title())
    shown_prefix = ask(f"Database prefix for '{name}' (2-4 letters, permanent)", prefix)
    if shown_prefix != prefix:
        if shown_prefix in existing_prefixes:
            warn(f"Prefix '{shown_prefix}' already in use. Using '{prefix}'.")
            shown_prefix = prefix
        prefix = shown_prefix

    existing_paths = {
        os.path.realpath(resolve_project_path(cfg, p)) for p in existing_projects
    }

    print()
    print("  Where do this project's files actually live? If it's an existing real")
    print("  folder somewhere else on disk, point at it directly - CLAUDE.md and a")
    print("  chat-files/ folder get added into it, nothing nested gets created, and")
    print("  anything already in that folder is left untouched.")
    print()
    default_path = str(root / name)
    while True:
        raw_path = ask(f"Project folder path for '{name}'", default_path)
        path = os.path.expanduser(raw_path)
        real_path = os.path.realpath(path)

        if os.path.isfile(path):
            warn(f"'{path}' is a file, not a folder.")
            continue
        if real_path in existing_paths:
            warn(f"Another project is already registered at '{path}'.")
            continue
        containing = [p for p in existing_paths
                      if real_path != p and (real_path.startswith(p + os.sep) or p.startswith(real_path + os.sep))]
        if containing:
            warn(f"'{path}' is nested inside/around an already-registered project path "
                 f"({containing[0]}). This is allowed but unusual - CLAUDE.md/chat-files "
                 f"will coexist with that project's.")
            if not ask_yn("Continue anyway?", "n"):
                continue
        break

    print(f"\n  Will add: {bold(name)} (prefix: {prefix}, label: {label})")
    print(f"  Folder: {path}")
    if not ask_yn("Proceed?"):
        info("Cancelled.")
        return

    # Steps 3-7: folder, CLAUDE.md, config.yaml, project_registry, MCP registration
    try:
        create_project(cfg, config_path, name, prefix, label, path=path)
    except ValueError as e:
        fail(str(e))
        return
    project_dir = Path(path)

    # Step 8: offer to copy over any OTHER (non-remember-server) MCP
    # servers already registered on the root project path. remember-server
    # itself needs nothing here - it's one global user-scope registration,
    # not per-project.
    home = Path.home()
    claude_json_path = home / ".claude.json"
    claude_json = {}
    if claude_json_path.exists():
        with open(claude_json_path) as f:
            claude_json = json.load(f)
    projects_section = claude_json.get("projects", {})
    project_path_str = str(project_dir)
    mcp_servers = projects_section.get(project_path_str, {}).get("mcpServers", {})

    root_str = str(root)
    other_mcps = {}
    if root_str in projects_section:
        root_mcp = projects_section[root_str].get("mcpServers", {})
        for mcp_name, mcp_config in root_mcp.items():
            if mcp_name != "remember-server":
                other_mcps[mcp_name] = mcp_config

    if other_mcps:
        print(f"\n  Found {len(other_mcps)} other MCP server(s) on your root project:")
        for mcp_name in other_mcps:
            print(f"    - {mcp_name}")

        if ask_yn(f"Register these for {name} too?"):
            for mcp_name, mcp_config in other_mcps.items():
                if mcp_name not in mcp_servers:
                    mcp_servers[mcp_name] = mcp_config
                    ok(f"{mcp_name} registered")
                else:
                    ok(f"{mcp_name} already registered")
            projects_section.setdefault(project_path_str, {
                "allowedTools": [], "mcpContextUris": [], "mcpServers": {},
                "enabledMcpjsonServers": [], "disabledMcpjsonServers": [],
                "hasTrustDialogAccepted": False,
            })
            projects_section[project_path_str]["mcpServers"] = mcp_servers
            claude_json["projects"] = projects_section
            with open(claude_json_path, "w") as f:
                json.dump(claude_json, f, indent=4)
            ok("~/.claude.json updated")

    # Done
    print(f"\n{'='*60}")
    print(f"  {green('Done!')} Project '{name}' added to claude_remembers.")
    print(f"{'='*60}")
    print(f"\n  To use the memory from this project:")
    print(f"    cd {project_dir}")
    print(f"    claude")
    print(f"\n  The memory will work automatically -- hooks capture data,")
    print(f"  MCP tools let Claude query your memory, CLAUDE.md provides")
    print(f"  instructions. Just start working.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new project to claude_remembers")
    parser.add_argument("path", nargs="?", help="Folder path the project's files live in")
    parser.add_argument("--name", help="Project name (default: derived from folder basename)")
    parser.add_argument("--prefix", help="Database prefix, 2-4 letters (default: auto-generated)")
    parser.add_argument("--label", help="Display name (default: Title Cased name)")
    cli_args = parser.parse_args()

    if cli_args.path:
        main_noninteractive(cli_args)
    else:
        main_interactive()
