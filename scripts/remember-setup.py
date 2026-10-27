#!/usr/bin/env python3
"""
remember-setup.py - First-run installer for claude_remembers.

Creates database, config, directories, hooks, MCP registration,
and seeds initial profile data. Idempotent - safe to re-run.

Usage:
    python3 scripts/remember-setup.py
    python3 scripts/remember-setup.py --questionnaire  (import filled questionnaire)
"""

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from config_utils import ensure_mcp_user_registration, remove_mcp_local_registrations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_PYTHON = (3, 10)
HOOK_EVENTS = ["SessionStart", "UserPromptSubmit", "Stop", "SessionEnd", "PreCompact", "PostCompact"]
HOOK_SCRIPTS = ["session-start.py", "user-prompt-submit.py", "stop.py", "session-end.py", "pre-compact.py", "post-compact.py"]

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

def phase_header(num, total, title):
    print(f"\n{'='*60}")
    print(f"  [{num}/{total}] {title}")
    print(f"{'='*60}")

def ok(msg):
    print(f"  {green('✓')} {msg}")

def fail(msg):
    print(f"  {red('✗')} {msg}")

def warn(msg):
    print(f"  {yellow('!')} {msg}")

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
    flat_name = re.sub(r"[-_ ]+", "", name).lower()
    parts = [p for p in re.split(r"[-_ ]+", name.lower()) if p]
    # Try first letters of each word (2-3 chars)
    if len(parts) >= 2:
        candidate = "".join(p[0] for p in parts[:3])
    else:
        candidate = flat_name[:3]

    if candidate not in existing_prefixes:
        return candidate

    # Collision - try adding more letters
    for length in range(2, min(len(flat_name), 5)):
        candidate = flat_name[:length]
        if candidate not in existing_prefixes:
            return candidate

    # Last resort - append digit
    base = flat_name[:2]
    for i in range(2, 100):
        candidate = f"{base}{i}"
        if candidate not in existing_prefixes:
            return candidate
    return base

# ---------------------------------------------------------------------------
# Phase 1: Pre-flight
# ---------------------------------------------------------------------------

def phase_preflight():
    phase_header(1, 9, "PRELIMINARY CHECKS")
    errors = []

    # Python version
    v = sys.version_info
    if v >= MIN_PYTHON:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} - need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+")
        errors.append("Python version too old")

    # Claude Code
    claude_path = shutil.which("claude")
    if claude_path:
        try:
            result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10)
            version = result.stdout.strip() or result.stderr.strip()
            ok(f"Claude Code found: {version[:60]}")
        except Exception:
            ok(f"Claude Code found at {claude_path}")
    else:
        fail("Claude Code not installed")
        info("  Install: https://code.claude.com/docs/en/overview")
        errors.append("Claude Code not found")

    # pip
    pip_cmd = None
    if shutil.which("pip3"):
        pip_cmd = ["pip3"]
        ok("pip3 available")
    else:
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"],
                           capture_output=True, timeout=10, check=True)
            pip_cmd = [sys.executable, "-m", "pip"]
            ok("pip available (via python3 -m pip)")
        except Exception:
            fail("pip not found")
            errors.append("pip not available")

    if errors:
        print(f"\n  {red('HARD STOP')}: Fix the above errors before continuing.")
        sys.exit(1)

    return pip_cmd

# ---------------------------------------------------------------------------
# Phase 2: Dependencies
# ---------------------------------------------------------------------------

def phase_dependencies(pip_cmd):
    phase_header(2, 9, "DEPENDENCIES")

    requirements_path = Path(__file__).resolve().parent.parent / "requirements.txt"
    if not requirements_path.exists():
        fail(f"requirements.txt not found at {requirements_path}")
        sys.exit(1)

    info("Installing from requirements.txt (includes sentence-transformers + torch")
    info("for semantic search - a 500MB+ download on first run)...")
    try:
        subprocess.run(
            pip_cmd + ["install", "--quiet", "-r", str(requirements_path)],
            capture_output=True, text=True, timeout=600, check=True
        )
        ok("All dependencies installed")
    except subprocess.CalledProcessError as e:
        fail("Dependency install failed")
        print(e.stderr[-2000:] if e.stderr else str(e))
        print(f"\n  {red('ERROR')}: Try manually: pip3 install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        fail(f"Dependency install failed: {e}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 3: Interactive Projects & Storage
# ---------------------------------------------------------------------------

def phase_projects():
    phase_header(3, 9, "PROJECT SETUP")

    # --- Storage mode ---
    print("\n  Storage mode - where your memory lives:")
    print("    synced - Root folder inside Dropbox/OneDrive/etc. so the SAME memory")
    print("             follows you to other computers. The database file itself is")
    print("             kept OUTSIDE the synced folder (sync + SQLite don't mix).")
    print("    local  - Everything in one plain folder, this computer only.")
    print("  Pick 'local' unless you actively use Claude Code on more than one machine.")
    storage_mode = ask("Storage mode", "synced").lower()
    while storage_mode not in ("synced", "local"):
        storage_mode = ask("Enter 'synced' or 'local'", "synced").lower()

    # --- Detect root path ---
    script_dir = Path(__file__).resolve().parent.parent
    home = Path.home()

    if storage_mode == "synced":
        # Try to detect if we're already inside the repo
        if (script_dir / "mcp" / "server.py").exists():
            default_root = str(script_dir)
        else:
            # Common sync folder locations
            for candidate in [
                home / "Dropbox" / "claude_remembers",
                home / "OneDrive" / "claude_remembers",
                home / "Google Drive" / "claude_remembers",
            ]:
                if candidate.parent.exists():
                    default_root = str(candidate)
                    break
            else:
                default_root = str(home / "Dropbox" / "claude_remembers")

        print("\n  This is the one folder everything lives in: the database, project")
        print("  folders, backups, and logs. Defaults to where you're running this from.")
        root_path = ask("Root folder path", default_root)
        default_db = str(home / "claude_remembers-local" / "claude_remembers.db")
        db_path = ask("Local database path (NOT in sync folder)", default_db)
    else:
        default_root = str(home / "claude_remembers")
        print("\n  This is the one folder everything lives in: the database, project")
        print("  folders, backups, and logs. Defaults to where you're running this from.")
        root_path = ask("Root folder path", default_root)
        root_path = os.path.expanduser(root_path)
        db_path = os.path.join(root_path, "claude_remembers.db")
        info(f"Database will be at: {db_path}")

    root_path = os.path.expanduser(root_path)
    db_path = os.path.expanduser(db_path)

    # --- Projects ---
    print("\n  PROJECTS - what this means:")
    print("  A 'project' is one body of work (a real repo, a client, or just 'general'")
    print("  for everything else). Whenever you later run 'claude' from inside a folder")
    print("  whose path contains the project name you type below, that session is")
    print("  AUTOMATICALLY tagged with its prefix - no manual categorizing. It does NOT")
    print("  have to be a folder created by this script; it can be any real repo")
    print("  anywhere on disk. No match found anywhere? Defaults to 'oth'.")
    print()
    print("  HOW TO PICK THE NAME: type it exactly as the real folder's name appears -")
    print("  e.g. for a repo at '~/code/Listing Checker', type 'Listing Checker' with")
    print("  the same case and spacing. This is an EXACT, case-sensitive match against")
    print("  the folder name, so 'Listing Checker' will NOT match a folder named")
    print("  'listing-checker' or 'listing_checker' - type it precisely.")
    print()
    print("  After the name, you'll also be asked for a DISPLAY NAME and a PREFIX:")
    print("    - Display name: a human-readable title, shown as the heading in that")
    print("      project's CLAUDE.md and in summaries/pickers. Purely cosmetic - it")
    print("      plays no part in matching your folder. Defaults to the name you just")
    print("      typed with hyphens/underscores turned into spaces and Title Cased;")
    print("      press Enter to accept that, or type your own.")
    print("    - Prefix: the short permanent code (2-4 letters) used internally to")
    print("      tag sessions and facts for this project in the database.")
    print()
    print("  That prefix then lets Claude scope its own memory lookups to just this")
    print("  project automatically - wired into its CLAUDE.md, nothing you manage.")
    print("  It's effectively permanent once sessions are tagged with it, so pick a")
    print("  word you'll keep using for that folder.")
    print()
    print("  You'll add projects ONE AT A TIME below (type a name, Enter, repeat) -")
    print("  there's no comma-separated shortcut for adding several at once.")
    print()
    print("  Not sure? Just add one project called 'general' and move on - sessions run")
    print("  from inside this script's own 'general' folder will tag themselves 'gen'")
    print("  automatically. Add more anytime by rerunning this script or /remember-add-project.")
    print()
    print("  Type it exactly as the real folder appears (any case, spaces allowed).")
    print("  Just avoid slashes ('/' or '\\\\') and leading/trailing spaces.")
    print("  Press Enter with no name when done. At least 1 project is required.")
    print()

    projects = []
    prefixes_used = set()
    pattern = re.compile(r'^[^/\\]+$')

    while True:
        default_hint = "general" if not projects else ""
        prompt = "Project name" + (f" (default: {default_hint})" if default_hint else " (Enter to finish)")
        sep()
        name = input(f"  {prompt}: ").strip()

        if not name:
            if not projects:
                name = "general"
                info("Using default: general")
            else:
                break

        if not pattern.match(name):
            warn("Invalid name. Avoid slashes.")
            continue

        if name in [p["folder_name"] for p in projects]:
            warn(f"'{name}' already added.")
            continue

        prefix = generate_prefix(name, prefixes_used)
        label = ask(f"  Display name for '{name}'", name.replace("-", " ").title())
        shown_prefix = ask(f"  Database prefix for '{name}' (2-4 letters, permanent)", prefix)
        if shown_prefix != prefix:
            if shown_prefix in prefixes_used:
                warn(f"Prefix '{shown_prefix}' already in use. Using '{prefix}'.")
                shown_prefix = prefix
            prefix = shown_prefix

        default_path = str(Path(root_path) / name)
        info("Where do this project's files actually live? Point at an existing real")
        info("folder if you have one - CLAUDE.md/chat-files get added into it, nothing")
        info("nested gets created, and anything already there is left untouched.")
        existing_paths = {os.path.realpath(os.path.expanduser(p.get("path", str(Path(root_path) / p["folder_name"]))))
                           for p in projects}
        while True:
            raw_path = ask(f"  Project folder path for '{name}'", default_path)
            path = os.path.expanduser(raw_path)
            real_path = os.path.realpath(path)
            if os.path.isfile(path):
                warn(f"'{path}' is a file, not a folder.")
                continue
            if real_path in existing_paths:
                warn(f"Another project is already registered at '{path}'.")
                continue
            break

        projects.append({
            "folder_name": name,
            "prefix": prefix,
            "label": label,
            "path": path
        })
        prefixes_used.add(prefix)
        ok(f"Added: {name} (prefix: {prefix}, path: {path})")

    # Auto-add "other" if not present
    if "other" not in [p["folder_name"] for p in projects]:
        oth_prefix = generate_prefix("other", prefixes_used)
        projects.append({
            "folder_name": "other",
            "prefix": oth_prefix,
            "label": "Uncategorized",
            "path": str(Path(root_path) / "other")
        })
        prefixes_used.add(oth_prefix)
        ok(f"Auto-added: other (prefix: {oth_prefix}) - catch-all for unmatched sessions")

    # --- Quick identity questions ---
    print("\n  OPTIONAL - lets Claude recall who you are automatically every session.")
    print("  Skip anything with Enter; fill it in anytime later via /remember-questionnaire.\n")
    identity = {}
    name_answer = ask("Your name")
    if name_answer:
        identity["full_name"] = name_answer
    location_answer = ask("Your location (city, state/country)")
    if location_answer:
        identity["location"] = location_answer
    role_answer = ask("What you do (role/profession)")
    if role_answer:
        identity["role"] = role_answer

    # --- Import ~/.claude/CLAUDE.md preferences ---
    claude_md_path = home / ".claude" / "CLAUDE.md"
    claude_md_content = None
    if claude_md_path.exists():
        info(f"\nFound your global CLAUDE.md at {claude_md_path}")
        info("Importing copies that text into the memory database too, so it's")
        info("searchable alongside everything else. Your file itself is untouched.")
        if ask_yn("Import its contents as remembered preferences?"):
            claude_md_content = claude_md_path.read_text(encoding="utf-8")
            ok("Will import CLAUDE.md preferences")

    # --- Confirm ---
    print(f"\n  {bold('Summary:')}")
    print(f"    Storage: {storage_mode}")
    print(f"    Root:    {root_path}")
    print(f"    DB:      {db_path}")
    print(f"    Projects:")
    for p in projects:
        print(f"      {p['folder_name']} ({p['prefix']}) - {p['label']} -> {p['path']}")
    if identity:
        print(f"    Identity: {', '.join(f'{k}={v}' for k, v in identity.items())}")
    print()

    if not ask_yn("Proceed?"):
        print("  Aborted.")
        sys.exit(0)

    return {
        "storage_mode": storage_mode,
        "root_path": root_path,
        "db_path": db_path,
        "projects": projects,
        "identity": identity,
        "claude_md_content": claude_md_content,
    }

# ---------------------------------------------------------------------------
# Phase 4: Directories
# ---------------------------------------------------------------------------

def phase_directories(cfg):
    phase_header(4, 9, "DIRECTORIES")
    root = Path(cfg["root_path"])
    db_dir = Path(cfg["db_path"]).parent

    dirs_to_create = [
        root,
        root / "scripts",
        root / "hooks",
        root / "mcp",
        root / "logs",
        root / "db-backup",
        root / "imports",
        root / "imports" / "completed",
        root / "verification",
        db_dir,
    ]

    # Project directories - p["path"] may point anywhere on disk, not
    # necessarily nested under root (see config_utils.resolve_project_path)
    for p in cfg["projects"]:
        project_dir = Path(p["path"])
        dirs_to_create.append(project_dir)
        dirs_to_create.append(project_dir / "chat-files")

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    ok(f"Created {len(dirs_to_create)} directories")

    # .gitkeep in empty dirs
    for name in ["db-backup", "logs"]:
        gitkeep = root / name / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    completed_gitkeep = root / "imports" / "completed" / ".gitkeep"
    if not completed_gitkeep.exists():
        completed_gitkeep.touch()

    ok(".gitkeep files in place")

# ---------------------------------------------------------------------------
# Phase 5: Database
# ---------------------------------------------------------------------------

DDL_STATEMENTS = [
    # --- System tables ---
    """CREATE TABLE IF NOT EXISTS sys_sessions (
        session_id       TEXT PRIMARY KEY,
        project          TEXT,
        started_at       TEXT,
        ended_at         TEXT,
        cwd              TEXT,
        claude_version   TEXT,
        model            TEXT,
        source           TEXT,
        message_count    INTEGER DEFAULT 0,
        created_at       TEXT,
        quality_score    INTEGER DEFAULT NULL,
        quality_tags     TEXT DEFAULT NULL,
        notes            TEXT,
        tags             TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS sys_ingest_log (
        file_path        TEXT PRIMARY KEY,
        file_size        INTEGER,
        file_type        TEXT,
        records_imported INTEGER,
        ingested_at      TEXT
    )""",
    # sys_session_summaries REMOVED - sys_sessions.notes is the single source of truth
    """CREATE TABLE IF NOT EXISTS project_registry (
        folder_name      TEXT PRIMARY KEY,
        prefix           TEXT UNIQUE,
        label            TEXT,
        registered_at    TEXT,
        summary          TEXT,
        summary_updated_at TEXT,
        status           TEXT DEFAULT 'active',
        health           TEXT DEFAULT 'green',
        path             TEXT
    )""",

    # --- Transcript tables ---
    """CREATE TABLE IF NOT EXISTS transcripts (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id       TEXT,
        project          TEXT,
        uuid             TEXT UNIQUE,
        parent_uuid      TEXT,
        type             TEXT,
        subtype          TEXT,
        role             TEXT,
        content          TEXT,
        model            TEXT,
        timestamp        TEXT,
        token_input      INTEGER,
        token_output     INTEGER,
        stop_reason      TEXT,
        is_subagent      INTEGER DEFAULT 0,
        source_file      TEXT,
        raw_json         TEXT,
        created_at       TEXT,
        source           TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS tool_results (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id       TEXT,
        project          TEXT,
        tool_use_id      TEXT,
        content          TEXT,
        source_file      TEXT,
        created_at       TEXT
    )""",

    # --- Memory tables ---
    """CREATE TABLE IF NOT EXISTS remember_facts (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        category         TEXT,
        key              TEXT,
        value            TEXT,
        project          TEXT DEFAULT NULL,
        source           TEXT,
        source_session   TEXT,
        confidence       TEXT,
        created_at       TEXT,
        updated_at       TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS remember_preferences (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        category         TEXT,
        preference       TEXT,
        project          TEXT DEFAULT NULL,
        source           TEXT,
        created_at       TEXT,
        updated_at       TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS decisions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        project          TEXT,
        decision_number  INTEGER,
        session_id       TEXT,
        description      TEXT,
        rationale        TEXT,
        created_at       TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS facts (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        project          TEXT,
        category         TEXT,
        key              TEXT,
        value            TEXT,
        source           TEXT,
        session_id       TEXT,
        created_at       TEXT,
        updated_at       TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS transcript_embeddings (
        transcript_id    INTEGER PRIMARY KEY,
        embedding        BLOB NOT NULL,
        model            TEXT DEFAULT 'all-MiniLM-L6-v2',
        created_at       TEXT
    )""",
]

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_transcripts_project ON transcripts(project)",
    "CREATE INDEX IF NOT EXISTS idx_transcripts_timestamp ON transcripts(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_transcripts_type ON transcripts(type)",
    "CREATE INDEX IF NOT EXISTS idx_transcripts_uuid ON transcripts(uuid)",
]

DDL_FTS = """CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
    content,
    content=transcripts,
    content_rowid=id
)"""

DDL_FTS_VOCAB = """CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts_vocab \
USING fts5vocab(transcripts_fts, row)"""

DDL_TRIGGERS = [
    """CREATE TRIGGER IF NOT EXISTS transcripts_ai AFTER INSERT ON transcripts BEGIN
        INSERT INTO transcripts_fts(rowid, content) VALUES (new.id, new.content);
    END""",
    """CREATE TRIGGER IF NOT EXISTS transcripts_au AFTER UPDATE ON transcripts BEGIN
        INSERT INTO transcripts_fts(transcripts_fts, rowid, content) VALUES ('delete', old.id, old.content);
        INSERT INTO transcripts_fts(rowid, content) VALUES (new.id, new.content);
    END""",
    """CREATE TRIGGER IF NOT EXISTS transcripts_ad AFTER DELETE ON transcripts BEGIN
        INSERT INTO transcripts_fts(transcripts_fts, rowid, content) VALUES ('delete', old.id, old.content);
    END""",
]


def phase_database(cfg):
    phase_header(5, 9, "DATABASE")
    db_path = cfg["db_path"]
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tables
    for ddl in DDL_STATEMENTS:
        cursor.execute(ddl)
    ok(f"{len(DDL_STATEMENTS)} tables verified")

    # Indexes
    for ddl in DDL_INDEXES:
        cursor.execute(ddl)
    ok(f"{len(DDL_INDEXES)} indexes verified")

    # FTS5
    try:
        cursor.execute(DDL_FTS)
        ok("FTS5 virtual table verified")
    except sqlite3.OperationalError as e:
        if "already exists" in str(e):
            ok("FTS5 virtual table already exists")
        else:
            fail(f"FTS5 creation failed: {e}")

    # FTS5 vocab table (for fuzzy search)
    try:
        cursor.execute(DDL_FTS_VOCAB)
        ok("FTS5 vocab table verified")
    except sqlite3.OperationalError as e:
        if "already exists" in str(e):
            ok("FTS5 vocab table already exists")
        else:
            fail(f"FTS5 vocab creation failed: {e}")

    # Triggers
    for ddl in DDL_TRIGGERS:
        try:
            cursor.execute(ddl)
        except sqlite3.OperationalError as e:
            if "already exists" in str(e):
                pass
            else:
                fail(f"Trigger failed: {e}")
    ok(f"{len(DDL_TRIGGERS)} triggers verified")

    # Seed project_registry
    for p in cfg["projects"]:
        cursor.execute(
            "INSERT OR IGNORE INTO project_registry (folder_name, prefix, label, registered_at, path) VALUES (?, ?, ?, ?, ?)",
            (p["folder_name"], p["prefix"], p["label"], now, p["path"])
        )
    conn.commit()
    ok(f"project_registry seeded ({len(cfg['projects'])} projects)")

    # Seed identity from quick questions
    identity = cfg.get("identity", {})
    if identity:
        for key, value in identity.items():
            cursor.execute(
                "INSERT OR IGNORE INTO remember_facts (category, key, value, source, confidence, created_at, updated_at) "
                "SELECT ?, ?, ?, 'remember-setup', 'confirmed', ?, ? "
                "WHERE NOT EXISTS (SELECT 1 FROM remember_facts WHERE category='identity' AND key=?)",
                ("identity", key, value, now, now, key)
            )
        conn.commit()
        ok(f"Identity seeded ({len(identity)} facts)")

    # Import CLAUDE.md preferences
    claude_md = cfg.get("claude_md_content")
    if claude_md:
        lines = [l.strip() for l in claude_md.splitlines() if l.strip() and not l.strip().startswith("#")]
        count = 0
        for line in lines:
            if len(line) > 10:  # Skip very short lines
                cursor.execute(
                    "INSERT INTO remember_preferences (category, preference, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    ("imported-claude-md", line, "remember-setup (~/.claude/CLAUDE.md)", now, now)
                )
                count += 1
        conn.commit()
        ok(f"Imported {count} preferences from CLAUDE.md")

    conn.close()
    ok(f"Database ready: {db_path}")

# ---------------------------------------------------------------------------
# Phase 6: Config + CLAUDE.md
# ---------------------------------------------------------------------------

def phase_config(cfg):
    phase_header(6, 9, "CONFIG & CLAUDE.MD FILES")
    root = Path(cfg["root_path"])

    # --- Generate config.yaml ---
    import yaml

    projects_list = []
    for p in cfg["projects"]:
        projects_list.append({
            "folder_name": p["folder_name"],
            "prefix": p["prefix"],
            "label": p["label"],
            "path": p["path"],
        })

    # Build jsonl_project_mapping from projects
    jsonl_mapping = {}
    for p in cfg["projects"]:
        jsonl_mapping[p["folder_name"]] = p["prefix"]

    config_data = {
        "storage": {
            "mode": cfg["storage_mode"],
            "root_path": cfg["root_path"],
        },
        "database": {
            "write_every_response": True,
            "read_priority": [
                "structured_tables",
                "raw_transcripts",
                "project_files",
                "tell_user_not_found",
            ],
        },
        "backup": {
            "max_copies": 2,
            "verify_after_copy": True,
        },
        "projects": projects_list,
        "remember": {
            "prefix": "remember",
            "categories": [
                "identity", "family", "professional", "working-style",
                "technical-setup", "goals", "preferences", "lessons-learned",
                "contacts", "health",
            ],
        },
        "jsonl": {
            "source_paths": [
                os.path.join(str(Path.home()), ".claude", "projects"),
            ],
            "ingest_subagents": True,
            "ingest_tool_results": True,
        },
        "jsonl_project_mapping": jsonl_mapping,
        "semantic_search": {
            "enabled": True,
            "model": "all-MiniLM-L6-v2",
        },
        "file_versioning": {
            "enabled": True,
            "folder_pattern": "{date}_{time}_{session_id_short}",
            "date_format": "YYYY-MM-DD",
            "time_format": "HHMMSS",
            "session_id_length": 8,
        },
        "logging": {
            "enabled": True,
            "level": "info",
        },
        "startup": {
            "run_on_session_start": True,
            "blocking": True,
            "verify_folders": True,
        },
        "scripts": {
            "commands": {
                "remember-check": "startup_check.py",
                "remember-sync": "remember_sync.py",
                "remember-status": "status.py",
            },
        },
        "meta": {
            "project_name": "claude_remembers",
            "version": "0.1.0",
            "license": "MIT",
            "github": "https://github.com/yourusername/claude_remembers",
        },
    }

    if cfg["storage_mode"] == "synced":
        config_data["storage"]["local_db_path"] = cfg["db_path"]

    config_path = root / "config.yaml"
    write_config = True
    if config_path.exists():
        warn(f"config.yaml already exists at {config_path}")
        write_config = ask_yn("Overwrite?", "n")

    if write_config:
        with open(config_path, "w") as f:
            f.write("# claude_remembers Configuration File\n")
            f.write("# Generated by remember-setup.py on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("# See config.yaml.example for documentation of all options.\n\n")
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        ok(f"config.yaml written")
    else:
        ok("config.yaml skipped (kept existing)")

    # --- Generate config.yaml.example ---
    example_data = {
        "storage": {
            "mode": "synced",
            "root_path": "~/Dropbox/claude_remembers",
            "local_db_path": "~/claude_remembers-local/claude_remembers.db",
        },
        "database": config_data["database"],
        "backup": config_data["backup"],
        "projects": [
            {"folder_name": "my-project", "prefix": "mp", "label": "My Project"},
            {"folder_name": "general", "prefix": "gen", "label": "General Conversations"},
            {"folder_name": "other", "prefix": "oth", "label": "Uncategorized"},
        ],
        "remember": config_data["remember"],
        "jsonl": {
            "source_paths": ["~/.claude/projects"],
            "ingest_subagents": True,
            "ingest_tool_results": True,
        },
        "jsonl_project_mapping": {
            "my-project": "mp",
            "general": "gen",
        },
        "semantic_search": config_data["semantic_search"],
        "file_versioning": config_data["file_versioning"],
        "logging": config_data["logging"],
        "startup": config_data["startup"],
        "scripts": config_data["scripts"],
        "meta": {
            "project_name": "claude_remembers",
            "version": "0.1.0",
            "license": "MIT",
            "github": "https://github.com/yourusername/claude_remembers",
            "author": "Your Name",
        },
    }

    example_path = root / "config.yaml.example"
    if example_path.exists():
        ok("config.yaml.example already exists (shipped with repo, skipping)")
    else:
        with open(example_path, "w") as f:
            f.write("# claude_remembers Configuration File - EXAMPLE\n")
            f.write("# Copy this to config.yaml and edit with your values.\n")
            f.write("# Or run: python3 scripts/remember-setup.py (generates config for you)\n\n")
            yaml.dump(example_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        ok("config.yaml.example written")

    # --- Generate CLAUDE.md per project ---
    for p in cfg["projects"]:
        project_dir = Path(p["path"])
        claude_md = project_dir / "CLAUDE.md"

        template = textwrap.dedent(f"""\
            # {p['label']}

            PROJECT: {p['folder_name']}
            PREFIX: {p['prefix']}

            ## RULE #1 - USE THE MEMORY FIRST. ALWAYS.
            **You have persistent memory across sessions. USE IT before doing anything substantive.**
            - Before proposing a plan, starting a task, debugging, answering questions, or suggesting changes: **search the memory.**
            - Use `search_transcripts`, `search_semantic`, `get_recent_summaries`, `lookup_decision`, `lookup_fact`.
            - Files tell you WHAT exists. The memory tells you WHY - strategic context, prior decisions, rejected approaches.
            - Not using the memory is the same as not having it.

            ## MEMORY CONNECTION
            MCP server "remember-server" provides persistent memory.
            Database: {cfg['db_path']}

            ## SESSION START
            1. **Search the memory** - query search_transcripts + get_recent_summaries for this project. This is step ONE.
            2. Call get_profile('{p['prefix']}') and get_project_state('{p['prefix']}') to load full
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
            - "My preferences" / "My profile" -> get_profile('{p['prefix']}')
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
            1. Search the memory: search_transcripts + get_recent_summaries for project '{p['prefix']}'
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
            When the user says "end session" (or similar), complete ALL steps:
            1. Write session notes + tags to the database:
               python3 {root}/scripts/write_session_notes.py --notes "<what was done, decisions, next steps>" --tags "<1-3 comma-separated tags>"
            2. Update project summary (if significant progress):
               python3 {root}/scripts/write_project_summary.py --prefix {p['prefix']} --summary "<current state>"
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
              [filepath] --project {p['prefix']} --session $SESSION_ID
        """)

        write_it = True
        if claude_md.exists():
            warn(f"CLAUDE.md already exists in {p['folder_name']}/")
            write_it = ask_yn(f"  Overwrite {p['folder_name']}/CLAUDE.md?", "n")

        if write_it:
            with open(claude_md, "w") as f:
                f.write(template)
            ok(f"CLAUDE.md written for {p['folder_name']}")
        else:
            ok(f"CLAUDE.md skipped for {p['folder_name']} (kept existing)")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Phase 7: Email Digests (optional)
# ---------------------------------------------------------------------------

def phase_email(cfg):
    phase_header(7, 9, "EMAIL DIGESTS (optional)")

    print("""
  EMAIL DIGESTS (fully optional - safe to skip):
  Schedules a job that emails YOU a summary pulled from your own memory database -
  what's pending, what happened recently - without opening Claude Code to ask.
  Daily weekday morning standup, or a Monday weekly recap. Nothing else in this
  tool depends on this; skipping it changes nothing else.

  Requires a Gmail account and a 16-character Gmail "App Password" (NOT your
  normal password - you'll be walked through generating one if you continue).
""")

    if not ask_yn("Set up email digests now? (can add later via config.yaml)", "n"):
        info("Skipped. You can set this up later in config.yaml.")
        cfg["email_enabled"] = False
        return

    # Get email address
    from_addr = ""
    while not from_addr:
        sep()
        from_addr = input("  Gmail address: ").strip()
        if "@" not in from_addr:
            warn("That doesn't look like an email address.")
            from_addr = ""

    to_addr = from_addr
    if ask_yn(f"  Send digests to the same address ({from_addr})?"):
        pass
    else:
        sep()
        to_addr = input("  Recipient email address: ").strip() or from_addr

    # Get app password
    print(f"""
  You need a Gmail App Password (NOT your regular password).
  To create one:
    1. Go to myaccount.google.com -> Security -> 2-Step Verification
    2. At the bottom, click "App passwords"
    3. Create one for "Mail" / "Other" -> name it "claude_remembers"
    4. Copy the 16-character password

  WHERE THIS GOES, SO YOU KNOW IT'S SAFE:
  - It's saved in plain text in your local config.yaml - nowhere else. It never
    leaves this machine, is never uploaded anywhere, and config.yaml is in
    .gitignore so it can't accidentally get committed or pushed to GitHub.
  - It's used for exactly one thing: logging into smtp.gmail.com over an
    encrypted (TLS) connection to send the digest email, same as any mail app.
  - An App Password is NOT your real Google password - it's a separate token
    scoped only to mail sending. If you ever want to shut this off, revoke it
    at myaccount.google.com -> Security -> App Passwords; your real login is
    never affected either way.
""")
    sep()
    app_password = input("  Gmail App Password: ").strip()
    if not app_password:
        warn("No password entered. Email will be disabled.")
        cfg["email_enabled"] = False
        return

    # Test connection
    info("Testing SMTP connection...")
    try:
        import smtplib
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_addr, app_password)
        ok("SMTP connection successful!")
    except Exception as e:
        warn(f"SMTP test failed: {e}")
        warn("Email config saved anyway - you can fix the password later in config.yaml.")

    # Write email config directly to config.yaml
    config_path = Path(cfg["root_path"]) / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}
        config_data["email"] = {
            "enabled": True,
            "from_address": from_addr,
            "to_address": to_addr,
            "gmail_app_password": app_password,
        }
        with open(config_path, "w") as f:
            f.write("# claude_remembers Configuration File\n")
            f.write("# Updated by remember-setup.py on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("# See config.yaml.example for documentation of all options.\n\n")
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        ok("Email config written to config.yaml")
    else:
        warn("config.yaml not found - run full setup first. Email config skipped.")
        return

    # Offer to set up cron
    print()
    root = cfg["root_path"]
    digest_script = os.path.join(root, "scripts", "remember_digest.py")
    log_dir = os.path.dirname(cfg["db_path"])
    daily_cron = f'0 8 * * 1-5 /usr/bin/python3 {digest_script} --daily >> {log_dir}/digest.log 2>&1'
    weekly_cron = f'0 8 * * 1 /usr/bin/python3 {digest_script} >> {log_dir}/digest.log 2>&1'

    info("The cron job is what actually triggers the digest on a schedule - without")
    info("it, the digest script exists but nothing runs it automatically.")

    if sys.platform == "win32":
        warn("cron isn't available on Windows - set this up with Task Scheduler instead:")
        info(f'  Program:   "{sys.executable}"')
        info(f'  Arguments: "{digest_script}" --daily')
        info("  Trigger:   Daily, weekdays, 8:00 AM (add a second task without --daily, weekly on Monday)")
        info("  (Task Scheduler app -> Create Basic Task -> point it at the above)")
        return

    if ask_yn("  Set up daily standup cron (weekdays 8am)?", "y"):
        try:
            import subprocess
            existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
            if "remember_digest" not in existing and "--daily" not in existing:
                new_cron = existing.rstrip("\n") + "\n" + daily_cron + "\n"
                subprocess.run(["crontab", "-"], input=new_cron, text=True)
                ok("Daily cron installed (weekdays 8am)")
            else:
                ok("Daily cron already exists, skipped")
        except Exception as e:
            warn(f"Could not set cron: {e}")
            info(f"  Add manually: {daily_cron}")

    if ask_yn("  Set up weekly digest cron (Monday 8am)?", "y"):
        try:
            existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
            if "remember_digest" not in existing or "--daily" in existing:
                # Check if weekly already there
                if "remember_digest.py >>" in existing and "--daily" not in existing.split("remember_digest.py >>")[0].split("\n")[-1]:
                    ok("Weekly cron already exists, skipped")
                else:
                    new_cron = existing.rstrip("\n") + "\n" + weekly_cron + "\n"
                    subprocess.run(["crontab", "-"], input=new_cron, text=True)
                    ok("Weekly cron installed (Monday 8am)")
        except Exception as e:
            warn(f"Could not set cron: {e}")
            info(f"  Add manually: {weekly_cron}")

    ok("Email digests configured")


# Phase 8: Registration + Ingestion
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def phase_registration(cfg):
    phase_header(8, 9, "REGISTRATION & INGESTION")
    root = Path(cfg["root_path"])
    home = Path.home()

    # --- Register hooks in ~/.claude/settings.json ---
    settings_path = home / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
    else:
        settings = {}

    hooks = settings.get("hooks", {})
    hooks_added = 0

    for event, script in zip(HOOK_EVENTS, HOOK_SCRIPTS):
        hook_cmd = f'"{sys.executable}" "{root / "hooks" / script}"'
        existing = hooks.get(event, [])

        # Check if already registered with same command
        already = False
        for entry in existing:
            for h in entry.get("hooks", []):
                if h.get("command") == hook_cmd:
                    already = True
                    break

        if not already:
            new_entry = {
                "matcher": "",
                "hooks": [{"type": "command", "command": hook_cmd}]
            }
            if existing:
                existing.append(new_entry)
            else:
                existing = [new_entry]
            hooks[event] = existing
            hooks_added += 1

    if hooks_added > 0:
        settings["hooks"] = hooks
        info(f"Will add {hooks_added} hook(s) to {settings_path}")
        info("Hooks are what capture sessions automatically (start/end/compact) -")
        info("without them, nothing gets saved to memory on its own.")
        if ask_yn("Register hooks?"):
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=4)
            ok(f"{hooks_added} hooks registered")
        else:
            warn("Hooks not registered - memory won't auto-capture sessions")
    else:
        ok("All 6 hooks already registered")

    # --- Register MCP server (global "user" scope - one entry covers
    # every project folder, now and for any added later) ---
    info("This is what lets Claude QUERY memory during a session (search,")
    info("lookup_fact, etc.) - without it, sessions get captured but not searchable.")
    if ensure_mcp_user_registration(str(root)):
        ok("MCP server registered (user scope, applies to all projects)")
    else:
        ok("MCP server already registered")
    removed = remove_mcp_local_registrations()
    if removed:
        info(f"Cleaned up {len(removed)} stale per-project MCP registration(s) that would")
        info("otherwise conflict with the user-scope one above.")

    # --- Install custom slash commands ---
    commands_dir = home / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    slash_commands = {
        "remember-import": textwrap.dedent(f"""\
            Import conversations from Claude.ai, ChatGPT, or Gemini into the memory database.

            **Claude.ai:** Uses JSON files exported via Chrome extension.
            Step 1: List JSON files in the imports folder:
              ls {root}/imports/*.json

            If no JSON files are found, do NOT fail silently or throw a bare error - tell the
            user exactly how to resolve it for whichever platform(s) they want, using these full
            instructions (not a one-line pointer):

            "No JSON files found in {root}/imports/. Here's how to get your history in, for
            whichever platform(s) you used:

            CLAUDE.AI - single conversation (quick path):
            1. Install the 'AI Chat Exporter' Chrome extension from the Chrome Web Store.
            2. Go to claude.ai, open the conversation, click the extension icon, export as JSON.
            3. Move the downloaded .json file to: {root}/imports/
            4. Then run /remember-import again.

            CLAUDE.AI - full account export (every conversation at once):
            1. In Claude: click your initials (bottom-left) -> Settings -> Privacy -> Export Data.
               An email arrives with a download link (expires in 24 hours).
            2. That link downloads a manifest.json listing several zips - find the one with
               \\"category\\": \\"conversations\\" (conversations-000.zip) and download/extract just
               that one. Extracting it gives you conversations.json.
            3. Run: python3 {root}/scripts/import_claude_ai.py --scan <path-to-conversations.json>
               This builds claude_ai_import_map.xlsx (title/about/tags per conversation, empty
               folder column) - nothing is imported yet.
            4. Open the xlsx, type a folder name into the folder column for anything you want to
               keep (existing name, or a new one), delete rows you don't want, save it.
            5. Preview: python3 {root}/scripts/import_claude_ai.py --import <path> --map claude_ai_import_map.xlsx --dry-run
            6. Looks good? Run the same command again minus --dry-run.

            CHATGPT:
            1. In ChatGPT: Settings -> Data Controls -> Export. An email with a download link
               arrives within a day or two.
            2. Extract the zip. Copy just the conversations-*.json files into their own folder
               under imports/, e.g. {root}/imports/chatgpt-import/.
            3. Run: python3 {root}/scripts/import_chatgpt.py --scan {root}/imports/chatgpt-import/
               This builds chatgpt_import_map.xlsx - nothing is imported yet.
            4. Open the xlsx, type a folder name per conversation you want to keep, delete the
               rest, save it in that same chatgpt-import/ folder.
            5. Preview: python3 {root}/scripts/import_chatgpt.py --import {root}/imports/chatgpt-import/ --map {root}/imports/chatgpt-import/chatgpt_import_map.xlsx --dry-run
            6. Looks good? Run the same command again minus --dry-run.

            GEMINI:
            1. Go to Google Takeout, select only 'My Activity' -> 'Gemini Apps', export it, wait
               for the email, download the zip.
            2. Extract it, e.g. to {root}/imports/gemini-export/.
            3. Run: python3 {root}/scripts/import_gemini.py --scan {root}/imports/gemini-export/
               This builds gemini_import_map.xlsx - nothing is imported yet.
            4. Open the xlsx, type a folder name per conversation to keep, delete the rest, save.
            5. Preview with --dry-run, then run the same command again without it, same as ChatGPT.

            Then run /remember-import again."
            And stop here.

            Step 2: If files ARE found, show the user a numbered list of filenames.
            Ask: "Which file do you want to import?" (they can say a number or filename)

            Step 3: Show context, then ask which folder to put it in. Run (no --folder, this only
              shows info, it does not import anything):
                python3 {root}/scripts/import_claude_ai.py "<chosen_file_path>"
              This prints the conversation's title, a short "about" description of what it was
              about, suggested tags, and the list of existing projects (folder name, prefix, label).
              There is no automatic matching or scoring - show the user the title/about/tags plus
              the existing project list, and ask them to type a folder name: an existing one from
              that list, or a brand-new name. Never assign a folder without them deciding.

            Step 4: Run the import with their chosen folder name - this works whether it's an
              existing project or a brand-new one (creates it automatically, including folder,
              CLAUDE.md, config entry, database + MCP registration, all in this one command):
                python3 {root}/scripts/import_claude_ai.py "<chosen_file_path>" --folder <folder-name>

            Step 5: Show the result. If successful, tell the user the file has been moved to imports/completed/.
            Ask if they want to import another file (if more remain in imports/).
        """),
        "remember-status": textwrap.dedent(f"""\
            Run the memory status check. Execute this command:
            python3 {root}/scripts/status.py

            Show the user the full output. This displays:
            - Total sessions and messages in the database
            - Messages per project
            - Last session date per project
            - Database size and backup status
            - Semantic search embedding count
        """),
        "remember-add-project": textwrap.dedent(f"""\
            Register and initialize a new project in claude_remembers - no interactive
            prompts, this is a single non-interactive command.

            Ask the user for the folder path their project's files live in (an existing
            real folder anywhere on disk - nothing nested gets created, anything already
            there is left untouched). Then run:
              python3 {root}/scripts/add-project.py "<folder-path>"

            This derives the project name from the folder's basename, auto-generates a
            database prefix, and Title-Cases the label - then does everything in one
            pass: folder/CLAUDE.md, config.yaml entry, database registration, MCP
            registration. If the user wants to override any of the derived values,
            add --name <name>, --prefix <pfx>, and/or --label "<Display Name>" to the
            same command.

            Show the result. If it fails because the name/prefix/path is already taken,
            ask the user for a different value and re-run with the matching override flag.

            (For fixing a broken hook/MCP registration or re-running the full installer,
            use `python3 {root}/scripts/remember-setup.py` directly instead - that's the
            first-run installer, not this command's job.)
        """),
        "remember-questionnaire": textwrap.dedent(f"""\
            Help the user fill out their questionnaire.

            The questionnaire file is at: {root}/remember-questionnaire.txt

            First, read the file to see what's already filled in.
            Then ask the user what they'd like to add or update.
            Edit the file with their answers.

            When they're done, run:
            python3 {root}/scripts/remember-setup.py --questionnaire

            This imports their answers into the memory database so Claude
            knows who they are across every session.
        """),
    }

    cmds_added = 0
    for name, content in slash_commands.items():
        cmd_path = commands_dir / f"{name}.md"
        if not cmd_path.exists():
            with open(cmd_path, "w") as f:
                f.write(content)
            cmds_added += 1

    if cmds_added > 0:
        ok(f"{cmds_added} slash commands installed: " + ", ".join(f"/{n}" for n in slash_commands))
    else:
        ok("All slash commands already installed")

    # --- Scan for existing JSONL files ---
    jsonl_dir = home / ".claude" / "projects"
    if jsonl_dir.exists():
        jsonl_files = list(jsonl_dir.rglob("*.jsonl"))
        if jsonl_files:
            info(f"\nFound {len(jsonl_files)} JSONL session files in {jsonl_dir}")
            info("These are your past Claude Code sessions - importing now pulls them")
            info("into memory immediately instead of waiting for your next session.")
            ingest_script = root / "scripts" / "ingest_jsonl.py"
            startup_script = root / "scripts" / "startup_check.py"

            if startup_script.exists():
                if ask_yn("Import existing sessions now?"):
                    info("Running startup_check.py to ingest sessions...")
                    try:
                        result = subprocess.run(
                            [sys.executable, str(startup_script)],
                            capture_output=True, text=True, timeout=120,
                            cwd=str(root)
                        )
                        if result.returncode == 0:
                            ok("Sessions ingested")
                            if result.stdout.strip():
                                for line in result.stdout.strip().split("\n")[-5:]:
                                    info(f"  {line}")
                        else:
                            warn(f"Ingestion returned code {result.returncode}")
                            if result.stderr.strip():
                                info(f"  {result.stderr.strip()[:200]}")
                    except Exception as e:
                        warn(f"Ingestion error: {e}")
                else:
                    info("Sessions will be imported automatically on your first Claude Code session.")
            else:
                info("startup_check.py not found - sessions will be imported on first Claude Code session.")
    else:
        info("No existing JSONL files found (new Claude Code installation)")

    # --- Check for claude.ai exports ---
    imports_dir = root / "imports"
    json_files = list(imports_dir.glob("*.json")) if imports_dir.exists() else []
    if json_files:
        info(f"\nFound {len(json_files)} JSON file(s) in imports/ folder")
        import_script = root / "scripts" / "import_claude_ai.py"
        if import_script.exists():
            info(f"Run: python3 {import_script} to import them")
        else:
            info("import_claude_ai.py not found - will be available after full setup")

# ---------------------------------------------------------------------------
# Phase 8: Health Check + Next Steps
# ---------------------------------------------------------------------------

def phase_health_check(cfg):
    phase_header(9, 9, "HEALTH CHECK & NEXT STEPS")
    root = Path(cfg["root_path"])
    home = Path.home()
    checks = []

    # DB
    try:
        conn = sqlite3.connect(cfg["db_path"])
        conn.execute("SELECT 1")
        tables = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        conn.close()
        checks.append(("Database", True, f"{tables} tables"))
    except Exception as e:
        checks.append(("Database", False, str(e)))

    # Config
    config_path = root / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                yaml.safe_load(f)
            checks.append(("config.yaml", True, "valid YAML"))
        except Exception as e:
            checks.append(("config.yaml", False, str(e)))
    else:
        checks.append(("config.yaml", False, "not found"))

    # Hooks
    settings_path = home / ".claude" / "settings.json"
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
        registered = sum(1 for e in HOOK_EVENTS if e in settings.get("hooks", {}))
        if registered == len(HOOK_EVENTS):
            checks.append(("Hooks", True, f"all {registered} registered"))
        else:
            checks.append(("Hooks", False, f"only {registered}/{len(HOOK_EVENTS)} registered"))
    else:
        checks.append(("Hooks", False, "settings.json not found"))

    # MCP server file
    server_py = root / "mcp" / "server.py"
    if server_py.exists():
        checks.append(("MCP server.py", True, "exists"))
    else:
        checks.append(("MCP server.py", False, "not found"))

    # MCP registration - one global "user" scope entry covers every project
    claude_json_path = home / ".claude.json"
    if claude_json_path.exists():
        with open(claude_json_path) as f:
            cj = json.load(f)
        if "remember-server" in cj.get("mcpServers", {}):
            checks.append(("MCP registered", True, "user scope (all projects)"))
        else:
            checks.append(("MCP registered", False, "not registered"))
    else:
        checks.append(("MCP registered", False, ".claude.json not found"))

    # Project directories
    missing_dirs = []
    for p in cfg["projects"]:
        pdir = Path(p["path"])
        if not pdir.exists():
            missing_dirs.append(p["folder_name"])
    if missing_dirs:
        checks.append(("Project dirs", False, f"missing: {', '.join(missing_dirs)}"))
    else:
        checks.append(("Project dirs", True, f"all {len(cfg['projects'])} exist"))

    # CLAUDE.md files
    missing_md = []
    for p in cfg["projects"]:
        if not (Path(p["path"]) / "CLAUDE.md").exists():
            missing_md.append(p["folder_name"])
    if missing_md:
        checks.append(("CLAUDE.md files", False, f"missing in: {', '.join(missing_md)}"))
    else:
        checks.append(("CLAUDE.md files", True, f"all {len(cfg['projects'])} deployed"))

    # DB stats
    try:
        conn = sqlite3.connect(cfg["db_path"])
        sessions = conn.execute("SELECT COUNT(*) FROM sys_sessions").fetchone()[0]
        transcripts = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        facts = conn.execute("SELECT COUNT(*) FROM remember_facts").fetchone()[0]
        conn.close()
        checks.append(("DB contents", True, f"{sessions} sessions, {transcripts} messages, {facts} facts"))
    except Exception:
        pass

    # Print results
    print()
    all_pass = True
    for name, passed, detail in checks:
        status = green("PASS") if passed else red("FAIL")
        print(f"  {status}  {name}: {detail}")
        if not passed:
            all_pass = False

    # --- Questionnaire info ---
    print(f"\n{'='*60}")
    print(f"  PERSONALIZE YOUR MEMORY")
    print(f"{'='*60}")
    questionnaire_path = root / "remember-questionnaire.txt"
    if not questionnaire_path.exists():
        _write_questionnaire_template(questionnaire_path, cfg["db_path"])
    print(f"""
  A questionnaire template has been created at:
    {questionnaire_path}

  Fill it out to teach the system about you - your background,
  preferences, family, goals, and working style. This makes Claude
  actually know who you are across every session.

  Once you're in Claude Code, just type:  /remember-questionnaire
  Claude will help you fill it out and import your answers.
""")

    # --- Slash commands ---
    print(f"{'='*60}")
    print(f"  SLASH COMMANDS (use inside Claude Code)")
    print(f"{'='*60}")
    print(f"""
  These commands work inside any Claude Code session:

    /remember-import         Import conversations (Claude.ai, ChatGPT, Gemini)
    /remember-status         Check memory health and statistics
    /remember-add-project    Register a new project (folder path, no prompts)
    /remember-questionnaire  Fill out or update your memory profile
""")

    # --- Next steps ---
    print(f"{'='*60}")
    print(f"  WHAT TO DO NEXT")
    print(f"{'='*60}")
    print(f"""
  1. START USING IT
     Open Claude Code in any project folder:
       cd {root / cfg['projects'][0]['folder_name']}
       claude
     The memory activates automatically - hooks fire, MCP connects,
     context loads. Nothing else to configure.

  2. IMPORT CONVERSATIONS (optional)
     You can import conversation history from Claude.ai, ChatGPT, and Gemini.

     CLAUDE.AI:
     Step 1 - Install the Chrome extension:
       Open Chrome and go to:
         https://chromewebstore.google.com/detail/claude-exporter-save-clau/elhmfakncmnghlnabnolalcjkdpfjnin
       Or search the Chrome Web Store for "AI Chat Exporter".
       Click "Add to Chrome" and confirm the install.
       You'll see a small icon appear in your browser toolbar.

     Step 2 - Export conversations:
       Go to https://claude.ai and open any conversation.
       Click the AI Chat Exporter icon in your toolbar
       (top-right of Chrome, may be under the puzzle piece menu).
       Select "JSON" as the export format.
       Click "Export" - a .json file downloads to your
       Downloads folder.

       Repeat for each conversation you want to import.

     Step 3 - Move the exported files:
       Move all the .json files into your imports folder:
         mv ~/Downloads/*.json {root / 'imports'}/

     Step 4 - Import into memory:
       Open Claude Code in any project folder and type:
         /remember-import
       Claude will walk you through assigning each conversation
       to a project. Successfully imported files are moved to
       imports/completed/ automatically.

     CHATGPT:
       Export your data from OpenAI (Settings > Data > Export).
       Then run: python3 scripts/import_chatgpt.py --scan <path-to-export>
       Or just tell Claude: "import my ChatGPT conversations"

     GEMINI:
       Export via Google Takeout (select Gemini Apps).
       Then run: python3 scripts/import_gemini.py --scan <path-to-takeout>
       Or just tell Claude: "import my Gemini conversations"

  3. FILL OUT THE QUESTIONNAIRE (optional)
     Open Claude Code and type:  /remember-questionnaire
     Claude will help you fill out your profile interactively.

  4. CHECK MEMORY STATUS (anytime)
     Inside Claude Code, type:  /remember-status
""")

    if all_pass:
        print(f"  {green('Setup complete. Your memory is ready.')}")
    else:
        print(f"  {yellow('Setup finished with warnings. Review FAIL items above.')}")

    return all_pass


def _write_questionnaire_template(path, db_path=None):
    """Write a blank questionnaire template for the user to fill out."""
    project_line = ""
    if db_path and os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            prefixes = [r[0] for r in conn.execute(
                "SELECT prefix FROM project_registry ORDER BY prefix"
            ).fetchall()]
            conn.close()
            if prefixes:
                project_line = "# Your registered project prefixes: " + ", ".join(prefixes) + "\n"
        except Exception:
            pass

    template = textwrap.dedent(f"""\
        # Remember Questionnaire
        # Fill out as much or as little as you want.
        # Lines starting with # are comments and will be ignored.
        # After filling out, run: python3 scripts/remember-setup.py --questionnaire
        #
        # Sections under a plain "## CATEGORY" header apply to EVERY project.
        # To make one section apply to only ONE project instead, use:
        #   ## PROJECT:<prefix> CATEGORY
        # e.g. "## PROJECT:lc WORKING STYLE" with "Communication preference: terse"
        # underneath makes that answer apply only in project lc, overriding the
        # global value of the same question everywhere else. A plain "## CATEGORY"
        # header afterward goes back to applying globally.
        {project_line}
        ## IDENTITY
        # Full name:
        # Nickname / preferred name:
        # Location (city, state/country):
        # Age or birth year:

        ## FAMILY
        # Spouse/partner name:
        # Children (names and ages):
        # Pets:
        # Other family members to know about:

        ## PROFESSIONAL
        # Current role/title:
        # Company/organization:
        # Industry:
        # Key skills:
        # Career background (brief):

        ## WORKING STYLE
        # Communication preference (direct, detailed, casual, formal):
        # How you like feedback (blunt, gentle, structured):
        # Decision-making style:
        # Pet peeves in AI responses:
        # What you value most in AI assistance:

        ## TECHNICAL SETUP
        # Primary OS:
        # Editor/IDE:
        # Primary programming languages:
        # Key tools you use daily:

        ## GOALS
        # Current main project(s):
        # What you're trying to accomplish this month:
        # Long-term goals:

        ## HEALTH (optional)
        # Any health context that affects your work:

        ## PREFERENCES
        # Timezone:
        # Work hours:
        # Anything else the memory system should know:
    """)
    with open(path, "w") as f:
        f.write(template)


PROJECT_HEADER_RE = re.compile(r'^##\s+PROJECT:(\S+)\s+(.+)$')


def import_questionnaire(root_path, db_path):
    """Import a filled-out questionnaire into remember_facts.

    Supports an optional per-project override syntax:
        ## PROJECT:<prefix> CATEGORY
    scopes everything under it to that project (overriding the global value
    of the same category+key elsewhere) until the next "##" header. A plain
    "## CATEGORY" header resets back to global scope (project=NULL).
    """
    q_path = Path(root_path) / "remember-questionnaire.txt"
    if not q_path.exists():
        fail(f"Questionnaire not found: {q_path}")
        sys.exit(1)

    content = q_path.read_text(encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    valid_prefixes = {r[0] for r in cursor.execute("SELECT prefix FROM project_registry").fetchall()}

    current_category = "general"
    current_project = None  # None = global scope
    count = 0

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            # Check if it's a category header
            if line.startswith("## "):
                m = PROJECT_HEADER_RE.match(line)
                if m:
                    prefix_candidate, cat_text = m.group(1), m.group(2)
                    if prefix_candidate in valid_prefixes:
                        current_project = prefix_candidate
                    else:
                        warn(f"Unknown project prefix '{prefix_candidate}' in questionnaire "
                             f"header - treating section as global")
                        current_project = None
                    current_category = cat_text.strip().lower().replace(" ", "-")
                else:
                    current_project = None
                    current_category = line[3:].strip().lower().replace(" ", "-")
            continue

        # Parse "Key: Value" format
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if value:  # Only insert if there's a value
                # Upsert - update if same category+key+project exists. "IS ?"
                # (not "= ?") is required for NULL-safe matching when
                # current_project is None.
                existing = cursor.execute(
                    "SELECT id FROM remember_facts WHERE category=? AND key=? AND project IS ?",
                    (current_category, key, current_project)
                ).fetchone()
                if existing:
                    cursor.execute(
                        "UPDATE remember_facts SET value=?, source='questionnaire', updated_at=? WHERE id=?",
                        (value, now, existing[0])
                    )
                else:
                    cursor.execute(
                        "INSERT INTO remember_facts (category, key, value, project, source, confidence, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, 'questionnaire', 'confirmed', ?, ?)",
                        (current_category, key, value, current_project, now, now)
                    )
                count += 1

    conn.commit()
    conn.close()
    ok(f"Imported {count} facts from questionnaire")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{bold('claude_remembers setup')}")
    print("Persistent memory system for Claude Code\n")

    # Handle --questionnaire mode
    if "--questionnaire" in sys.argv:
        # Need to find root and db paths
        script_dir = Path(__file__).resolve().parent.parent
        config_path = script_dir / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            root_path = config["storage"]["root_path"]
            if config["storage"]["mode"] == "synced":
                db_path = config["storage"]["local_db_path"]
            else:
                db_path = os.path.join(root_path, "claude_remembers.db")
            import_questionnaire(root_path, db_path)
        else:
            fail("config.yaml not found. Run remember-setup.py first (without --questionnaire).")
            sys.exit(1)
        return

    # Normal setup flow
    pip_cmd = phase_preflight()
    phase_dependencies(pip_cmd)
    cfg = phase_projects()
    phase_directories(cfg)
    phase_database(cfg)
    phase_config(cfg)
    phase_email(cfg)
    phase_registration(cfg)
    phase_health_check(cfg)


if __name__ == "__main__":
    main()
