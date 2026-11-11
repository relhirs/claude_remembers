#!/usr/bin/env python3
"""
startup_check.py - Session start orchestrator for claude_remembers.

Scans for new JSONL files, calls ingest_jsonl.ingest() for each,
verifies required folders, triggers DB backup, prints summary.

Usage:
    python3 startup_check.py

Triggered by: hooks/session-start.py
Exit codes: 0 = success, 1 = warnings, 2 = fatal error
"""

import logging
import os
import pathlib
import shutil
import socket
import sqlite3
import subprocess
import sys
import time

import yaml

from metrics_utils import ensure_metrics_schema
from config_utils import (
    resolve_db_path,
    migrate_config_project_paths,
    ensure_mcp_user_registration,
    remove_mcp_local_registrations,
)

# ---------------------------------------------------------------------------
# Config / Logging / DB - same patterns as ingest_jsonl.py
# ---------------------------------------------------------------------------

def get_root_path():
    return str(pathlib.Path(__file__).resolve().parent.parent)


def load_config(root_path):
    config_path = os.path.join(root_path, "config.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(f"FATAL: config.yaml not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    for key_path in [
        ("storage", "root_path"),
        ("jsonl", "source_paths"),
    ]:
        obj = config
        for k in key_path:
            if not isinstance(obj, dict) or k not in obj:
                raise SystemExit(f"FATAL: Missing config key: {'.'.join(key_path)}")
            obj = obj[k]
    return config


def setup_logging(root_path):
    hostname = socket.gethostname()
    log_dir = os.path.join(root_path, "logs", hostname)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "startup_check.log")

    logger = logging.getLogger("startup_check")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%Y-%m-%dT%H:%M:%SZ")
        fmt.converter = time.gmtime
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def connect_db(db_path):
    if not os.path.exists(db_path):
        raise SystemExit(f"FATAL: Database not found at {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------
# Self-healing column additions. SQLite has no "ADD COLUMN IF NOT EXISTS",
# so each column is guarded by a PRAGMA table_info check. Safe to run on
# every startup.

SYS_SESSIONS_MIGRATIONS = [
    # (column_name, column_ddl)
    ("end_reason", "end_reason TEXT"),
    ("clear_recovery_shown", "clear_recovery_shown INTEGER DEFAULT 0"),
]

PROJECT_REGISTRY_MIGRATIONS = [
    # (column_name, column_ddl)
    ("path", "path TEXT"),
]

REMEMBER_FACTS_MIGRATIONS = [
    # (column_name, column_ddl)
    ("project", "project TEXT DEFAULT NULL"),
]

REMEMBER_PREFERENCES_MIGRATIONS = [
    # (column_name, column_ddl)
    ("project", "project TEXT DEFAULT NULL"),
]


def migrate_schema(conn, logger):
    """Add any missing columns to sys_sessions. Idempotent."""
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(sys_sessions)").fetchall()}
    added = 0
    for col_name, col_ddl in SYS_SESSIONS_MIGRATIONS:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE sys_sessions ADD COLUMN {col_ddl}")
            logger.info("Migration: added sys_sessions.%s", col_name)
            added += 1
    if added:
        conn.commit()

    # project_registry.path - denormalized copy of config.yaml's project
    # path field, kept in sync by add-project.py/remember-setup.py. Added
    # alongside the config.yaml `path` field (see config_utils.py) that
    # decouples a project's physical folder from root_path/folder_name.
    existing_proj_cols = {row[1] for row in conn.execute("PRAGMA table_info(project_registry)").fetchall()}
    proj_added = 0
    for col_name, col_ddl in PROJECT_REGISTRY_MIGRATIONS:
        if col_name not in existing_proj_cols:
            conn.execute(f"ALTER TABLE project_registry ADD COLUMN {col_ddl}")
            logger.info("Migration: added project_registry.%s", col_name)
            proj_added += 1
    if proj_added:
        conn.commit()

    # remember_facts/remember_preferences.project - lets the questionnaire
    # (and anything else) scope a fact/preference to one project instead of
    # applying globally. NULL (the default) means global, unchanged for
    # every existing row. See scripts/profile_utils.py for how this is merged.
    existing_rf_cols = {row[1] for row in conn.execute("PRAGMA table_info(remember_facts)").fetchall()}
    rf_added = 0
    for col_name, col_ddl in REMEMBER_FACTS_MIGRATIONS:
        if col_name not in existing_rf_cols:
            conn.execute(f"ALTER TABLE remember_facts ADD COLUMN {col_ddl}")
            logger.info("Migration: added remember_facts.%s", col_name)
            rf_added += 1
    if rf_added:
        conn.commit()

    existing_rp_cols = {row[1] for row in conn.execute("PRAGMA table_info(remember_preferences)").fetchall()}
    rp_added = 0
    for col_name, col_ddl in REMEMBER_PREFERENCES_MIGRATIONS:
        if col_name not in existing_rp_cols:
            conn.execute(f"ALTER TABLE remember_preferences ADD COLUMN {col_ddl}")
            logger.info("Migration: added remember_preferences.%s", col_name)
            rp_added += 1
    if rp_added:
        conn.commit()

    # transcripts has an AFTER DELETE trigger keeping transcripts_fts in sync
    # (transcripts_ad) but nothing equivalent existed for transcript_embeddings,
    # so deleting transcripts (e.g. via remove-project.py) left orphaned
    # embedding rows behind. Add the missing trigger, idempotently.
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS transcripts_ad_embeddings
        AFTER DELETE ON transcripts
        BEGIN
            DELETE FROM transcript_embeddings WHERE transcript_id = old.id;
        END
    """)
    conn.commit()

    # Measurement layer tables (retrieval_events, tool_events, api_usage,
    # metrics_snapshots). CREATE IF NOT EXISTS, so safe on every startup.
    ensure_metrics_schema(conn)

    return added


# ---------------------------------------------------------------------------
# Folder verification
# ---------------------------------------------------------------------------

REQUIRED_FOLDERS = [
    "",             # ROOT itself
    "scripts",
    "hooks",
    "mcp",
    "logs",
    "db-backup",
    "verification",
]


def verify_folders(root_path, logger):
    """Check required folders exist. Create missing ones with warning."""
    warnings = 0
    for folder in REQUIRED_FOLDERS:
        full_path = os.path.join(root_path, folder) if folder else root_path
        if not os.path.isdir(full_path):
            logger.warning("Missing folder, creating: %s", full_path)
            os.makedirs(full_path, exist_ok=True)
            warnings += 1
    return warnings


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def get_ingested_files(conn):
    """Return set of file_path values already in sys_ingest_log."""
    cur = conn.execute("SELECT file_path FROM sys_ingest_log")
    return {row[0] for row in cur.fetchall()}


def discover_files(source_paths, config, logger):
    """Walk source paths and discover JSONL and tool-result files.

    Returns list of (file_path, expected_type) tuples.
    """
    ingest_subagents = config.get("jsonl", {}).get("ingest_subagents", True)
    ingest_tool_results = config.get("jsonl", {}).get("ingest_tool_results", True)

    discovered = []

    for source_path in source_paths:
        source_path = os.path.expanduser(source_path)
        if not os.path.isdir(source_path):
            logger.warning("Source path does not exist: %s", source_path)
            continue

        for dirpath, dirnames, filenames in os.walk(source_path):
            # Skip memory directories
            if "/memory/" in dirpath or dirpath.endswith("/memory"):
                continue

            for filename in filenames:
                full_path = os.path.join(dirpath, filename)

                if filename.endswith(".jsonl"):
                    # Check if subagent
                    if "/subagents/" in full_path or "/subagent/" in full_path:
                        if ingest_subagents:
                            discovered.append((full_path, "subagent"))
                    else:
                        discovered.append((full_path, "jsonl"))

                elif filename.endswith(".txt"):
                    if ingest_tool_results and ("/tool-results/" in full_path or "/tool_results/" in full_path):
                        discovered.append((full_path, "tool_result"))

    return discovered


# ---------------------------------------------------------------------------
# Abandoned-session backfill
# ---------------------------------------------------------------------------
# hooks/session-end.py only fires on a graceful exit (/exit, /clear,
# /logout) - Claude Code explicitly triggers it before the process closes.
# If a terminal tab gets closed, the app force-quit, or the machine
# sleeps/crashes mid-session, that hook never runs, end_reason stays blank
# forever, and the session permanently has no summary/notes - nothing else
# ever revisits it.
#
# This closes that gap without needing a true SessionEnd signal: every
# session-start is itself proof that a previous process is gone, so any
# OTHER session still missing end_reason, whose last message is old enough
# that it can't just be sitting idle in another open terminal tab, is safe
# to treat as over and run the normal summarization pipeline against -
# same script, same result, just triggered late instead of never.

ABANDONED_STALE_MINUTES = 30
ABANDONED_BATCH_LIMIT = 5  # cap per startup run so this can't stall session-start


def backfill_abandoned_sessions(conn, root_path, logger):
    """Summarize sessions that never got a graceful SessionEnd. Returns count processed."""
    rows = conn.execute(
        """SELECT session_id, message_count FROM sys_sessions
           WHERE (end_reason IS NULL OR end_reason = '')
             AND message_count > 0
             AND ended_at IS NOT NULL
             AND ended_at < datetime('now', ?)
           ORDER BY ended_at ASC
           LIMIT ?""",
        (f"-{ABANDONED_STALE_MINUTES} minutes", ABANDONED_BATCH_LIMIT),
    ).fetchall()

    if not rows:
        return 0

    summarize_path = os.path.join(root_path, "scripts", "summarize_session.py")
    processed = 0
    for session_id, msg_count in rows:
        try:
            result = subprocess.run(
                [sys.executable, summarize_path,
                 "--session-id", session_id, "--reason", "inferred_abandoned"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                logger.warning("Abandoned-session backfill failed for %s: %s",
                                session_id, (result.stderr or "")[-500:])
            else:
                logger.info("Abandoned-session backfill: summarized %s (%d msgs)",
                             session_id, msg_count)
                processed += 1
        except Exception as e:
            logger.warning("Abandoned-session backfill error for %s: %r", session_id, e)

    return processed


# ---------------------------------------------------------------------------
# Database backup
# ---------------------------------------------------------------------------

def run_backup(root_path, config, logger):
    """Run database backup with rotation.

    Inline implementation (separate from remember_sync.py).
    Rotation: max 2 copies. .bak2 deleted, .bak1 renamed to .bak2, new copy to .bak1.
    """
    db_path = resolve_db_path(config)
    backup_dir = os.path.join(root_path, "db-backup")
    os.makedirs(backup_dir, exist_ok=True)

    if not os.path.exists(db_path):
        logger.error("Cannot backup: DB not found at %s", db_path)
        return False, 0

    db_name = os.path.basename(db_path)
    bak1 = os.path.join(backup_dir, f"{db_name}.bak1")
    bak2 = os.path.join(backup_dir, f"{db_name}.bak2")

    # Rotate
    if os.path.exists(bak2):
        os.remove(bak2)
    if os.path.exists(bak1):
        os.rename(bak1, bak2)

    # Copy
    shutil.copy2(db_path, bak1)

    # Verify
    backup_size = os.path.getsize(bak1)
    if backup_size == 0:
        logger.error("Backup file is empty: %s", bak1)
        return False, 0

    verify = config.get("backup", {}).get("verify_after_copy", True)
    if verify:
        try:
            conn = sqlite3.connect(bak1)
            result = conn.execute("PRAGMA integrity_check;").fetchone()[0]
            conn.close()
            if result != "ok":
                logger.error("Backup integrity check failed: %s", result)
                return False, backup_size
        except Exception as e:
            logger.error("Backup integrity check error: %s", e)
            return False, backup_size

    logger.info("Backup complete: %s (%d bytes)", bak1, backup_size)
    return True, backup_size




# ---------------------------------------------------------------------------
# Main startup check
# ---------------------------------------------------------------------------

def startup_check(root_path=None, config=None):
    """Run the full startup check sequence.

    Returns dict with summary info and exit_code.
    """
    if root_path is None:
        root_path = get_root_path()
    if config is None:
        config = load_config(root_path)

    logger = setup_logging(root_path)
    logger.info("=== Startup check begin ===")

    # 0. Backfill config.yaml project `path` fields (self-healing, idempotent)
    config_path = os.path.join(root_path, "config.yaml")
    if migrate_config_project_paths(config, config_path):
        logger.info("Migration: backfilled project `path` fields in config.yaml")

    # 0b. MCP registration: one global "user" scope entry replaces the old
    # per-project ones (which conflict with it once both exist). Self-heal
    # any installs still carrying the old per-project registrations.
    if ensure_mcp_user_registration(root_path):
        logger.info("Migration: registered remember-server at MCP user scope")
    cleaned = remove_mcp_local_registrations()
    if cleaned:
        logger.info("Migration: removed %d stale per-project MCP registration(s)", len(cleaned))

    # Import ingest module
    import importlib.util
    ingest_path = os.path.join(root_path, "scripts", "ingest_jsonl.py")
    spec = importlib.util.spec_from_file_location("ingest_jsonl", ingest_path)
    ingest_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest_mod)

    warnings = 0
    errors = 0

    # 1. Verify folders
    folder_warnings = verify_folders(root_path, logger)
    warnings += folder_warnings

    # 2. Connect to DB
    db_path = resolve_db_path(config)
    try:
        conn = connect_db(db_path)
    except SystemExit as e:
        logger.error(str(e))
        return {"source_paths_scanned": 0, "new_files": 0,
                "records_ingested": 0, "errors": 1,
                "backup_ok": False, "backup_size": 0,
                "exit_code": 2}

    try:
        # 2b. Migrate schema (self-healing, idempotent)
        migrate_schema(conn, logger)

        # 3. Get already-ingested files
        ingested = get_ingested_files(conn)

        # 4. Discover files from source paths
        source_paths = config["jsonl"]["source_paths"]
        all_discovered = discover_files(source_paths, config, logger)

        # 5. Filter to new files only
        new_files = [(fp, ft) for fp, ft in all_discovered
                     if os.path.abspath(fp) not in ingested]

        # 6. Ingest new files
        total_imported = 0
        ingest_errors = 0
        for file_path, file_type in new_files:
            result = ingest_mod.ingest(
                file_path, type_override=file_type,
                config=config, root_path=root_path
            )
            total_imported += result["records_imported"]
            if result["exit_code"] != 0:
                ingest_errors += 1
                errors += 1

        # 6b. Catch up any sessions that never got a graceful SessionEnd
        # (closed terminal, force-quit, crash, sleep) - see comment above
        # backfill_abandoned_sessions() for why this is safe to run here.
        try:
            backfilled = backfill_abandoned_sessions(conn, root_path, logger)
            if backfilled:
                logger.info("Abandoned-session backfill: %d session(s) summarized", backfilled)
        except Exception as e:
            logger.warning("Abandoned-session backfill step failed: %r", e)

    finally:
        conn.close()

    # 7. Run backup
    backup_ok, backup_size = run_backup(root_path, config, logger)
    if not backup_ok:
        warnings += 1

    # 8. Determine exit code
    if errors > 0:
        exit_code = 1
    else:
        exit_code = 0

    logger.info("=== Startup check complete: %d new files, %d records, %d errors ===",
                len(new_files), total_imported, errors)

    return {
        "source_paths_scanned": len(source_paths),
        "new_files": len(new_files),
        "records_ingested": total_imported,
        "errors": errors,
        "backup_ok": backup_ok,
        "backup_size": backup_size,
        "exit_code": exit_code,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    result = startup_check()

    backup_status = f"OK ({result['backup_size']} bytes)" if result["backup_ok"] else "FAILED"

    print("=== Claude Remembers Startup Check ===")
    print(f"Source paths scanned: {result['source_paths_scanned']}")
    print(f"New files found: {result['new_files']}")
    print(f"Records ingested: {result['records_ingested']}")
    print(f"Errors: {result['errors']}")
    print(f"Backup: {backup_status}")
    print("===================================")

    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()
