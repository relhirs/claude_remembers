#!/usr/bin/env python3
"""
import_claude_ai.py - Import claude.ai JSON conversation exports into claude_remembers.

Parses claude.ai export JSON (single conversation per file), maps messages,
writes to sys_sessions + transcripts + sys_ingest_log, moves file to completed/.

Usage:
    python3 import_claude_ai.py <json_file> --project <prefix>
    python3 import_claude_ai.py <json_file>              # prints a ranked
                                                           # project suggestion
                                                           # instead of importing
    python3 import_claude_ai.py <json_file> --new-project <folder-name> [--prefix <prefix>] [--label <label>]
                                                           # creates the project
                                                           # on the fly, then
                                                           # imports into it

Exit codes: 0 = success, 1 = error, 2 = no --project given (suggestion printed)
"""

import argparse
import datetime
import importlib.util
import json
import logging
import os
import pathlib
import re
import shutil
import socket
import sqlite3
import sys
import time

import yaml

from config_utils import resolve_db_path
from project_classifier import suggest_project, slugify, CONFIDENCE_THRESHOLD
from write_exchange import embed_message, SEMANTIC_AVAILABLE

# ---------------------------------------------------------------------------
# Config / Logging / DB
# ---------------------------------------------------------------------------

def get_root_path():
    return str(pathlib.Path(__file__).resolve().parent.parent)


def load_config(root_path):
    config_path = os.path.join(root_path, "config.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(f"FATAL: config.yaml not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "storage" not in config or "root_path" not in config["storage"]:
        raise SystemExit("FATAL: Missing config key: storage.root_path")
    return config


def setup_logging(root_path):
    hostname = socket.gethostname()
    log_dir = os.path.join(root_path, "logs", hostname)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "import_claude_ai.log")

    logger = logging.getLogger("import_claude_ai")
    logger.setLevel(logging.DEBUG)

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
# Import logic
# ---------------------------------------------------------------------------

SENDER_MAP = {
    "human": ("user", "user"),
    "assistant": ("assistant", "assistant"),
}


def extract_user_text_samples(chat_messages, conversation_name, limit=20):
    """Pull conversation title + up to `limit` user-message texts, for
    feeding into the project-suggestion embedding comparison."""
    samples = [conversation_name] if conversation_name else []
    count = 0
    for msg in chat_messages or []:
        if count >= limit:
            break
        if not isinstance(msg, dict) or msg.get("sender") != "human":
            continue
        content = ""
        content_blocks = msg.get("content")
        if isinstance(content_blocks, list):
            parts = [b["text"] for b in content_blocks if isinstance(b, dict) and b.get("text")]
            content = "\n\n".join(parts)
        if not content:
            content = msg.get("text", "")
        if content:
            samples.append(content[:1000])
            count += 1
    return samples


def import_export(file_path, project, root_path=None, config=None, move_on_success=True):
    """Import a claude.ai JSON export file.

    If `project` is None, no import is performed - instead, ranked
    project suggestions are returned for a human to confirm before
    re-running with an explicit project.

    Returns dict: {records_imported, conversation_name, exit_code}
    (when project is None: {suggestions, conversation_name, exit_code: 2})
    """
    if root_path is None:
        root_path = get_root_path()
    if config is None:
        config = load_config(root_path)

    logger = setup_logging(root_path)
    db_path = resolve_db_path(config)
    file_path = os.path.abspath(file_path)

    # Check file exists
    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        return {"records_imported": 0, "conversation_name": "", "exit_code": 1}

    # Parse JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("Invalid JSON in %s: %s", file_path, e)
        return {"records_imported": 0, "conversation_name": "", "exit_code": 1}

    # Validate required fields
    if not isinstance(data, dict):
        logger.error("Expected JSON object in %s", file_path)
        return {"records_imported": 0, "conversation_name": "", "exit_code": 1}

    chat_messages = data.get("chat_messages")
    if not isinstance(chat_messages, list):
        logger.error("Missing or invalid chat_messages in %s", file_path)
        return {"records_imported": 0, "conversation_name": "", "exit_code": 1}

    session_id = data.get("uuid", "")
    conversation_name = data.get("name", "")
    created_at = data.get("created_at", "")
    export_model = data.get("model")

    if project is None:
        try:
            conn = connect_db(db_path)
        except SystemExit as e:
            logger.error(str(e))
            return {"suggestions": [], "conversation_name": conversation_name, "exit_code": 1}
        try:
            samples = extract_user_text_samples(chat_messages, conversation_name)
            suggestions = suggest_project(config, conn, samples)
        finally:
            conn.close()
        return {"suggestions": suggestions, "conversation_name": conversation_name, "exit_code": 2}

    # Auto-suggest tags from conversation name + first few messages
    tag_keywords = {
        "book-editing": ["edit", "polish", "chapter", "manuscript", "copyedit"],
        "memoir": ["memoir", "johnny", "goods", "maffia", "mob", "harlem", "three fingers"],
        "job-search": ["resume", "job", "interview", "career", "hiring"],
        "finance": ["money", "bank", "invest", "stock", "crypto", "tax"],
        "coding": ["python", "script", "code", "programming", "api", "github"],
        "ai-tools": ["chatgpt", "claude", "gpt", "anthropic", "gemini", "llm"],
        "tech-setup": ["linux", "fedora", "install", "setup", "config", "terminal"],
        "family": ["mom", "mother", "father", "wife", "daughter", "family"],
        "health": ["doctor", "medical", "health", "therapy", "leg"],
        "legal": ["lawyer", "legal", "court", "contract", "estate"],
        "remember-project": ["brain", "mcp", "hook", "session", "memory", "database"],
    }
    tag_text = (conversation_name or "").lower()
    # Add first few message contents for better matching
    for msg in (chat_messages or [])[:5]:
        if isinstance(msg, dict):
            msg_content = ""
            content_blocks = msg.get("content")
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("text"):
                        msg_content += block["text"][:200] + " "
            if not msg_content:
                msg_content = msg.get("text", "")[:200]
            tag_text += " " + msg_content.lower()
    suggested_tags = ", ".join(
        tag for tag, kws in tag_keywords.items()
        if any(kw in tag_text for kw in kws)
    )[:3 * 30]  # max 3 tags

    if not session_id:
        logger.error("Missing uuid in export %s", file_path)
        return {"records_imported": 0, "conversation_name": conversation_name, "exit_code": 1}

    # Connect to DB
    try:
        conn = connect_db(db_path)
    except SystemExit as e:
        logger.error(str(e))
        return {"records_imported": 0, "conversation_name": conversation_name, "exit_code": 1}

    try:
        # Check ingest log
        cur = conn.execute(
            "SELECT file_path FROM sys_ingest_log WHERE file_path = ?", (file_path,)
        )
        if cur.fetchone():
            logger.info("Already imported, skipping: %s", file_path)
            return {"records_imported": 0, "conversation_name": conversation_name, "exit_code": 0}

        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Create session row
        conn.execute(
            """INSERT OR IGNORE INTO sys_sessions
               (session_id, project, started_at, model, source, created_at, notes, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, project, created_at, export_model, "claude_ai_import", now,
             f"Claude.ai import: {conversation_name}", suggested_tags),
        )

        # Import messages
        records_imported = 0
        for msg in chat_messages:
            if not isinstance(msg, dict):
                continue

            msg_uuid = msg.get("uuid")
            if not msg_uuid:
                continue

            sender = msg.get("sender", "")
            type_role = SENDER_MAP.get(sender)
            if not type_role:
                logger.warning("Unknown sender '%s' in %s, skipping", sender, file_path)
                continue

            msg_type, role = type_role
            # Content: prefer content[] blocks (newer format), fall back to text field
            content = ""
            content_blocks = msg.get("content")
            if isinstance(content_blocks, list):
                parts = []
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("text"):
                        parts.append(block["text"])
                content = "\n\n".join(parts)
            if not content:
                content = msg.get("text", "")
            timestamp = msg.get("created_at", "")
            parent_uuid = msg.get("parent_message_uuid")
            raw_json = json.dumps(msg)

            changes_before = conn.total_changes
            cur = conn.execute(
                """INSERT OR IGNORE INTO transcripts
                   (session_id, project, uuid, parent_uuid, type, role,
                    content, model, timestamp, is_subagent,
                    source_file, raw_json, created_at, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, project, msg_uuid, parent_uuid, msg_type, role,
                 content, export_model, timestamp, 0,
                 file_path, raw_json, now, "claude_ai"),
            )
            if conn.total_changes > changes_before:
                records_imported += 1
                if SEMANTIC_AVAILABLE:
                    embed_message(config, conn, cur.lastrowid, content, logger)

        # Update session message count
        total = conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE sys_sessions SET message_count = ? WHERE session_id = ?",
            (total, session_id),
        )

        # Record in ingest log
        file_size = os.path.getsize(file_path)
        conn.execute(
            """INSERT INTO sys_ingest_log
               (file_path, file_size, file_type, records_imported, ingested_at)
               VALUES (?, ?, ?, ?, ?)""",
            (file_path, file_size, "claude_ai_import", records_imported, now),
        )

        conn.commit()

        # Move file to imports/completed/
        if move_on_success and records_imported > 0:
            completed_dir = os.path.join(root_path, "imports", "completed")
            os.makedirs(completed_dir, exist_ok=True)
            dest = os.path.join(completed_dir, os.path.basename(file_path))
            # Avoid overwriting if filename exists
            if os.path.exists(dest):
                base, ext = os.path.splitext(os.path.basename(file_path))
                dest = os.path.join(completed_dir, f"{base}_{session_id[:8]}{ext}")
            shutil.move(file_path, dest)
            logger.info("Moved %s -> %s", file_path, dest)

        logger.info("Imported %d messages from '%s' -> project %s",
                     records_imported, conversation_name, project)

        return {"records_imported": records_imported,
                "conversation_name": conversation_name, "exit_code": 0}

    except Exception as e:
        logger.error("Error importing %s: %s", file_path, e)
        return {"records_imported": 0, "conversation_name": conversation_name, "exit_code": 1}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# On the fly project creation (--new-project)
# ---------------------------------------------------------------------------

def load_add_project_module(root_path):
    """Dynamically load add-project.py (hyphenated filename, not a valid
    module name for a plain import) so its create_project()/generate_prefix()
    can be reused instead of duplicating project-creation logic here."""
    add_project_path = os.path.join(root_path, "scripts", "add-project.py")
    spec = importlib.util.spec_from_file_location("add_project", add_project_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_new_project(root_path, folder_name, prefix=None, label=None):
    """Create a project on the fly and return its prefix, or raise ValueError."""
    name_pattern = re.compile(r'^[a-z][a-z0-9-]*[a-z0-9]$|^[a-z]$')
    name = folder_name.strip().lower()
    if not name_pattern.match(name):
        raise ValueError(f"Invalid project folder name '{name}' - use lowercase letters, numbers, hyphens, starting with a letter.")

    config = load_config(root_path)
    config_path = pathlib.Path(root_path) / "config.yaml"
    add_project = load_add_project_module(root_path)

    existing_prefixes = {p["prefix"] for p in config.get("projects", [])}
    final_prefix = prefix or add_project.generate_prefix(name, existing_prefixes)
    final_label = label or name.replace("-", " ").title()

    add_project.create_project(config, config_path, name, final_prefix, final_label)
    return name, final_prefix, final_label


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Import claude.ai JSON export into claude_remembers DB")
    parser.add_argument("json_file", help="Path to claude.ai JSON export file")
    parser.add_argument("--project", default=None, help="Project prefix to assign; omit to get a ranked suggestion instead")
    parser.add_argument("--new-project", default=None, metavar="FOLDER_NAME",
                         help="Create a new project folder on the fly and import into it. Mutually exclusive with --project.")
    parser.add_argument("--prefix", default=None,
                         help="Database prefix for --new-project (2-4 letters, permanent). Auto-generated from the folder name if omitted.")
    parser.add_argument("--label", default=None,
                         help="Display name for --new-project. Defaults to a title-cased version of the folder name.")
    args = parser.parse_args()

    if args.new_project and args.project:
        parser.error("--new-project and --project are mutually exclusive - creating a project already decides which one you import into.")

    if args.new_project:
        try:
            name, prefix, label = create_new_project(get_root_path(), args.new_project, args.prefix, args.label)
        except ValueError as e:
            print(f"Could not create project: {e}")
            sys.exit(1)
        print(f"Created new project '{name}' (prefix: {prefix}, label: {label})")
        args.project = prefix

    result = import_export(args.json_file, args.project)

    if args.project is None and result["exit_code"] == 2:
        suggestions = result.get("suggestions", [])
        good_match = suggestions and suggestions[0][1] >= CONFIDENCE_THRESHOLD
        if suggestions:
            print(f"Suggestions for '{result['conversation_name']}':")
            for prefix, score in suggestions:
                flag = "" if score >= CONFIDENCE_THRESHOLD else "  (weak match)"
                print(f"  {prefix}  (similarity {score:.2f}){flag}")
        else:
            print(f"No suggestion available for '{result['conversation_name']}' (not enough embedded content yet, or no projects exist yet).")

        slug = slugify(result['conversation_name']) if result['conversation_name'] else None
        new_project_cmd = (
            f'python3 {sys.argv[0]} "{args.json_file}" --new-project {slug or "<folder-name>"}'
        )
        if good_match:
            print(f"\nUse --project <prefix> for one of the above, or if none actually fit, create a new")
            print(f"project from this conversation's own content instead:\n  {new_project_cmd}")
        else:
            print(f"\nNone of these look like a confident match - consider creating a new project from this")
            print(f"conversation's own content instead:\n  {new_project_cmd}")
            print(f"(or use --project <prefix> to force one of the weak matches above anyway)")
        print("(add --prefix/--label to the --new-project command to override the auto-generated prefix or display name)")
    elif args.project is None:
        print(f"Could not read '{args.json_file}' - nothing to suggest a project for.")
    else:
        print(f"Imported {result['records_imported']} messages from '{result['conversation_name']}' -> project {args.project}")

    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()
