#!/usr/bin/env python3
"""
import_claude_ai.py - Import claude.ai conversations into claude_remembers.

Two input shapes, two matching modes:

  Single conversation (e.g. the "AI Chat Exporter" Chrome extension's export
  of one open conversation -- a JSON object with a top-level chat_messages
  list):
    python3 import_claude_ai.py <json_file>                 # prints title +
                                                               # about + tags,
                                                               # no import
    python3 import_claude_ai.py <json_file> --folder <folder-name>
                                                               # imports into
                                                               # that folder,
                                                               # creating it
                                                               # if it's new

  Bulk account export (Settings -> Privacy -> Export Data -> conversations.json,
  a JSON list of every conversation):
    python3 import_claude_ai.py --scan <file_or_dir>
    python3 import_claude_ai.py --import <path> --map claude_ai_import_map.xlsx [--dry-run]

Nothing auto-assigns a project. You type the folder name yourself -- an
existing one or a brand-new name (created on import). Leave it blank (bulk
mode) to use the default project.

Exit codes (single-conversation mode): 0 = success, 1 = error,
2 = no --folder given (title/about/tags printed instead)
"""

import argparse
import csv
import datetime
import json
import logging
import os
import pathlib
import shutil
import socket
import sqlite3
import sys
import time
from pathlib import Path

import yaml

from config_utils import resolve_db_path
from import_common import green, yellow, red, suggest_tags, build_about, load_add_project_module, resolve_or_create_project
from write_exchange import embed_message, SEMANTIC_AVAILABLE

# ---------------------------------------------------------------------------
# Config / Logging / DB
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DEFAULT_PROJECT = "gen"

SENDER_MAP = {
    "human": ("user", "user"),
    "assistant": ("assistant", "assistant"),
}


def get_root_path():
    return str(ROOT_DIR)


def load_config(root_path=None):
    config_path = os.path.join(root_path or str(ROOT_DIR), "config.yaml")
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


def print_project_reference(db_path):
    """Print existing projects (folder_name/prefix/label) so a human can
    pick a valid one, or decide to type a brand-new name instead."""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT folder_name, prefix, label FROM project_registry ORDER BY folder_name"
        ).fetchall()
        conn.close()
    except Exception:
        rows = []
    if rows:
        print("Existing projects:")
        for folder_name, prefix, label in rows:
            print(f"  {folder_name}  (prefix: {prefix}, label: {label})")
    else:
        print("No projects exist yet.")


# ---------------------------------------------------------------------------
# Shared conversation parsing / insert logic
# ---------------------------------------------------------------------------

def extract_messages(conversation):
    """Extract messages from a claude.ai conversation's chat_messages list.

    Returns list of (sender, content, timestamp, uuid, parent_uuid) tuples,
    in the order claude.ai already stored them.
    """
    chat_messages = conversation.get("chat_messages")
    if not isinstance(chat_messages, list):
        return []

    messages = []
    for msg in chat_messages:
        if not isinstance(msg, dict):
            continue
        sender = msg.get("sender", "")
        if sender not in SENDER_MAP:
            continue

        content = ""
        content_blocks = msg.get("content")
        if isinstance(content_blocks, list):
            parts = [b["text"] for b in content_blocks if isinstance(b, dict) and b.get("text")]
            content = "\n\n".join(parts)
        if not content:
            content = msg.get("text", "")
        if not content:
            continue

        msg_uuid = msg.get("uuid")
        if not msg_uuid:
            continue

        messages.append((sender, content, msg.get("created_at", ""), msg_uuid, msg.get("parent_message_uuid")))

    return messages


def import_conversation(conn, config, conversation, project, source_file, now, tags="", logger=None):
    """Insert one conversation (session row + transcript rows + embeddings).
    Shared by both single-conversation mode and the bulk --import loop.

    Returns records_imported (0 if the session already existed).
    """
    conv_id = conversation.get("uuid", "")
    title = conversation.get("name", "") or "(untitled)"
    created_at = conversation.get("created_at", "")
    messages = extract_messages(conversation)

    existing = conn.execute("SELECT 1 FROM sys_sessions WHERE session_id = ?", (conv_id,)).fetchone()
    if existing:
        return 0

    conn.execute(
        """INSERT INTO sys_sessions
           (session_id, project, started_at, source, created_at, notes, tags)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (conv_id, project, created_at, "claude_ai_import", now,
         f"Claude.ai import: {title}", tags),
    )

    records_imported = 0
    for sender, content, timestamp, msg_uuid, parent_uuid in messages:
        msg_type, role = SENDER_MAP[sender]
        cur = conn.execute(
            """INSERT OR IGNORE INTO transcripts
               (session_id, project, uuid, parent_uuid, type, role,
                content, timestamp, is_subagent, source_file, created_at, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (conv_id, project, msg_uuid, parent_uuid, msg_type, role,
             content, timestamp, 0, str(source_file), now, "claude_ai"),
        )
        if cur.rowcount:
            records_imported += 1
            if SEMANTIC_AVAILABLE:
                embed_message(config, conn, cur.lastrowid, content, logger or logging.getLogger("import_claude_ai"))

    conn.execute(
        "UPDATE sys_sessions SET message_count = ? WHERE session_id = ?",
        (records_imported, conv_id),
    )
    return records_imported


# ---------------------------------------------------------------------------
# Single-conversation mode (Chrome-extension export)
# ---------------------------------------------------------------------------

def single_conversation_import(file_path, folder, root_path=None, config=None, move_on_success=True):
    """Import one claude.ai conversation JSON file (a dict, not a list).

    If `folder` is None, nothing is imported - title/about/tags + the
    existing-projects list are returned instead, for a human to decide.

    Returns dict: {records_imported, conversation_name, exit_code}
    (when folder is None: {about, tags, conversation_name, exit_code: 2})
    """
    if root_path is None:
        root_path = get_root_path()
    if config is None:
        config = load_config(root_path)

    logger = setup_logging(root_path)
    db_path = resolve_db_path(config)
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        return {"records_imported": 0, "conversation_name": "", "exit_code": 1}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("Invalid JSON in %s: %s", file_path, e)
        return {"records_imported": 0, "conversation_name": "", "exit_code": 1}

    if isinstance(data, list):
        logger.error("%s contains a list of conversations, not one - use --scan instead", file_path)
        print(red(f"{file_path} is a bulk export (a list of conversations), not a single conversation."))
        print(f"Use: python3 {sys.argv[0]} --scan {file_path}")
        return {"records_imported": 0, "conversation_name": "", "exit_code": 1}

    if not isinstance(data, dict):
        logger.error("Expected a JSON object in %s", file_path)
        return {"records_imported": 0, "conversation_name": "", "exit_code": 1}

    chat_messages = data.get("chat_messages")
    if not isinstance(chat_messages, list):
        logger.error("Missing or invalid chat_messages in %s", file_path)
        return {"records_imported": 0, "conversation_name": "", "exit_code": 1}

    conversation_name = data.get("name", "")

    if folder is None:
        messages = extract_messages(data)
        first_human_text = next((content for sender, content, *_ in messages if sender == "human" and content), "")
        about = build_about(data.get("summary"), first_human_text, conversation_name)
        search_text = (conversation_name + " " + first_human_text[:500]).lower()
        tags = suggest_tags(search_text)
        return {"about": about, "tags": tags, "conversation_name": conversation_name, "exit_code": 2}

    if not data.get("uuid"):
        logger.error("Missing uuid in export %s", file_path)
        return {"records_imported": 0, "conversation_name": conversation_name, "exit_code": 1}

    try:
        conn = connect_db(db_path)
    except SystemExit as e:
        logger.error(str(e))
        return {"records_imported": 0, "conversation_name": conversation_name, "exit_code": 1}

    try:
        cur = conn.execute("SELECT file_path FROM sys_ingest_log WHERE file_path = ?", (file_path,))
        if cur.fetchone():
            logger.info("Already imported, skipping: %s", file_path)
            return {"records_imported": 0, "conversation_name": conversation_name, "exit_code": 0}

        messages = extract_messages(data)
        first_human_text = next((content for sender, content, *_ in messages if sender == "human" and content), "")
        search_text = (conversation_name + " " + first_human_text[:500]).lower()
        tags = suggest_tags(search_text)

        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        add_project = load_add_project_module()
        project, label, created = resolve_or_create_project(config, pathlib.Path(root_path) / "config.yaml", folder, add_project)
        if created:
            print(f"{green('Created new project')} '{folder}' (prefix: {project}, label: {label})")

        records_imported = import_conversation(conn, config, data, project, file_path, now, tags, logger)

        file_size = os.path.getsize(file_path)
        conn.execute(
            """INSERT INTO sys_ingest_log
               (file_path, file_size, file_type, records_imported, ingested_at)
               VALUES (?, ?, ?, ?, ?)""",
            (file_path, file_size, "claude_ai_import", records_imported, now),
        )
        conn.commit()

        if move_on_success and records_imported > 0:
            completed_dir = os.path.join(root_path, "imports", "completed")
            os.makedirs(completed_dir, exist_ok=True)
            dest = os.path.join(completed_dir, os.path.basename(file_path))
            if os.path.exists(dest):
                base, ext = os.path.splitext(os.path.basename(file_path))
                dest = os.path.join(completed_dir, f"{base}_{data.get('uuid', '')[:8]}{ext}")
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
# Bulk mode: SCAN
# ---------------------------------------------------------------------------

def find_conversations_file(path):
    """Accept either a direct path to conversations.json, or a directory
    containing one (however the export zip was extracted)."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        candidate = p / "conversations.json"
        if candidate.exists():
            return candidate
    print(red(f"conversations.json not found at {path}"))
    sys.exit(1)


def load_conversations(path):
    """Load the list of conversations from claude.ai's conversations.json."""
    f = find_conversations_file(path)
    with open(f, encoding="utf-8") as fh:
        convos = json.load(fh)
    if not isinstance(convos, list):
        print(red(f"Expected a JSON list of conversations in {f}"))
        sys.exit(1)
    print(f"  Loaded {len(convos)} conversations from {f.name}")
    return convos, f


def cmd_scan(args):
    """Scan a claude.ai conversations.json export and generate mapping xlsx."""
    print(f"\nScanning claude.ai export: {args.path}")
    print()

    convos, _ = load_conversations(args.path)

    config = load_config()
    db_path = resolve_db_path(config)
    existing_ids = set()
    try:
        conn = sqlite3.connect(db_path)
        for row in conn.execute("SELECT session_id FROM sys_sessions WHERE source = 'claude_ai_import'"):
            existing_ids.add(row[0])
        conn.close()
    except Exception:
        pass  # DB not available, nothing excluded as already-imported

    rows = []
    already_imported = 0
    for c in convos:
        conv_id = c.get("uuid", "")
        if conv_id in existing_ids:
            already_imported += 1
            continue

        title = c.get("name") or "(untitled)"
        created_at = c.get("created_at", "")
        date_str = created_at[:10] if created_at else "unknown"
        messages = extract_messages(c)
        msg_count = len(messages)

        first_human_text = next((content for sender, content, *_ in messages if sender == "human" and content), "")
        about = build_about(c.get("summary"), first_human_text, title)
        search_text = (title + " " + first_human_text[:500]).lower()
        suggested_tags = suggest_tags(search_text)

        rows.append({
            "conversation_id": conv_id,
            "date": date_str,
            "messages": msg_count,
            "title": title,
            "about": about,
            "tags": suggested_tags,
            "folder": "",
        })

    rows.sort(key=lambda x: x["date"])

    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    output_path = Path(args.output) if args.output else ROOT_DIR / "imports" / "claude_ai_import_map.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()

    ws1 = wb.active
    ws1.title = "Conversations"
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    edit_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    headers = ["conversation_id", "date", "messages", "title", "about", "tags", "folder"]
    for col, h in enumerate(headers, 1):
        cell = ws1.cell(row=1, column=col, value=h)
        cell.fill = header_fill
        cell.font = header_font

    for row_num, r in enumerate(rows, 2):
        ws1.cell(row=row_num, column=1, value=r["conversation_id"])
        ws1.cell(row=row_num, column=2, value=r["date"])
        ws1.cell(row=row_num, column=3, value=r["messages"])
        ws1.cell(row=row_num, column=4, value=r["title"])
        ws1.cell(row=row_num, column=5, value=r["about"])
        cell_tags = ws1.cell(row=row_num, column=6, value=r["tags"])
        cell_tags.fill = edit_fill
        cell_folder = ws1.cell(row=row_num, column=7, value=r["folder"])
        cell_folder.fill = edit_fill

    ws1.column_dimensions["A"].width = 40
    ws1.column_dimensions["B"].width = 12
    ws1.column_dimensions["C"].width = 10
    ws1.column_dimensions["D"].width = 30
    ws1.column_dimensions["E"].width = 70
    ws1.column_dimensions["F"].width = 30
    ws1.column_dimensions["G"].width = 25

    # --- Tab 2: Project Reference ---
    ws2 = wb.create_sheet("Project Reference")
    ws2.cell(row=1, column=1, value="EXISTING PROJECTS -- TYPE A FOLDER_NAME INTO THE FOLDER COLUMN").font = Font(bold=True, size=14)
    ws2.merge_cells("A1:D1")

    ref_headers = ["prefix", "folder_name", "label", "status"]
    for col, h in enumerate(ref_headers, 1):
        cell = ws2.cell(row=3, column=col, value=h)
        cell.fill = header_fill
        cell.font = header_font

    try:
        conn = sqlite3.connect(db_path)
        proj_rows = conn.execute("SELECT prefix, folder_name, label, status FROM project_registry ORDER BY folder_name").fetchall()
        for row_num, r in enumerate(proj_rows, 4):
            for col, val in enumerate(r, 1):
                ws2.cell(row=row_num, column=col, value=val)
        last_row = 4 + len(proj_rows)
        conn.close()
    except Exception:
        last_row = 4

    ws2.column_dimensions["A"].width = 10
    ws2.column_dimensions["B"].width = 25
    ws2.column_dimensions["C"].width = 50
    ws2.column_dimensions["D"].width = 10

    ws2.cell(row=last_row + 1, column=1, value="").font = Font(bold=True)
    ws2.cell(row=last_row + 2, column=1, value="INSTRUCTIONS:").font = Font(bold=True)
    ws2.cell(row=last_row + 3, column=1, value="1. Go to the Conversations tab")
    ws2.cell(row=last_row + 4, column=1, value="2. Read the TITLE and ABOUT columns to remember what each one was")
    ws2.cell(row=last_row + 5, column=1, value="3. Type a folder name into the yellow FOLDER column for any you want")
    ws2.cell(row=last_row + 6, column=1, value="   to keep: an existing folder_name from this tab, or a brand-new name")
    ws2.cell(row=last_row + 7, column=1, value="   (that creates the project automatically on import). Use the SAME")
    ws2.cell(row=last_row + 8, column=1, value="   new name across multiple rows to group them into one project.")
    ws2.cell(row=last_row + 9, column=1, value="   Leave FOLDER blank to use the default project.")
    ws2.cell(row=last_row + 10, column=1, value="4. Review the yellow TAGS column -- edit, add, or remove tags")
    ws2.cell(row=last_row + 11, column=1, value="5. Delete entire rows for conversations you do NOT want imported")
    ws2.cell(row=last_row + 12, column=1, value="6. Save the file")
    ws2.cell(row=last_row + 13, column=1, value="7. Tell Claude you are ready for dry-run")

    wb.save(output_path)

    tag_counts = {}
    total_msgs = 0
    for r in rows:
        total_msgs += r["messages"]
        for tag in [t.strip() for t in r["tags"].split(",") if t.strip()]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"\n{green('Mapping xlsx generated:')} {output_path}")
    print(f"\n  Conversations: {len(rows)} new ({already_imported} already imported, excluded from the sheet)")
    print(f"  Total messages: {total_msgs}")
    print(f"\n  Tag suggestions ({len(tag_counts)} unique tags):")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"    {tag}: {count} conversations")

    print(f"\n{yellow('NEXT STEP:')} Review and edit {output_path.name}")
    print(f"  - Read TITLE/ABOUT, type a destination folder for each row you want to keep")
    print(f"  - Delete rows you don't want imported")
    print(f"  - Then run: python3 scripts/import_claude_ai.py --import {args.path} --map {output_path} --dry-run")


# ---------------------------------------------------------------------------
# Bulk mode: IMPORT
# ---------------------------------------------------------------------------

def cmd_import(args):
    """Import conversations using the reviewed mapping xlsx/csv."""
    if not args.map:
        print(red("--map <xlsx_file> is required for import mode"))
        sys.exit(1)

    map_path = Path(args.map)
    if not map_path.exists():
        print(red(f"Mapping file not found: {map_path}"))
        print(f"Run --scan first to generate it.")
        sys.exit(1)

    mapping = {}  # conv_id -> {"folder": ..., "tags": ...}
    if map_path.suffix == ".xlsx":
        from openpyxl import load_workbook
        wb = load_workbook(map_path)
        ws = wb["Conversations"]
        headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
        for row in range(2, ws.max_row + 1):
            conv_id = ws.cell(row=row, column=1).value
            if not conv_id:
                continue
            folder = (ws.cell(row=row, column=headers.index("folder") + 1).value or "").strip()
            tags = ws.cell(row=row, column=headers.index("tags") + 1).value or ""
            mapping[conv_id] = {"folder": folder, "tags": tags}
    else:
        with open(map_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row["conversation_id"]] = {
                    "folder": (row.get("folder") or "").strip(),
                    "tags": row.get("tags", ""),
                }

    print(f"\nMapping file: {map_path.name} ({len(mapping)} conversations)")

    print(f"Loading conversations from: {args.path}")
    convos, source_file = load_conversations(args.path)

    to_import = []
    skipped_not_in_map = 0
    skipped_zero_msgs = 0

    for c in convos:
        conv_id = c.get("uuid", "")
        if conv_id not in mapping:
            skipped_not_in_map += 1
            continue

        messages = extract_messages(c)
        if not messages:
            skipped_zero_msgs += 1
            continue

        to_import.append({
            "conversation": c,
            "messages": messages,
            "folder": mapping[conv_id]["folder"],
            "tags": mapping[conv_id].get("tags", ""),
            "conv_id": conv_id,
        })

    print(f"\n  To import: {len(to_import)} conversations")
    print(f"  Skipped (not in mapping): {skipped_not_in_map}")
    print(f"  Skipped (0 messages): {skipped_zero_msgs}")

    config = load_config()
    db_path = resolve_db_path(config)

    if args.dry_run:
        print(f"\n{yellow('DRY RUN')} -- showing what would be imported:\n")
        existing_folders = {p["folder_name"] for p in config.get("projects", [])}
        folder_summary = {}
        for item in to_import:
            f = item["folder"] or f"(default: {DEFAULT_PROJECT})"
            if f not in folder_summary:
                folder_summary[f] = {"count": 0, "messages": 0}
            folder_summary[f]["count"] += 1
            folder_summary[f]["messages"] += len(item["messages"])

        print(f"  {'Folder':<30} {'Conversations':>14} {'Messages':>10}")
        print(f"  {'-'*30} {'-'*14} {'-'*10}")
        for f in sorted(folder_summary.keys()):
            s = folder_summary[f]
            marker = "  (will be created)" if f and f not in existing_folders and not f.startswith("(default") else ""
            print(f"  {f:<30} {s['count']:>14} {s['messages']:>10}{marker}")
        total_msgs = sum(s["messages"] for s in folder_summary.values())
        print(f"  {'TOTAL':<30} {len(to_import):>14} {total_msgs:>10}")

        print(f"\n  Sample conversations:")
        for item in to_import[:10]:
            c = item["conversation"]
            title = (c.get("name") or "(untitled)")[:60]
            label = item["folder"] or f"default: {DEFAULT_PROJECT}"
            print(f"    [{label}] {len(item['messages']):>3} msgs | {title}")
        if len(to_import) > 10:
            print(f"    ... and {len(to_import) - 10} more")

        print(f"\n{yellow('To execute:')} remove --dry-run flag")
        return

    # Actually import
    conn = sqlite3.connect(db_path)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    import_logger = logging.getLogger("import_claude_ai")

    folder_names = {item["folder"] for item in to_import if item["folder"]}
    folder_prefixes = {}
    if folder_names:
        add_project = load_add_project_module()
        for folder_name in sorted(folder_names):
            prefix, label, created = resolve_or_create_project(config, ROOT_DIR / "config.yaml", folder_name, add_project)
            folder_prefixes[folder_name] = prefix
            if created:
                print(f"  {green('Created new project')} '{folder_name}' (prefix: {prefix}, label: {label})")

    imported = 0
    skipped_dup = 0
    total_transcripts = 0

    for item in to_import:
        project = folder_prefixes[item["folder"]] if item["folder"] else DEFAULT_PROJECT
        records_imported = import_conversation(
            conn, config, item["conversation"], project, source_file, now,
            tags=item.get("tags", ""), logger=import_logger,
        )
        if records_imported:
            imported += 1
            total_transcripts += records_imported
        else:
            skipped_dup += 1

    conn.commit()
    conn.close()

    print(f"\n{green('Import complete:')}")
    print(f"  Conversations imported: {imported}")
    print(f"  Transcripts created: {total_transcripts}")
    print(f"  Skipped (already imported): {skipped_dup}")
    print(f"  Source tag: 'claude_ai' (transcripts), 'claude_ai_import' (sessions)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Import claude.ai conversations into claude_remembers DB")
    parser.add_argument("json_file", nargs="?", help="Path to a single-conversation claude.ai JSON export")
    parser.add_argument("--folder", default=None, metavar="FOLDER_NAME",
                         help="Destination folder for single-conversation mode; creates it if new")
    parser.add_argument("--scan", dest="scan_path", metavar="FILE_OR_DIR",
                        help="Scan a bulk conversations.json export (or its containing dir) and generate mapping xlsx")
    parser.add_argument("--import", dest="import_path", metavar="FILE_OR_DIR",
                        help="Import a bulk conversations.json export using mapping xlsx")
    parser.add_argument("--map", metavar="XLSX",
                        help="Mapping xlsx file (required for --import)")
    parser.add_argument("--output", metavar="FILE",
                        help="Output path for mapping xlsx (default: imports/claude_ai_import_map.xlsx)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be imported without touching the database")
    args = parser.parse_args()

    if args.scan_path:
        args.path = args.scan_path
        cmd_scan(args)
        return
    if args.import_path:
        args.path = args.import_path
        cmd_import(args)
        return

    if not args.json_file:
        print("Single conversation: python3 import_claude_ai.py <json_file> [--folder <folder-name>]")
        print("Bulk export:         python3 import_claude_ai.py --scan <conversations.json>")
        sys.exit(1)

    result = single_conversation_import(args.json_file, args.folder)

    if args.folder is None and result["exit_code"] == 2:
        print(f"Title: {result['conversation_name'] or '(untitled)'}")
        print(f"About: {result['about']}")
        if result["tags"]:
            print(f"Suggested tags: {result['tags']}")
        print()
        db_path = resolve_db_path(load_config())
        print_project_reference(db_path)
        print(f"\nRe-run with --folder <name> (existing or brand-new) to import:")
        print(f'  python3 {sys.argv[0]} "{args.json_file}" --folder <folder-name>')
    elif args.folder is None:
        print(f"Could not read '{args.json_file}' - nothing to show.")
    else:
        print(f"Imported {result['records_imported']} messages from '{result['conversation_name']}' -> folder {args.folder}")

    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()
