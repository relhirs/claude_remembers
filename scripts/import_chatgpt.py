#!/usr/bin/env python3
"""
import_chatgpt.py - Import ChatGPT conversation exports into the memory database.

Supports the full data export from OpenAI (Settings → Data → Export).
Text conversations only -- audio, images, and system messages are skipped.

Three modes:
  --scan <dir>     Generate mapping xlsx with a descriptive "about" blurb +
                    suggested tags, for you to type a destination folder into
  --import <dir>   Import conversations using the reviewed mapping xlsx
  --dry-run        Preview import without touching the database

Nothing auto-assigns a project. You type the folder name yourself in the
`folder` column -- an existing one (see the Project Reference tab) or a
brand-new name, which gets created on import. Leave it blank to use the
default project.

Workflow:
  1. Export from ChatGPT (Settings → Data → Export → wait for email → download zip)
  2. Extract the zip to a folder
  3. Run: python3 scripts/import_chatgpt.py --scan imports/chatgpt-export/
  4. Review chatgpt_import_map.xlsx -- type folder names, edit tags, delete unwanted rows
  5. Run: python3 scripts/import_chatgpt.py --import imports/chatgpt-export/ --map chatgpt_import_map.xlsx --dry-run
  6. When satisfied: python3 scripts/import_chatgpt.py --import imports/chatgpt-export/ --map chatgpt_import_map.xlsx
"""

import argparse
import csv
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import resolve_db_path
from import_common import green, yellow, red, suggest_tags, build_about, load_add_project_module, resolve_or_create_project
from write_exchange import embed_message, SEMANTIC_AVAILABLE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DEFAULT_PROJECT = "gen"

# ChatGPT roles we import (skip system and tool)
IMPORTABLE_ROLES = {"user", "assistant"}

# Content types we extract text from
TEXT_CONTENT_TYPES = {"text", "multimodal_text", "code", "execution_output",
                      "thoughts", "reasoning_recap"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_conversation_files(directory):
    """Find all conversations-NNN.json files in a directory."""
    d = Path(directory)
    files = sorted(d.glob("conversations-*.json"))
    if not files:
        # Maybe the zip was extracted flat -- check for conversations.json
        single = d / "conversations.json"
        if single.exists():
            files = [single]
    return files


def load_conversations(directory):
    """Load all conversations from JSON files in directory."""
    files = find_conversation_files(directory)
    if not files:
        print(red(f"No conversations-*.json files found in {directory}"))
        sys.exit(1)

    all_convos = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            convos = json.load(fh)
            all_convos.extend(convos)
        print(f"  Loaded {len(convos)} conversations from {f.name}")

    return all_convos


def extract_messages(conversation):
    """Extract messages from ChatGPT's tree-based mapping structure.

    Returns list of (role, content, timestamp, node_id) tuples in
    chronological order. Skips system messages, tool messages, and empty
    content.
    """
    mapping = conversation.get("mapping", {})
    if not mapping:
        return []

    messages = []
    for node_id, node in mapping.items():
        msg = node.get("message")
        if not msg:
            continue

        role = msg.get("author", {}).get("role", "")
        if role not in IMPORTABLE_ROLES:
            continue

        content_type = msg.get("content", {}).get("content_type", "")
        if content_type not in TEXT_CONTENT_TYPES:
            continue

        parts = msg.get("content", {}).get("parts", [])
        text_parts = []
        for part in parts:
            if isinstance(part, str) and part.strip():
                text_parts.append(part.strip())
            elif isinstance(part, dict):
                if "text" in part:
                    text_parts.append(part["text"].strip())

        content = "\n".join(text_parts)
        if not content:
            continue

        timestamp = msg.get("create_time", 0)
        messages.append((role, content, timestamp, node_id))

    messages.sort(key=lambda x: x[2])
    return messages


def load_config():
    """Load config.yaml. Returns (config_dict, db_path)."""
    config_path = ROOT_DIR / "config.yaml"
    if not config_path.exists():
        print(red("config.yaml not found. Run remember-setup.py first."))
        sys.exit(1)

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config, resolve_db_path(config)


# ---------------------------------------------------------------------------
# SCAN mode: generate mapping xlsx
# ---------------------------------------------------------------------------

def cmd_scan(args):
    """Scan ChatGPT export and generate mapping xlsx for review."""
    print(f"\nScanning ChatGPT export in: {args.directory}")
    print()

    convos = load_conversations(args.directory)

    config, db_path = load_config()
    existing_ids = set()
    try:
        conn = sqlite3.connect(db_path)
        for row in conn.execute("SELECT session_id FROM sys_sessions WHERE source = 'chatgpt_import'"):
            existing_ids.add(row[0].replace("chatgpt_", ""))
        conn.close()
    except Exception:
        pass  # DB not available, nothing excluded as already-imported

    rows = []
    already_imported = 0
    for c in convos:
        title = c.get("title") or "(untitled)"
        conv_id = c.get("conversation_id", c.get("id", ""))
        if conv_id in existing_ids:
            already_imported += 1
            continue

        create_time = c.get("create_time", 0)
        messages = extract_messages(c)
        msg_count = len(messages)
        date_str = datetime.fromtimestamp(create_time).strftime("%Y-%m-%d") if create_time else "unknown"

        first_user_text = next((content for role, content, _, _ in messages if role == "user" and content), "")
        about = build_about(None, first_user_text, title)
        search_text = (title + " " + first_user_text[:500]).lower()
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

    output_path = Path(args.output) if args.output else ROOT_DIR / "imports" / "chatgpt_import_map.xlsx"
    wb = Workbook()

    # --- Tab 1: Conversations ---
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
        _, db_path = load_config()
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

    # Summary
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
    print(f"  - Then run: python3 scripts/import_chatgpt.py --import {args.directory} --map {output_path} --dry-run")


# ---------------------------------------------------------------------------
# IMPORT mode: read mapping xlsx/csv and ingest
# ---------------------------------------------------------------------------

def cmd_import(args):
    """Import conversations using reviewed mapping xlsx/csv."""
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

    print(f"Loading conversations from: {args.directory}")
    convos = load_conversations(args.directory)

    to_import = []
    skipped_not_in_map = 0
    skipped_zero_msgs = 0

    for c in convos:
        conv_id = c.get("conversation_id", c.get("id", ""))
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

    config, db_path = load_config()

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
            title = (c.get("title") or "(untitled)")[:60]
            label = item["folder"] or f"default: {DEFAULT_PROJECT}"
            print(f"    [{label}] {len(item['messages']):>3} msgs | {title}")
        if len(to_import) > 10:
            print(f"    ... and {len(to_import) - 10} more")

        print(f"\n{yellow('To execute:')} remove --dry-run flag")
        return

    # Actually import
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Resolve every distinct folder name to a prefix up front (creating new
    # projects as needed) so the main loop below can just look them up.
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
        c = item["conversation"]
        conv_id = item["conv_id"]
        project = folder_prefixes[item["folder"]] if item["folder"] else DEFAULT_PROJECT
        messages = item["messages"]
        title = c.get("title", "") or "(untitled)"
        create_time = c.get("create_time", 0)

        existing = conn.execute(
            "SELECT 1 FROM sys_sessions WHERE session_id = ?",
            (f"chatgpt_{conv_id}",)
        ).fetchone()

        if existing:
            skipped_dup += 1
            continue

        tags = item.get("tags", "")
        started = datetime.fromtimestamp(create_time).strftime("%Y-%m-%dT%H:%M:%SZ") if create_time else now
        ended = datetime.fromtimestamp(messages[-1][2]).strftime("%Y-%m-%dT%H:%M:%SZ") if messages[-1][2] else now

        conn.execute(
            """INSERT INTO sys_sessions
               (session_id, project, started_at, ended_at, cwd, claude_version, model, source, message_count, created_at, notes, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"chatgpt_{conv_id}", project, started, ended,
             "chatgpt", "chatgpt", "gpt", "chatgpt_import",
             len(messages), now, f"ChatGPT import: {title}", tags)
        )

        for i, (role, content, timestamp, node_id) in enumerate(messages):
            ts = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ") if timestamp else now
            sender = "human" if role == "user" else "assistant"

            cur = conn.execute(
                """INSERT INTO transcripts
                   (session_id, project, uuid, type, role, content, timestamp, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (f"chatgpt_{conv_id}", project, node_id,
                 "text", sender, content, ts, "chatgpt", now)
            )
            total_transcripts += 1
            if SEMANTIC_AVAILABLE:
                embed_message(config, conn, cur.lastrowid, content, logging.getLogger("import_chatgpt"))

        imported += 1

    conn.commit()
    conn.close()

    print(f"\n{green('Import complete:')}")
    print(f"  Conversations imported: {imported}")
    print(f"  Transcripts created: {total_transcripts}")
    print(f"  Skipped (already imported): {skipped_dup}")
    print(f"  Source tag: 'chatgpt' (transcripts), 'chatgpt_import' (sessions)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Import ChatGPT conversations into the memory database"
    )
    parser.add_argument("--scan", dest="scan_dir", metavar="DIR",
                        help="Scan ChatGPT export directory and generate mapping xlsx")
    parser.add_argument("--import", dest="import_dir", metavar="DIR",
                        help="Import conversations using mapping xlsx")
    parser.add_argument("--map", metavar="XLSX",
                        help="Mapping xlsx file (required for --import)")
    parser.add_argument("--output", metavar="FILE",
                        help="Output path for mapping xlsx (default: chatgpt_import_map.xlsx)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be imported without touching the database")

    args = parser.parse_args()

    if args.scan_dir:
        args.directory = args.scan_dir
        cmd_scan(args)
    elif args.import_dir:
        args.directory = args.import_dir
        cmd_import(args)
    else:
        parser.print_help()
        print(f"\n{yellow('Workflow:')}")
        print(f"  1. python3 scripts/import_chatgpt.py --scan <chatgpt-export-dir>")
        print(f"  2. Review chatgpt_import_map.xlsx (type folder names, edit tags)")
        print(f"  3. python3 scripts/import_chatgpt.py --import <dir> --map chatgpt_import_map.xlsx --dry-run")
        print(f"  4. python3 scripts/import_chatgpt.py --import <dir> --map chatgpt_import_map.xlsx")


if __name__ == "__main__":
    main()
