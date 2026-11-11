#!/usr/bin/env python3
"""
backfill_metrics.py - Fill tool_events and api_usage from history already in the database.

New messages record these automatically (write_exchange.py / ingest_jsonl.py).
This covers everything captured before the measurement layer existed, by
re-reading each stored message's raw_json. It is manual on purpose: it parses
the whole raw_json column once, which is too heavy to run at session start.

Safe to re-run: every write is an upsert keyed on tool_use_id / message_id.
It only ADDS rows to the metrics tables. It never edits or deletes anything in
transcripts (raw_json is untouched).

Usage:
    python3 scripts/backfill_metrics.py --dry-run   # count only, write nothing
    python3 scripts/backfill_metrics.py             # backfill, then verify

Exit codes: 0 = done and verified (or dry run), 1 = verification found a gap
"""

import argparse
import json
import os
import pathlib
import sqlite3
import sys
import time

import yaml

from config_utils import resolve_db_path
from metrics_utils import (
    ensure_metrics_schema,
    extract_api_usage,
    extract_tool_blocks,
    write_capture_metrics,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
BATCH = 2000

SELECT_ROWS = """
    SELECT id, session_id, project, timestamp, is_subagent, raw_json
    FROM transcripts
    WHERE type IN ('assistant', 'user') AND raw_json IS NOT NULL
      AND (type = 'assistant'
           OR raw_json LIKE '%"tool_result"%')
    ORDER BY id
"""


def backfill(db_path, dry_run=False, out=print):
    """Backfill and verify. Returns a summary dict (see keys below)."""
    reader = sqlite3.connect(pathlib.Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    writer = None
    if not dry_run:
        writer = sqlite3.connect(db_path, timeout=10.0)
        writer.execute("PRAGMA busy_timeout=10000;")
        ensure_metrics_schema(writer)

    seen_uses, seen_results = set(), set()
    seen_usage = {}  # message_id -> (input, cache_creation, cache_read, output), max per field
    rows_read = parse_errors = 0
    started = time.time()

    cursor = reader.execute(SELECT_ROWS)
    while True:
        batch = cursor.fetchmany(BATCH)
        if not batch:
            break
        for row_id, session_id, project, ts, is_subagent, raw_json in batch:
            rows_read += 1
            try:
                data = json.loads(raw_json)
            except (json.JSONDecodeError, TypeError):
                parse_errors += 1
                continue
            uses, results = extract_tool_blocks(data)
            usage = extract_api_usage(data)
            seen_uses.update(u["tool_use_id"] for u in uses)
            seen_results.update(r["tool_use_id"] for r in results)
            if usage:
                fields = (usage["input_tokens"], usage["cache_creation_tokens"],
                          usage["cache_read_tokens"], usage["output_tokens"])
                old = seen_usage.get(usage["message_id"], (0, 0, 0, 0))
                seen_usage[usage["message_id"]] = tuple(max(a, b) for a, b in zip(old, fields))
            if writer is not None:
                write_capture_metrics(writer, data, session_id, project, ts, row_id, is_subagent)
        if writer is not None:
            writer.commit()
        out(f"  ...{rows_read:,} rows read")

    summary = {
        "rows_read": rows_read,
        "parse_errors": parse_errors,
        "tool_calls_found": len(seen_uses),
        "tool_results_found": len(seen_results),
        "api_responses_found": len(seen_usage),
        "dry_run": dry_run,
        "seconds": round(time.time() - started, 1),
        "missing_tool_calls": None,
        "missing_api_responses": None,
        "usage_totals_match": None,
    }

    if writer is not None:
        # Verify against the database itself, not against the writer's own counters.
        stored_uses = {r[0] for r in writer.execute(
            "SELECT tool_use_id FROM tool_events WHERE tool_name IS NOT NULL")}
        stored_usage = {r[0]: tuple(r[1:]) for r in writer.execute(
            "SELECT message_id, input_tokens, cache_creation_tokens, cache_read_tokens, "
            "output_tokens FROM api_usage")}
        summary["missing_tool_calls"] = len(seen_uses - stored_uses)
        summary["missing_api_responses"] = len(set(seen_usage) - set(stored_usage))
        summary["usage_totals_match"] = all(
            stored_usage.get(mid) == fields for mid, fields in seen_usage.items())
        writer.close()
    reader.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Backfill tool_events and api_usage from stored history")
    parser.add_argument("--dry-run", action="store_true", help="count what would be written; write nothing")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = resolve_db_path(config)
    if not os.path.exists(db_path):
        print(f"FATAL: database not found at {db_path}")
        sys.exit(2)

    print(f"{'DRY RUN - ' if args.dry_run else ''}Backfilling metrics tables from {db_path}")
    summary = backfill(db_path, dry_run=args.dry_run)

    print()
    print(f"Rows scanned:           {summary['rows_read']:,}  ({summary['parse_errors']} unreadable)")
    print(f"Tool calls found:       {summary['tool_calls_found']:,}")
    print(f"Tool results found:     {summary['tool_results_found']:,}")
    print(f"API responses found:    {summary['api_responses_found']:,}")
    print(f"Time:                   {summary['seconds']}s")
    if args.dry_run:
        print("\nDry run: nothing was written.")
        return

    ok = (summary["missing_tool_calls"] == 0 and summary["missing_api_responses"] == 0
          and summary["usage_totals_match"])
    print()
    print(f"Verify - tool calls missing from tool_events:  {summary['missing_tool_calls']}")
    print(f"Verify - responses missing from api_usage:     {summary['missing_api_responses']}")
    print(f"Verify - token counts match what was read:     {summary['usage_totals_match']}")
    print("\nVerified: every tool call and API response in history is in the metrics tables."
          if ok else "\nVERIFICATION FAILED: see the counts above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
