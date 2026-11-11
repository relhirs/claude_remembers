#!/usr/bin/env python3
"""session-start.py - Claude Code SessionStart hook for claude_remembers.

Fires once when a Claude Code session starts.
1. Runs startup_check.py (scan for new JSONL, ingest, backup)
2. Queries recent session summaries from the database
3. Returns summaries as additionalContext for Claude's awareness

RULE: stdout is SACRED. Only valid JSON goes to stdout.
"""

import json
import os
import re
import socket
import sqlite3
import subprocess
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


def _verify_pending_items(conn, next_session_content):
    """Parse NEXT_SESSION.md for pending items, verify each against the memory DB.

    Searches the memory for each pending item individually. If the memory has
    evidence that an item is already resolved, flags it so Claude does NOT
    present it as pending to the user.

    Parsing: Only captures numbered/bulleted list items under sections whose
    headers START with keywords like REMAINING, PENDING, DECISIONS.
    Verification: Topic keyword AND resolution word must appear in the SAME
    line of a transcript message — prevents false positives from unrelated
    mentions of "done" or "complete" elsewhere in the message.

    Returns list of dicts: {item, date, snippet} for items that appear resolved.
    """
    if not next_session_content:
        return []

    # Step 1: Extract list items from pending/remaining/decisions sections
    pending_items = []
    in_pending_section = False

    for line in next_session_content.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        # Section header must START with the keyword (prevents false triggers
        # like "REVIEW FINAL FIXES + REMAINING TASKS" in a title)
        if any(upper.startswith(kw) for kw in [
            "REMAINING", "PENDING", "DECISIONS",
            "STILL OPEN", "UNRESOLVED", "TODO",
        ]):
            in_pending_section = True
            continue

        # Exit on separator lines or known non-pending section headers
        if in_pending_section and stripped.startswith("===="):
            in_pending_section = False
            continue
        if in_pending_section and any(upper.startswith(kw) for kw in [
            "WHAT WAS DONE", "COMPLETED", "DETAILED STATUS",
            "THREE QUALITY", "IMMEDIATE ACTION",
        ]):
            in_pending_section = False
            continue

        # Only capture numbered items (1. ...) or bulleted items (- ...)
        if (in_pending_section and stripped and len(stripped) < 200
                and (re.match(r'^\d+\.?\s', stripped) or stripped.startswith('-')
                     or stripped.startswith('•'))):
            pending_items.append(stripped)

    if not pending_items:
        return []

    # Step 2: Get recent session notes (structured summaries — high precision)
    # Session notes say things like "Removed: X" and "Step Y: COMPLETE",
    # which are much more reliable than raw transcript searches.
    try:
        notes_rows = conn.execute("""
            SELECT notes, started_at
            FROM sys_sessions
            WHERE notes IS NOT NULL AND notes != ''
            AND started_at > datetime('now', '-90 days')
            ORDER BY started_at DESC
            LIMIT 30
        """).fetchall()
    except Exception:
        return []

    if not notes_rows:
        return []

    skip_words = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "is",
        "it", "or", "and", "not", "yet", "still", "mike", "waiting",
        "confirm", "needed", "open", "pending", "after", "from", "with",
        "has", "been", "that", "this", "will", "need", "needs", "ready",
        "order", "send", "run", "use", "check", "page", "step", "when",
        "but", "its", "are", "was", "our", "can", "may", "via", "just",
        "create", "created", "produce", "produced", "new", "update",
    }

    # Resolution words — chosen carefully to avoid ambiguity.
    # "cut" excluded: matches both "should we cut?" and "was cut".
    # "done" excluded: too common in non-resolution contexts.
    resolution_words = {
        "removed", "complete", "fixed", "resolved", "applied",
        "deleted", "finished", "dropped", "eliminated", "stripped",
    }

    results = []

    for item in pending_items:
        # Clean punctuation, extract distinctive words
        clean = re.sub(r'[^\w\s]', ' ', item)
        words = clean.split()
        keywords = [
            w for w in words
            if w.lower() not in skip_words and len(w) > 2 and not w.isdigit()
        ]

        if len(keywords) < 2:
            continue

        # Check each session note for same-SENTENCE evidence.
        # Session notes are often one giant paragraph, so splitting on
        # newlines alone is not enough — split into sentences too.
        found = False
        for notes, ts in notes_rows:
            if found:
                break
            # Split into sentences: first by newlines, then by ". " boundaries
            sentences = []
            for nline in (notes or "").split("\n"):
                sentences.extend(re.split(r'(?<=\.)\s+(?=[A-Z])', nline))
            for sentence in sentences:
                sl = sentence.lower()
                # Require 2+ topic keywords AND a resolution word, same sentence
                topic_hits = sum(1 for kw in keywords[:3] if kw.lower() in sl)
                has_res = any(rw in sl for rw in resolution_words)
                if topic_hits >= 2 and has_res:
                    date = ts[:10] if ts else "unknown"
                    results.append({
                        "item": item[:120],
                        "date": date,
                        "snippet": sentence.strip()[:150],
                    })
                    found = True
                    break

    return results


def main():
    # Read stdin (hook protocol requires it; SessionStart sends {})
    sys.stdin.read()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Run startup check - ingest new files, verify folders, backup DB
    #    hook's own stdout stays suppressed (hook stdout is SACRED, JSON
    #    only), but stderr goes to startup_check.py's own log file instead
    #    of being discarded, so a crash before its internal logger starts
    #    is still captured somewhere.
    try:
        hostname = socket.gethostname()
        log_dir = os.path.join(root, "logs", hostname)
        os.makedirs(log_dir, exist_ok=True)
        sc_log = open(os.path.join(log_dir, "startup_check.log"), "a")
        subprocess.run(
            [sys.executable, os.path.join(root, "scripts", "startup_check.py")],
            stdout=subprocess.DEVNULL,
            stderr=sc_log,
        )
        sc_log.close()
    except Exception as e:
        _log_hook_error(root, "session-start.py", f"startup_check.py launch failed: {e!r}")

    # 2. Query last session notes + recent summaries and output JSON
    try:
        config_path = os.path.join(root, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # semantic_search.mode "always" keeps the embedding model warm in a
        # background helper; start it now so the first prompt already has it.
        # Never blocks and never fails the hook.
        try:
            sem_cfg = config.get("semantic_search") or {}
            if sem_cfg.get("enabled") is not False and str(sem_cfg.get("mode", "")).strip().lower() == "always":
                sys.path.insert(0, os.path.join(root, "scripts"))
                from semantic_helper import DEFAULT_IDLE_MIN, ensure_running
                ensure_running(root, idle_min=sem_cfg.get("idle_shutdown_min", DEFAULT_IDLE_MIN))
        except Exception as e:
            _log_hook_error(root, "session-start.py", f"semantic helper start failed: {e!r}")

        db_path = resolve_db_path(config)
        if not os.path.exists(db_path):
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ""}}))
            return

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.row_factory = sqlite3.Row

        lines = []

        # Verification checklist - always first, non-negotiable
        lines.append("## Session Start Checklist")
        lines.append("- **MANDATORY: Search the memory (search_transcripts, search_semantic, get_recent_summaries) BEFORE your first substantive response.** The memory has 24,000+ transcripts. Use it.")
        lines.append("- **SHOW unfinished items and next-session notes to the user prominently.** They must SEE and react to them -- not just a checkmark.")
        lines.append("- Before acting on inherited work: verify the premise independently")
        lines.append("- Before debugging: identify WHICH component is actually failing")
        lines.append("- Do NOT trust prior session notes blindly - they may contain wrong assumptions")
        lines.append("- **NEXT_SESSION.md is a HINT, not truth. The memory is the source of truth.** Verify EACH pending item/decision against the memory individually before presenting it to the user. One broad search is NOT enough.")
        lines.append("")

        # Read NEXT_SESSION.md from CWD if it exists
        next_session_content = ""
        cwd = os.environ.get("CWD", os.getcwd())
        next_session_path = os.path.join(cwd, "NEXT_SESSION.md")
        if os.path.isfile(next_session_path):
            try:
                with open(next_session_path) as nsf:
                    next_session_content = nsf.read().strip()
                if next_session_content:
                    lines.append("## Next Session Notes (from NEXT_SESSION.md)")
                    lines.append("The previous session left these notes for you:")
                    lines.append("")
                    lines.append(next_session_content)
                    lines.append("")
            except Exception:
                pass

        # Verify pending items from NEXT_SESSION.md against the memory DB
        if next_session_content:
            try:
                verified = _verify_pending_items(conn, next_session_content)
                if verified:
                    lines.append("## Memory Verification: Pending Items May Be Stale")
                    lines.append("These items from NEXT_SESSION.md were listed as pending,")
                    lines.append("but the memory has evidence they may ALREADY be resolved.")
                    lines.append("**Do NOT present these as pending without re-verifying.**")
                    lines.append("")
                    for v in verified:
                        lines.append(f"- **POSSIBLY RESOLVED:** {v['item']}")
                        lines.append(f"  Memory evidence ({v['date']}): {v['snippet']}")
                    lines.append("")
            except Exception:
                pass

        # Get last session notes (most valuable context for continuity)
        notes_text = None
        try:
            notes_row = conn.execute("""
                SELECT session_id, project, notes, started_at
                FROM sys_sessions
                WHERE notes IS NOT NULL AND notes != ''
                ORDER BY started_at DESC LIMIT 1
            """).fetchone()
            if notes_row:
                date = notes_row[3][:10] if notes_row[3] else "unknown"
                notes_text = notes_row[2] or ""
                lines.append("## Last Session Notes")
                lines.append(f"Date: {date} | Project: {notes_row[1]}")
                lines.append("")
                lines.append(notes_text)
                lines.append("")
        except Exception:
            pass

        # Scan last session notes for unfinished/unverified items
        if notes_text:
            unfinished_patterns = [
                r"\bNOT YET VERIFIED\b", r"\bNOT DONE\b", r"\bunverified\b",
                r"\buntested\b", r"\bneeds verification\b", r"\bVERIFY\b",
                r"\bnot yet tested\b", r"\bstill broken\b", r"\bstill failing\b",
            ]
            flagged = []
            for note_line in notes_text.split("\n"):
                stripped = note_line.strip()
                # Skip section headers (## Blockers, etc.)
                if stripped.startswith("#"):
                    continue
                if stripped and any(
                    re.search(p, stripped, re.IGNORECASE)
                    for p in unfinished_patterns
                ):
                    flagged.append(stripped)
            if flagged:
                lines.append("## Unfinished Items From Last Session")
                lines.append("VERIFY these independently before continuing:")
                for f in flagged:
                    lines.append(f"- {f}")
                lines.append("")

        # Gap detection: a count, not a list. The startup sweep summarizes old
        # sessions a few at a time, and the 12-character IDs this used to list
        # weren't enough for Claude to act on anyway (the old LIMIT 20 also
        # made the count wrong whenever more than 20 were missing).
        try:
            (missing,) = conn.execute(
                "SELECT COUNT(*) FROM sys_sessions WHERE notes IS NULL OR notes = ''"
            ).fetchone()
            if missing:
                lines.append("## Sessions Missing Notes")
                lines.append(f"{missing} past session(s) have no notes yet; the startup sweep fills "
                             "them in over time. Only mention this if the user asks about memory gaps.")
                lines.append("")
        except Exception:
            pass

        # Fallback notes detection: force Claude to rewrite placeholder notes
        FALLBACK_MARKER = "AUTO-GENERATED FALLBACK"
        try:
            fallbacks = conn.execute("""
                SELECT session_id, project, started_at,
                       (SELECT COUNT(*) FROM transcripts t WHERE t.session_id = s.session_id) as msg_count
                FROM sys_sessions s
                WHERE notes LIKE ?
                ORDER BY started_at DESC
                LIMIT 5
            """, (FALLBACK_MARKER + "%",)).fetchall()
            if fallbacks:
                lines.append("## MANDATORY FIRST TASK: Rewrite Placeholder Notes")
                lines.append("The following session(s) ended without Claude writing proper notes.")
                lines.append("You MUST fix these BEFORE doing any other work:")
                lines.append("")
                for sid, proj, started, msg_count in fallbacks:
                    date = started[:10] if started else "unknown"
                    lines.append(f"- **{date} | {proj} | {msg_count} messages | {sid}**")
                lines.append("")
                lines.append("For EACH session above:")
                lines.append("1. Read the transcript: `get_session(session_id)`")
                lines.append("2. Write real notes: `python3 scripts/write_session_notes.py --notes \"<notes>\" --tags \"<tags>\"`")
                lines.append("3. Update project summary: `python3 scripts/write_project_summary.py --prefix <prefix> --summary \"<summary>\"`")
                lines.append("")
        except Exception:
            pass

        # Project summary injection: show current project context
        cwd_project = None
        try:
            cwd = os.environ.get("CWD", os.getcwd())
            norm_cwd = normalize_for_match(cwd)
            mapping = config.get("jsonl_project_mapping", {})
            # "oth" matches write_exchange.py's detect_project() fallback -
            # without this, a folder with no jsonl_project_mapping entry
            # left cwd_project as None here (but still wrote project="oth"
            # when capturing messages), silently disabling the project
            # summary/recovery sections below for that folder.
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
                    lines.append("## Current Project Summary")
                    lines.append(proj_row[0])
                    lines.append("")
        except Exception:
            pass

        # Project-scoped profile injection: mirrors get_profile(project)'s
        # merge (project-specific facts override global ones sharing the
        # same key), force-injected the same way project_registry.summary
        # is above - otherwise these preferences would only be visible if
        # Claude proactively called the MCP tool.
        try:
            if cwd_project:
                from profile_utils import fetch_merged_profile, format_profile_markdown
                facts, prefs = fetch_merged_profile(conn, cwd_project)
                profile_md = format_profile_markdown(
                    facts, prefs, heading=f"## Project Profile ({cwd_project})"
                )
                if profile_md:
                    lines.append(profile_md)
                    lines.append("")
        except Exception:
            pass

        # Get last 10 sessions with notes, grouped by project
        rows = conn.execute("""
            SELECT project, notes, started_at
            FROM sys_sessions
            WHERE notes IS NOT NULL AND notes != ''
            ORDER BY started_at DESC
            LIMIT 10
        """).fetchall()
        conn.close()

        if rows:
            # Group by project
            by_project = {}
            for project, notes, started_at in rows:
                if project not in by_project:
                    by_project[project] = []
                by_project[project].append((notes, started_at))

            lines.append("## Recent Session Context")
            lines.append("")
            for project, entries in by_project.items():
                lines.append(f"### {project}")
                for summary, created_at in entries[:5]:
                    date = created_at[:10] if created_at else "unknown"
                    topic = ""
                    for sline in (summary or "").split("\n"):
                        sline = sline.strip()
                        if sline.startswith("Topic:"):
                            topic = sline[6:].strip()
                            break
                    if not topic:
                        for sline in (summary or "").split("\n"):
                            sline = sline.strip()
                            if sline and not sline.startswith("Session:") and not sline.startswith("Project:") and not sline.startswith("Time:"):
                                topic = sline[:120]
                                break
                    if topic:
                        lines.append(f"- [{date}] {topic}")
                lines.append("")

        # Update checker intentionally removed: this install has diverged
        # from the upstream repo it was cloned from (custom /clear recovery,
        # summarization, search changes), so "run git pull" would silently
        # overwrite local changes rather than offer a safe upstream update.

        if lines:
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "\n".join(lines)}}))
        else:
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ""}}))

    except Exception as e:
        _log_hook_error(root, "session-start.py", f"context build failed, returned empty: {e!r}")
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ""}}))


if __name__ == "__main__":
    main()
