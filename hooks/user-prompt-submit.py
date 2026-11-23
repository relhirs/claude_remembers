#!/usr/bin/env python3
"""user-prompt-submit.py - Claude Code UserPromptSubmit hook for claude_remembers.

Fires before every user message is sent to Claude.
1. Extracts user prompt text from stdin JSON
2. Post-/clear recovery: if the previous session in this project ended via
   /clear and hasn't been shown yet (event-based, no time window), inject
   its generated summary merged with a normal relevance search.
3. Otherwise: keyword (FTS5) search first; only fall back to semantic
   (embedding) search when keyword results are weak/empty, since semantic
   search's cold model-load cost is only worth paying when the fast path
   already struck out.
4. Returns results as additionalContext, capped by a relevance floor
   rather than always padding to a fixed count.

RULE: stdout is SACRED. Only valid JSON goes to stdout.
"""

import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone

import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRIEVAL_LOG_PATH = os.path.join(ROOT_DIR, "logs", "retrieval.log")
RETRIEVAL_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5MB; one line per message otherwise grows forever


def _rotate_log_if_needed(path, max_bytes=RETRIEVAL_LOG_MAX_BYTES):
    """Keep the log from growing unbounded by dropping its older half once
    it crosses max_bytes. Never raises - log rotation must not break the hook."""
    try:
        if os.path.getsize(path) <= max_bytes:
            return
        with open(path, "rb") as f:
            data = f.read()
        tail = data[-(max_bytes // 2):]
        newline = tail.find(b"\n")
        if newline != -1:
            tail = tail[newline + 1:]
        with open(path, "wb") as f:
            f.write(tail)
    except Exception:
        pass


def _log_retrieval(session_id, outcome, keywords, prompt_text, n_results=0, ts=None):
    """Append one line per prompt recording which retrieval path fired and
    what it found, so keyword-vs-semantic coverage can actually be audited
    instead of just assumed. Never raises - logging must not break the hook."""
    try:
        os.makedirs(os.path.dirname(RETRIEVAL_LOG_PATH), exist_ok=True)
        if os.path.exists(RETRIEVAL_LOG_PATH):
            _rotate_log_if_needed(RETRIEVAL_LOG_PATH)
        ts = ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        snippet = (prompt_text or "")[:120].replace("\n", " ")
        kw = ",".join(keywords or [])
        with open(RETRIEVAL_LOG_PATH, "a") as f:
            f.write(
                f"{ts} [{session_id or '?'}] {outcome} n={n_results} "
                f"keywords=[{kw}] prompt={snippet!r}\n"
            )
    except Exception:
        pass

# Add scripts/ to path for semantic_search / metrics imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from config_utils import MACHINE_GENERATED_PREFIXES, is_machine_generated, resolve_db_path
try:
    from semantic_search import semantic_search
except ImportError:
    semantic_search = None
# Measurement layer: optional, so a missing metrics_utils.py can never break
# retrieval itself (same pattern as semantic_search above).
try:
    from metrics_utils import (
        CHARS_PER_TOKEN,
        is_system_generated_prompt,
        metrics_enabled,
        record_retrieval_event,
    )
except ImportError:
    record_retrieval_event = None

# All retrieval steps live in scripts/retrieval.py, shared with the eval, the
# replay and /remember-explain. The import is guarded: if that file is missing
# or broken the hook must not crash (memory would silently stop on every
# prompt). It degrades instead: it prints {}, exits 0, records outcome
# "retrieval_unavailable" in retrieval.log and retrieval_events, and writes one
# line to stderr. Post-/clear recovery (below) does not depend on the module.
RETRIEVAL_IMPORT_ERROR = None
try:
    from retrieval import (  # noqa: F401  (re-exported: tests and older callers use hook.X)
        FTS_RANK_FLOOR, GATE_MIN_COVERAGE, MAX_RESULTS, MEMORY_PHRASING, MIN_STRENGTH_PER_LOG_DOC,
        RECENCY_SCALE_DAYS, RETRIEVAL_MODE, RRF_K, SEMANTIC_CANDIDATES, SEMANTIC_CUTOFF,
        SEMANTIC_PROJECT_BOOST, STOP_WORDS,
        _age_days, _confidence_floor, _coverage, _keyword_search, build_search_keywords,
        extract_keywords, format_memories, merge_results, retrieve, semantic_mode, semantic_rows,
        should_use_semantic,
    )
except Exception as _e:  # noqa: BLE001 - any failure must degrade, never crash the hook
    retrieve = None
    RETRIEVAL_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"


def normalize_for_match(s):
    """Lowercase and strip everything except letters/digits/slashes.

    Makes folder-name matching robust to however the real folder is actually
    spelled - spaces, hyphens, underscores, commas, mixed case, any symbol -
    by collapsing all of that down to one canonical form before comparing.
    """
    return re.sub(r'[^a-z0-9/]', '', s.replace('\\', '/').lower())


def project_for_cwd(config, cwd):
    """Project prefix for a working folder: the first jsonl_project_mapping
    entry (in file order) whose folder appears in it. "oth" matches
    write_exchange.py's detect_project() fallback - without it, a folder with
    no mapping entry left the project None here (but still wrote project="oth"
    when capturing messages), silently disabling post-/clear recovery and
    project-biased ranking for that folder."""
    cwd_project = "oth"
    norm_cwd = normalize_for_match(cwd)
    for folder, prefix in config.get("jsonl_project_mapping", {}).items():
        if normalize_for_match(folder) in norm_cwd:
            cwd_project = prefix
            break
    return cwd_project


# ---------------------------------------------------------------------------
# Post-/clear recovery
# Detection: event-based (sys_sessions.end_reason='clear' AND
# clear_recovery_shown=0), no time window - fires on your very next prompt
# whenever you return, 30 seconds or 3 days later.
# ---------------------------------------------------------------------------

def _find_pending_clear(conn, cwd_project, current_session_id):
    """The previous session in this project that ended via /clear and hasn't
    been shown yet, as (session_id, notes, ended_at, message_count), or None.
    Read-only: marking it shown is _build_post_clear_context's job. Also used
    by /remember-explain, which must say whether recovery would fire without
    firing it."""
    return conn.execute(
        """SELECT session_id, notes, ended_at, message_count
           FROM sys_sessions
           WHERE project = ?
             AND session_id != ?
             AND end_reason = 'clear'
             AND clear_recovery_shown = 0
           ORDER BY ended_at DESC
           LIMIT 1""",
        (cwd_project, current_session_id),
    ).fetchone()


def _build_post_clear_context(conn, cwd_project, current_session_id, keywords):
    """Detect post-/clear state and build recovery context, merged with a
    normal relevance search against the user's NEW prompt (not a fixed
    override that replaces search for the turn)."""
    try:
        row = _find_pending_clear(conn, cwd_project, current_session_id)

        if not row:
            return None

        old_session_id, notes, ended_at, msg_count = row

        lines = []
        lines.append("## POST-/CLEAR CONTEXT RECOVERY")
        lines.append("")
        lines.append(
            f"A /clear happened. Previous session `{old_session_id}` "
            f"ended at {(ended_at or '')[:19]} with {msg_count} messages."
        )
        lines.append("")

        if notes and "AUTO-GENERATED FALLBACK" not in notes:
            lines.append("### Session Summary")
            lines.append(notes)
            lines.append("")
        else:
            lines.append(
                f"### Session Summary\nStill generating or unavailable for `{old_session_id}`. "
                f"Call `get_session('{old_session_id}')` via the memory MCP if you need detail now."
            )
            lines.append("")

        proj_row = conn.execute(
            """SELECT summary FROM project_registry
               WHERE prefix = ? AND summary IS NOT NULL AND summary != ''""",
            (cwd_project,),
        ).fetchone()
        if proj_row:
            lines.append("### Current Project Summary")
            lines.append(proj_row[0])
            lines.append("")

        # Merge in a normal relevance search against the actual new prompt,
        # deduplicated against anything the summary already covers.
        if keywords:
            fts_query = " OR ".join(f'"{kw}"' for kw in keywords)
            try:
                search_rows = _keyword_search(conn, fts_query, cwd_project,
                                              exclude_session=current_session_id)
            except Exception:
                search_rows = []
            summary_lower = (notes or "").lower()
            fresh = [
                r for r in search_rows
                if not ((r[2] or "")[:80].lower() in summary_lower)
            ]
            if fresh:
                lines.append("### Relevant to Your New Message")
                for proj, ts, content, _rank, *_ in fresh:
                    preview = (content or "")[:400].replace("\n", " ")
                    lines.append(f"- [{(ts or '')[:10]}, {proj}] {preview}")
                lines.append("")

        conn.execute(
            "UPDATE sys_sessions SET clear_recovery_shown = 1 WHERE session_id = ?",
            (old_session_id,),
        )
        conn.commit()

        return "\n".join(lines)
    except Exception:
        return None


def _finish_event(ev, started):
    """Write this prompt's retrieval_events row. Never raises."""
    try:
        db_path = ev.pop("_db_path", None)
        enabled = ev.pop("_metrics_enabled", False)
        if not db_path or not enabled or record_retrieval_event is None:
            return
        ev.setdefault("outcome", "error")
        ev["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        ev["injected_tokens_est"] = (ev.get("injected_chars") or 0) // CHARS_PER_TOKEN
        record_retrieval_event(db_path, ev)
    except Exception:
        pass


def _helper_vector(root, text):
    """Query vector from the warm background helper, or None if it isn't up."""
    try:
        import numpy as np
        from semantic_helper import query
        vec = query(root, text)
        return np.array(vec, dtype=np.float32) if vec else None
    except Exception:
        return None


def _start_helper(root, config):
    """Ask for the helper to be started for the NEXT prompt. Never blocks."""
    try:
        from semantic_helper import DEFAULT_IDLE_MIN, ensure_running
        idle = (config.get("semantic_search") or {}).get("idle_shutdown_min", DEFAULT_IDLE_MIN)
        ensure_running(root, idle_min=idle)
    except Exception:
        pass


def _run(raw_input, ev):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        try:
            input_data = json.loads(raw_input)
        except json.JSONDecodeError:
            print("{}")
            return

        prompt_text = input_data.get("prompt", "")
        if not isinstance(prompt_text, str):
            prompt_text = ""
        prompt_text = prompt_text.strip()

        current_session_id = input_data.get("session_id", "") or ""
        ev["session_id"] = current_session_id
        ev["prompt_chars"] = len(prompt_text)

        config_path = os.path.join(root, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        db_path = resolve_db_path(config)
        if not os.path.exists(db_path):
            print("{}")
            return
        if record_retrieval_event is not None:
            ev["_db_path"] = db_path
            ev["_metrics_enabled"] = metrics_enabled(config)
            ev["prompt_kind"] = "system_generated" if is_system_generated_prompt(prompt_text) else "user"
        sem_mode = semantic_mode(config) if retrieve is not None else None
        ev["mode"] = f"{sem_mode}_v3" if retrieve is not None else "retrieval_unavailable"

        cwd_project = project_for_cwd(config, os.environ.get("CWD", os.getcwd()))
        ev["project"] = cwd_project

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000;")

        # Extract keywords once - used by both post-/clear merge and the
        # normal search path below.
        keywords = build_search_keywords(prompt_text, db_path) if retrieve is not None else []
        ev["n_keywords"] = len(keywords)

        # Post-/clear recovery: event-based, no time window. Runs before the
        # short-prompt skip, same as before - "continue" after /clear must
        # still trigger it.
        if cwd_project and current_session_id:
            post_clear_ctx = _build_post_clear_context(
                conn, cwd_project, current_session_id, keywords
            )
            if post_clear_ctx:
                conn.close()
                ev["outcome"] = "post_clear"
                ev["injected_chars"] = len(post_clear_ctx)
                print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": post_clear_ctx}}))
                return

        # Skip short/trivial prompts (only reached if not post-/clear)
        if len(prompt_text) < 15:
            conn.close()
            ev["outcome"] = "skipped_short"
            print("{}")
            return

        if retrieve is None:
            # Degraded: retrieval.py failed to import (see RETRIEVAL_IMPORT_ERROR).
            conn.close()
            ev["outcome"] = "retrieval_unavailable"
            _log_retrieval(current_session_id, "retrieval_unavailable", [], prompt_text)
            print(f"user-prompt-submit: retrieval unavailable ({RETRIEVAL_IMPORT_ERROR}); "
                  "no memories injected", file=sys.stderr)
            print("{}")
            return

        if not keywords:
            conn.close()
            ev["outcome"] = "skipped_no_keywords"
            _log_retrieval(current_session_id, "skipped_no_keywords", keywords, prompt_text)
            print("{}")
            return

        # Every retrieval step (keyword search, weak-match gate, meaning search per
        # config.yaml `semantic_search.mode`, merge) is retrieve(); it has no side
        # effects. What it did is in `trace`, and this hook does the side effects:
        #   on_miss - load the model in this process, but only when the gate
        #             says keyword results can't be trusted (costs ~3 s then)
        #   always  - ask the warm background helper for the query vector and
        #             merge with keyword results on every prompt; if the helper
        #             isn't up, this prompt stays keyword-only and the helper
        #             is started for next time (a prompt never waits on a start)
        # The finally block also runs when retrieve() raises, so what was logged
        # or recorded before the failure still is.
        trace = {}
        try:
            result = retrieve(conn, config, prompt_text, project=cwd_project,
                              exclude_session=current_session_id, mode=sem_mode,
                              query_vec_fn=lambda text: _helper_vector(root, text),
                              semantic_fn=semantic_search, gate_fn=should_use_semantic, trace=trace)
        finally:
            ev.update(trace.get("ev", {}))
            for event in trace.get("log_events", []):
                _log_retrieval(current_session_id, event["outcome"], keywords, prompt_text,
                               event["n"], ts=event["ts"])
            if trace.get("helper_unavailable"):
                _start_helper(root, config)
        rows = result.rows

        conn.close()

        if not rows:
            print("{}")
            return

        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                                  "additionalContext": trace["injected_text"]}}))

    except Exception:
        ev["outcome"] = "error"
        print("{}")


def main():
    raw_input = sys.stdin.read()
    started = time.perf_counter()
    ev = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": "live"}
    try:
        _run(raw_input, ev)
    finally:
        _finish_event(ev, started)


if __name__ == "__main__":
    main()
