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


def _log_retrieval(session_id, outcome, keywords, prompt_text, n_results=0):
    """Append one line per prompt recording which retrieval path fired and
    what it found, so keyword-vs-semantic coverage can actually be audited
    instead of just assumed. Never raises - logging must not break the hook."""
    try:
        os.makedirs(os.path.dirname(RETRIEVAL_LOG_PATH), exist_ok=True)
        if os.path.exists(RETRIEVAL_LOG_PATH):
            _rotate_log_if_needed(RETRIEVAL_LOG_PATH)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "shall", "not", "no", "yes", "ok",
    "okay", "this", "that", "it", "i", "you", "we", "they", "me", "my",
    "your", "our", "their", "and", "or", "but", "if", "then", "so", "for",
    "of", "to", "in", "on", "at", "by", "with", "from", "up", "out",
    "into", "about", "what", "how", "when", "where", "which", "who",
    "go", "just", "let", "get", "make", "know", "think", "want", "need",
    "use", "try", "see", "look", "tell", "give", "take", "come", "also",
    "now", "here", "there", "than", "more", "some", "any", "all", "each",
    "very", "much", "well", "still", "already", "please", "thanks",
    "sure", "right", "good", "new", "first", "last", "next", "other",
}

# A weak FTS5 rank (closer to 0 is better) below this is treated as "no
# good keyword match" and triggers the semantic fallback.
FTS_RANK_FLOOR = -2.0
MAX_RESULTS = 5

# The best keyword result must clear this bar or nothing is injected: with no
# real match, BM25 still returns the 5 least-bad rows, which is junk. BM25
# scores grow with database size (rarer words score higher in a bigger
# corpus), so the bar is expressed per ln(number of messages): about one
# fully distinctive keyword's worth of evidence. At ~14K messages that is a
# BM25 score of -10. A guess tuned on 46 golden questions, not a law: on that
# set it keeps every question it found and removes all junk on unanswerable ones.
MIN_STRENGTH_PER_LOG_DOC = 1.05

# Older memories need a stronger match to outrank newer ones: the BM25 rank is
# divided by (1 + age_days / RECENCY_SCALE_DAYS), so a year-old memory has to
# score about twice as well as a brand-new one. (This used to multiply the
# negative rank by a growing factor, which pushed OLD matches to the top.)
RECENCY_SCALE_DAYS = 365

# Labels retrieval_events rows so before/after comparisons can tell which
# retrieval behavior produced them. "legacy_fallback" was the original.
RETRIEVAL_MODE = "fallback_v2"

def normalize_for_match(s):
    """Lowercase and strip everything except letters/digits/slashes.

    Makes folder-name matching robust to however the real folder is actually
    spelled - spaces, hyphens, underscores, commas, mixed case, any symbol -
    by collapsing all of that down to one canonical form before comparing.
    """
    return re.sub(r'[^a-z0-9/]', '', s.replace('\\', '/').lower())


def extract_keywords(prompt_text):
    """Search keywords for a prompt: 3+ letter words minus stop-words, max 8.

    Pure function (no DB, no typo correction) so scripts/remember_baseline.py
    replays history through exactly the code the live hook runs.
    """
    words = re.findall(r'[a-zA-Z](?:[a-zA-Z\-]*[a-zA-Z])?', prompt_text.lower())
    keywords = []
    for w in words:
        if len(w) < 3 or w.replace("-", "") in STOP_WORDS or w in STOP_WORDS or w in keywords:
            continue
        keywords.append(w)
    return keywords[:8]


def build_search_keywords(prompt_text, db_path=None):
    """Keywords for a prompt, with no repeats.

    The one place the live hook, the baseline replay and the eval build their
    keywords, so they can't drift apart. There is deliberately no typo
    correction: on the golden-set eval it found no extra memories, ran ~8x
    slower, and rewrote unknown words into unrelated real ones ("kubernetes"
    became "subagents"), so nonsense still matched something. `db_path` is
    unused and kept so existing callers keep working.
    """
    return extract_keywords(prompt_text)


def format_memories(rows):
    """The 'Relevant Memories' block injected into Claude's context."""
    lines = ["## Relevant Memories", ""]
    for i, (project, timestamp, content, _rank, *_) in enumerate(rows, 1):
        preview = (content or "")[:400].replace("\n", " ")
        lines.append(f"{i}. [{(timestamp or '')[:10]}, {project or ''}] {preview}")
    lines.append("")
    return "\n".join(lines)


def _coverage(keywords, content):
    """Share of the distinct keywords that appear in `content` as words."""
    distinct = set(keywords)
    if not distinct:
        return None
    present = set(re.findall(r'[a-z]+', (content or "").lower()))
    covered = sum(1 for kw in distinct if all(part in present for part in kw.split("-")))
    return round(covered / len(distinct), 3)


def _age_days(timestamp, now=None):
    """Days between a stored ISO timestamp and now (or None if unparseable)."""
    try:
        then = datetime.fromisoformat((timestamp or "").replace("Z", "+00:00"))
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return round(((now or datetime.now(timezone.utc)) - then).total_seconds() / 86400, 1)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Post-/clear recovery
# Detection: event-based (sys_sessions.end_reason='clear' AND
# clear_recovery_shown=0), no time window - fires on your very next prompt
# whenever you return, 30 seconds or 3 days later.
# ---------------------------------------------------------------------------

def _keyword_search(conn, fts_query, cwd_project, as_of=None, exclude_session=None,
                    info=None, gated=True):
    """Same project-biased FTS5 search used by the normal path below.

    Rows are (project, timestamp, content, rank, transcript_id).
    `exclude_session` hides the current session: those messages are already
    in Claude's context, so injecting them again only wastes tokens.
    Machine-generated rows (subagent hand-backs, command artifacts) are never
    returned; they aren't anything the user said.
    `as_of` is for replaying history (scripts/remember_baseline.py): it hides
    anything stored at or after that ISO timestamp and measures age from it
    instead of from now. The live hook never passes it.
    `gated` (default) returns nothing when even the best result is weak (see
    MIN_STRENGTH_PER_LOG_DOC); pass a dict as `info` to learn that this
    happened (info["weak"] is set). gated=False is for the eval to compare.
    """
    age_ref = as_of or "now"
    time_filter = "AND t.timestamp < ?" if as_of else ""
    time_args = (as_of,) if as_of else ()
    session_filter = "AND t.session_id != ?" if exclude_session else ""
    session_args = (exclude_session,) if exclude_session else ()
    # Only the first 80 characters are inspected: the markers sit at the very
    # start, and lowercasing whole (sometimes huge) messages made this ~6x slower.
    machine_filter = "AND NOT (" + " OR ".join(
        f"LOWER(LTRIM(SUBSTR(t.content, 1, 80))) LIKE '{prefix}%'"
        for prefix in MACHINE_GENERATED_PREFIXES) + ")"

    rows = []
    if cwd_project:
        rows = conn.execute(f"""
            SELECT t.project, t.timestamp, t.content, fts.rank, t.id
            FROM transcripts_fts fts
            JOIN transcripts t ON t.rowid = fts.rowid
            WHERE transcripts_fts MATCH ? AND t.project = ? {session_filter} {machine_filter} {time_filter}
            ORDER BY fts.rank / (1.0 + MAX(0, julianday(?) - julianday(t.timestamp)) / ?)
            LIMIT 3
        """, (fts_query, cwd_project, *session_args, *time_args, age_ref, RECENCY_SCALE_DAYS)).fetchall()

    global_rows = conn.execute(f"""
        SELECT t.project, t.timestamp, t.content, fts.rank, t.id
        FROM transcripts_fts fts
        JOIN transcripts t ON t.rowid = fts.rowid
        WHERE transcripts_fts MATCH ? {session_filter} {machine_filter} {time_filter}
        ORDER BY fts.rank / (1.0 + MAX(0, julianday(?) - julianday(t.timestamp)) / ?)
        LIMIT 7
    """, (fts_query, *session_args, *time_args, age_ref, RECENCY_SCALE_DAYS)).fetchall()

    seen = set()
    merged = []
    for r in list(rows) + list(global_rows):
        key = (r[0], (r[2] or "")[:50])
        if key in seen:
            continue
        seen.add(key)
        if r[3] is not None and r[3] > FTS_RANK_FLOOR:
            continue  # below the relevance floor - skip rather than pad
        merged.append(r)
        if len(merged) >= MAX_RESULTS:
            break
    if gated and merged:
        floor = _confidence_floor(conn)
        best = min((r[3] for r in merged if r[3] is not None), default=None)
        if floor is not None and best is not None and best > floor:
            if info is not None:
                info["weak"] = True
                info["weak_top_rank"] = best
            return []
    return merged


def _confidence_floor(conn):
    """BM25 score the best keyword result must beat (negative; None = no gate)."""
    try:
        n = conn.execute("SELECT COUNT(*) FROM transcripts_fts_docsize").fetchone()[0]
    except sqlite3.Error:
        return None
    return -MIN_STRENGTH_PER_LOG_DOC * math.log(max(n, 2))


def _build_post_clear_context(conn, cwd_project, current_session_id, keywords):
    """Detect post-/clear state and build recovery context, merged with a
    normal relevance search against the user's NEW prompt (not a fixed
    override that replaces search for the turn)."""
    try:
        row = conn.execute(
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


def _note_results(ev, rows, keywords, cwd_project):
    """Copy what a search returned into the telemetry event.

    top_rank is the strongest BM25 score among the results (most negative is
    best) and top_coverage the best share of the prompt's keywords that any
    single result contains; both feed the later "is keyword search confident?"
    gate. A semantic result has no BM25 rank, so top_rank stays None there.
    """
    now = datetime.now(timezone.utc)
    ranks = [r[3] for r in rows if r[3] is not None]
    coverages = [c for c in (_coverage(keywords, r[2]) for r in rows) if c is not None]
    ev["n_results"] = len(rows)
    ev["result_ids"] = [r[4] if len(r) > 4 else None for r in rows]
    ev["result_projects"] = [r[0] for r in rows]
    ev["result_ages_days"] = [_age_days(r[1], now) for r in rows]
    ev["same_project_n"] = sum(1 for r in rows if r[0] == cwd_project)
    ev["top_rank"] = min(ranks) if ranks else None
    ev["top_coverage"] = max(coverages) if coverages else None


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
        ev["mode"] = RETRIEVAL_MODE

        # "oth" matches write_exchange.py's detect_project() fallback -
        # without this, a folder with no jsonl_project_mapping entry left
        # cwd_project as None here (but still wrote project="oth" when
        # capturing messages), silently disabling post-/clear recovery and
        # project-biased search ranking below for that folder.
        cwd_project = "oth"
        cwd = os.environ.get("CWD", os.getcwd())
        norm_cwd = normalize_for_match(cwd)
        mapping = config.get("jsonl_project_mapping", {})
        for folder, prefix in mapping.items():
            if normalize_for_match(folder) in norm_cwd:
                cwd_project = prefix
                break
        ev["project"] = cwd_project

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000;")

        # Extract keywords once - used by both post-/clear merge and the
        # normal search path below.
        keywords = build_search_keywords(prompt_text, db_path)
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

        if not keywords:
            conn.close()
            ev["outcome"] = "skipped_no_keywords"
            _log_retrieval(current_session_id, "skipped_no_keywords", keywords, prompt_text)
            print("{}")
            return

        fts_query = " OR ".join(f'"{kw}"' for kw in keywords)
        started_kw = time.perf_counter()
        kw_info = {}
        rows = _keyword_search(conn, fts_query, cwd_project, exclude_session=current_session_id,
                               info=kw_info)
        ev["keyword_ms"] = round((time.perf_counter() - started_kw) * 1000, 1)
        if kw_info.get("weak"):
            ev["outcome"] = "weak_keyword"     # replaced by semantic_hit if the fallback finds something
            ev["top_rank"] = kw_info["weak_top_rank"]
            _log_retrieval(current_session_id, "weak_keyword", keywords, prompt_text)

        if rows:
            ev["outcome"] = "keyword_hit"
            _log_retrieval(current_session_id, "keyword_hit", keywords, prompt_text, len(rows))

        # Semantic fallback: only when keyword search came up empty/weak -
        # this is the one place per-message search can be slow (embedding
        # model cold start), by design limited to messages that already
        # struck out on the fast path.
        if not rows and semantic_search:
            try:
                started_sem = time.perf_counter()
                sem_results = semantic_search(conn, config, prompt_text, project=cwd_project, limit=MAX_RESULTS)
                ev["semantic_ms"] = round((time.perf_counter() - started_sem) * 1000, 1)
                if sem_results:
                    ev["top_similarity"] = round(sem_results[0]["similarity"], 4)
                rows = [
                    (r["project"], r["timestamp"], r["content"], None, r.get("id"))
                    for r in sem_results
                    if r["similarity"] > 0.35
                    and r.get("session_id") != current_session_id
                    and not is_machine_generated(r["content"])
                ]
                if rows:
                    ev["outcome"] = "semantic_hit"
                    _log_retrieval(current_session_id, "semantic_hit", keywords, prompt_text, len(rows))
            except Exception:
                pass
        elif not rows and not semantic_search:
            ev["outcome"] = "no_match_semantic_unavailable"
            _log_retrieval(current_session_id, "no_match_semantic_unavailable", keywords, prompt_text)

        conn.close()

        if not rows:
            ev.setdefault("outcome", "no_match")
            _log_retrieval(current_session_id, "no_match", keywords, prompt_text)
            print("{}")
            return

        _note_results(ev, rows, keywords, cwd_project)
        context = format_memories(rows)
        ev["injected_chars"] = len(context)

        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}))

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
