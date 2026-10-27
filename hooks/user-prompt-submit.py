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
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRIEVAL_LOG_PATH = os.path.join(ROOT_DIR, "logs", "retrieval.log")


def _log_retrieval(session_id, outcome, keywords, prompt_text, n_results=0):
    """Append one line per prompt recording which retrieval path fired and
    what it found, so keyword-vs-semantic coverage can actually be audited
    instead of just assumed. Never raises - logging must not break the hook."""
    try:
        os.makedirs(os.path.dirname(RETRIEVAL_LOG_PATH), exist_ok=True)
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

# Add scripts/ to path for fuzzy_search / semantic_search imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from config_utils import resolve_db_path
try:
    from fuzzy_search import fuzzy_correct
except ImportError:
    fuzzy_correct = None
try:
    from semantic_search import semantic_search
except ImportError:
    semantic_search = None

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

# ---------------------------------------------------------------------------
# Frustration detection - circuit breaker
# ---------------------------------------------------------------------------

FRUSTRATION_PATTERNS = [
    r"\bwhat the hell\b",
    r"\bwhy the hell\b",
    r"\bhow the hell\b",
    r"\bfor god'?s? sake",
    r"\bstupid\b",
    r"\bdumb\b",
    r"\bidiot\b",
    r"\bretard",
    r"\bare you even\b",
    r"\bwhy didn'?t you\b",
    r"\bhow can you\b",
    r"\bwhat are you doing\b",
    r"\bstop being\b",
    r"\bfix it\b",
    r"\bwast(?:ed?|ing)\s+(?:my\s+)?time\b",
    r"\bhow many times\b",
    r"\bnot listen",
    r"\bcan'?t you\b",
    r"\bfor (?:fuck|christ)\b",
]

FRUSTRATION_WORDS = {
    "hell", "stupid", "dumb", "idiot", "retard", "retarded",
    "god", "sakes", "sake", "damn", "crap", "shit", "fuck",
    "why", "how", "what", "come", "stop", "being", "even",
    "fix", "wasted", "waste", "wasting", "time", "christ",
    "listen", "listening", "cant",
}


def normalize_for_match(s):
    """Lowercase and strip everything except letters/digits/slashes.

    Makes folder-name matching robust to however the real folder is actually
    spelled - spaces, hyphens, underscores, commas, mixed case, any symbol -
    by collapsing all of that down to one canonical form before comparing.
    """
    return re.sub(r'[^a-z0-9/]', '', s.replace('\\', '/').lower())


def detect_frustration(text):
    """Check if the user's message indicates frustration."""
    text_lower = text.lower()

    for pattern in FRUSTRATION_PATTERNS:
        if re.search(pattern, text_lower):
            return True

    if len(text) > 20:
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.5:
                return True

    if text.count("!") >= 3 or text.count("?") >= 3:
        return True

    return False


def extract_topic_keywords(text):
    """Extract topic keywords from a frustrated message, stripping frustration words."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    keywords = [w for w in words if w not in STOP_WORDS and w not in FRUSTRATION_WORDS]
    return keywords[:8]


def handle_frustration(prompt_text, root):
    """When frustration detected: search memory for topic context, return STOP directive."""
    topic_keywords = extract_topic_keywords(prompt_text)

    context_lines = [
        "## FRUSTRATION DETECTED - STOP AND REASSESS",
        "Something is wrong with your current approach.",
        "Do NOT continue what you were doing. Instead:",
        "1. Review the memory context below",
        "2. Verify your assumptions are correct",
        "3. Present your findings to Mike BEFORE taking any action",
        "",
    ]

    try:
        config_path = os.path.join(root, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        db_path = resolve_db_path(config)
        if not os.path.exists(db_path):
            return "\n".join(context_lines)

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000;")

        if topic_keywords:
            fts_query = " OR ".join(f'"{kw}"' for kw in topic_keywords)
            try:
                rows = conn.execute(
                    """SELECT t.project, t.timestamp,
                              substr(t.content, 1, 600) as preview
                       FROM transcripts_fts fts
                       JOIN transcripts t ON t.rowid = fts.rowid
                       WHERE transcripts_fts MATCH ?
                       ORDER BY fts.rank
                       LIMIT 5""",
                    (fts_query,),
                ).fetchall()
                if rows:
                    context_lines.append("## Memory Context for Current Topic")
                    for proj, ts, preview in rows:
                        date = ts[:10] if ts else "?"
                        text = (preview or "").replace("\n", " ").strip()
                        context_lines.append(f"- [{date}, {proj}] {text}")
                    context_lines.append("")
            except Exception:
                pass

        try:
            notes_rows = conn.execute(
                """SELECT notes, started_at FROM sys_sessions
                   WHERE notes IS NOT NULL AND notes != ''
                   ORDER BY started_at DESC LIMIT 3"""
            ).fetchall()
            if notes_rows:
                context_lines.append("## Recent Session Notes")
                for notes, date in notes_rows:
                    d = date[:10] if date else "?"
                    context_lines.append(f"### [{d}]")
                    context_lines.append((notes or "")[:500])
                    context_lines.append("")
        except Exception:
            pass

        conn.close()

    except Exception:
        pass

    return "\n".join(context_lines)


# ---------------------------------------------------------------------------
# Post-/clear recovery
# Detection: event-based (sys_sessions.end_reason='clear' AND
# clear_recovery_shown=0), no time window - fires on your very next prompt
# whenever you return, 30 seconds or 3 days later.
# ---------------------------------------------------------------------------

def _keyword_search(conn, fts_query, cwd_project):
    """Same project-biased FTS5 search used by the normal path below."""
    rows = []
    if cwd_project:
        rows = conn.execute("""
            SELECT t.project, t.timestamp, t.content, fts.rank
            FROM transcripts_fts fts
            JOIN transcripts t ON t.rowid = fts.rowid
            WHERE transcripts_fts MATCH ? AND t.project = ?
            ORDER BY fts.rank * (1.0 + 0.1 * MAX(0, julianday('now') - julianday(t.timestamp)))
            LIMIT 3
        """, (fts_query, cwd_project)).fetchall()

    global_rows = conn.execute("""
        SELECT t.project, t.timestamp, t.content, fts.rank
        FROM transcripts_fts fts
        JOIN transcripts t ON t.rowid = fts.rowid
        WHERE transcripts_fts MATCH ?
        ORDER BY fts.rank * (1.0 + 0.1 * MAX(0, julianday('now') - julianday(t.timestamp)))
        LIMIT 7
    """, (fts_query,)).fetchall()

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
    return merged


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
                search_rows = _keyword_search(conn, fts_query, cwd_project)
            except Exception:
                search_rows = []
            summary_lower = (notes or "").lower()
            fresh = [
                r for r in search_rows
                if not ((r[2] or "")[:80].lower() in summary_lower)
            ]
            if fresh:
                lines.append("### Relevant to Your New Message")
                for proj, ts, content, _rank in fresh:
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


def main():
    raw_input = sys.stdin.read()

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

        # Frustration circuit breaker - check BEFORE length filter
        if prompt_text and detect_frustration(prompt_text):
            context = handle_frustration(prompt_text, root)
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}))
            return

        config_path = os.path.join(root, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        db_path = resolve_db_path(config)
        if not os.path.exists(db_path):
            print("{}")
            return

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

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000;")

        # Extract keywords once - used by both post-/clear merge and the
        # normal search path below.
        words = re.findall(r'[a-zA-Z](?:[a-zA-Z\-]*[a-zA-Z])?', prompt_text.lower())
        words = [w for w in words if len(w) >= 3]
        keywords = [w for w in words if w.replace("-", "") not in STOP_WORDS and w not in STOP_WORDS]
        keywords = keywords[:8]
        if fuzzy_correct and keywords:
            try:
                keywords, _ = fuzzy_correct(keywords, db_path)
            except Exception:
                pass

        # Post-/clear recovery: event-based, no time window. Runs before the
        # short-prompt skip, same as before - "continue" after /clear must
        # still trigger it.
        if cwd_project and current_session_id:
            post_clear_ctx = _build_post_clear_context(
                conn, cwd_project, current_session_id, keywords
            )
            if post_clear_ctx:
                conn.close()
                print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": post_clear_ctx}}))
                return

        # Skip short/trivial prompts (only reached if not post-/clear)
        if len(prompt_text) < 15:
            conn.close()
            print("{}")
            return

        if not keywords:
            conn.close()
            _log_retrieval(current_session_id, "skipped_no_keywords", keywords, prompt_text)
            print("{}")
            return

        fts_query = " OR ".join(f'"{kw}"' for kw in keywords)
        rows = _keyword_search(conn, fts_query, cwd_project)

        if rows:
            _log_retrieval(current_session_id, "keyword_hit", keywords, prompt_text, len(rows))

        # Semantic fallback: only when keyword search came up empty/weak -
        # this is the one place per-message search can be slow (embedding
        # model cold start), by design limited to messages that already
        # struck out on the fast path.
        if not rows and semantic_search:
            try:
                sem_results = semantic_search(conn, config, prompt_text, project=cwd_project, limit=MAX_RESULTS)
                rows = [
                    (r["project"], r["timestamp"], r["content"], None)
                    for r in sem_results if r["similarity"] > 0.35
                ]
                if rows:
                    _log_retrieval(current_session_id, "semantic_hit", keywords, prompt_text, len(rows))
            except Exception:
                pass
        elif not rows and not semantic_search:
            _log_retrieval(current_session_id, "no_match_semantic_unavailable", keywords, prompt_text)

        conn.close()

        if not rows:
            _log_retrieval(current_session_id, "no_match", keywords, prompt_text)
            print("{}")
            return

        lines = ["## Relevant Memories", ""]
        for i, (project, timestamp, content, _rank) in enumerate(rows, 1):
            preview = (content or "")[:400].replace("\n", " ")
            lines.append(f"{i}. [{(timestamp or '')[:10]}, {project or ''}] {preview}")
        lines.append("")

        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "\n".join(lines)}}))

    except Exception:
        print("{}")


if __name__ == "__main__":
    main()
