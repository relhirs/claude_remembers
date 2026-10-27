#!/usr/bin/env python3
"""
summarize_session.py - Generate a structured session summary via Claude Haiku.

Called as a detached background subprocess by hooks/session-end.py for
EVERY session end (not just /clear). Replaces the old placeholder-note
approach: instead of leaving a stub note for a human/Claude to rewrite
later, this writes a real structured summary immediately.

Writes:
  - sys_sessions.notes   (structured markdown summary)
  - sys_sessions.tags    (model-generated tags)
  - sys_sessions.end_reason, clear_recovery_shown (bookkeeping columns)
  - facts (category='lesson')  one row per durable, reusable lesson/pattern
  - project_registry.summary   only if the model flags a material change

If the API call fails for any reason (no credentials, rate limit, network,
bad parse), falls back to a simple placeholder note + keyword-based tags
so a session is never left with nothing.

Usage:
    python3 summarize_session.py --session-id <id> --reason <reason>

Exit codes: 0 = success (model path or fallback path), 1 = fatal error
"""

import argparse
import os
import pathlib
import re
import sqlite3
import sys
import traceback
from datetime import datetime, timezone

import yaml

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
from scripts.write_session_notes import write_notes          # noqa: E402
from scripts.write_project_summary import write_summary       # noqa: E402
from scripts.config_utils import resolve_db_path              # noqa: E402

LOG_PATH = os.path.join(ROOT, "logs", "summarize_session.log")

REQUIRED_HEADERS = [
    "## Decisions", "## Done", "## In Progress",
    "## Blockers", "## Next Step", "## Key References",
]


def _log(session_id, level, message):
    """Append a timestamped line to logs/summarize_session.log. Never raises."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(LOG_PATH, "a") as f:
            f.write(f"{ts} [{session_id}] {level}: {message}\n")
    except Exception:
        pass


def is_valid_summary(text):
    """True only if the model actually followed the required template shape."""
    if "Topic:" not in text:
        return False
    return all(header in text for header in REQUIRED_HEADERS)

# ---------------------------------------------------------------------------
# Config / DB
# ---------------------------------------------------------------------------

def load_config():
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


# ---------------------------------------------------------------------------
# Fallback note/tags - last resort if the model call fails.
# Keyword tag list and logic moved here from hooks/session-end.py, which no
# longer owns note/tag generation.
# ---------------------------------------------------------------------------

FALLBACK_MARKER = "AUTO-GENERATED FALLBACK (model summarization unavailable)"

TAG_KEYWORDS = {
    "book-editing": ["edit", "polish", "chapter", "manuscript", "copyedit", "proofread", "rewrite", "revision"],
    "memoir": ["memoir", "johnny", "goods", "maffia", "mob", "harlem", "gangster", "three fingers"],
    "job-search": ["resume", "job search", "interview", "career", "hiring", "salary", "recruiter", "linkedin"],
    "finance": ["money", "bank", "invest", "stock", "crypto", "budget", "tax", "financial"],
    "coding": ["python", "script", "code", "programming", "debug", "function", "api", "github"],
    "ai-tools": ["chatgpt", "claude", "gpt", "openai", "anthropic", "gemini", "llm", "prompt"],
    "tech-setup": ["linux", "fedora", "install", "setup", "config", "terminal", "laptop", "asus"],
    "family": ["mom", "mother", "father", "wife", "daughter", "son", "family"],
    "health": ["doctor", "medical", "health", "therapy", "leg", "injury"],
    "legal": ["lawyer", "legal", "court", "contract", "lawsuit", "attorney"],
    "home": ["house", "home", "repair", "plumbing", "electric", "hvac", "mortgage"],
    "auto": ["car", "toyota", "highlander", "vehicle", "mechanic"],
    "music": ["music", "song", "guitar", "band", "album"],
    "business": ["business", "company", "startup", "entrepreneur", "marketing"],
    "writing": ["writing", "write", "story", "narrative", "voice", "author", "publisher"],
    "research": ["research", "compare", "analysis", "review", "recommend"],
    "remember-project": ["brain", "mcp", "hook", "session-start", "session-end", "ingest", "sqlite"],
    "launch": ["launch", "readme", "github", "public", "tweet", "reddit", "hacker news"],
}


def _suggest_tags_keyword(user_messages):
    search_text = " ".join((m or "")[:500].lower() for m in user_messages)
    matched = [tag for tag, kws in TAG_KEYWORDS.items() if any(kw in search_text for kw in kws)]
    return ", ".join(matched[:3])


def write_fallback(conn, session_id, project, user_messages, msg_count, first_ts, last_ts):
    """Simple placeholder note + keyword tags. Last resort only."""
    preview = (user_messages[0] if user_messages else "(no content)")[:150].replace("\n", " ").strip()
    note = (
        f"{FALLBACK_MARKER}\n"
        f"Project: {project or 'unknown'} | Messages: {msg_count}\n"
        f"Time: {(first_ts or '')[:19]} to {(last_ts or '')[:19]}\n"
        f"First message: \"{preview}\"\n"
        f"NOTE: the model call failed for this session - rewrite these notes manually if needed."
    )
    tags = _suggest_tags_keyword(user_messages)
    conn.execute("UPDATE sys_sessions SET notes = ?, tags = ? WHERE session_id = ?",
                 (note, tags, session_id))
    conn.commit()


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """You are summarizing one Claude Code session for a persistent memory system. \
Read the transcript below and produce a structured summary in EXACTLY this markdown shape \
(omit a section's bullets if genuinely empty, but keep the heading):

Topic: <one line - what this session was actually about>

## Decisions
- <anything locked in, with a one-line rationale>

## Done
- <what actually got completed>

## In Progress
- <unfinished work, exact state - not just "some work happened">

## Blockers
- <anything unresolved, open questions>

## Next Step
- <the literal "pick up here" line for next time>

## Key References
- <file paths, topics, anything needed to go find more detail>

TAGS: tag1, tag2, tag3

LESSONS:
- <short label>: <a durable, reusable lesson or pattern from this session - e.g. "X failed because Y" \
or "pattern: Z works well for W". Only include genuinely reusable insights, not session trivia. \
Omit this whole LESSONS section if there are none worth keeping.>

Only include a PROJECT_UPDATE section if this session materially changed the overall state of the \
project (not just normal session-to-session progress) - a fresh 2-4 sentence overview a new session \
should see immediately:

PROJECT_UPDATE: <2-4 sentences, or omit this line entirely if nothing material changed>

Be concise. Do not pad sections with filler when there's nothing to say."""


FORMAT_RETRY_NOTE = """

IMPORTANT: Your previous attempt did not follow the required format. You MUST \
start the response with "Topic:" and include every one of these headings, in \
order, even if a section's bullets are empty: ## Decisions, ## Done, \
## In Progress, ## Blockers, ## Next Step, ## Key References."""


def call_model(config, transcript_text, retry=False):
    import anthropic

    sum_cfg = config.get("summarization", {})
    model = sum_cfg.get("model", "claude-haiku-4-5")
    max_tokens = sum_cfg.get("max_tokens", 2048)

    # Prefer an explicit config key (this is a background script, not an
    # interactive shell, so it won't see an `ant auth login` profile - only
    # a real ANTHROPIC_API_KEY env var or this config override will work).
    api_key = sum_cfg.get("api_key") or None
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    system = SUMMARY_PROMPT + (FORMAT_RETRY_NOTE if retry else "")
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": transcript_text}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text


def parse_model_output(text):
    """Split the model's markdown response into (notes, tags, lessons, project_update)."""
    tags_match = re.search(r"^TAGS:\s*(.*)$", text, re.MULTILINE)
    lessons_match = re.search(r"^LESSONS:\s*\n(.*?)(?:\n\nPROJECT_UPDATE:|\Z)", text, re.MULTILINE | re.DOTALL)
    project_match = re.search(r"^PROJECT_UPDATE:\s*(.+)$", text, re.MULTILINE)

    # notes = everything before the TAGS: line
    if tags_match:
        notes = text[:tags_match.start()].strip()
    else:
        notes = text.strip()

    tags = tags_match.group(1).strip() if tags_match else ""

    lessons = []
    if lessons_match:
        for line in lessons_match.group(1).strip().split("\n"):
            line = line.strip().lstrip("-").strip()
            if not line:
                continue
            # Model typically wraps the label in **bold**: "**Label:** rest of text"
            bold_match = re.match(r"^\*\*(.+?)\*\*:?\s*", line)
            if bold_match:
                label = bold_match.group(1)
            elif ":" in line:
                label, _, _ = line.partition(":")
            else:
                label = line
            lessons.append((label.strip("* :")[:80], line))

    project_update = project_match.group(1).strip() if project_match else None

    return notes, tags, lessons, project_update


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a structured session summary")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--reason", default="other")
    args = parser.parse_args()

    config = load_config()
    db_path = resolve_db_path(config)
    if not os.path.exists(db_path):
        sys.exit(1)

    conn = connect_db(db_path)
    try:
        row = conn.execute(
            "SELECT project, message_count, started_at, ended_at FROM sys_sessions WHERE session_id = ?",
            (args.session_id,),
        ).fetchone()
        if not row:
            sys.exit(1)
        project, msg_count, started_at, ended_at = row

        rows = conn.execute(
            """SELECT role, content FROM transcripts
               WHERE session_id = ? AND role IN ('user', 'assistant')
                 AND content IS NOT NULL AND trim(content) != ''
               ORDER BY timestamp""",
            (args.session_id,),
        ).fetchall()
        user_messages = [c for r, c in rows if r == "user"]

        # Always record the end reason + reset the recovery flag, regardless
        # of which path (model / fallback) below succeeds.
        conn.execute(
            "UPDATE sys_sessions SET end_reason = ?, clear_recovery_shown = 0 WHERE session_id = ?",
            (args.reason, args.session_id),
        )
        conn.commit()

        sum_cfg = config.get("summarization", {})
        if not sum_cfg.get("enabled", True) or not rows:
            write_fallback(conn, args.session_id, project, user_messages, msg_count, started_at, ended_at)
            sys.exit(0)

        transcript_text = "\n\n".join(
            f"{'User' if r == 'user' else 'Claude'}: {c}" for r, c in rows
        )

        try:
            model_output = call_model(config, transcript_text)
            if not is_valid_summary(model_output):
                _log(args.session_id, "WARN",
                     f"output failed format validation, retrying once. "
                     f"First 200 chars: {model_output[:200]!r}")
                model_output = call_model(config, transcript_text, retry=True)
                if not is_valid_summary(model_output):
                    raise ValueError(
                        f"output still invalid after retry: {model_output[:200]!r}")

            notes, tags, lessons, project_update = parse_model_output(model_output)
            if not notes.strip():
                raise ValueError("empty summary from model")
        except Exception as e:
            _log(args.session_id, "ERROR", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            write_fallback(conn, args.session_id, project, user_messages, msg_count, started_at, ended_at)
            sys.exit(0)

        write_notes(args.session_id, notes, db_path, tags=tags or None)

        # Dedup against this session's own already-written lessons. A
        # session can get summarized more than once (e.g. the abandoned-
        # session backfill writes a provisional summary, then the real
        # SessionEnd overwrites notes/end_reason later) - without this,
        # the same durable insight could get inserted twice across those
        # two passes. Scoped to this session_id only: a lesson genuinely
        # worth re-noting in a different session is still allowed through.
        existing_labels = {
            (row[0] or "").strip().lower()
            for row in conn.execute(
                "SELECT key FROM facts WHERE session_id = ? AND category = 'lesson'",
                (args.session_id,),
            ).fetchall()
        }

        now = datetime.now(timezone.utc).isoformat()
        for label, value in lessons:
            norm_label = label.strip().lower()
            if norm_label in existing_labels:
                continue
            conn.execute(
                """INSERT INTO facts (project, category, key, value, source, session_id, created_at, updated_at)
                   VALUES (?, 'lesson', ?, ?, 'summarize_session.py', ?, ?, ?)""",
                (project, label, value, args.session_id, now, now),
            )
            existing_labels.add(norm_label)
        conn.commit()

        if project_update:
            write_summary(project, project_update, db_path)

    finally:
        conn.close()

    sys.exit(0)


if __name__ == "__main__":
    main()
