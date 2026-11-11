#!/usr/bin/env python3
"""
metrics_utils.py - Shared helpers for the measurement layer of claude_remembers.

Everything here exists so the system can answer "is memory actually helping?"
with numbers instead of a feeling. Four tables:

  retrieval_events   one row per prompt the UserPromptSubmit hook handled
                     (which path fired, how long it took, what it injected)
  tool_events        one row per tool call Claude made, distilled from the
                     raw JSONL (name + target + success + result size)
  api_usage          one row per API response, with the token counts the
                     usage line reported (deduplicated by message id)
  metrics_snapshots  saved baseline / report results, for before-and-after

Design rules:
  - Optional everywhere. hooks and capture scripts import this inside
    try/except ImportError, so a missing file can never break memory itself.
  - Never raises into a caller. Every writer returns True/False.
  - No command text and no prompt text is stored, only names, ids, lengths
    and counts. tool_events.target holds a file path, or for Bash just the
    executable name.
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone

from config_utils import is_machine_generated

# Rough rule of thumb used for every "estimated tokens" figure in the metrics
# layer (the API's real counts are in api_usage). Kept in one place on purpose.
CHARS_PER_TOKEN = 4

# MCP server names whose tools are memory lookups. A tool call whose name
# contains one of these counts toward "Claude searched memory by hand".
MEMORY_TOOL_MARKERS = ("remember-server", "remember_server", "brain-server", "brain_server")

DDL_METRICS = [
    """CREATE TABLE IF NOT EXISTS retrieval_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'live',
        session_id TEXT,
        project TEXT,
        mode TEXT,
        outcome TEXT NOT NULL,
        prompt_kind TEXT,
        prompt_chars INTEGER,
        n_keywords INTEGER,
        n_results INTEGER,
        top_rank REAL,
        top_similarity REAL,
        top_coverage REAL,
        keyword_ms REAL,
        semantic_ms REAL,
        latency_ms REAL,
        result_ids TEXT,
        result_projects TEXT,
        result_ages_days TEXT,
        same_project_n INTEGER,
        injected_chars INTEGER,
        injected_tokens_est INTEGER
    )""",
    "CREATE INDEX IF NOT EXISTS idx_retrieval_events_ts ON retrieval_events(ts)",
    "CREATE INDEX IF NOT EXISTS idx_retrieval_events_session ON retrieval_events(session_id)",
    """CREATE TABLE IF NOT EXISTS tool_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tool_use_id TEXT NOT NULL,
        session_id TEXT,
        project TEXT,
        ts TEXT,
        transcript_id INTEGER,
        tool_name TEXT,
        target TEXT,
        is_memory_tool INTEGER DEFAULT 0,
        ok INTEGER,
        result_chars INTEGER
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_events_use ON tool_events(tool_use_id)",
    "CREATE INDEX IF NOT EXISTS idx_tool_events_session ON tool_events(session_id)",
    """CREATE TABLE IF NOT EXISTS api_usage (
        message_id TEXT PRIMARY KEY,
        session_id TEXT,
        project TEXT,
        ts TEXT,
        model TEXT,
        is_subagent INTEGER DEFAULT 0,
        input_tokens INTEGER DEFAULT 0,
        cache_creation_tokens INTEGER DEFAULT 0,
        cache_read_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_api_usage_session ON api_usage(session_id)",
    """CREATE TABLE IF NOT EXISTS metrics_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        kind TEXT NOT NULL,
        label TEXT,
        git_commit TEXT,
        params TEXT,
        results TEXT
    )""",
]

# Project-scoped tables, so remove-project.py knows what to clear.
PROJECT_METRICS_TABLES = ["retrieval_events", "tool_events", "api_usage"]


def ensure_metrics_schema(conn):
    """Create the metrics tables if missing. Idempotent; safe on every start."""
    for ddl in DDL_METRICS:
        conn.execute(ddl)
    conn.commit()


def metrics_enabled(config):
    """metrics.enabled in config.yaml; on unless explicitly turned off."""
    if not isinstance(config, dict):
        return True
    metrics = config.get("metrics")
    if not isinstance(metrics, dict):
        return True
    return bool(metrics.get("enabled", True))


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_system_generated_prompt(text):
    """True for harness-generated prompts (subagent hand-backs and the like)."""
    return is_machine_generated(text)


# ---------------------------------------------------------------------------
# retrieval_events (written by hooks/user-prompt-submit.py)
# ---------------------------------------------------------------------------

_RETRIEVAL_COLUMNS = (
    "ts", "source", "session_id", "project", "mode", "outcome", "prompt_kind",
    "prompt_chars", "n_keywords", "n_results", "top_rank", "top_similarity",
    "top_coverage", "keyword_ms", "semantic_ms", "latency_ms", "result_ids",
    "result_projects", "result_ages_days", "same_project_n", "injected_chars",
    "injected_tokens_est",
)
_RETRIEVAL_INSERT = (
    "INSERT INTO retrieval_events (" + ", ".join(_RETRIEVAL_COLUMNS) + ") VALUES ("
    + ", ".join(":" + c for c in _RETRIEVAL_COLUMNS) + ")"
)


def record_retrieval_event(db_path, event):
    """Insert one retrieval_events row. Never raises; returns True on success.

    Opens its own short-lived connection with a 1s lock timeout, so a busy
    database can delay a prompt by at most a second, and creates the tables
    on first use (fresh installs and the CI fixture DB don't have them yet).
    """
    try:
        values = {c: event.get(c) for c in _RETRIEVAL_COLUMNS}
        values["source"] = values["source"] or "live"
        values["ts"] = values["ts"] or utc_now()
        for key in ("result_ids", "result_projects", "result_ages_days"):
            if isinstance(values[key], (list, tuple)):
                values[key] = json.dumps(values[key])
        conn = sqlite3.connect(db_path, timeout=1.0)
        try:
            for attempt in (1, 2):
                try:
                    conn.execute(_RETRIEVAL_INSERT, values)
                    conn.commit()
                    return True
                except sqlite3.OperationalError as e:
                    if attempt == 1 and "no such table" in str(e):
                        ensure_metrics_schema(conn)
                        continue
                    raise
        finally:
            conn.close()
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# tool_events + api_usage (written by the capture paths, one JSONL record at a time)
# ---------------------------------------------------------------------------

_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_EXECUTABLE = re.compile(r"^[\w./+-]+$")


def _executable_name(command):
    """First real word of a shell command (skipping FOO=bar prefixes and a
    leading `cd <dir> &&`), and only that. The rest of the command line is
    never stored."""
    tokens = (command or "").split()
    if tokens and tokens[0] == "cd":
        # Claude usually writes `cd /some/dir && real_command`; the cd says
        # nothing about what was done, so report the command after it.
        for i, token in enumerate(tokens):
            if token in ("&&", ";"):
                tokens = tokens[i + 1:]
                break
            if token.endswith(";") or token.endswith("&&"):
                tokens = tokens[i + 1:]
                break
        else:
            return "cd"
    for token in tokens:
        if _ENV_ASSIGNMENT.match(token):
            continue
        if _EXECUTABLE.match(token):
            return os.path.basename(token) or None
        return None
    return None


def _tool_target(name, tool_input):
    """What a tool call acted on: a file path, a skill/agent name, or for
    Bash the executable name. Anything else is None."""
    if not isinstance(tool_input, dict):
        return None
    if name == "Bash":
        return _executable_name(tool_input.get("command"))
    if name == "Agent":
        target = tool_input.get("subagent_type")
    elif name == "Skill":
        target = tool_input.get("skill")
    else:
        target = (tool_input.get("file_path") or tool_input.get("notebook_path")
                  or tool_input.get("path"))
    return str(target)[:300] if target else None


def _result_chars(content):
    """Size of a tool_result payload in characters (text blocks only)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) for b in content
                   if isinstance(b, dict) and b.get("type") == "text")
    return 0


def extract_tool_blocks(data):
    """Pull tool calls and tool results out of one parsed JSONL record.

    Returns (uses, results). A record can hold both several uses (assistant
    row) or several results (user row); either list may be empty.
    """
    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return [], []
    uses, results = [], []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "tool_use" and block.get("id"):
            name = block.get("name") or ""
            uses.append({
                "tool_use_id": block["id"],
                "tool_name": name,
                "target": _tool_target(name, block.get("input")),
                "is_memory_tool": int(any(m in name.lower() for m in MEMORY_TOOL_MARKERS)),
            })
        elif kind == "tool_result" and block.get("tool_use_id"):
            results.append({
                "tool_use_id": block["tool_use_id"],
                "ok": 0 if block.get("is_error") else 1,
                "result_chars": _result_chars(block.get("content")),
            })
    return uses, results


def extract_api_usage(data):
    """The token usage line of one assistant record, or None."""
    if not isinstance(data, dict) or data.get("type") != "assistant":
        return None
    message = data.get("message")
    if not isinstance(message, dict):
        return None
    message_id, usage = message.get("id"), message.get("usage")
    if not message_id or not isinstance(usage, dict):
        return None

    def count(key):
        value = usage.get(key)
        return int(value) if isinstance(value, (int, float)) else 0

    return {
        "message_id": message_id,
        "model": message.get("model"),
        "input_tokens": count("input_tokens"),
        "cache_creation_tokens": count("cache_creation_input_tokens"),
        "cache_read_tokens": count("cache_read_input_tokens"),
        "output_tokens": count("output_tokens"),
    }


# A tool call and its result can be captured in different passes, in either
# order, so each side is an upsert that only sets the fields it owns.
_TOOL_USE_UPSERT = """
INSERT INTO tool_events (tool_use_id, session_id, project, ts, transcript_id,
                         tool_name, target, is_memory_tool)
VALUES (:tool_use_id, :session_id, :project, :ts, :transcript_id,
        :tool_name, :target, :is_memory_tool)
ON CONFLICT(tool_use_id) DO UPDATE SET
    session_id = excluded.session_id, project = excluded.project,
    ts = excluded.ts, transcript_id = excluded.transcript_id,
    tool_name = excluded.tool_name, target = excluded.target,
    is_memory_tool = excluded.is_memory_tool
"""
_TOOL_RESULT_UPSERT = """
INSERT INTO tool_events (tool_use_id, session_id, project, ts, ok, result_chars)
VALUES (:tool_use_id, :session_id, :project, :ts, :ok, :result_chars)
ON CONFLICT(tool_use_id) DO UPDATE SET
    ok = excluded.ok, result_chars = excluded.result_chars,
    session_id = COALESCE(tool_events.session_id, excluded.session_id),
    project = COALESCE(tool_events.project, excluded.project)
"""
# One API response is logged as several JSONL rows that repeat the same usage
# line, so keep the largest value seen per field instead of adding them up.
_USAGE_UPSERT = """
INSERT INTO api_usage (message_id, session_id, project, ts, model, is_subagent,
                       input_tokens, cache_creation_tokens, cache_read_tokens, output_tokens)
VALUES (:message_id, :session_id, :project, :ts, :model, :is_subagent,
        :input_tokens, :cache_creation_tokens, :cache_read_tokens, :output_tokens)
ON CONFLICT(message_id) DO UPDATE SET
    input_tokens = MAX(api_usage.input_tokens, excluded.input_tokens),
    cache_creation_tokens = MAX(api_usage.cache_creation_tokens, excluded.cache_creation_tokens),
    cache_read_tokens = MAX(api_usage.cache_read_tokens, excluded.cache_read_tokens),
    output_tokens = MAX(api_usage.output_tokens, excluded.output_tokens)
"""


def write_capture_metrics(conn, data, session_id, project, ts, transcript_id=None,
                          is_subagent=0):
    """Upsert the tool_events / api_usage rows for one parsed JSONL record.

    Does not commit; the caller's transaction covers it. Raises on SQL errors
    so record_capture_metrics (and the backfill script) can decide what to do.
    """
    uses, results = extract_tool_blocks(data)
    base = {"session_id": session_id, "project": project, "ts": ts}
    for use in uses:
        conn.execute(_TOOL_USE_UPSERT, {**base, **use, "transcript_id": transcript_id})
    for result in results:
        conn.execute(_TOOL_RESULT_UPSERT, {**base, **result})
    usage = extract_api_usage(data)
    if usage:
        conn.execute(_USAGE_UPSERT, {**base, **usage, "is_subagent": int(bool(is_subagent))})
    return len(uses), len(results), 1 if usage else 0


def record_capture_metrics(conn, config, data, session_id, project, ts,
                           transcript_id=None, is_subagent=0, logger=None):
    """Capture-path wrapper: honors metrics.enabled, creates tables on first
    use, and never raises, because losing a metric must not lose a message."""
    if not metrics_enabled(config):
        return False
    try:
        for attempt in (1, 2):
            try:
                write_capture_metrics(conn, data, session_id, project, ts,
                                      transcript_id, is_subagent)
                return True
            except sqlite3.OperationalError as e:
                if attempt == 1 and "no such table" in str(e):
                    ensure_metrics_schema(conn)
                    continue
                raise
    except Exception as e:
        if logger is not None:
            logger.warning("Metrics capture failed for session %s: %s", session_id, e)
    return False
