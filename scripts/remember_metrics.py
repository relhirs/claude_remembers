#!/usr/bin/env python3
"""
remember_metrics.py - How memory retrieval is doing right now, from live data.

Reads the telemetry the hooks write as you work (retrieval_events, tool_events,
api_usage) and reports in plain language: how often memory was injected, how
fresh and relevant it was, how fast, what it cost in tokens, and whether the
capture pipeline is healthy. Read-only: it never writes to the database.

remember_baseline.py replays history to build a "before" picture. This script
reports what is actually happening today. It does not claim tokens SAVED:
savings cannot be observed directly, so it shows the measured cost of the
injections and says what is still needed to measure the benefit.

Usage:
    python3 scripts/remember_metrics.py                   # last 7 days
    python3 scripts/remember_metrics.py --days 30
    python3 scripts/remember_metrics.py --project fin
    python3 scripts/remember_metrics.py --json            # machine-readable
    python3 scripts/remember_metrics.py --markdown        # for slides / docs
"""

import argparse
import bisect
import json
import os
import pathlib
import sqlite3
import statistics
import sys
from datetime import datetime, timedelta, timezone

import yaml

from config_utils import resolve_db_path
from remember_baseline import database_facts, _percentile

ROOT = pathlib.Path(__file__).resolve().parent.parent
THIN_DATA_EVENTS = 30   # below this many prompts, numbers are anecdotes, not rates


def _connect_readonly(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _has(conn, table):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone() is not None


def _json_list(text):
    try:
        return json.loads(text) if text else []
    except (ValueError, TypeError):
        return []


def retrieval_facts(conn, since, project):
    sql = ("SELECT ts, session_id, outcome, n_results, top_coverage, latency_ms, "
           "result_projects, result_ages_days, same_project_n, injected_tokens_est "
           "FROM retrieval_events WHERE source = 'live' AND ts >= ?")
    args = [since]
    if project:
        sql += " AND project = ?"
        args.append(project)
    rows = conn.execute(sql, args).fetchall()
    if not rows:
        return {"events": 0}
    outcomes = {}
    for r in rows:
        outcomes[r[2]] = outcomes.get(r[2], 0) + 1
    injected = [r for r in rows if (r[3] or 0) > 0]
    ages = [a for r in injected for a in _json_list(r[7]) if isinstance(a, (int, float))]
    results = sum(r[3] for r in injected)
    same = sum(r[8] or 0 for r in injected)
    coverage = [r[4] for r in injected if r[4] is not None]
    latency = [r[5] for r in rows if r[5] is not None]
    return {
        "events": len(rows),
        "first": min(r[0] for r in rows), "last": max(r[0] for r in rows),
        "outcomes": outcomes,
        "injection_rate": round(len(injected) / len(rows), 3),
        "avg_results_per_injection": round(results / len(injected), 2) if injected else None,
        "median_age_days": round(statistics.median(ages), 1) if ages else None,
        "share_older_than_90d": round(sum(a > 90 for a in ages) / len(ages), 3) if ages else None,
        "same_project_share": round(same / results, 3) if results else None,
        "median_best_coverage": round(statistics.median(coverage), 3) if coverage else None,
        "best_coverage_under_half": (round(sum(c < 0.5 for c in coverage) / len(coverage), 3)
                                     if coverage else None),
        "latency_ms_p50": round(_percentile(latency, 50), 1) if latency else None,
        "latency_ms_p95": round(_percentile(latency, 95), 1) if latency else None,
    }


def token_cost_facts(conn, since, project, retrieval):
    """Measured cost of what the hook injected. Injected text stays in context, so
    every later reply in that session re-reads it: cost = tokens x later replies."""
    if not retrieval.get("events") or not _has(conn, "api_usage"):
        return {"available": False}
    sql = ("SELECT session_id, ts, injected_tokens_est FROM retrieval_events "
           "WHERE source = 'live' AND ts >= ? AND injected_tokens_est > 0")
    args = [since]
    if project:
        sql += " AND project = ?"
        args.append(project)
    events = conn.execute(sql, args).fetchall()
    sessions = {s for s, _, _ in events}
    replies = {}
    for sid, ts, ctx in conn.execute(
            "SELECT session_id, ts, input_tokens + cache_creation_tokens + cache_read_tokens "
            "FROM api_usage WHERE is_subagent = 0 AND ts IS NOT NULL ORDER BY session_id, ts"):
        if sid in sessions:
            replies.setdefault(sid, []).append((ts, ctx))
    stamps = {sid: [t for t, _ in v] for sid, v in replies.items()}
    injected = sum(t for _, _, t in events)
    carried = 0
    for sid, ts, tokens in events:
        if sid in stamps:
            carried += tokens * (len(stamps[sid]) - bisect.bisect_left(stamps[sid], ts))
    window_context = sum(c for v in replies.values() for _, c in v)
    return {
        "available": True,
        "injections": len(events),
        "injected_tokens": injected,
        "avg_tokens_per_injection": round(injected / len(events)) if events else None,
        "carry_tokens_reread": carried,
        "carry_share_of_context_in_those_sessions": (round(carried / window_context, 4)
                                                     if window_context else None),
        "sessions_with_usage_data": len(replies),
    }


def search_anyway_facts(conn, since, project, retrieval):
    """How often Claude went and searched memory by hand during the window."""
    if not _has(conn, "tool_events"):
        return {"available": False}
    # Count only from the first recorded prompt on, so calls and prompts cover the same period.
    sql = "SELECT COUNT(*) FROM tool_events WHERE is_memory_tool = 1 AND ts >= ?"
    args = [retrieval.get("first") or since]
    if project:
        sql += " AND project = ?"
        args.append(project)
    manual = conn.execute(sql, args).fetchone()[0]
    prompts = retrieval.get("events") or 0
    return {"available": True, "manual_memory_searches": manual,
            "per_100_prompts": round(manual / prompts * 100, 1) if prompts else None}


def collect(config, days, project):
    db_path = resolve_db_path(config)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = _connect_readonly(db_path)
    try:
        retrieval = (retrieval_facts(conn, since, project)
                     if _has(conn, "retrieval_events") else {"events": 0})
        return {
            "window_days": days, "project": project, "since": since,
            "retrieval": retrieval,
            "tokens": token_cost_facts(conn, since, project, retrieval),
            "search_anyway": search_anyway_facts(conn, since, project, retrieval),
            "pipeline": database_facts(conn, os.path.getsize(db_path)),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x):
    return "n/a" if x is None else f"{round(x * 100)}%"


def _pct1(x):
    """Percent with one decimal, for small shares that would round to 0%."""
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _verdict(r):
    ret, tok = r["retrieval"], r["tokens"]
    if not ret.get("events"):
        return ["No prompts recorded in this window yet. Use Claude Code normally and run this again."]
    lines = []
    if ret["events"] < THIN_DATA_EVENTS:
        lines.append(f"Only {ret['events']} prompts so far. These are examples, not rates; "
                     f"wait for {THIN_DATA_EVENTS}+ before quoting a percentage.")
    if ret.get("best_coverage_under_half") is not None and ret["best_coverage_under_half"] > 0.4:
        lines.append(f"{_pct(ret['best_coverage_under_half'])} of injections were weak matches "
                     "(best result shared under half your keywords).")
    if ret.get("share_older_than_90d") is not None and ret["share_older_than_90d"] > 0.5:
        lines.append("Most injected memories are old; check the recency ranking.")
    if tok.get("available"):
        lines.append(f"Measured cost: {tok['injected_tokens']:,} tokens injected, "
                     f"{tok['carry_tokens_reread']:,} re-read by later replies "
                     f"({_pct1(tok['carry_share_of_context_in_those_sessions'])} of those sessions' context).")
    lines.append("Tokens SAVED is not shown: it needs the golden-set eval and the used-or-ignored "
                 "test (next phases). Until then this report measures cost and quality, not benefit.")
    return lines


def render_text(r):
    ret, tok, sa, pipe = r["retrieval"], r["tokens"], r["search_anyway"], r["pipeline"]
    scope = f" for project '{r['project']}'" if r["project"] else ""
    L = ["=" * 66, f"  MEMORY METRICS: last {r['window_days']} days{scope}", "=" * 66]
    if not ret.get("events"):
        L += [""] + _verdict(r)
        return "\n".join(L)
    L += ["", "HOW OFTEN IT HELPS",
          f"  Prompts seen:                    {ret['events']}",
          f"  Got memories injected:           {_pct(ret['injection_rate'])}",
          f"  Memories per injection:          {ret['avg_results_per_injection']}"]
    others = {k: v for k, v in ret["outcomes"].items() if k != "keyword_hit"}
    if others:
        L.append("  Other outcomes:                  " + ", ".join(f"{k} {v}" for k, v in others.items()))
    L += ["", "HOW GOOD THE MEMORIES ARE",
          f"  Typical age:                     {ret['median_age_days']} days",
          f"  Older than 90 days:              {_pct(ret['share_older_than_90d'])}",
          f"  From the current project:        {_pct(ret['same_project_share'])}",
          f"  Best result matched under half your keywords:  {_pct(ret['best_coverage_under_half'])}",
          "", "HOW FAST",
          f"  Typical / slowest 1-in-20:       {ret['latency_ms_p50']} ms / {ret['latency_ms_p95']} ms"]
    if tok.get("available"):
        L += ["", "WHAT IT COSTS (measured)",
              f"  Tokens injected:                 {tok['injected_tokens']:,}"
              f" ({tok['avg_tokens_per_injection']:,} per injection)",
              f"  Re-read by later replies:        {tok['carry_tokens_reread']:,}"
              f" ({_pct1(tok['carry_share_of_context_in_those_sessions'])} of those sessions' context)"]
    if sa.get("available") and sa["per_100_prompts"] is not None:
        L += ["", "DOES CLAUDE SEARCH ANYWAY? (includes its start-of-session lookup)",
              f"  Manual memory searches:          {sa['manual_memory_searches']}"
              f" ({sa['per_100_prompts']} per 100 prompts)"]
    L += ["", "IS THE PIPELINE HEALTHY",
          f"  Searchable messages embedded:    {_pct(pipe['embedding_coverage_of_embeddable'])}",
          f"  Sessions without notes:          {pipe['sessions_without_notes']} of {pipe['sessions']}",
          "", "VERDICT"] + [f"  {line}" for line in _verdict(r)]
    return "\n".join(L)


def render_markdown(r):
    ret, tok, pipe = r["retrieval"], r["tokens"], r["pipeline"]
    scope = f" ({r['project']})" if r["project"] else ""
    out = [f"## Memory metrics: last {r['window_days']} days{scope}", ""]
    if not ret.get("events"):
        return "\n".join(out + _verdict(r))
    out += ["| Measure | Value |", "|---|---|",
            f"| Prompts seen | {ret['events']} |",
            f"| Injection rate | {_pct(ret['injection_rate'])} |",
            f"| Typical age of a memory | {ret['median_age_days']} days |",
            f"| Older than 90 days | {_pct(ret['share_older_than_90d'])} |",
            f"| From current project | {_pct(ret['same_project_share'])} |",
            f"| Weak matches (under half the keywords) | {_pct(ret['best_coverage_under_half'])} |",
            f"| Latency p50 / p95 | {ret['latency_ms_p50']} / {ret['latency_ms_p95']} ms |"]
    if tok.get("available"):
        out += [f"| Tokens injected | {tok['injected_tokens']:,} |",
                f"| Tokens re-read by later replies | {tok['carry_tokens_reread']:,} |"]
    out += [f"| Embedding coverage | {_pct(pipe['embedding_coverage_of_embeddable'])} |",
            f"| Sessions without notes | {pipe['sessions_without_notes']} of {pipe['sessions']} |",
            "", "**Verdict**", ""] + [f"- {line}" for line in _verdict(r)]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Live memory retrieval metrics")
    parser.add_argument("--days", type=int, default=7, help="window length (default 7)")
    parser.add_argument("--project", help="only this project")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--markdown", action="store_true", help="print a markdown table")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    if not os.path.exists(resolve_db_path(config)):
        print(f"FATAL: database not found at {resolve_db_path(config)}")
        sys.exit(2)

    result = collect(config, args.days, args.project)
    if args.json:
        print(json.dumps(result, indent=2))
    elif args.markdown:
        print(render_markdown(result))
    else:
        print(render_text(result))


if __name__ == "__main__":
    main()
