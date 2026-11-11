#!/usr/bin/env python3
"""
remember_baseline.py - A "before" picture of how memory retrieval behaves.

Replays your real past prompts through the same keyword-search code the live
hook runs (hooks/user-prompt-submit.py), against the memory as it existed at
the moment each prompt was typed, and reports in plain language what it
would have injected: how often it found something, how old and how relevant
those memories were, and what that cost in tokens. Run it before changing
retrieval, save the result, and run it again afterwards to see what the
change actually did.

What it does NOT do:
  - It never touches your live database during the replay. It works on a
    consistent snapshot copy (SQLite backup API) that is deleted afterwards.
    The only live write is saving the finished result to metrics_snapshots,
    which --no-save skips.
  - It does not run the semantic (meaning-based) fallback. It only counts
    how often the fallback WOULD have fired, because loading the model for
    thousands of replays says little about live behavior.
  - It skips the /clear-recovery path (that path marks a session as "shown").

Known limits, printed in the report: BM25 word-rarity statistics and the typo
corrector's vocabulary come from today's whole database, so the replay is
very slightly kinder to old prompts than reality was.

Usage:
    python3 scripts/remember_baseline.py                    # all prompts
    python3 scripts/remember_baseline.py --limit 500        # most recent 500
    python3 scripts/remember_baseline.py --since 2026-10-27 --label before-fix
    python3 scripts/remember_baseline.py --json             # machine-readable
"""

import argparse
import bisect
import importlib.util
import json
import os
import pathlib
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import yaml

from config_utils import resolve_db_path
from metrics_utils import (
    CHARS_PER_TOKEN,
    ensure_metrics_schema,
    is_system_generated_prompt,
    utc_now,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOOK_PATH = ROOT / "hooks" / "user-prompt-submit.py"
RETRIEVAL_LOG = ROOT / "logs" / "retrieval.log"

SHORT_PROMPT_CHARS = 15        # the live hook skips anything shorter
SEMANTIC_CUTOFF = 0.35         # cosine cutoff hard-coded in the hook (recorded, not used)
GENERIC_DOC_SHARE = 0.05       # a word in >5% of messages is "generic"
OLD_MEMORY_DAYS = (30, 90)
DEFAULT_PRICE_RATIOS = {"input": 1.0, "cache_write": 1.25, "cache_read": 0.1, "output": 5.0}

LOG_LINE = re.compile(r"^(\S+) \[[^\]]*\] (\w+) n=(\d+)")


def load_hook():
    """Import hooks/user-prompt-submit.py (hyphenated name, so by path)."""
    spec = importlib.util.spec_from_file_location("ups_hook", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_snapshot(db_path, dest_dir):
    """Consistent point-in-time copy of the live DB, safe to read freely."""
    dest = os.path.join(dest_dir, "snapshot.db")
    src = sqlite3.connect(pathlib.Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return dest


def _parse_ts(ts):
    try:
        dt = datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def fetch_prompts(conn, since=None, limit=None):
    sql = ("SELECT id, session_id, project, timestamp, content FROM transcripts "
           "WHERE source = 'claude_code' AND type = 'user' AND role = 'user' "
           "AND is_subagent = 0 AND LENGTH(TRIM(content)) > 0")
    args = []
    if since:
        sql += " AND timestamp >= ?"
        args.append(since)
    if limit:
        rows = conn.execute(sql + " ORDER BY timestamp DESC LIMIT ?", (*args, limit)).fetchall()
        return list(reversed(rows))
    return conn.execute(sql + " ORDER BY timestamp", args).fetchall()


def replay(hook, conn, snapshot_path, prompts, progress=None):
    """Run each prompt through the hook's keyword path as of its own timestamp.

    Returns one record per prompt. Mirrors the hook's order of checks (short
    prompt, no keywords, keyword search) so the outcome names match
    retrieval_events.
    """
    total_docs = conn.execute(
        "SELECT COUNT(*) FROM transcripts WHERE LENGTH(TRIM(content)) > 0").fetchone()[0]
    doc_freq = {}

    def generic(term):
        if term not in doc_freq:
            row = conn.execute(
                "SELECT doc FROM transcripts_fts_vocab WHERE term = ?", (term,)).fetchone()
            doc_freq[term] = row[0] if row else 0
        return total_docs > 0 and doc_freq[term] / total_docs > GENERIC_DOC_SHARE

    records = []
    for i, (row_id, session_id, project, ts, content) in enumerate(prompts, 1):
        prompt = content.strip()
        rec = {"id": row_id, "ts": ts, "project": project,
               "kind": "system_generated" if is_system_generated_prompt(prompt) else "user"}
        if len(prompt) < SHORT_PROMPT_CHARS:
            rec["outcome"] = "skipped_short"
        else:
            keywords = hook.build_search_keywords(prompt, snapshot_path)
            if not keywords:
                rec["outcome"] = "skipped_no_keywords"
            else:
                fts_query = " OR ".join(f'"{kw}"' for kw in keywords)
                started = time.perf_counter()
                info = {}
                rows = hook._keyword_search(conn, fts_query, project, as_of=ts, exclude_session=session_id,
                                            info=info)
                rec["keyword_ms"] = (time.perf_counter() - started) * 1000
                # Would mode=on_miss load the meaning model for this prompt, and why?
                rec["gate_fires"], rec["gate_reason"] = hook.should_use_semantic(
                    prompt, keywords, rows, info, conn)
                rec["generic_keywords"] = sum(1 for kw in keywords if generic(kw))
                rec["n_keywords"] = len(keywords)
                if not rows:
                    rec["outcome"] = "no_keyword_hit"     # live hook would try semantic here
                else:
                    now = _parse_ts(ts)
                    ranks = [r[3] for r in rows if r[3] is not None]
                    coverages = [c for c in (hook._coverage(keywords, r[2]) for r in rows)
                                 if c is not None]
                    rec.update(
                        outcome="keyword_hit",
                        n_results=len(rows),
                        ages=[a for a in (hook._age_days(r[1], now) for r in rows) if a is not None],
                        same_project=sum(1 for r in rows if r[0] == project),
                        top_rank=min(ranks) if ranks else None,
                        best_coverage=max(coverages) if coverages else None,
                        injected_chars=len(hook.format_memories(rows)),
                    )
        records.append(rec)
        if progress and i % 500 == 0:
            progress(f"  ...replayed {i:,} of {len(prompts):,} prompts")
    return records


def summarize_replay(records):
    """Aggregate per-prompt records into the headline numbers."""
    users = [r for r in records if r["kind"] == "user"]
    searched = [r for r in users if r["outcome"] in ("keyword_hit", "no_keyword_hit")]
    hits = [r for r in searched if r["outcome"] == "keyword_hit"]
    ages = [a for r in hits for a in r["ages"]]
    n_injected = sum(r["n_results"] for r in hits)
    chars = [r["injected_chars"] for r in hits]
    ms = [r["keyword_ms"] for r in searched]
    generic_share = [r["generic_keywords"] / r["n_keywords"] for r in searched if r["n_keywords"]]
    coverages = [r["best_coverage"] for r in hits if r["best_coverage"] is not None]
    # BM25 ranks are negative; more negative is a stronger match.
    strengths = [r["top_rank"] for r in hits if r["top_rank"] is not None]

    def share(part, whole):
        return round(part / whole, 4) if whole else None

    return {
        "prompts_replayed": len(records),
        "system_generated_prompts": len(records) - len(users),
        "user_prompts": len(users),
        "skipped_short": sum(1 for r in users if r["outcome"] == "skipped_short"),
        "skipped_no_keywords": sum(1 for r in users if r["outcome"] == "skipped_no_keywords"),
        "searched": len(searched),
        "keyword_hit": len(hits),
        "keyword_hit_rate": share(len(hits), len(searched)),
        "semantic_would_fire_rate": share(len(searched) - len(hits), len(searched)),
        "on_miss_gate_fire_rate": share(sum(1 for r in searched if r.get("gate_fires")), len(searched)),
        "on_miss_gate_reasons": {
            reason: sum(1 for r in searched if r.get("gate_reason") == reason)
            for reason in sorted({r.get("gate_reason") for r in searched if r.get("gate_fires")})},
        "avg_results_per_hit": round(n_injected / len(hits), 2) if hits else None,
        "median_age_days": round(statistics.median(ages), 1) if ages else None,
        "share_older_than_30d": share(sum(1 for a in ages if a > OLD_MEMORY_DAYS[0]), len(ages)),
        "share_older_than_90d": share(sum(1 for a in ages if a > OLD_MEMORY_DAYS[1]), len(ages)),
        "share_from_same_project": share(sum(r["same_project"] for r in hits), n_injected),
        "median_best_coverage": round(statistics.median(coverages), 3) if coverages else None,
        "share_best_coverage_under_half": share(sum(1 for c in coverages if c < 0.5), len(coverages)),
        "top_rank_strongest_tenth": round(_percentile(strengths, 10), 2) if strengths else None,
        "top_rank_median": round(_percentile(strengths, 50), 2) if strengths else None,
        "top_rank_weakest_tenth": round(_percentile(strengths, 90), 2) if strengths else None,
        "avg_generic_keyword_share": round(statistics.mean(generic_share), 3) if generic_share else None,
        "avg_injected_tokens_per_hit": round(statistics.mean(chars) / CHARS_PER_TOKEN) if chars else None,
        "total_injected_tokens": round(sum(chars) / CHARS_PER_TOKEN),
        "keyword_ms_p50": round(_percentile(ms, 50), 1) if ms else None,
        "keyword_ms_p95": round(_percentile(ms, 95), 1) if ms else None,
        "first_prompt": records[0]["ts"] if records else None,
        "last_prompt": records[-1]["ts"] if records else None,
    }


# ---------------------------------------------------------------------------
# Other facts
# ---------------------------------------------------------------------------

def read_live_log(since_ts=None):
    """Outcome counts from the real logs/retrieval.log (what actually happened)."""
    counts, first, last = {}, None, None
    if not RETRIEVAL_LOG.exists():
        return {"lines": 0}
    for line in RETRIEVAL_LOG.read_text(errors="replace").splitlines():
        m = LOG_LINE.match(line)
        if not m:
            continue
        ts, outcome = m.group(1), m.group(2)
        if since_ts and ts < since_ts:
            continue
        counts[outcome] = counts.get(outcome, 0) + 1
        first, last = first or ts, ts
    return {"lines": sum(counts.values()), "outcomes": counts, "first": first, "last": last}


def database_facts(conn, db_bytes):
    one = lambda sql: conn.execute(sql).fetchone()[0] or 0
    rows = one("SELECT COUNT(*) FROM transcripts")
    embeddable = one("SELECT COUNT(*) FROM transcripts WHERE LENGTH(TRIM(content)) >= 50")
    embedded = one("SELECT COUNT(*) FROM transcripts t JOIN transcript_embeddings e "
                   "ON e.transcript_id = t.id WHERE LENGTH(TRIM(t.content)) >= 50")
    empty = one("SELECT COUNT(*) FROM transcripts WHERE content IS NULL OR TRIM(content) = ''")
    sessions = one("SELECT COUNT(*) FROM sys_sessions")
    no_notes = one("SELECT COUNT(*) FROM sys_sessions WHERE notes IS NULL OR TRIM(notes) = ''")
    fallback = one("SELECT COUNT(*) FROM sys_sessions WHERE notes LIKE 'AUTO-GENERATED FALLBACK%'")
    raw_json = one("SELECT SUM(LENGTH(raw_json)) FROM transcripts")
    return {
        "db_mb": round(db_bytes / 1e6),
        "transcript_rows": rows,
        "rows_with_no_text": empty,
        "rows_with_no_text_share": round(empty / rows, 3) if rows else None,
        "embeddable_rows": embeddable,
        "embedded_rows": embedded,
        "embedding_coverage_of_embeddable": round(embedded / embeddable, 4) if embeddable else None,
        "sessions": sessions,
        "sessions_without_notes": no_notes,
        "sessions_with_fallback_notes": fallback,
        "raw_json_mb": round(raw_json / 1e6),
        "raw_json_share_of_db": round(raw_json / db_bytes, 3) if db_bytes else None,
    }


def token_facts(conn, ratios):
    """Where the tokens go, from api_usage and tool_events (run backfill_metrics.py first)."""
    try:
        n, i, w, r, o = conn.execute(
            "SELECT COUNT(*), SUM(input_tokens), SUM(cache_creation_tokens), "
            "SUM(cache_read_tokens), SUM(output_tokens) FROM api_usage").fetchone()
    except sqlite3.OperationalError:
        n, i, w, r, o = 0, 0, 0, 0, 0
    if not n:
        return {"available": False}
    i, w, r, o = (i or 0), (w or 0), (r or 0), (o or 0)
    weighted = {"cache_read": r * ratios["cache_read"], "cache_write": w * ratios["cache_write"],
                "input": i * ratios["input"], "output": o * ratios["output"]}
    total_weighted = sum(weighted.values()) or 1
    volume = (i + w + r + o) or 1
    # Context size at the first and last reply of each real session: the first
    # one is the fixed overhead (system prompt, tool definitions, CLAUDE.md,
    # hook context) that gets re-read on every single reply.
    replies = {}  # session -> [(timestamp, context size, output tokens)], main thread only
    for sid, ts, ctx, out in conn.execute(
            "SELECT session_id, ts, input_tokens + cache_creation_tokens + cache_read_tokens, "
            "output_tokens FROM api_usage WHERE is_subagent = 0 AND ts IS NOT NULL "
            "ORDER BY session_id, ts"):
        replies.setdefault(sid, []).append((ts, ctx, out))
    real = {sid: v for sid, v in replies.items() if len(v) >= 5}
    sessions = [[ctx for _, ctx, _ in v] for v in real.values()]
    all_context = sum(sum(v) for v in sessions) or 1

    # Where the growth comes from. Every tool result and every reply is re-read
    # by each later reply in its session, so weight its size by how many came
    # after it. These are ceilings, not estimates: compaction drops old content
    # and thinking blocks are mostly not carried forward.
    stamps = {sid: [r[0] for r in v] for sid, v in real.items()}
    tool_ceiling = 0.0
    for sid, ts, chars in conn.execute(
            "SELECT session_id, ts, result_chars FROM tool_events "
            "WHERE tool_name IS NOT NULL AND result_chars IS NOT NULL"):
        if sid in real:
            later = len(real[sid]) - bisect.bisect_right(stamps[sid], ts)
            tool_ceiling += chars / CHARS_PER_TOKEN * later
    output_ceiling = sum(out * (len(v) - 1 - i) for v in real.values()
                         for i, (_, _, out) in enumerate(v))
    start_share = sum(v[0] * len(v) for v in sessions) / all_context
    tools = conn.execute(
        "SELECT COUNT(*), SUM(is_memory_tool), SUM(ok = 0), SUM(ok IS NOT NULL), "
        "SUM(result_chars) FROM tool_events WHERE tool_name IS NOT NULL").fetchone()
    return {
        "available": True,
        "api_responses": n,
        "cache_read_tokens": r, "cache_write_tokens": w, "input_tokens": i, "output_tokens": o,
        "avg_context_reread_per_response": round(r / n),
        "session_start_context_median": round(statistics.median(v[0] for v in sessions)) if sessions else None,
        "session_end_context_median": round(statistics.median(v[-1] for v in sessions)) if sessions else None,
        "start_context_share_of_all_context": round(start_share, 3),
        "tool_output_share_ceiling": round(tool_ceiling / all_context, 3),
        "claude_output_share_ceiling": round(output_ceiling / all_context, 3),
        "other_share_residual": round(max(0.0, 1 - start_share - tool_ceiling / all_context
                                          - output_ceiling / all_context), 3),
        "volume_share": {"cache_read": round(r / volume, 4), "cache_write": round(w / volume, 4),
                         "output": round(o / volume, 4)},
        "cost_share": {k: round(v / total_weighted, 4) for k, v in weighted.items()},
        "price_ratios": ratios,
        "tool_calls": tools[0] or 0,
        "memory_tool_calls": tools[1] or 0,
        "tool_calls_failed": tools[2] or 0,
        "tool_result_tokens_est": round((tools[4] or 0) / CHARS_PER_TOKEN),
    }


def git_stamp():
    """Commit + whether the tree had uncommitted changes, so a later run can
    tell exactly which code produced this baseline."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return f"{commit}{'+uncommitted' if dirty else ''}" if commit else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x):
    return "n/a" if x is None else f"{round(x * 100)}%"


def _n(x):
    return f"{x:,}"


def render_report(result):
    s, live, db, tok = result["replay"], result["live_log"], result["database"], result["tokens"]
    L = []
    add = L.append
    add("=" * 66)
    add("  MEMORY BASELINE: how retrieval behaves today")
    add("=" * 66)
    add(f"I replayed {_n(s['user_prompts'])} of your real prompts "
        f"({s['first_prompt'][:10]} to {s['last_prompt'][:10]}) through the same search the")
    add("live hook runs, against the memory as it was when you typed each one.")
    if s["system_generated_prompts"]:
        add(f"({_n(s['system_generated_prompts'])} machine-generated prompts, like subagent "
            "hand-backs, were set aside.)")
    add("")
    add("WHAT IT DOES")
    add(f"  Prompts that got searched:          {_n(s['searched'])}"
        f"   (skipped {_n(s['skipped_short'])} too-short, {_n(s['skipped_no_keywords'])} with no usable words)")
    add(f"  ...that found something to inject:  {_pct(s['keyword_hit_rate'])}")
    add(f"  ...where the backup (meaning-based) search would have kicked in:  "
        f"{_pct(s['semantic_would_fire_rate'])}")
    if s.get("on_miss_gate_fire_rate") is not None:
        add(f"  ...where mode 'on_miss' would load the meaning model (about 3 s each):  "
            f"{_pct(s['on_miss_gate_fire_rate'])}")
    add(f"  Memories injected per prompt:       {s['avg_results_per_hit']}")
    add("")
    add("WARNING SIGNS")
    add(f"  Typical age of an injected memory:  {s['median_age_days']} days old")
    add(f"  Older than 30 days / 90 days:       {_pct(s['share_older_than_30d'])} / {_pct(s['share_older_than_90d'])}")
    add(f"  Best result matched under half of your keywords:  {_pct(s['share_best_coverage_under_half'])} of prompts")
    add(f"  Your keywords that are generic (in >5% of all messages):  {_pct(s['avg_generic_keyword_share'])}")
    add(f"  Injected memories from the project you were in:  {_pct(s['share_from_same_project'])}")
    add("")
    add("COST")
    add(f"  Average injection:                  ~{s['avg_injected_tokens_per_hit']} tokens")
    add(f"  Total injected over the replay:     ~{_n(s['total_injected_tokens'])} tokens")
    add(f"  Search speed (keyword path):        {s['keyword_ms_p50']} ms typical, {s['keyword_ms_p95']} ms slowest 1 in 20")
    if tok.get("available"):
        add("")
        add("WHERE YOUR TOKENS GO  (from the API's own usage counts)")
        add(f"  API responses recorded:             {_n(tok['api_responses'])}")
        add(f"  Average context re-read per reply:  {_n(tok['avg_context_reread_per_response'])} tokens")
        if tok["session_start_context_median"]:
            add(f"  Context before you type anything:  ~{_n(tok['session_start_context_median'])} tokens "
                "(tools, instructions, CLAUDE.md), re-read on every reply")
            add(f"  Context by the end of a session:    ~{_n(tok['session_end_context_median'])} tokens; that fixed starting "
                f"context is {_pct(tok['start_context_share_of_all_context'])} of everything re-read")
            add("  Where that re-read context comes from:")
            add(f"    fixed starting context:   {_pct(tok['start_context_share_of_all_context'])}   (measured)")
            add(f"    tool output:              up to {_pct(tok['tool_output_share_ceiling'])}   (ceiling: ignores compaction)")
            add(f"    Claude's own output:      up to {_pct(tok['claude_output_share_ceiling'])}   (ceiling: includes thinking, mostly dropped later)")
            add(f"    everything else:          ~{_pct(tok['other_share_residual'])}   (your messages, hook context, structure)")
        vs, cs = tok["volume_share"], tok["cost_share"]
        add(f"  Re-reading old context:             {_pct(vs['cache_read'])} of all tokens, ~{_pct(cs['cache_read'])} of cost")
        add(f"  Claude's own output:                {_pct(vs['output'])} of all tokens, ~{_pct(cs['output'])} of cost")
        add(f"  Tool calls: {_n(tok['tool_calls'])} total, {_n(tok['memory_tool_calls'])} were memory lookups by hand, "
            f"{_n(tok['tool_calls_failed'])} failed")
        add(f"  Tool output that entered the conversation:  ~{_n(tok['tool_result_tokens_est'])} tokens")
        add("  (Cost shares use the standard price ratios; check them against your plan.)")
    else:
        add("")
        add("WHERE YOUR TOKENS GO: no data yet. Run python3 scripts/backfill_metrics.py first.")
    add("")
    add("THE DATABASE")
    add(f"  Size:                               {_n(db['db_mb'])} MB, of which {_pct(db['raw_json_share_of_db'])} is raw tool/JSON detail nothing reads")
    add(f"  Rows with no text (tool-only turns): {_pct(db['rows_with_no_text_share'])} of {_n(db['transcript_rows'])}")
    add(f"  Meaning-search coverage:            {_pct(db['embedding_coverage_of_embeddable'])} of the {_n(db['embeddable_rows'])} rows long enough to embed")
    add(f"  Sessions with no summary:           {_n(db['sessions_without_notes'])} of {_n(db['sessions'])}")
    if live.get("lines"):
        add("")
        add("SANITY CHECK AGAINST WHAT REALLY HAPPENED")
        mix = ", ".join(f"{k} {v}" for k, v in sorted(live["outcomes"].items(), key=lambda kv: -kv[1]))
        add(f"  The live log recorded {_n(live['lines'])} real prompts ({live['first'][:10]} to {live['last'][:10]}): {mix}.")
        if result.get("replay_same_window"):
            w = result["replay_same_window"]
            add(f"  The replay over that same window found something {_pct(w['keyword_hit_rate'])} of the time.")
    add("")
    add("HOW TO READ THIS")
    add("  'Found something' only means the search returned results, not that they were")
    add("  the right ones. The warning signs are the better guide: old, loosely matching,")
    add("  generic-word results. Run this again after any retrieval change to compare.")
    add("")
    add("LIMITS OF THE REPLAY")
    for note in result["caveats"]:
        add(f"  - {note}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(config, since=None, limit=None, progress=print):
    db_path = resolve_db_path(config)
    hook = load_hook()
    tmpdir = tempfile.mkdtemp(prefix="remember_baseline_")
    try:
        snapshot = make_snapshot(db_path, tmpdir)
        conn = sqlite3.connect(snapshot)
        try:
            prompts = fetch_prompts(conn, since=since, limit=limit)
            if not prompts:
                return None
            progress(f"Replaying {len(prompts):,} prompts...")
            records = replay(hook, conn, snapshot, prompts, progress=progress)
            summary = summarize_replay(records)
            live = read_live_log()
            window = None
            if live.get("first"):
                window = summarize_replay([r for r in records if r["ts"] >= live["first"]])
            ratios = {**DEFAULT_PRICE_RATIOS, **((config.get("metrics") or {}).get("price_ratios") or {})}
            return {
                "created_at": utc_now(),
                "git_commit": git_stamp(),
                "params": {
                    "mode": hook.RETRIEVAL_MODE,
                    "recency_scale_days": hook.RECENCY_SCALE_DAYS,
                    "fts_rank_floor": hook.FTS_RANK_FLOOR,
                    "max_results": hook.MAX_RESULTS,
                    "semantic_cutoff": SEMANTIC_CUTOFF,
                    "since": since, "limit": limit,
                },
                "replay": summary,
                "replay_same_window": window,
                "live_log": live,
                "database": database_facts(conn, os.path.getsize(snapshot)),
                "tokens": token_facts(conn, ratios),
                "caveats": [
                    "Word-rarity (BM25) statistics and the typo corrector's vocabulary come from "
                    "today's whole database, so old prompts are replayed slightly more kindly than "
                    "they were served.",
                    "The meaning-based fallback is counted ('would have kicked in'), not run.",
                    "The /clear recovery path is not replayed.",
                    "Imported chats (Claude.ai, ChatGPT) with differently formatted timestamps can "
                    "leak a few same-day memories into the replay.",
                ],
            }
        finally:
            conn.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def save_result(config, result, label):
    conn = sqlite3.connect(resolve_db_path(config), timeout=10.0)
    try:
        ensure_metrics_schema(conn)
        conn.execute(
            "INSERT INTO metrics_snapshots (created_at, kind, label, git_commit, params, results) "
            "VALUES (?, 'baseline', ?, ?, ?, ?)",
            (result["created_at"], label, result["git_commit"],
             json.dumps(result["params"]), json.dumps(result)))
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Baseline picture of memory retrieval behavior")
    parser.add_argument("--since", help="only replay prompts on/after this date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="only replay the most recent N prompts")
    parser.add_argument("--label", default="baseline", help="name for the saved snapshot")
    parser.add_argument("--no-save", action="store_true", help="print only; don't save to metrics_snapshots")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    if not os.path.exists(resolve_db_path(config)):
        print(f"FATAL: database not found at {resolve_db_path(config)}")
        sys.exit(2)

    result = run_baseline(config, since=args.since, limit=args.limit,
                          progress=lambda m: print(m, file=sys.stderr))
    if result is None:
        print("No prompts found to replay.")
        sys.exit(1)
    if not args.no_save:
        save_result(config, result, args.label)
    print(json.dumps(result, indent=2) if args.json else render_report(result))
    if not args.no_save:
        print(f"\nSaved as '{args.label}' in metrics_snapshots ({result['git_commit']}).", file=sys.stderr)


if __name__ == "__main__":
    main()
