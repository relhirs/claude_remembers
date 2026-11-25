#!/usr/bin/env python3
"""retrieval.py - the one place memory retrieval is decided.

The live hook (hooks/user-prompt-submit.py), the eval (remember_eval.py), the
replay (remember_baseline.py) and /remember-explain all call retrieve(), so
what the explanation says is what the hook does. This module is a library: it
has no command line.

retrieve() has NO side effects. It writes no retrieval.log line, no
retrieval_events row, never marks a /clear recovery as shown and never starts
the warm helper. It returns the results plus a plain-dict trace of every step
with the real values. The hook does its logging and telemetry afterwards from
that trace (trace["log_events"], trace["ev"], trace["helper_unavailable"]).

Every number below is a guess measured by scripts/remember_eval.py, not a law.
"""

import math
import os
import re
import sqlite3
import sys
import time
from collections import namedtuple
from datetime import datetime, timezone

# The hook runs with scripts/ on sys.path; the MCP server only has the repo
# root, so it reaches these as scripts.X. Support both so one embedding model
# is shared (never a second copy under another module name).
try:
    from config_utils import MACHINE_GENERATED_PREFIXES, is_machine_generated
except ImportError:
    from scripts.config_utils import MACHINE_GENERATED_PREFIXES, is_machine_generated

try:
    from semantic_search import semantic_search as _default_semantic
except ImportError:
    try:
        from scripts.semantic_search import semantic_search as _default_semantic
    except ImportError:
        _default_semantic = None

# The metrics layer is optional: a missing metrics_utils.py must never stop retrieval.
try:
    from metrics_utils import CHARS_PER_TOKEN
except ImportError:
    try:
        from scripts.metrics_utils import CHARS_PER_TOKEN
    except ImportError:
        CHARS_PER_TOKEN = 4

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
MAX_KEYWORDS = 8
SHORT_PROMPT_CHARS = 15        # the live hook skips anything shorter

# The best keyword result must clear this bar or nothing is injected: with no
# real match, BM25 still returns the 5 least-bad rows, which is junk. BM25
# scores grow with database size (rarer words score higher in a bigger
# corpus), so the bar is expressed per ln(number of messages): about one
# fully distinctive keyword's worth of evidence. At ~14K messages that is a
# BM25 score of -10. A guess tuned on 46 golden questions, not a law: on that
# set it keeps every question it found and removes all junk on unanswerable ones.
MIN_STRENGTH_PER_LOG_DOC = 1.05

# Meaning-based (semantic) search. The cutoff applies to the raw cosine
# similarity; the current-project boost only reorders results that already
# cleared it. 30 candidates are scored so that dropping the current session
# and machine rows still leaves enough to fill MAX_RESULTS. All are guesses
# measured by scripts/remember_eval.py, not laws.
SEMANTIC_CUTOFF = 0.35
SEMANTIC_CANDIDATES = 30
SEMANTIC_PROJECT_BOOST = 0.05
RRF_K = 60

# `mode: on_miss` runs meaning search only when one of these says keyword
# search can't be trusted (see should_use_semantic). On real prompts, 0.25
# fires the model load on ~1 prompt in 4; the stricter 0.5 plus a "moderate
# score" signal fired on 3 in 4 for one or two extra found memories, which
# makes on_miss as slow as always.
GATE_MIN_COVERAGE = 0.25       # best result contains under a quarter of the prompt's keywords
# Multi-word phrases only: single words like "earlier" or "before" are in
# almost every prompt.
MEMORY_PHRASING = re.compile(
    r"\b(what did we|did we (decide|ever|already|talk|discuss)|do you remember|remember when|"
    r"last time|we (talked|discussed|decided|agreed) (about|on)|a (while|few days|few weeks) ago|"
    r"the other day|you (said|mentioned|told me)|as we discussed|where did we|when did we|"
    r"why did we|how did we|earlier (today|this|we|you))\b", re.IGNORECASE)

# Older memories need a stronger match to outrank newer ones: the BM25 rank is
# divided by (1 + age_days / RECENCY_SCALE_DAYS), so a year-old memory has to
# score about twice as well as a brand-new one. (This used to multiply the
# negative rank by a growing factor, which pushed OLD matches to the top.)
RECENCY_SCALE_DAYS = 365

# Labels retrieval_events rows so before/after comparisons can tell which
# retrieval behavior produced them. "legacy_fallback" was the original.
RETRIEVAL_MODE = "hybrid_v3"

# Marks "use this module's own function" for semantic_fn / gate_fn. An explicit
# semantic_fn=None means "meaning search is unavailable" (as when its import fails).
_DEFAULT = object()

Retrieval = namedtuple("Retrieval", "rows trace")


# ---------------------------------------------------------------------------
# Keywords and formatting
# ---------------------------------------------------------------------------

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
    return keywords[:MAX_KEYWORDS]


def keyword_breakdown(prompt_text):
    """Same decisions as extract_keywords, with the reason for every word set
    aside: (kept, [(word, reason)]). Reasons: too_short, stop_word, repeat,
    over_cap. For explanations only; the search uses extract_keywords."""
    words = re.findall(r'[a-zA-Z](?:[a-zA-Z\-]*[a-zA-Z])?', prompt_text.lower())
    kept, dropped = [], []
    for w in words:
        if len(w) < 3:
            dropped.append((w, "too_short"))
        elif w.replace("-", "") in STOP_WORDS or w in STOP_WORDS:
            dropped.append((w, "stop_word"))
        elif w in kept:
            dropped.append((w, "repeat"))
        else:
            kept.append(w)
    for w in kept[MAX_KEYWORDS:]:
        dropped.append((w, "over_cap"))
    return kept[:MAX_KEYWORDS], dropped


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


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------

def _doc_count(conn):
    """Number of indexed messages, or None if the index has no docsize table."""
    try:
        return conn.execute("SELECT COUNT(*) FROM transcripts_fts_docsize").fetchone()[0]
    except sqlite3.Error:
        return None


def _confidence_floor(conn):
    """BM25 score the best keyword result must beat (negative; None = no gate)."""
    n = _doc_count(conn)
    if n is None:
        return None
    return -MIN_STRENGTH_PER_LOG_DOC * math.log(max(n, 2))


def _strength_gate(conn, merged):
    """The weak-match gate's numbers: {"n_docs","floor","best","weak"}. `weak`
    is True when even the best result is weaker than the bar."""
    n = _doc_count(conn)
    floor = None if n is None else -MIN_STRENGTH_PER_LOG_DOC * math.log(max(n, 2))
    best = min((r[3] for r in merged if r[3] is not None), default=None)
    return {"n_docs": n, "floor": floor, "best": best,
            "weak": bool(floor is not None and best is not None and best > floor)}


def _keyword_candidates(conn, fts_query, cwd_project, as_of=None, exclude_session=None, diag=None):
    """The FTS5 search up to (not including) the weak-match gate. `cwd_project` is
    favored in the ranking; None favors no project.

    Rows are (project, timestamp, content, rank, transcript_id). `diag`, when
    given, receives what was set aside and why (explanations only).
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

    if diag is not None:
        diag["project_rows"] = list(rows)
        diag["global_rows"] = list(global_rows)
        diag["dropped"] = []          # (reason, row): duplicate / below_floor / over_cap

    seen = set()
    merged = []
    ordered = list(rows) + list(global_rows)
    for idx, r in enumerate(ordered):
        key = (r[0], (r[2] or "")[:50])
        if key in seen:
            if diag is not None:
                diag["dropped"].append(("duplicate", r))
            continue
        seen.add(key)
        if r[3] is not None and r[3] > FTS_RANK_FLOOR:
            if diag is not None:
                diag["dropped"].append(("below_floor", r))
            continue  # below the relevance floor - skip rather than pad
        merged.append(r)
        if len(merged) >= MAX_RESULTS:
            if diag is not None:
                diag["dropped"].extend(("over_cap", x) for x in ordered[idx + 1:])
            break
    return merged


def _keyword_search(conn, fts_query, cwd_project, as_of=None, exclude_session=None,
                    info=None, gated=True):
    """Same FTS5 search used by the normal path (`cwd_project` favored, None: no project).

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
    merged = _keyword_candidates(conn, fts_query, cwd_project, as_of=as_of, exclude_session=exclude_session)
    if gated and merged:
        gate = _strength_gate(conn, merged)
        if gate["weak"]:
            if info is not None:
                info["weak"] = True
                info["weak_top_rank"] = gate["best"]
            return []
    return merged


# ---------------------------------------------------------------------------
# Meaning search helpers
# ---------------------------------------------------------------------------

def should_use_semantic(prompt_text, keywords, rows, kw_info, conn):
    """on_miss gate: (run meaning search?, why). Any one signal is enough.

    Four signals: score strength (no result, or even the best one under the
    weak-match bar), coverage (the best result contains under a quarter of the
    prompt's keywords: 1 of 6 is weak), memory phrasing (questions about the
    past, where the words rarely match and the meaning does), and length
    (prompts under 15 characters never get here: they skip retrieval).
    """
    if not rows:
        return True, "weak_keyword" if kw_info.get("weak") else "no_keyword_result"
    coverages = [c for c in (_coverage(keywords, r[2]) for r in rows) if c is not None]
    if coverages and max(coverages) < GATE_MIN_COVERAGE:
        return True, "low_coverage"
    if MEMORY_PHRASING.search(prompt_text or ""):
        return True, "memory_phrasing"
    return False, "keyword_confident"


def semantic_rows(results, cwd_project, current_session_id):
    """Meaning-search results -> hook rows. Cutoff on raw similarity, then the
    current project's results are ordered slightly ahead (keyword search also
    favors the project you're in), and the current session and machine rows
    are dropped."""
    kept = [r for r in results
            if r["similarity"] > SEMANTIC_CUTOFF
            and r.get("session_id") != current_session_id
            and not is_machine_generated(r["content"])]
    kept.sort(key=lambda r: r["similarity"] + (SEMANTIC_PROJECT_BOOST if r["project"] == cwd_project else 0),
              reverse=True)
    return [(r["project"], r["timestamp"], r["content"], None, r.get("id")) for r in kept]


def merge_results(keyword_rows, sem_rows):
    """Reciprocal Rank Fusion: each list votes 1/(k + rank) for a row, so a
    memory both searches agree on rises above one only a single search found.
    Keyword rows come first so a row found by both keeps its BM25 rank."""
    score, rows = {}, {}
    for lst in (keyword_rows, sem_rows):
        for rank, row in enumerate(lst, 1):
            key = row[4] if row[4] is not None else (row[0], (row[2] or "")[:50])
            score[key] = score.get(key, 0.0) + 1.0 / (RRF_K + rank)
            rows.setdefault(key, row)
    merged, seen = [], set()
    for key in sorted(score, key=lambda k: -score[k]):
        row = rows[key]
        sig = (row[0], (row[2] or "")[:50])
        if sig in seen:
            continue
        seen.add(sig)
        merged.append(row)
        if len(merged) >= MAX_RESULTS:
            break
    return merged


def semantic_mode(config):
    """"on_miss" (default), "always" or "off" from config.yaml. Anything
    unrecognized, or a config written before this setting existed, is on_miss."""
    sem = (config or {}).get("semantic_search") or {}
    if sem.get("enabled") is False:
        return "off"
    mode = str(sem.get("mode") or "on_miss").strip().lower()
    return mode if mode in ("on_miss", "always") else "on_miss"


# ---------------------------------------------------------------------------
# Trace helpers (display only: nothing here changes what is returned)
# ---------------------------------------------------------------------------

def _now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(ts):
    try:
        dt = datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _snippet(content, n=100):
    return (content or "").replace("\n", " ").strip()[:n]


def row_view(row, ref_now):
    """A result row as a plain dict, with age and the recency-adjusted rank
    (rank / (1 + age/365), what the SQL ORDER BY uses) for display."""
    then = _parse_ts(row[1])
    age = max(0.0, (ref_now - then).total_seconds() / 86400) if then else None
    rank = row[3]
    adjusted = None if rank is None or age is None else rank / (1.0 + age / RECENCY_SCALE_DAYS)
    return {"id": row[4] if len(row) > 4 else None, "project": row[0], "timestamp": row[1],
            "rank": rank, "age_days": None if age is None else round(age, 1),
            "adjusted_rank": adjusted, "snippet": _snippet(row[2])}


def gate_fields(prompt, keywords, rows):
    """The numbers behind the on_miss gate's coverage and memory-phrasing
    signals, for the trace (display only)."""
    coverages = [c for c in (_coverage(keywords, r[2]) for r in rows) if c is not None]
    m = MEMORY_PHRASING.search(prompt or "")
    return {"min_coverage": GATE_MIN_COVERAGE, "best_coverage": max(coverages) if coverages else None,
            "n_keywords": len(set(keywords)), "phrase": m.group(0) if m else None}


def gate_details(prompt, keywords, rows, kw_info, conn, gate_fn):
    """The on_miss gate's decision plus the numbers behind it, for the trace.
    Used where the hook doesn't itself act on the gate (modes off and always):
    any failure here is recorded, never raised, so a display-only value can't
    change what the hook returns."""
    out = {"fires": None, "reason": None, "error": None, **gate_fields(prompt, keywords, rows)}
    try:
        out["fires"], out["reason"] = gate_fn(prompt, keywords, rows, kw_info, conn)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def classify_semantic(results, cwd_project, current_session_id):
    """Why each meaning-search candidate was kept or dropped, in the order
    semantic_rows checks: cutoff, then current session, then machine message."""
    out = []
    for r in results:
        if not r["similarity"] > SEMANTIC_CUTOFF:
            status = "below_cutoff"
        elif r.get("session_id") == current_session_id:
            status = "current_session"
        elif is_machine_generated(r["content"]):
            status = "machine_message"
        else:
            status = "kept"
        out.append({"id": r.get("id"), "session_id": r.get("session_id"), "project": r["project"],
                    "similarity": r["similarity"], "status": status, "snippet": _snippet(r["content"]),
                    "timestamp": r["timestamp"]})
    return out


def merge_view(keyword_rows, sem_rows, merged):
    """Where each merged row came from and its RRF score."""
    def key(row):
        return row[4] if row[4] is not None else (row[0], (row[2] or "")[:50])
    kw_rank = {key(r): i for i, r in enumerate(keyword_rows, 1)}
    sem_rank = {key(r): i for i, r in enumerate(sem_rows, 1)}
    out = []
    for row in merged:
        k = key(row)
        score = sum(1.0 / (RRF_K + rk[k]) for rk in (kw_rank, sem_rank) if k in rk)
        source = "both" if k in kw_rank and k in sem_rank else "keyword" if k in kw_rank else "meaning"
        out.append({"id": row[4], "source": source, "keyword_rank": kw_rank.get(k),
                    "meaning_rank": sem_rank.get(k), "rrf_score": score})
    return out


def count_filtered(conn, fts_query, as_of, exclude_session):
    """Explain-only: how many matching rows the search's SQL filters removed
    before ranking (current session, machine message, newer than as_of). Never
    called on the hook path."""
    out = {}
    try:
        base = "SELECT COUNT(*) FROM transcripts_fts fts JOIN transcripts t ON t.rowid = fts.rowid WHERE transcripts_fts MATCH ?"
        out["matching_total"] = conn.execute(base, (fts_query,)).fetchone()[0]
        if exclude_session:
            out["current_session"] = conn.execute(base + " AND t.session_id = ?", (fts_query, exclude_session)).fetchone()[0]
        machine = "(" + " OR ".join(f"LOWER(LTRIM(SUBSTR(t.content, 1, 80))) LIKE '{p}%'"
                                    for p in MACHINE_GENERATED_PREFIXES) + ")"
        out["machine_message"] = conn.execute(base + " AND " + machine, (fts_query,)).fetchone()[0]
        if as_of:
            out["after_as_of"] = conn.execute(base + " AND t.timestamp >= ?", (fts_query, as_of)).fetchone()[0]
    except sqlite3.Error as e:
        out["error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------

def retrieve(conn, config, prompt, *, project, exclude_session=None, as_of=None, mode=None,
             query_vec_fn=None, semantic_fn=_DEFAULT, gate_fn=_DEFAULT, row_filter=None,
             detail=False, trace=None):
    """Run every retrieval step for `prompt` and return Retrieval(rows, trace).

    No side effects (see the module docstring). Arguments:
      project, exclude_session, as_of  as in _keyword_search
      mode          "off" | "on_miss" | "always"; None = from config
      query_vec_fn  always mode: text -> query vector (the hook passes the warm
                    helper's; None return = helper down, the prompt stays
                    keyword-only). No function = embed in-process.
      semantic_fn   meaning search; default this module's. None = unavailable.
      gate_fn       the on_miss gate; default should_use_semantic
      row_filter    eval only: row_filter("keyword", row) / ("semantic", dict)
                    -> keep? Applied after the weak-match gate and after the
                    meaning-search fetch, where the eval always filtered.
      detail        explain only: extra read-only counting queries
      trace         dict to fill in; a crash halfway still leaves what was done
    """
    if trace is None:
        trace = {}
    if semantic_fn is _DEFAULT:
        semantic_fn = _default_semantic
    if gate_fn is _DEFAULT:
        gate_fn = should_use_semantic
    mode_given = mode is not None
    if mode is None:
        mode = semantic_mode(config)

    ev = trace.setdefault("ev", {})              # telemetry fields, set only where the hook sets them
    log_events = trace.setdefault("log_events", [])

    def log(outcome, n=0):
        log_events.append({"outcome": outcome, "n": n, "ts": _now_utc()})

    ref_now = _parse_ts(as_of) or datetime.now(timezone.utc)
    trace["inputs"] = {"prompt_chars": len(prompt), "project": project, "exclude_session": exclude_session,
                       "as_of": as_of, "mode": mode, "mode_source": "given" if mode_given else "config"}
    trace["helper_unavailable"] = False
    trace["gate"] = None
    trace["rows"] = []
    trace["injected_text"] = ""
    trace["injected_chars"] = 0
    trace["tokens_est"] = 0

    _, dropped = keyword_breakdown(prompt)
    keywords = build_search_keywords(prompt)
    trace["keywords"] = {"kept": keywords, "dropped": dropped, "fts_query": None}
    trace["skip"] = None
    if len(prompt) < SHORT_PROMPT_CHARS:
        trace["skip"] = "skipped_short"
        trace["outcome"] = "skipped_short"
        return Retrieval([], trace)
    if not keywords:
        trace["skip"] = "skipped_no_keywords"
        trace["outcome"] = "skipped_no_keywords"
        return Retrieval([], trace)

    fts_query = " OR ".join(f'"{kw}"' for kw in keywords)
    trace["keywords"]["fts_query"] = fts_query

    # --- keyword search ----------------------------------------------------
    started_kw = time.perf_counter()
    diag = {} if detail else None
    candidates = _keyword_candidates(conn, fts_query, project, as_of=as_of, exclude_session=exclude_session,
                                     diag=diag)
    gate = _strength_gate(conn, candidates) if candidates else \
        {"n_docs": _doc_count(conn), "floor": None, "best": None, "weak": False}
    kw_info = {}
    if gate["weak"]:
        kw_info["weak"] = True
        kw_info["weak_top_rank"] = gate["best"]
    pre_gate = candidates
    rows = [] if gate["weak"] else list(candidates)
    ungated = list(candidates)
    if row_filter is not None:
        rows = [r for r in rows if row_filter("keyword", r)]
        ungated = [r for r in ungated if row_filter("keyword", r)]
    ev["keyword_ms"] = round((time.perf_counter() - started_kw) * 1000, 1)
    trace["keyword"] = {
        "ms": ev["keyword_ms"],
        "candidates": [row_view(r, ref_now) for r in candidates],
        "strength_gate": {**gate, "applied": bool(candidates)},
        "rows_before_gate": [r[4] for r in pre_gate],
    }
    trace["ungated_rows"] = ungated
    if diag is not None:
        trace["keyword"]["project_candidates"] = [row_view(r, ref_now) for r in diag["project_rows"]]
        trace["keyword"]["global_candidates"] = [row_view(r, ref_now) for r in diag["global_rows"]]
        trace["keyword"]["dropped"] = [{"reason": why, **row_view(r, ref_now)} for why, r in diag["dropped"]]
        trace["keyword"]["sql_filtered"] = count_filtered(conn, fts_query, as_of, exclude_session)
    if kw_info.get("weak"):
        ev["outcome"] = "weak_keyword"     # replaced by semantic_hit if the fallback finds something
        ev["top_rank"] = kw_info["weak_top_rank"]
        log("weak_keyword")
    if rows:
        ev["outcome"] = "keyword_hit"
        log("keyword_hit", len(rows))
    keyword_rows = list(rows)

    # --- meaning search ----------------------------------------------------
    sem_found, reason = [], None
    sem = {"ran": False, "reason": None, "candidates": [], "error": None, "ms": None,
           "top_similarity": None, "usable_first_similarity": None, "kept_rows": []}
    trace["semantic"] = sem
    gate_in_try = False
    if semantic_fn and mode != "off":
        try:
            query_vec = None
            if mode == "always":
                if query_vec_fn is not None:
                    query_vec = query_vec_fn(prompt)
                    if query_vec is None:
                        reason = "helper_unavailable"
                        trace["helper_unavailable"] = True
                    else:
                        reason = "always"
                    run_semantic = query_vec is not None
                else:
                    reason = "always"
                    run_semantic = True
            else:
                gate_in_try = True
                trace["gate"] = {"fires": None, "reason": None, "error": None,
                                 **gate_fields(prompt, keywords, rows)}
                try:
                    run_semantic, reason = gate_fn(prompt, keywords, rows, kw_info, conn)
                except Exception as e:
                    trace["gate"]["error"] = f"{type(e).__name__}: {e}"
                    raise
                trace["gate"]["fires"], trace["gate"]["reason"] = run_semantic, reason
            sem["reason"] = reason
            if run_semantic:
                sem["ran"] = True
                started_sem = time.perf_counter()
                sem_results = semantic_fn(conn, config, prompt, project=None,
                                          limit=SEMANTIC_CANDIDATES, query_vec=query_vec)
                ev["semantic_ms"] = round((time.perf_counter() - started_sem) * 1000, 1)
                sem["ms"] = ev["semantic_ms"]
                if sem_results:
                    ev["top_similarity"] = round(sem_results[0]["similarity"], 4)
                    sem["top_similarity"] = sem_results[0]["similarity"]
                sem["candidates"] = classify_semantic(sem_results, project, exclude_session)
                usable = sem_results
                if row_filter is not None:
                    usable = [r for r in sem_results if row_filter("semantic", r)]
                    sem["usable_first_similarity"] = usable[0]["similarity"] if usable else None
                sem_found = semantic_rows(usable, project, exclude_session)
        except Exception as e:
            sem["error"] = f"{type(e).__name__}: {e}"
    elif not rows and not semantic_fn:
        ev["outcome"] = "no_match_semantic_unavailable"
        log("no_match_semantic_unavailable")
    ev["semantic_reason"] = reason
    sem["reason"] = reason
    sem["kept_rows"] = [r[4] for r in sem_found]
    sem["unavailable"] = not semantic_fn
    if sem_found:
        ev["outcome"] = "hybrid_hit" if rows else "semantic_hit"
        rows = merge_results(rows, sem_found)
        log(ev["outcome"], len(rows))
        trace["merge"] = merge_view(keyword_rows, sem_found, rows)

    # --- on_miss gate, for the record, in every mode the hook doesn't act on it
    if not gate_in_try:
        trace["gate"] = gate_details(prompt, keywords, keyword_rows, kw_info, conn, gate_fn)

    # What the eval reads to score every method from one pass (remember_eval.py).
    trace["keyword_rows"] = keyword_rows
    trace["semantic_rows"] = sem_found

    # --- result -------------------------------------------------------------
    if not rows:
        ev.setdefault("outcome", "no_match")
        log("no_match")
        trace["outcome"] = ev["outcome"]
        return Retrieval([], trace)

    _note_results(ev, rows, keywords, project)
    context = format_memories(rows)
    ev["injected_chars"] = len(context)
    trace["outcome"] = ev.get("outcome")
    trace["rows"] = [row_view(r, ref_now) for r in rows]
    if "merge" not in trace:
        trace["merge"] = merge_view(keyword_rows, [], rows)
    trace["injected_text"] = context
    trace["injected_chars"] = len(context)
    trace["tokens_est"] = len(context) // CHARS_PER_TOKEN
    return Retrieval(rows, trace)
